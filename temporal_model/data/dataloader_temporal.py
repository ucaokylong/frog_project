import h5py
import numpy as np
import math
import torch
from torch.utils.data import Dataset

# =========================================================================
# 1. HÀM TOÁN HỌC VÀ TIỀN XỬ LÝ
# =========================================================================

if "clip" in dir(np.core.umath):
    _clip = np.core.umath.clip
else:
    _clip = np.clip

def scans_to_cutout(
    scans, scan_phi, stride=1, centered=True, fixed=False,
    window_width=1.0, window_depth=0.5, num_cutout_pts=56,
    padding_val=29.99, area_mode=True,
):
    """ Ép mật độ 720 khung hình về chuẩn 56 điểm """
    num_scans, num_pts = scans.shape
    dists = scans[:, ::stride] if fixed else np.tile(scans[-1, ::stride], num_scans).reshape(num_scans, -1)
    half_alpha = np.arctan(0.5 * window_width / np.maximum(dists, 1e-2))

    delta_alpha = 2.0 * half_alpha / (num_cutout_pts - 1)
    ang_ct = scan_phi[::stride] - half_alpha + np.arange(num_cutout_pts).reshape(num_cutout_pts, 1, 1) * delta_alpha
    ang_ct = (ang_ct + np.pi) % (2.0 * np.pi) - np.pi  
    inds_ct = (ang_ct - scan_phi[0]) / (scan_phi[1] - scan_phi[0])
    outbound_mask = np.logical_or(inds_ct < 0, inds_ct > num_pts - 1)

    inds_ct_low = _clip(np.floor(inds_ct), 0, num_pts - 1).astype(np.int32)
    inds_ct_high = _clip(inds_ct_low + 1, 0, num_pts - 1).astype(np.int32)
    inds_ct_ratio = _clip(inds_ct - inds_ct_low, 0.0, 1.0)
    inds_offset = np.arange(num_scans).reshape(1, num_scans, 1) * num_pts
    
    ct_low = np.take(scans, inds_ct_low + inds_offset)
    ct_high = np.take(scans, inds_ct_high + inds_offset)
    ct = ct_low + inds_ct_ratio * (ct_high - ct_low)

    if area_mode:
        num_pts_in_window = inds_ct[-1] - inds_ct[0]
        area_mask = num_pts_in_window > num_cutout_pts
        if np.sum(area_mask) > 0:
            s_area = int(math.ceil(np.max(num_pts_in_window) / num_cutout_pts))
            num_ct_pts_area = s_area * num_cutout_pts
            delta_alpha_area = 2.0 * half_alpha / (num_ct_pts_area - 1)
            ang_ct_area = scan_phi[::stride] - half_alpha + np.arange(num_ct_pts_area).reshape(num_ct_pts_area, 1, 1) * delta_alpha_area
            ang_ct_area = (ang_ct_area + np.pi) % (2.0 * np.pi) - np.pi
            inds_ct_area = np.rint(_clip((ang_ct_area - scan_phi[0]) / (scan_phi[1] - scan_phi[0]), 0, num_pts - 1)).astype(np.int32)
            ct_area = np.take(scans, inds_ct_area + inds_offset).reshape(num_cutout_pts, s_area, num_scans, dists.shape[1]).mean(axis=1)
            ct[:, area_mask] = ct_area[:, area_mask]

    ct[outbound_mask] = padding_val
    ct = _clip(ct, dists - window_depth, dists + window_depth)
    if centered:
        ct = (ct - dists) / window_depth

    return np.ascontiguousarray(ct.transpose((2, 1, 0)), dtype=np.float32)

def global_to_canonical(scan_r, scan_phi, dets_r, dets_phi):
    """ Chuyển đổi Tọa độ Toàn cục (Global) -> Địa phương (Canonical) """
    dx = np.sin(dets_phi - scan_phi) * dets_r
    dy = np.cos(dets_phi - scan_phi) * dets_r - scan_r
    return dx, dy

# =========================================================================
# 2. LOADER & DATASET CORE
# =========================================================================

class FrogDataLoader:
    def __init__(self, filename, min_people=0):
        f = h5py.File(filename, 'r')
        self.f = f
        self.scans       = f['scans']
        self.timestamps  = f['timestamps'][:]
        self.circle_idxs = f['circle_idx'][:]
        self.circle_nums = f['circle_num'][:]
        self.circles     = f['circles']
        split            = f.get('split')
        self.split = split[:] if split is not None else None

        self.SCAN_WIDTH = self.scans.shape[1]
        if self.SCAN_WIDTH == 450:
            self.SCAN_NEAR, self.SCAN_FAR = 0.2, 15.0
            self.SCAN_FOV = np.radians(225)
            self.HARDCODED_PERSON_RADIUS = 0.35
        elif self.SCAN_WIDTH == 720:
            self.SCAN_NEAR, self.SCAN_FAR = 0.2, 10.0
            self.SCAN_FOV = np.radians(180)
            self.HARDCODED_PERSON_RADIUS = 0.4
        else:
            raise Exception("Unknown dataset structure")

        self.selection = np.nonzero(self.circle_nums >= min_people)[0]
        self.SCAN_ANGMIN, self.SCAN_ANGMAX = -self.SCAN_FOV/2, +self.SCAN_FOV/2
        self.SCAN_ANGLES = np.linspace(self.SCAN_ANGMIN, self.SCAN_ANGMAX, self.SCAN_WIDTH, endpoint=False, dtype=np.float32)
        self.SCAN_POINTS = np.stack([np.cos(self.SCAN_ANGLES), np.sin(self.SCAN_ANGLES)], axis=-1)

    def get_split(self, split_id):
        return np.nonzero(self.split[self.selection] == split_id)[0] if self.split is not None else np.arange(len(self.selection))

    def __len__(self): return self.selection.shape[0]

    def __getitem__(self, i):
        idx = self.selection[i]
        scan = np.nan_to_num(self.scans[idx, :], posinf=100.0)
        circles = self.circles[self.circle_idxs[idx]: self.circle_idxs[idx] + self.circle_nums[idx]]
        return scan, circles


class TemporalLocDataset(Dataset):
    """ 
    Dataset Localization chuyên dụng cho CUTOUT + CANONICAL
    Trả về:
      - scans_tensor: (T, 720, 1, 56)
      - target: (720, 3) -> [objectness, local_dx, local_dy]
    """
    def __init__(self, loader, split=0, T=5):
        super().__init__()
        self.loader = loader
        self.T = T  
        self.indices = loader.get_split(split).astype(int)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        current_idx = self.indices[idx]
        
        # 1. Thu thập chuỗi T frames
        scans_seq = []
        for t in range(self.T):
            lookback_idx = max(0, current_idx - (self.T - 1) + t)
            scan, _ = self.loader[lookback_idx]
            
            ct = scans_to_cutout(
                scan[None, ...], self.loader.SCAN_ANGLES,
                stride=1, centered=True, fixed=True,
                window_width=1.0, window_depth=0.5,
                num_cutout_pts=56, padding_val=29.99, area_mode=True,
            )
            ct_tensor = torch.from_numpy(ct).float() 
            scans_seq.append(ct_tensor)
            
        scans_tensor = torch.stack(scans_seq, dim=0) # (T, 720, 1, 56)
        
        # 2. Xử lý Ground Truth cho Frame hiện tại
        scan, gt_circles = self.loader[current_idx]
        gt_centers_xy = gt_circles[:, 0:2]
        scan_xy = scan[:, None] * self.loader.SCAN_POINTS # (720, 2)
        
        target = np.zeros((self.loader.SCAN_WIDTH, 3), dtype=np.float32)
        
        if gt_centers_xy.shape[0] > 0:
            # Đo khoảng cách từ từng tia quét tới từng người
            dist = np.hypot(*(scan_xy[:, None, :] - gt_centers_xy[None, :, :]).T).T
            min_dist = np.min(dist, axis=1)
            gt_assign = np.argmin(dist, axis=1) # Tìm người gần nhất với mỗi tia
            
            # ĐÃ SỬA: Loại bỏ nhãn -1.0. Dùng Hard Threshold dứt khoát 1.0 và 0.0 chuẩn DR-SPAAM
            target[:, 0] = (min_dist <= self.loader.HARDCODED_PERSON_RADIUS).astype(np.float32)
            
            # Tính nhãn Regression (dx, dy) theo hệ tọa độ CANONICAL
            scan_r = scan
            scan_phi = self.loader.SCAN_ANGLES
            
            assigned_gt_xy = gt_centers_xy[gt_assign]
            assigned_gt_r = np.hypot(assigned_gt_xy[:, 0], assigned_gt_xy[:, 1])
            assigned_gt_phi = np.arctan2(assigned_gt_xy[:, 1], assigned_gt_xy[:, 0])
            
            dx, dy = global_to_canonical(scan_r, scan_phi, assigned_gt_r, assigned_gt_phi)
            
            # Scale lên 10 lần để Gradient ổn định (model dễ học hơn)
            target[:, 1] = dx * 10.0
            target[:, 2] = dy * 10.0

        return scans_tensor, torch.from_numpy(target)

def temporal_collate(batch):
    scans = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    return scans, targets
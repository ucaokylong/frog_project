import h5py
import numpy as np
import math
import torch
from torch.utils.data import Dataset

# =========================================================================
# 1. HÀM TIỀN XỬ LÝ CUTOUT (Được nhúng trực tiếp, không cần import)
# =========================================================================

if "clip" in dir(np.core.umath):
    _clip = np.core.umath.clip
else:
    _clip = np.clip

def scans_to_cutout(
    scans,
    scan_phi,
    stride=1,
    centered=True,
    fixed=False,
    window_width=1.66,
    window_depth=1.0,
    num_cutout_pts=48,
    padding_val=29.99,
    area_mode=False,
):
    num_scans, num_pts = scans.shape

    # size (width) of the window
    dists = (
        scans[:, ::stride]
        if fixed
        else np.tile(scans[-1, ::stride], num_scans).reshape(num_scans, -1)
    )
    half_alpha = np.arctan(0.5 * window_width / np.maximum(dists, 1e-2))

    # cutout indices
    delta_alpha = 2.0 * half_alpha / (num_cutout_pts - 1)
    ang_ct = (
        scan_phi[::stride]
        - half_alpha
        + np.arange(num_cutout_pts).reshape(num_cutout_pts, 1, 1) * delta_alpha
    )
    ang_ct = (ang_ct + np.pi) % (2.0 * np.pi) - np.pi  # warp angle
    inds_ct = (ang_ct - scan_phi[0]) / (scan_phi[1] - scan_phi[0])
    outbound_mask = np.logical_or(inds_ct < 0, inds_ct > num_pts - 1)

    # cutout (linear interp)
    inds_ct_low = _clip(np.floor(inds_ct), 0, num_pts - 1).astype(np.int32)
    inds_ct_high = _clip(inds_ct_low + 1, 0, num_pts - 1).astype(np.int32)
    inds_ct_ratio = _clip(inds_ct - inds_ct_low, 0.0, 1.0)
    inds_offset = (
        np.arange(num_scans).reshape(1, num_scans, 1) * num_pts
    )  # because np.take flattens array
    ct_low = np.take(scans, inds_ct_low + inds_offset)
    ct_high = np.take(scans, inds_ct_high + inds_offset)
    ct = ct_low + inds_ct_ratio * (ct_high - ct_low)

    # use area sampling for down-sampling (close points)
    if area_mode:
        num_pts_in_window = inds_ct[-1] - inds_ct[0]
        area_mask = num_pts_in_window > num_cutout_pts
        if np.sum(area_mask) > 0:
            s_area = int(math.ceil(np.max(num_pts_in_window) / num_cutout_pts))
            num_ct_pts_area = s_area * num_cutout_pts
            delta_alpha_area = 2.0 * half_alpha / (num_ct_pts_area - 1)
            ang_ct_area = (
                scan_phi[::stride]
                - half_alpha
                + np.arange(num_ct_pts_area).reshape(num_ct_pts_area, 1, 1)
                * delta_alpha_area
            )
            ang_ct_area = (ang_ct_area + np.pi) % (2.0 * np.pi) - np.pi
            inds_ct_area = (ang_ct_area - scan_phi[0]) / (scan_phi[1] - scan_phi[0])
            inds_ct_area = np.rint(_clip(inds_ct_area, 0, num_pts - 1)).astype(np.int32)
            ct_area = np.take(scans, inds_ct_area + inds_offset)
            ct_area = ct_area.reshape(
                num_cutout_pts, s_area, num_scans, dists.shape[1]
            ).mean(axis=1)
            ct[:, area_mask] = ct_area[:, area_mask]

    # normalize cutout
    ct[outbound_mask] = padding_val
    ct = _clip(ct, dists - window_depth, dists + window_depth)
    if centered:
        ct = ct - dists
        ct = ct / window_depth

    return np.ascontiguousarray(
        ct.transpose((2, 1, 0)), dtype=np.float32
    )  # (scans, times, cutouts)


# =========================================================================
# 2. LOADER CORE
# =========================================================================

class FrogDataLoader:
    def __init__(self, filename, min_people=0, points_per_sector=6):
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


def circle_overlap(circles1, circles2, radius):
    if circles2.shape[0] == 0: return np.zeros((circles1.shape[0], 0), dtype=np.float32)
    diff = circles1[:, None, :] - circles2[None, :, :]
    distances = np.hypot(diff[:, :, 0], diff[:, :, 1])
    return np.maximum(0.0, 1.0 - distances / (2 * radius))


# =========================================================================
# 3. DATASETS TÍCH HỢP CUTOUT
# =========================================================================

class TemporalSegDataset(Dataset):
    """ 
    Dataset dành cho bài toán phân đoạn (Segmentation) sử dụng CUTOUT.
    Trả về chuỗi T frames dưới dạng (T, 720, 1, 56). Mask: (1, 720)
    """
    def __init__(self, loader, split=0, T=5):
        super().__init__()
        self.loader = loader
        self.T = T
        self.indices = loader.get_split(split).astype(int)

    def __len__(self): 
        return len(self.indices)

    def __getitem__(self, idx):
        current_idx = self.indices[idx]
        
        # 1. Thu thập chuỗi T frames
        scans_seq = []
        for t in range(self.T):
            lookback_idx = max(0, current_idx - (self.T - 1) + t)
            scan, _ = self.loader[lookback_idx]
            
            # Sử dụng Cutout
            ct = scans_to_cutout(
                scan[None, ...],
                self.loader.SCAN_ANGLES,
                stride=1, centered=True, fixed=True,
                window_width=1.0, window_depth=0.5,
                num_cutout_pts=56, padding_val=29.99, area_mode=True,
            )
            
            ct_tensor = torch.from_numpy(ct).float() # (720, 1, 56)
            scans_seq.append(ct_tensor)
            
        scans_tensor = torch.stack(scans_seq, dim=0) # (T, 720, 1, 56)
        
        # 2. Tính GT (Mask) cho frame hiện tại
        scan, gt_circles = self.loader[current_idx]
        scan_xy = scan[:, None] * self.loader.SCAN_POINTS
        
        gt_people_xy = gt_circles[:, 0:2]
        gt_people_r  = gt_circles[:, 2]
        gt_people_d  = gt_circles[:, 3]

        removed = np.nonzero(gt_people_d > self.loader.SCAN_FAR)[0]
        point_dist = np.hypot(*(scan_xy[:, None, :] - gt_people_xy[None, :, :]).T).T
        if removed.size > 0: 
            point_dist[:, removed] = 100.0

        seg_gt = np.any(point_dist <= gt_people_r[None, :], axis=1).astype(np.float32)
        
        return scans_tensor, torch.from_numpy(seg_gt).unsqueeze(0)


class TemporalLocDataset(Dataset):
    """ 
    Dataset dành cho bài toán Localization sử dụng CUTOUT.
    Trả về chuỗi T frames (T, 720, 1, 56). Target: (720, 3)
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
            
            # Sử dụng Cutout
            ct = scans_to_cutout(
                scan[None, ...],
                self.loader.SCAN_ANGLES,
                stride=1, centered=True, fixed=True,
                window_width=1.0, window_depth=0.5,
                num_cutout_pts=56, padding_val=29.99, area_mode=True,
            )
            
            ct_tensor = torch.from_numpy(ct).float() # (720, 1, 56)
            scans_seq.append(ct_tensor)
            
        scans_tensor = torch.stack(scans_seq, dim=0) # (T, 720, 1, 56)
        
        # 2. Xử lý Ground Truth (720 tia)
        scan, gt_circles = self.loader[current_idx]
        gt_centers_xy = gt_circles[:, 0:2]
        scan_xy = scan[:, None] * self.loader.SCAN_POINTS # (720, 2)
        
        # Mặc định tất cả là 0.0 (Negative)
        target = np.zeros((self.loader.SCAN_WIDTH, 3), dtype=np.float32)
        target[:, 0] = 0.0 
        
        if gt_centers_xy.shape[0] > 0:
            target[:, 0] = -1.0 # Tạm thời gán -1 (Ignore) cho cảnh có người
            
            dist = np.hypot(*(scan_xy[:, None, :] - gt_centers_xy[None, :, :]).T).T
            min_dist = np.min(dist, axis=1)
            gt_assign = np.argmin(dist, axis=1)
            
            # Phân loại Positive và Negative
            target[min_dist < 0.35, 0] = 1.0 
            target[min_dist > 0.50, 0] = 0.0 
            
            # Tính khoảng cách (Offset) từ tâm người tới tia quét
            target[:, 1:3] = gt_centers_xy[gt_assign, :] - scan_xy

        return scans_tensor, torch.from_numpy(target)


def temporal_collate(batch):
    """ Gom batch cho dữ liệu chuỗi thời gian. Dùng chung được cho cả Seg và Loc """
    scans = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    return scans, targets
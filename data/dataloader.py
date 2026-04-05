import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class FrogDataLoader:
    """
    Xử lý nạp dữ liệu thô từ file .h5 và cấu hình lưới Anchor.
    """
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
        # Tự động nhận diện cấu hình cảm biến (DROW vs FROG)
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

        # Cấu hình Grid Anchor cho Localization
        if points_per_sector > 0:
            self.NUM_SECTORS = self.SCAN_WIDTH // points_per_sector
            self.NUM_ANCHORS_PER_SECTOR = int((self.SCAN_FAR - self.SCAN_NEAR) / (0.8 * self.HARDCODED_PERSON_RADIUS))
            self.ANCHOR_DEPTH = (self.SCAN_FAR - self.SCAN_NEAR) / self.NUM_ANCHORS_PER_SECTOR
            self.ANCHOR_REGRESSION_SCALE = 1.0 / self.ANCHOR_DEPTH
            
            g_dist = np.linspace(self.SCAN_NEAR + self.ANCHOR_DEPTH/2, self.SCAN_FAR - self.ANCHOR_DEPTH/2, self.NUM_ANCHORS_PER_SECTOR)
            self.SECTOR_AMPL = self.SCAN_FOV / self.NUM_SECTORS
            g_ang = np.linspace(self.SCAN_ANGMIN + self.SECTOR_AMPL/2, self.SCAN_ANGMAX - self.SECTOR_AMPL/2, self.NUM_SECTORS)

            dist_grid, ang_grid = np.meshgrid(g_dist, g_ang)
            self.grid_polar = np.stack((dist_grid, ang_grid), axis=2)
            self.grid_xy = np.stack((dist_grid * np.cos(ang_grid), dist_grid * np.sin(ang_grid)), axis=2)

    def get_split(self, split_id):
        return np.nonzero(self.split[self.selection] == split_id)[0] if self.split is not None else np.arange(len(self.selection))

    def normalize_scan(self, scan):
        return 1.0 - np.clip(scan, self.SCAN_NEAR, self.SCAN_FAR) / self.SCAN_FAR

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

class LocDataset(Dataset):
    """ Dataset dành cho bài toán định vị mỏ neo (Localization) """
    def __init__(self, loader, split=0, overlap_threshold=0.25):
        super().__init__()
        self.loader = loader
        self.overlap_threshold = overlap_threshold
        self.indices = loader.get_split(split).astype(int)
        self.anchor_xy = loader.grid_xy.reshape(-1, 2)
        self.anchor_polar = loader.grid_polar.reshape(-1, 2)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        sel_idx = self.indices[idx]
        scan, gt_circles = self.loader[sel_idx]
        gt_centers_xy = gt_circles[:, 0:2]
        gt_centers_polar = gt_circles[:, 3:5]

        overlaps = circle_overlap(self.anchor_xy, gt_centers_xy, self.loader.HARDCODED_PERSON_RADIUS)

        if gt_centers_xy.shape[0] == 0:
            objectness = np.zeros(self.anchor_xy.shape[0], dtype=np.float32)
            regr_target = np.zeros_like(self.anchor_polar, dtype=np.float32)
        else:
            max_ov = np.max(overlaps, axis=1)
            gt_assign = np.argmax(overlaps, axis=1)
            
            # Đảm bảo gán anchor tốt nhất cho mỗi GT
            max_ov_gt = np.max(overlaps, axis=0)
            best_anchor_idxs, _ = np.where(overlaps == max_ov_gt)
            
            objectness = np.full(self.anchor_xy.shape[0], -1.0, dtype=np.float32)
            objectness[max_ov < self.overlap_threshold] = 0.0
            objectness[max_ov >= self.overlap_threshold] = 1.0
            objectness[np.unique(best_anchor_idxs)] = 1.0

            regr_target = gt_centers_polar[gt_assign, :] - self.anchor_polar
            regr_target[:, 1] *= self.anchor_polar[:, 0] # Chuyển góc sang cung tròn

        grid_gt = np.empty((self.anchor_xy.shape[0], 3), dtype=np.float32)
        grid_gt[:, 0] = objectness
        grid_gt[:, 1:3] = self.loader.ANCHOR_REGRESSION_SCALE * (grid_gt[:, 0:1] == 1.0) * regr_target
        grid_gt = grid_gt.reshape(self.loader.NUM_SECTORS, self.loader.NUM_ANCHORS_PER_SECTOR, 3)

        scan_norm = torch.from_numpy(self.loader.normalize_scan(scan).astype(np.float32)).unsqueeze(0)
        return scan_norm, torch.from_numpy(grid_gt)

class SegDataset(Dataset):
    """ Dataset dành cho bài toán phân đoạn tia quét (Segmentation) """
    def __init__(self, loader, split=0):
        super().__init__()
        self.loader = loader
        self.indices = loader.get_split(split).astype(int)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        sel_idx = self.indices[idx]
        scan, gt_circles = self.loader[sel_idx]
        scan_xy = scan[:, None] * self.loader.SCAN_POINTS
        
        gt_people_xy = gt_circles[:, 0:2]
        gt_people_r  = gt_circles[:, 2]
        gt_people_d  = gt_circles[:, 3]

        # Mask bỏ qua những người ở quá xa giới hạn cảm biến
        removed = np.nonzero(gt_people_d > self.loader.SCAN_FAR)[0]
        point_dist = np.hypot(*(scan_xy[:, None, :] - gt_people_xy[None, :, :]).T).T
        if removed.size > 0: point_dist[:, removed] = 100.0

        seg_gt = np.any(point_dist <= gt_people_r[None, :], axis=1).astype(np.float32)
        scan_norm = torch.from_numpy(self.loader.normalize_scan(scan).astype(np.float32)).unsqueeze(0)
        return scan_norm, torch.from_numpy(seg_gt).unsqueeze(0)

# Các hàm hỗ trợ gom batch
def seg_collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

def loc_collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])
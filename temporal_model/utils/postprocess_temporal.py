import numpy as np
import cv2
from scipy.ndimage import maximum_filter

def canonical_to_global(scan_r, scan_phi, dx, dy):
    """ Chuyển đổi từ tọa độ địa phương (Canonical) về tọa độ toàn cục (Polar) """
    tmp_y = scan_r + dy
    tmp_phi = np.arctan2(dx, tmp_y)
    dets_phi = tmp_phi + scan_phi
    dets_r = tmp_y / np.cos(tmp_phi)
    return dets_r, dets_phi

def rphi_to_xy(r, phi):
    """ Chuyển tọa độ Cực sang Descartes (X, Y) """
    # ĐÃ SỬA LỖI CHÍ MẠNG: Dùng cos cho X và sin cho Y để khớp định dạng Ground Truth
    return r * np.cos(phi), r * np.sin(phi)

def parse_loc_voting(loader, locdata, raw_scan, min_thresh=0.01):
    """
    Thuật toán Voting NMS (Gom cụm và cộng dồn điểm mật độ) chuẩn của DR-SPAAM.
    - locdata: (720, 3) -> [objectness, dx_local, dy_local]
    - raw_scan: (720,) -> Bản đồ khoảng cách của frame hiện tại
    """
    # 1. Sigmoid đưa về xác suất
    pred_cls = 1.0 / (1.0 + np.exp(-locdata[:, 0]))  # Shape: (720,)
    
    # 2. Hoàn tác Scale (Chia cho 10.0 như đã cấu hình trong Dataloader)
    pred_reg = locdata[:, 1:3] / 10.0

    scan_r = raw_scan
    scan_phi = loader.SCAN_ANGLES

    # 3. Phục hồi tọa độ toàn cục (Global XY)
    pred_r, pred_phi = canonical_to_global(scan_r, scan_phi, pred_reg[:, 0], pred_reg[:, 1])
    pred_xs, pred_ys = rphi_to_xy(pred_r, pred_phi)

    # 4. Thông số lưới Grid chuẩn Benchmark
    bin_size = 0.1
    blur_sigma = 0.5
    
    # ĐÃ SỬA LỖI CHÍ MẠNG: Mở rộng lưới thành hình vuông 30m x 30m 
    # Để đảm bảo dù X hay Y là trục tiến (depth) hay ngang (lateral) thì cũng không bị cắt lẹm dữ liệu
    x_min, x_max = -15.0, 15.0
    y_min, y_max = -15.0, 15.0
    vote_collect_radius = 0.3

    # Lọc bỏ các tia có xác suất thấp hơn ngưỡng (Tối ưu tính toán)
    voters_inds = np.where(pred_cls >= min_thresh)[0]
    if len(voters_inds) == 0:
        return []

    pred_xs, pred_ys = pred_xs[voters_inds], pred_ys[voters_inds]
    pred_cls = pred_cls[voters_inds]

    # Khởi tạo không gian Lưới (Voting Grid)
    x_range = int((x_max - x_min) / bin_size)
    y_range = int((y_max - y_min) / bin_size)
    grid = np.zeros((x_range, y_range), dtype=np.float32)

    pred_x_inds = np.int64((pred_xs - x_min) / bin_size)
    pred_y_inds = np.int64((pred_ys - y_min) / bin_size)

    # Bỏ qua các dự đoán nằm ngoài Lưới
    mask = (0 <= pred_x_inds) & (pred_x_inds < x_range) & (0 <= pred_y_inds) & (pred_y_inds < y_range)
    pred_x_inds, pred_xs = pred_x_inds[mask], pred_xs[mask]
    pred_y_inds, pred_ys = pred_y_inds[mask], pred_ys[mask]
    pred_cls = pred_cls[mask]

    if len(pred_cls) == 0:
        return []

    # =======================================================
    # BƯỚC QUAN TRỌNG NHẤT: Bỏ phiếu cộng dồn (Density Voting)
    # =======================================================
    np.add.at(grid, (pred_x_inds, pred_y_inds), pred_cls)

    # Làm mờ Gauss để tạo các "đỉnh" mật độ không gian
    if blur_sigma > 0:
        blur_win = int(2 * ((blur_sigma * 5) // 2) + 1)
        grid = cv2.GaussianBlur(grid, (blur_win, blur_win), blur_sigma)
    
    # Tìm đỉnh (Local Maxima)
    grid_nms_val = maximum_filter(grid, size=3)
    grid_nms_inds = (grid == grid_nms_val) & (grid > 0)
    nms_xs, nms_ys = np.where(grid_nms_inds)

    if len(nms_xs) == 0:
        return []

    # Đưa tọa độ đỉnh từ Lưới trở về Hệ tọa độ thực (mét)
    nms_xs = nms_xs * bin_size + x_min + bin_size / 2
    nms_ys = nms_ys * bin_size + y_min + bin_size / 2

    # Gộp các tia vào đỉnh gần nhất và tính trung bình
    distance_to_center = np.hypot(pred_xs - nms_xs[:, None], pred_ys - nms_ys[:, None])
    detection_ids = np.argmin(distance_to_center, axis=0)

    results = []
    for ipeak in range(len(nms_xs)):
        voter_inds = np.where(detection_ids == ipeak)[0]
        voter_inds = voter_inds[distance_to_center[ipeak, voter_inds] < vote_collect_radius]
        
        if len(voter_inds) == 0:
            continue
            
        # Tính trung bình cộng tọa độ và điểm số của tất cả các tia trúng 1 người
        avg_x = np.mean(pred_xs[voter_inds])
        avg_y = np.mean(pred_ys[voter_inds])
        avg_score = np.mean(pred_cls[voter_inds])
        
        results.append((avg_score, avg_x, avg_y))

    return results
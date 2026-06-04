import numpy as np
from tqdm import tqdm

def proper_ap(recs, precs):
    """ 
    All-point interpolated AP (AUC - VOC 2010+ style).
    Tính diện tích chính xác dưới đường cong Precision-Recall 
    bằng cách quét qua tất cả các điểm thay vì chỉ lấy 11 mốc cố định.
    """
    # Thêm các điểm biên an toàn
    mrec = np.concatenate(([0.], recs, [1.]))
    mpre = np.concatenate(([0.], precs, [0.]))

    # Tạo đường bao bậc thang (Envelope) từ phải sang trái để làm mượt đường cong
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Tìm các vị trí mà Recall có sự thay đổi giá trị
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Tính tổng diện tích của các hình chữ nhật bậc thang
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def eer(recs, precs):
    """ Equal Error Rate: điểm P xấp xỉ R. """
    if len(precs) == 0: return 0.0
    idx = np.argmin(np.abs(precs - recs))
    return (precs[idx] + recs[idx]) / 2.0

def compute_pr_curve_expert(loader_frog, all_results, assoc_distance=0.5):
    """
    Tính PR Curve theo phong cách Global Sort của tác giả.
    all_results: list of (idx_scan, people_array)
    """
    all_scores, all_tp_fp = [], []
    total_gt = loader_frog.circles.shape[0] # Tổng số người thực tế trong file H5

    for idx_scan, people in tqdm(all_results, desc=f"Benchmark matching (d={assoc_distance}m)"):
        scan_idx = loader_frog.selection[idx_scan]
        start = loader_frog.circle_idxs[scan_idx]
        num   = loader_frog.circle_nums[scan_idx]
        gt_xy = loader_frog.circles[start:start+num, 0:2]

        if people is None or len(people) == 0: continue

        # Sort detections theo score giảm dần
        people = people[np.argsort(-people[:, 0])]
        pred_scores, pred_xy = people[:, 0], people[:, 1:3]
        used_gt = np.zeros(len(gt_xy), dtype=bool)

        for i_p in range(len(pred_xy)):
            all_scores.append(pred_scores[i_p])
            if len(gt_xy) == 0:
                all_tp_fp.append(0)
                continue

            dists = np.linalg.norm(gt_xy - pred_xy[i_p], axis=1)
            idx_min = np.argmin(dists)
            if dists[idx_min] < assoc_distance and not used_gt[idx_min]:
                all_tp_fp.append(1)
                used_gt[idx_min] = True
            else:
                all_tp_fp.append(0)

    # Tính toán đường cong toàn cục
    all_scores, all_tp_fp = np.array(all_scores), np.array(all_tp_fp)
    
    # Bắt lỗi nếu model không dự đoán được bất kỳ object nào
    if len(all_scores) == 0:
        return np.array([0.0]), np.array([0.0]), 0.0, 0.0, 0.0, 0, 0, total_gt, 0

    sort_idx = np.argsort(-all_scores)
    tp_fp_sorted = all_tp_fp[sort_idx]

    tp_cumsum = np.cumsum(tp_fp_sorted)
    precisions = tp_cumsum / np.arange(1, len(tp_cumsum) + 1)
    recalls = tp_cumsum / (total_gt + 1e-8)

    # ĐÃ SỬA: Gọi proper_ap All-point (đã loại bỏ tham số points=11)
    ap = proper_ap(recalls, precisions)
    eer_val = eer(recalls, precisions)
    f1_scores = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-8, 2.0)
    
    # Compute TP, FP, FN
    tp = int(np.sum(all_tp_fp))
    fp = len(all_tp_fp) - tp
    fn = total_gt - tp
    tn = 0  
    
    return recalls, precisions, ap, eer_val, np.max(f1_scores), tp, fp, fn, tn

def compute_mean_metrics(metrics_dict):
    """ Tính trung bình (Mean) cho AP, EER, và Peak F1 từ các mức khoảng cách. """
    aps, eers, f1s = [], [], []
    for dist, data in metrics_dict.items():
        aps.append(data["AP"])
        eers.append(data["EER"])
        f1s.append(data["F1"])
        
    return {
        "mAP": np.mean(aps) if aps else 0.0,
        "mEER": np.mean(eers) if eers else 0.0,
        "mF1": np.mean(f1s) if f1s else 0.0
    }
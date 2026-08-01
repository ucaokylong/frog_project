import numpy as np
from tqdm import tqdm

def proper_ap(recs, precs, points=11):
    """ 11-point interpolated AP matching the benchmark standard. """
    new_r = np.linspace(0.0, 1.0, num=points, endpoint=True)
    new_p = []
    for r_thr in new_r:
        idxs = np.nonzero(recs >= r_thr)[0]
        new_p.append(np.max(precs[idxs]) if len(idxs) > 0 else 0.0)
    return np.mean(new_p)

def eer(recs, precs):
    """ Equal Error Rate: point where Precision approximately equals Recall. """
    if len(precs) == 0: return 0.0
    idx = np.argmin(np.abs(precs - recs))
    return (precs[idx] + recs[idx]) / 2.0

def compute_pr_curve_expert(loader_frog, all_results, assoc_distance=0.5):
    """
    Computes the PR Curve using the author's Global Sort methodology 
    along with supplementary advanced quantitative evaluation metrics.
    """
    all_scores, all_tp_fp = [], []
    total_gt = loader_frog.circles.shape[0] # Total actual ground truth targets in the H5 file

    for idx_scan, people in tqdm(all_results, desc=f"Benchmark matching (d={assoc_distance}m)"):
        scan_idx = loader_frog.selection[idx_scan]
        start = loader_frog.circle_idxs[scan_idx]
        num   = loader_frog.circle_nums[scan_idx]
        gt_xy = loader_frog.circles[start:start+num, 0:2]

        if people is None or len(people) == 0: continue

        # Sort detections in descending order of confidence scores
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

    # Compute global curves
    all_scores, all_tp_fp = np.array(all_scores), np.array(all_tp_fp)
    sort_idx = np.argsort(-all_scores)
    tp_fp_sorted = all_tp_fp[sort_idx]

    tp_cumsum = np.cumsum(tp_fp_sorted)
    precisions = tp_cumsum / np.arange(1, len(tp_cumsum) + 1)
    recalls = tp_cumsum / (total_gt + 1e-8)

    ap = proper_ap(recalls, precisions)
    eer_val = eer(recalls, precisions)
    f1_scores = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-8, 2.0)
    
    # Compute standard confusion metrics
    tp = int(np.sum(all_tp_fp))
    fp = len(all_tp_fp) - tp
    fn = total_gt - tp
    
    # Compute complementary metrics at the final operational cutoff
    final_precision = tp / (tp + fp + 1e-8)
    final_recall = tp / (total_gt + 1e-8)
    final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-8)
    
    return recalls, precisions, ap, eer_val, np.max(f1_scores), tp, fp, fn, final_precision, final_recall, final_f1
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.dataloader_temporal import FrogDataLoader, scans_to_cutout
from models.loc_model_temporal import TemporalLocModel1D
# ĐÃ SỬA: Import hàm Voting NMS mới
from utils.postprocess_temporal import parse_loc_voting
from utils.metrics_temporal import compute_pr_curve_expert, compute_mean_metrics

# --- CẤU HÌNH ---
TEST_PATH    = "data/frog_16-41_test.h5"
WEIGHTS_PATH = "checkpoints/drspaam_loc_cutout_best_SOTA.pth"  
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_WINDOW = 5 
SCAN_FAR = 10.0 # Hạn chế tầm đánh giá theo quy ước nhãn dữ liệu FROG

def evaluate(batch_size=32):
    loader = FrogDataLoader(TEST_PATH, min_people=0)
    
    model = TemporalLocModel1D().to(DEVICE)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Model weights not found at {WEIGHTS_PATH}")
        return

    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.eval()

    all_results = []
    print(f"Running Temporal Inference with CUTOUT (T={T_WINDOW})...")

    with torch.no_grad():
        N = len(loader)
        for start in tqdm(range(0, N, batch_size), desc="Inference"):
            end = min(start + batch_size, N)
            
            batch_seqs = []
            idx_list   = []

            for i in range(start, end):
                seq = []
                for t in range(T_WINDOW):
                    lookback_idx = max(0, i - (T_WINDOW - 1) + t)
                    scan, _ = loader[lookback_idx]
                    
                    ct = scans_to_cutout(
                        scan[None, ...],
                        loader.SCAN_ANGLES,
                        stride=1, centered=True, fixed=True,
                        window_width=1.0, window_depth=0.5,
                        num_cutout_pts=56, padding_val=29.99, area_mode=True,
                    )
                    
                    ct_tensor = torch.from_numpy(ct).float()
                    seq.append(ct_tensor)
                
                batch_seqs.append(torch.stack(seq, dim=0)) 
                idx_list.append(i)

            x = torch.stack(batch_seqs, dim=0).to(DEVICE)
            preds = model(x).cpu().numpy()  # Cutout Output: (B, 720, 3)

            for b in range(preds.shape[0]):
                pred = preds[b]
                current_idx = idx_list[b]
                raw_scan, _ = loader[current_idx] 
                
                # GỌI HÀM VOTING NMS VỚI THRESHOLD 0.01 CHUẨN
                raw_people = parse_loc_voting(loader, pred, raw_scan, min_thresh=0.01)
                
                filtered_people = []
                for p in raw_people:
                    c_score, c_x, c_y = p[0], p[1], p[2]
                    c_dist = np.hypot(c_x, c_y)
                    
                    # CHỈ GIỮ LẠI dự đoán nằm trong phạm vi 10 mét
                    if c_dist <= SCAN_FAR:
                        filtered_people.append(p)
                
                if len(filtered_people) > 0:
                    filtered_people = np.array(filtered_people, dtype=np.float32)
                    filtered_people = filtered_people[np.argsort(-filtered_people[:, 0], kind='stable'), :]
                else:
                    filtered_people = np.array([], dtype=np.float32)
                    
                all_results.append((current_idx, filtered_people))

    # Benchmark
    metrics = {}
    for dist in [0.5, 0.3]:
        R, P, AP, EER, F1, TP, FP, FN, TN = compute_pr_curve_expert(loader, all_results, assoc_distance=dist)
        metrics[dist] = {"R": R, "P": P, "AP": AP, "EER": EER, "F1": F1}
        print(f"\n--- Result for Association Distance {dist}m ---")
        print(f"AP: {AP*100:.1f}% | Peak F1: {F1*100:.1f}% | EER: {EER*100:.1f}%")
        print(f"Chi tiết: TP={TP}, FP={FP}, FN={FN}")

    # Gọi hàm tính Mean Metrics và in bảng Tổng kết
    mean_res = compute_mean_metrics(metrics)
    print("\n================ TỔNG KẾT BENCHMARK (CUTOUT) ================")
    print(f"mAP      : {mean_res['mAP']*100:.2f}%")
    print(f"mPEAK F1 : {mean_res['mF1']*100:.2f}%")
    print(f"mEER     : {mean_res['mEER']*100:.2f}%")
    print("=============================================================")

    np.savez(os.path.join(RESULTS_DIR, "drspaam_cutout_test_metrics.npz"), metrics=metrics)

    plt.figure(figsize=(6, 6))
    plt.plot(metrics[0.5]["R"], metrics[0.5]["P"], label=f"d=0.5 (AP={metrics[0.5]['AP']*100:.1f}%)")
    plt.plot(metrics[0.3]["R"], metrics[0.3]["P"], label=f"d=0.3 (AP={metrics[0.3]['AP']*100:.1f}%)")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Temporal DR-SPAAM (Cutout) PR Curve")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(RESULTS_DIR, "drspaam_cutout_pr_curve.png"))
    print(f"\n>>> Results saved in {RESULTS_DIR}/")

if __name__ == "__main__":
    evaluate(batch_size=64)
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data import FrogDataLoader, LocDataset, temporal_collate
from models import LocModel1D
from utils import parse_loc, compute_pr_curve_expert


# --- CẤU HÌNH ---
PROJECT_ROOT = "/home/s2410433/frog_project"
TEST_PATH    = os.path.join(PROJECT_ROOT, "data/frog_16-41_test.h5")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints/lfe_mamba_loc_best.pth")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 5 # Độ dài chuỗi thời gian - phải khớp với lúc train
BATCH_SIZE = 64


def evaluate():
    # 1. Load Data
    print(f"--- ĐANG SETUP TEST PIPELINE (T={T}) ---")
    loader = FrogDataLoader(TEST_PATH, min_people=0)
    
    # Sử dụng LocDataset để tự động xử lý sliding window (B, T, 1, L)
    # Ở đây dùng split=0 vì thường file test chỉ có 1 split duy nhất
    test_ds = LocDataset(loader, split=0, sequence_length=T, overlap_threshold=0.35)
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=temporal_collate, 
        num_workers=4,
        pin_memory=True
    )

    # 2. Setup Model
    model = LocModel1D(
        num_anchors_per_sector=loader.NUM_ANCHORS_PER_SECTOR,
        glob=True,
        d_model=96
    ).to(DEVICE)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Model weights not found at {WEIGHTS_PATH}")
        return

    print(f"Loading weights from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    all_results = []
    print("Running Inference on Test Set...")

    # 3. Inference Loop
    with torch.no_grad():
        # idx_in_batch_global giúp map kết quả về đúng frame index trong file H5
        idx_counter = 0 
        
        for scans, _ in tqdm(test_loader, desc="Testing"):
            scans = scans.to(DEVICE) # Shape: (B, T, 1, L)
            
            # Model trả về (B, S, A, 3) cho frame cuối cùng của mỗi sequence
            preds = model(scans).cpu().numpy() 

            for b in range(preds.shape[0]):
                pred = preds[b] # (S, A, 3)
                people = parse_loc(loader, pred, threshold=0.01)
                
                # compute_pr_curve_expert cần index của mẫu trong dataset
                all_results.append((idx_counter, people))
                idx_counter += 1

    # 4. Tính toán Metrics (Benchmark tại 0.5m và 0.3m)
    metrics_log = {}
    plt.figure(figsize=(7, 7))

    for dist in [0.5, 0.3]:
        R, P, AP, EER, F1, TP, FP, FN, TN = compute_pr_curve_expert(loader, all_results, assoc_distance=dist)
        metrics_log[dist] = {"R": R, "P": P, "AP": AP, "EER": EER, "F1": F1, "TP": TP, "FP": FP, "FN": FN, "TN": TN}
        
        print(f"\n--- Kết quả với khoảng cách sai số {dist}m ---")
        print(f"AP: {AP*100:.2f}% | Peak F1: {F1*100:.2f}% | EER: {EER*100:.2f}%")
        print(f"TP: {TP} | FP: {FP} | FN: {FN} | TN: {TN}")
        
        plt.plot(R, P, label=f"d={dist}m (AP={AP*100:.1f}%)")

    # 5. Lưu và Vẽ biểu đồ
    np.savez(os.path.join(RESULTS_DIR, "test_metrics.npz"), 
             dist_05=metrics_log[0.5], 
             dist_03=metrics_log[0.3])

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Mamba-Temporal T={T})")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_DIR, "pr_curve_temporal.png")
    plt.savefig(plot_path)
    print(f"\n>>> Biểu đồ đã lưu tại: {plot_path}")


if __name__ == "__main__":
    evaluate()
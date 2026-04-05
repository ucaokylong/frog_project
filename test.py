import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import FrogDataLoader
from models import LocModel1D
from utils import parse_loc, compute_pr_curve_expert

# --- CẤU HÌNH ---
TEST_PATH = "data/frog_16-41_test.h5"
WEIGHTS_PATH = "checkpoints/lfe_ppn_best.pth"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    loader = FrogDataLoader(TEST_PATH, min_people=0)
    model = LocModel1D(num_anchors_per_sector=loader.NUM_ANCHORS_PER_SECTOR, glob=True).to(DEVICE)
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Model weights not found at {WEIGHTS_PATH}")
        return

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    all_results = []
    print("Running Inference...")
    with torch.no_grad():
        for i in tqdm(range(len(loader))):
            scan, _ = loader[i]
            x = torch.from_numpy(loader.normalize_scan(scan)).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # (1, S, A, 3) -> numpy (S, A, 3)
            pred = model(x).cpu().numpy()[0]
            
            # Merging NMS với threshold cực thấp để lấy đủ đường PR
            people = parse_loc(loader, pred, threshold=0.001)
            all_results.append((i, people))

    # Benchmark tại 2 mốc 0.5m và 0.3m
    metrics = {}
    for dist in [0.5, 0.3]:
        R, P, AP, EER, F1 = compute_pr_curve_expert(loader, all_results, assoc_distance=dist)
        metrics[dist] = {"R": R, "P": P, "AP": AP, "EER": EER, "F1": F1}
        print(f"\n--- Result for Association Distance {dist}m ---")
        print(f"AP: {AP*100:.1f}% | Peak F1: {F1*100:.1f}% | EER: {EER*100:.1f}%")

    # Lưu kết quả ra file NPZ để sau này vẽ biểu đồ
    np.savez(os.path.join(RESULTS_DIR, "test_metrics.npz"), metrics=metrics)
    
    # Vẽ biểu đồ nhanh
    plt.figure(figsize=(6,6))
    plt.plot(metrics[0.5]["R"], metrics[0.5]["P"], label=f"d=0.5 (AP={metrics[0.5]['AP']*100:.1f}%)")
    plt.plot(metrics[0.3]["R"], metrics[0.3]["P"], label=f"d=0.3 (AP={metrics[0.3]['AP']*100:.1f}%)")
    plt.plot([0,1], [0,1], 'k--', alpha=0.2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "pr_curve.png"))
    print(f"\n>>> Results saved in {RESULTS_DIR}/")

if __name__ == "__main__":
    evaluate()
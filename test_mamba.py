import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.dataloader_mamba import FrogDataLoader
from models.loc_model_mamba import LocModel1D
from utils.postprocess_mamba import parse_loc 
from utils.metrics_mamba import compute_pr_curve_expert

# --- CẤU HÌNH ---
TEST_PATH    = "data/frog_16-41_test.h5"
WEIGHTS_PATH = "checkpoints/lfe_mamba_loc_best.pth"  
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SỬA LỖI: Đồng bộ chiều dài chuỗi với lúc train Mamba (T=15)
T_WINDOW = 15 

def evaluate(batch_size=64):
    loader = FrogDataLoader(TEST_PATH, min_people=0)
    
    # SỬA LỖI: Bắt buộc phải truyền glob=True và d_model=96 để cấu trúc mạng khớp 100% với file weights
    model = LocModel1D(
        num_anchors_per_sector=loader.NUM_ANCHORS_PER_SECTOR,
        glob=True,
        d_model=96
    ).to(DEVICE)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Model weights not found at {WEIGHTS_PATH}")
        return

    # Load weights an toàn
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    # Loại bỏ tiền tố 'module.' nếu lúc train bạn dùng nn.DataParallel (Multi-GPU)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.eval()

    all_results = []
    print(f"Running Mamba Temporal Inference (T={T_WINDOW})...")

    with torch.no_grad():
        N = len(loader)
        for start in tqdm(range(0, N, batch_size), desc="Inference"):
            end = min(start + batch_size, N)
            
            batch_seqs = []
            idx_list   = []

            # 1) Gom batch bằng cách trượt cửa sổ T frames cho từng index
            for i in range(start, end):
                seq = []
                for t in range(T_WINDOW):
                    lookback_idx = max(0, i - (T_WINDOW - 1) + t)
                    scan, _ = loader[lookback_idx]
                    
                    # Mô hình Mamba dùng hàm normalize_scan gốc (End-to-End), KHÔNG dùng Cutout
                    scan_norm = loader.normalize_scan(scan).astype(np.float32)
                    seq.append(scan_norm)
                
                # Cấu trúc của seq: list của T array (L,)
                batch_seqs.append(np.stack(seq, axis=0)) # -> (T, L)
                idx_list.append(i)

            # 2) Stack toàn bộ batch: (B, T, L) -> Thêm kênh -> (B, T, 1, L)
            batch_np = np.stack(batch_seqs, axis=0)
            x = torch.from_numpy(batch_np).unsqueeze(2).to(DEVICE)

            # 3) Inference
            preds = model(x).cpu().numpy()  # Output shape: (B, S, A, 3)

            # 4) Parse kết quả
            for b in range(preds.shape[0]):
                pred = preds[b]
                people = parse_loc(loader, pred, threshold=0.001)
                all_results.append((idx_list[b], people))

    # Benchmark
    metrics = {}
    for dist in [0.5, 0.3]:
        R, P, AP, EER, F1, TP, FP, FN, TN = compute_pr_curve_expert(loader, all_results, assoc_distance=dist)
        metrics[dist] = {"R": R, "P": P, "AP": AP, "EER": EER, "F1": F1}
        print(f"\n--- Result for Association Distance {dist}m ---")
        print(f"AP: {AP*100:.1f}% | Peak F1: {F1*100:.1f}% | EER: {EER*100:.1f}%")
        print(f"Chi tiết: TP={TP}, FP={FP}, FN={FN}")

    # Lưu kết quả
    np.savez(os.path.join(RESULTS_DIR, "mamba_test_metrics.npz"), metrics=metrics)

    # Vẽ và lưu biểu đồ PR Curve
    plt.figure(figsize=(6, 6))
    plt.plot(metrics[0.5]["R"], metrics[0.5]["P"], label=f"d=0.5 (AP={metrics[0.5]['AP']*100:.1f}%)")
    plt.plot(metrics[0.3]["R"], metrics[0.3]["P"], label=f"d=0.3 (AP={metrics[0.3]['AP']*100:.1f}%)")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Mamba Temporal (T={T_WINDOW}) PR Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "mamba_pr_curve.png"))
    print(f"\n>>> Results saved in {RESULTS_DIR}/")

if __name__ == "__main__":
    # Bạn có thể giảm batch_size xuống 32 nếu bị OOM lúc cấp tensor (B, 15, 1, 720)
    evaluate(batch_size=64)
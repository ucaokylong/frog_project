import os
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from thop import profile
except ImportError:
    print("Warning: Please install thop via 'pip install thop' for FLOPs profiling.")
    profile = None

from data.dataloader_temporal import FrogDataLoader, scans_to_cutout
from models.loc_model_temporal import TemporalLocModel1D
from utils.postprocess_temporal import parse_loc_voting
from utils.metrics_temporal import compute_pr_curve_expert, compute_mean_metrics

# --- CONFIGURATION ---
TEST_PATH    = "data/frog_16-41_test.h5"
WEIGHTS_PATH = "checkpoints/drspaam_loc_cutout_best_SOTA.pth"  
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_WINDOW = 5 
SCAN_FAR = 10.0

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

    # =========================================================================
    # COMPUTE HARDWARE-INDEPENDENT COMPLEXITY (FLOPs & Parameters)
    # =========================================================================
    flops_str, params_str = "N/A", "N/A"
    if profile is not None:
        # Input sequence structure shape: (Batch=1, Time=5, Channels=720, Dimensionality=1, Points=56)
        dummy_input = torch.randn(1, T_WINDOW, loader.SCAN_WIDTH, 1, 56).to(DEVICE)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_str = f"{flops / 1e6:.2f} MFLOPs"
        params_str = f"{params / 1e3:.2f} K"
        print(f"\n[Model Complexity] Total Parameters: {params_str} | Computation: {flops_str}")

    all_results = []
    print(f"Running Temporal Inference with CUTOUT (T={T_WINDOW})...")

    # Arrays to record synchronized inference timing parameters
    pure_inference_times = []
    N = len(loader)

    with torch.no_grad():
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
            
            # CUDA SYNCHRONIZED TIMING INFERENCE PER BATCH
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            preds = model(x).cpu().numpy()  

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            
            # Compute average latency per single scan frame within this batch
            batch_latency = (t_end - t_start) / x.size(0)
            pure_inference_times.append(batch_latency)

            for b in range(preds.shape[0]):
                pred = preds[b]
                current_idx = idx_list[b]
                raw_scan, _ = loader[current_idx] 
                
                raw_people = parse_loc_voting(loader, pred, raw_scan, min_thresh=0.01)
                
                filtered_people = []
                for p in raw_people:
                    c_score, c_x, c_y = p[0], p[1], p[2]
                    c_dist = np.hypot(c_x, c_y)
                    
                    if c_dist <= SCAN_FAR:
                        filtered_people.append(p)
                
                if len(filtered_people) > 0:
                    filtered_people = np.array(filtered_people, dtype=np.float32)
                    filtered_people = filtered_people[np.argsort(-filtered_people[:, 0], kind='stable'), :]
                else:
                    filtered_people = np.array([], dtype=np.float32)
                    
                all_results.append((current_idx, filtered_people))

    avg_latency_ms = np.mean(pure_inference_times) * 1000.0

    # Benchmark Evaluations
    metrics = {}
    for dist in [0.5, 0.3]:
        R, P, AP, EER, F1, tp, fp, fn, p_final, r_final, f1_final = compute_pr_curve_expert(loader, all_results, assoc_distance=dist)
        metrics[dist] = {
            "R": R, "P": P, "AP": AP, "EER": EER, "F1": F1, "TP": tp, "FP": fp, "FN": fn,
            "P_Final": p_final, "R_Final": r_final, "F1_Final": f1_final
        }
        print(f"\n--- Result for Association Distance {dist}m ---")
        print(f"AP: {AP*100:.1f}% | Peak F1: {F1*100:.1f}% | EER: {EER*100:.1f}%")
        print(f"Precision: {p_final*100:.1f}% | Recall: {r_final*100:.1f}% | F1-Score: {f1_final*100:.1f}%")
        print(f"Confusion Summary: TP={tp} | FP={fp} | FN={fn}")

    mean_res = compute_mean_metrics(metrics)
    print("\n================ BENCHMARK SUMMARY (CUTOUT PARADIGM) ================")
    print(f"mAP                  : {mean_res['mAP']*100:.2f}%")
    print(f"mPEAK F1             : {mean_res['mF1']*100:.2f}%")
    print(f"mEER                 : {mean_res['mEER']*100:.2f}%")
    print(f"Average Edge Latency : {avg_latency_ms:.3f} ms")
    print("=====================================================================")

    np.savez(os.path.join(RESULTS_DIR, "drspaam_cutout_test_metrics.npz"), 
             metrics=metrics, mean_res=mean_res, latency=avg_latency_ms, FLOPs=flops_str, Params=params_str)

    # Export visualization figure
    plt.figure(figsize=(7, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(metrics[0.5]["R"], metrics[0.5]["P"], label=f"d=0.5m (AP={metrics[0.5]['AP']*100:.1f}%)", color='#1f77b4', lw=2)
    plt.plot(metrics[0.3]["R"], metrics[0.3]["P"], label=f"d=0.3m (AP={metrics[0.3]['AP']*100:.1f}%)", color='#ff7f0e', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Spatiotemporal Model Precision-Recall Curve", fontsize=13, fontweight='bold')
    plt.legend(loc='lower left', frameon=True)
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "drspaam_cutout_pr_curve.png"), dpi=200)
    plt.close()
    print(f"\n>>> Results and charts stored in {RESULTS_DIR}/")

if __name__ == "__main__":
    evaluate(batch_size=64)
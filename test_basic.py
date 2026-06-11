import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.dataloader_basic import FrogDataLoader
from models.loc_model_basic import LocModel1D
from utils.postprocess_basic import parse_loc
from utils.metrics_basic import compute_pr_curve_expert

# --- CẤU HÌNH ---
TEST_PATH    = "data/frog_16-41_test.h5"
WEIGHTS_PATH = "checkpoints/lfe_ppn_best.pth"
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(batch_size=64):
    loader = FrogDataLoader(TEST_PATH, min_people=0)
    model = LocModel1D(
        num_anchors_per_sector=loader.NUM_ANCHORS_PER_SECTOR,
        glob=True
    ).to(DEVICE)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Model weights not found at {WEIGHTS_PATH}")
        return

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    all_results = []
    print("Running Inference (Basic LFE-PPN)...")

    with torch.no_grad():
        N = len(loader)
        # duyệt theo batch index
        for start in tqdm(range(0, N, batch_size), desc="Inference"):
            end = min(start + batch_size, N)
            scans_list = []
            idx_list   = []

            # 1) gom batch bằng tay
            for i in range(start, end):
                scan, _ = loader[i]
                scan_norm = loader.normalize_scan(scan).astype(np.float32)
                scans_list.append(scan_norm)
                idx_list.append(i)

            # 2) stack thành tensor (B, 1, L)
            scans_np = np.stack(scans_list, axis=0)                
            x = torch.from_numpy(scans_np).unsqueeze(1).to(DEVICE)  

            # 3) chạy model một lần cho cả batch
            preds = model(x).cpu().numpy()                          

            # 4) parse từng phần tử trong batch
            for b in range(preds.shape[0]):
                pred = preds[b]                                     
                
                # SỬA: Lọc 10m và giữ Threshold 0.01
                raw_people = parse_loc(loader, pred, threshold=0.01)
                
                filtered_people = []
                if raw_people is not None:
                    for p in raw_people:
                        c_score, c_x, c_y = p[0], p[1], p[2]
                        c_dist = np.hypot(c_x, c_y)
                        
                        # CHỈ GIỮ LẠI dự đoán nằm trong phạm vi 10 mét
                        if c_dist <= 10.0:
                            filtered_people.append(p)
                
                if len(filtered_people) > 0:
                    filtered_people = np.array(filtered_people, dtype=np.float32)
                    filtered_people = filtered_people[np.argsort(-filtered_people[:, 0], kind='stable'), :]
                else:
                    filtered_people = np.array([], dtype=np.float32)
                    
                all_results.append((idx_list[b], filtered_people))

    # Benchmark tại 2 mốc 0.5m và 0.3m
    metrics = {}
    for dist in [0.5, 0.3]:
        R, P, AP, EER, F1, tp, fp, fn, tn = compute_pr_curve_expert(loader, all_results, assoc_distance=dist)
        metrics[dist] = {"R": R, "P": P, "AP": AP, "EER": EER, "F1": F1, "TP": tp, "FP": fp, "FN": fn}
        print(f"\n--- Result for Association Distance {dist}m ---")
        print(f"AP: {AP*100:.1f}% | Peak F1: {F1*100:.1f}% | EER: {EER*100:.1f}%")
        print(f"TP: {tp} | FP: {fp} | FN: {fn}")

    # Calculate macro-averaged metrics
    mAP = np.mean([metrics[dist]["AP"] for dist in metrics.keys()])
    mF1 = np.mean([metrics[dist]["F1"] for dist in metrics.keys()])
    mEER = np.mean([metrics[dist]["EER"] for dist in metrics.keys()])
    
    print(f"\n--- Macro-Averaged Metrics ---")
    print(f"mAP: {mAP*100:.1f}% | mPeak F1: {mF1*100:.1f}% | mEER: {mEER*100:.1f}%")

    # Lưu kết quả ra file NPZ để sau này vẽ biểu đồ
    np.savez(os.path.join(RESULTS_DIR, "test_metrics.npz"), 
             metrics=metrics, mAP=mAP, mF1=mF1, mEER=mEER)

    # Khởi tạo vẽ 4 biểu đồ Advanced
    _create_advanced_plots(metrics, mAP, mF1, mEER)
    
    print(f"\n>>> Results saved in {RESULTS_DIR}/")


def _create_advanced_plots(metrics, mAP, mF1, mEER):
    """Vẽ 4 biểu đồ Advanced độc lập, bao gồm Confusion Matrix"""
    plt.style.use('seaborn-v0_8-whitegrid') 
    colors = ['#1f77b4', '#ff7f0e']
    distances = sorted(metrics.keys())
    dist_labels = [f"d={d}m" for d in distances]

    # --- PLOT 1: PR CURVES ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for idx, dist in enumerate(distances):
        ax1.plot(metrics[dist]["R"], metrics[dist]["P"],
                 label=f"d={dist}m (AP={metrics[dist]['AP']*100:.1f}%)",
                 color=colors[idx], linewidth=2.5)
    
    ax1.axhline(y=mAP, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label=f"mAP={mAP*100:.1f}%")
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    ax1.set_xlabel("Recall", fontsize=12)
    ax1.set_ylabel("Precision", fontsize=12)
    ax1.set_title("Precision-Recall Curves", fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', frameon=True)
    ax1.grid(True, alpha=0.4)
    fig1.tight_layout()
    fig1.savefig(os.path.join(RESULTS_DIR, "plot_1_pr_curves.png"), dpi=200)

    # --- PLOT 2: METRICS COMPARISON ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    APs = [metrics[d]["AP"]*100 for d in distances]
    F1s = [metrics[d]["F1"]*100 for d in distances]
    EERs = [metrics[d]["EER"]*100 for d in distances]
    
    x = np.arange(len(dist_labels))
    width = 0.25
    ax2.bar(x - width, APs, width, label='AP', color='#1f77b4', alpha=0.85)
    ax2.bar(x, F1s, width, label='Peak F1', color='#ff7f0e', alpha=0.85)
    ax2.bar(x + width, EERs, width, label='EER', color='#2ca02c', alpha=0.85)
    
    ax2.axhline(y=mAP*100, color='#1f77b4', linestyle='--', alpha=0.6, label=f'mAP={mAP*100:.1f}%')
    ax2.axhline(y=mF1*100, color='#ff7f0e', linestyle='--', alpha=0.6, label=f'mPeak F1={mF1*100:.1f}%')
    ax2.axhline(y=mEER*100, color='#2ca02c', linestyle='--', alpha=0.6, label=f'mEER={mEER*100:.1f}%')
    
    ax2.set_xlabel("Association Distance", fontsize=12)
    ax2.set_ylabel("Score (%)", fontsize=12)
    ax2.set_title("Core Metrics Comparison", fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(dist_labels, fontsize=11)
    ax2.legend(loc='lower right', frameon=True)
    ax2.set_ylim([0, 110])
    fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "plot_2_metrics_comparison.png"), dpi=200)

    # --- PLOT 3: CONFUSION MATRIX ---
    fig3, axes3 = plt.subplots(1, len(distances), figsize=(5 * len(distances) + 2, 6))
    if len(distances) == 1: axes3 = [axes3]

    for idx, dist in enumerate(distances):
        ax = axes3[idx]
        tp, fp, fn = metrics[dist]["TP"], metrics[dist]["FP"], metrics[dist]["FN"]
        tn = "N/A"

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.invert_yaxis()

        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='#d4edda', edgecolor='black', lw=1))
        ax.text(0.5, 0.5, f"True Positive (TP)\n{tp:,}", ha='center', va='center', fontsize=12, fontweight='bold', color='#155724')

        ax.add_patch(plt.Rectangle((1, 0), 1, 1, facecolor='#f8d7da', edgecolor='black', lw=1))
        ax.text(1.5, 0.5, f"False Negative (FN)\n{fn:,}", ha='center', va='center', fontsize=12, fontweight='bold', color='#721c24')

        ax.add_patch(plt.Rectangle((0, 1), 1, 1, facecolor='#fff3cd', edgecolor='black', lw=1))
        ax.text(0.5, 1.5, f"False Positive (FP)\n{fp:,}", ha='center', va='center', fontsize=12, fontweight='bold', color='#856404')

        ax.add_patch(plt.Rectangle((1, 1), 1, 1, facecolor='#e2e3e5', edgecolor='black', lw=1))
        ax.text(1.5, 1.5, f"True Negative (TN)\n{tn}", ha='center', va='center', fontsize=12, fontweight='bold', color='#383d41')

        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(['Pred: Person', 'Pred: Background'], fontsize=11)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Actual: Person', 'Actual: Background'], rotation=90, va='center', fontsize=11)
        ax.set_title(f"Confusion Matrix (d = {dist}m)", fontsize=14, fontweight='bold', pad=15)
        ax.grid(False)

    fig3.tight_layout()
    fig3.savefig(os.path.join(RESULTS_DIR, "plot_3_confusion_matrix.png"), dpi=200)

    # --- PLOT 4: SUMMARY REPORT TỐI ƯU ---
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.axis('off')
    
    summary_text = "MODEL EVALUATION SUMMARY (LFE-PPN)\n"
    summary_text += "="*60 + "\n\n"
    summary_text += "➤ MACRO-AVERAGED METRICS:\n"
    summary_text += f"    • mAP      : {mAP*100:6.2f}%\n"
    summary_text += f"    • mPeak F1 : {mF1*100:6.2f}%\n"
    summary_text += f"    • mEER     : {mEER*100:6.2f}%\n\n"
    
    summary_text += "➤ PER-DISTANCE DETAILS:\n"
    summary_text += "-"*60 + "\n"
    
    for dist in distances:
        summary_text += f"  [ Distance {dist}m ]\n"
        summary_text += f"    • Scores   -> AP: {metrics[dist]['AP']*100:5.2f}% | F1: {metrics[dist]['F1']*100:5.2f}% | EER: {metrics[dist]['EER']*100:5.2f}%\n"
        summary_text += f"    • Counts   -> TP: {metrics[dist]['TP']:,} | FP: {metrics[dist]['FP']:,} | FN: {metrics[dist]['FN']:,}\n\n"
    
    ax4.text(0.05, 0.95, summary_text, fontsize=12, family='monospace',
             verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    fig4.tight_layout()
    fig4.savefig(os.path.join(RESULTS_DIR, "plot_4_summary_report.png"), dpi=200)
    
    plt.close('all')

if __name__ == "__main__":
    evaluate(batch_size=64)
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Results data from evaluation - ACTUAL DATA
results_data = {
    0.5: {"AP": 70.6, "Peak F1": 71.6, "EER": 71.6, "TP": 140379, "FP": 1407456, "FN": 13279},
    0.3: {"AP": 65.5, "Peak F1": 69.0, "EER": 69.0, "TP": 128879, "FP": 1418956, "FN": 24779},
}
macro_metrics = {"AP": 68.1, "Peak F1": 70.3, "EER": 70.3}

def plot_metrics_comparison():
    """Plot AP, Peak F1, EER comparison across distances."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    distances = [0.3, 0.5]
    colors_dist = ['#ff7f0e', '#1f77b4']
    
    metrics_names = ["AP", "Peak F1", "EER"]
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        
        # Per-distance bars
        values = [results_data[d][metric] for d in distances]
        bars = ax.bar([f"{d}m" for d in distances], values, 
                      color=colors_dist, alpha=0.8, width=0.5, label="Per-distance")
        
        # Macro-average line
        macro_val = macro_metrics[metric]
        ax.axhline(y=macro_val, color='red', linestyle='--', linewidth=2.5, 
                   label=f"Macro-avg: {macro_val:.1f}%", alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel("Score (%)", fontsize=12, fontweight='bold')
        ax.set_title(f"{metric}", fontsize=13, fontweight='bold')
        ax.set_ylim([0, 110])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_metrics_comparison.png"), dpi=150, bbox_inches='tight')
    print("✓ Saved: 01_metrics_comparison.png")
    plt.close()


def plot_unified_metrics_chart():
    """Plot all metrics in one unified chart with all macro-average lines and legend."""
    fig, ax = plt.subplots(figsize=(13, 7))
    
    distances = [0.3, 0.5]
    metrics = ["AP", "Peak F1", "EER"]
    x = np.arange(len(distances))
    width = 0.25
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, metric in enumerate(metrics):
        values = [results_data[d][metric] for d in distances]
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[idx], alpha=0.85)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add macro-average reference lines with different line styles to avoid overlap
    ax.axhline(y=macro_metrics["AP"], color='#1f77b4', linestyle='--', linewidth=2.5, 
              alpha=0.8, label=f'mAP: {macro_metrics["AP"]:.1f}%')
    ax.axhline(y=macro_metrics["Peak F1"], color='#ff7f0e', linestyle='-.', linewidth=2.5, 
              alpha=0.8, label=f'mPeak F1: {macro_metrics["Peak F1"]:.1f}%')
    ax.axhline(y=macro_metrics["EER"], color='#2ca02c', linestyle=':', linewidth=3, 
              alpha=0.8, label=f'mEER: {macro_metrics["EER"]:.1f}%')
    
    ax.set_xlabel("Distance Threshold", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score (%)", fontsize=12, fontweight='bold')
    ax.set_title("Model Performance: All Metrics Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}m" for d in distances])
    ax.set_ylim([0, 110])
    
    ax.legend(loc='upper right', fontsize=11, ncol=2, frameon=True, 
             fancybox=True, shadow=True, title='Legend', title_fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_unified_metrics.png"), dpi=150, bbox_inches='tight')
    print("✓ Saved: 02_unified_metrics.png")
    plt.close()


def plot_confusion_matrix():
    """Plot confusion matrix (TP, FP, FN) as 2x2 squares for each distance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    distances = [0.3, 0.5]
    
    for ax_idx, dist in enumerate(distances):
        tp = results_data[dist]["TP"]
        fp = results_data[dist]["FP"]
        fn = results_data[dist]["FN"]
        
        ax = axes[ax_idx]
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.invert_yaxis()
        
        # TP (top-left) - Green
        rect_tp = plt.Rectangle((0, 0), 1, 1, facecolor='#90EE90', edgecolor='black', linewidth=2)
        ax.add_patch(rect_tp)
        ax.text(0.5, 0.5, f'TP\n{tp:,}', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='#1a5c1a')
        
        # FN (top-right) - Red
        rect_fn = plt.Rectangle((1, 0), 1, 1, facecolor='#FFB6B6', edgecolor='black', linewidth=2)
        ax.add_patch(rect_fn)
        ax.text(1.5, 0.5, f'FN\n{fn:,}', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='#8B0000')
        
        # FP (bottom-left) - Yellow
        rect_fp = plt.Rectangle((0, 1), 1, 1, facecolor='#FFFFE0', edgecolor='black', linewidth=2)
        ax.add_patch(rect_fp)
        ax.text(0.5, 1.5, f'FP\n{fp:,}', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='#B8860B')
        
        # TN (bottom-right) - Gray
        rect_tn = plt.Rectangle((1, 1), 1, 1, facecolor='#D3D3D3', edgecolor='black', linewidth=2)
        ax.add_patch(rect_tn)
        ax.text(1.5, 1.5, f'TN\nN/A', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='#555555')
        
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(['Predicted\nPositive', 'Predicted\nNegative'], fontsize=11, fontweight='bold')
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Actual\nPositive', 'Actual\nNegative'], fontsize=11, fontweight='bold')
        ax.set_title(f"Distance {dist}m", fontsize=13, fontweight='bold')
        ax.grid(False)
    
    plt.suptitle("Confusion Matrix (2x2)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_confusion_matrix.png"), dpi=150, bbox_inches='tight')
    print("✓ Saved: 03_confusion_matrix.png")
    plt.close()


def plot_detection_performance():
    """Plot TP, FP, FN counts and evaluation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    distances = [0.3, 0.5]
    
    # --- Plot 1: Absolute counts (TP, FP, FN) ---
    ax1 = axes[0, 0]
    tp_vals = [results_data[d]["TP"] for d in distances]
    fp_vals = [results_data[d]["FP"] for d in distances]
    fn_vals = [results_data[d]["FN"] for d in distances]
    
    x = np.arange(len(distances))
    width = 0.25
    
    bars1 = ax1.bar(x - width, tp_vals, width, label='TP', color='#2ca02c', alpha=0.85)
    bars2 = ax1.bar(x, fp_vals, width, label='FP', color='#d62728', alpha=0.85)
    bars3 = ax1.bar(x + width, fn_vals, width, label='FN', color='#ff7f0e', alpha=0.85)
    
    ax1.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Distance Threshold", fontsize=11, fontweight='bold')
    ax1.set_title("Detection Counts (TP, FP, FN)", fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{d}m" for d in distances])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # --- Plot 2: Percentage distribution ---
    ax2 = axes[0, 1]
    total_vals = [tp_vals[i] + fp_vals[i] + fn_vals[i] for i in range(len(distances))]
    tp_pct = [tp_vals[i] / total_vals[i] * 100 for i in range(len(distances))]
    fp_pct = [fp_vals[i] / total_vals[i] * 100 for i in range(len(distances))]
    fn_pct = [fn_vals[i] / total_vals[i] * 100 for i in range(len(distances))]
    
    x_pos = np.arange(len(distances))
    ax2.bar(x_pos, tp_pct, label='TP', color='#2ca02c', alpha=0.85)
    ax2.bar(x_pos, fp_pct, bottom=tp_pct, label='FP', color='#d62728', alpha=0.85)
    ax2.bar(x_pos, fn_pct, bottom=[tp_pct[i] + fp_pct[i] for i in range(len(distances))], 
           label='FN', color='#ff7f0e', alpha=0.85)
    
    ax2.set_ylabel("Percentage (%)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Distance Threshold", fontsize=11, fontweight='bold')
    ax2.set_title("Distribution (%)", fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{d}m" for d in distances])
    ax2.set_ylim([0, 100])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 3: AP, Peak F1, EER (actual metrics) ---
    ax3 = axes[1, 0]
    aps = [results_data[d]["AP"] for d in distances]
    f1s = [results_data[d]["Peak F1"] for d in distances]
    eers = [results_data[d]["EER"] for d in distances]
    
    x = np.arange(len(distances))
    width = 0.25
    
    ax3.bar(x - width, aps, width, label='AP', color='#1f77b4', alpha=0.85)
    ax3.bar(x, f1s, width, label='Peak F1', color='#ff7f0e', alpha=0.85)
    ax3.bar(x + width, eers, width, label='EER', color='#2ca02c', alpha=0.85)
    
    ax3.set_ylabel("Score (%)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Distance Threshold", fontsize=11, fontweight='bold')
    ax3.set_title("Evaluation Metrics", fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{d}m" for d in distances])
    ax3.set_ylim([0, 100])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for d_idx, d in enumerate(distances):
        ax3.text(d_idx - width, aps[d_idx] + 1, f'{aps[d_idx]:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.text(d_idx, f1s[d_idx] + 1, f'{f1s[d_idx]:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.text(d_idx + width, eers[d_idx] + 1, f'{eers[d_idx]:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # --- Plot 4: Summary statistics table ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_stats = "DETECTION METRICS SUMMARY\n"
    summary_stats += "=" * 55 + "\n\n"
    
    for d in distances:
        tp = results_data[d]["TP"]
        fp = results_data[d]["FP"]
        fn = results_data[d]["FN"]
        
        summary_stats += f"Distance {d}m:\n"
        summary_stats += f"  TP: {tp:>12,}\n"
        summary_stats += f"  FP: {fp:>12,}\n"
        summary_stats += f"  FN: {fn:>12,}\n"
        summary_stats += f"  AP:       {results_data[d]['AP']:>7.2f}%\n"
        summary_stats += f"  Peak F1:  {results_data[d]['Peak F1']:>7.2f}%\n"
        summary_stats += f"  EER:      {results_data[d]['EER']:>7.2f}%\n"
        summary_stats += "-" * 55 + "\n\n"
    
    summary_stats += "Macro-Averaged Metrics:\n"
    summary_stats += f"  mAP:      {macro_metrics['AP']:>7.2f}%\n"
    summary_stats += f"  mPeak F1: {macro_metrics['Peak F1']:>7.2f}%\n"
    summary_stats += f"  mEER:     {macro_metrics['EER']:>7.2f}%\n"
    
    ax4.text(0.05, 0.95, summary_stats, fontsize=10, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='#f0f0f0', alpha=0.9, pad=1))
    
    plt.suptitle("Detection Performance Analysis", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_detection_performance.png"), dpi=150, bbox_inches='tight')
    print("✓ Saved: 06_detection_performance.png")
    plt.close()


def plot_metrics_heatmap():
    """Plot metrics as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    distances = ["0.3m", "0.5m", "Macro-avg"]
    metrics_names = ["AP", "Peak F1", "EER"]
    
    data = np.array([
        [results_data[0.3]["AP"], results_data[0.3]["Peak F1"], results_data[0.3]["EER"]],
        [results_data[0.5]["AP"], results_data[0.5]["Peak F1"], results_data[0.5]["EER"]],
        [macro_metrics["AP"], macro_metrics["Peak F1"], macro_metrics["EER"]],
    ])
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=60, vmax=75)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics_names)))
    ax.set_yticks(np.arange(len(distances)))
    ax.set_xticklabels(metrics_names, fontsize=12, fontweight='bold')
    ax.set_yticklabels(distances, fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(distances)):
        for j in range(len(metrics_names)):
            text = ax.text(j, i, f'{data[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax.set_title("Metrics Heatmap", fontsize=14, fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Score (%)", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_metrics_heatmap.png"), dpi=150, bbox_inches='tight')
    print("✓ Saved: 04_metrics_heatmap.png")
    plt.close()


def plot_summary_report():
    """Create a comprehensive text-based summary report."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    report_text = f"""
╔══════════════════════════════════════════════════════════════╗
║           MODEL EVALUATION COMPREHENSIVE REPORT              ║
╚══════════════════════════════════════════════════════════════╝

┌─ Per-Distance Results ──────────────────────────────────────┐
│                                                              │
│  Distance 0.5m (Higher Tolerance):                          │
│  ├─ AP:      70.6%  ✓                                       │
│  ├─ Peak F1: 71.6%  ✓                                       │
│  ├─ EER:     71.6%  ✓                                       │
│  └─ TP: 140,379 | FP: 1,407,456 | FN: 13,279               │
│                                                              │
│  Distance 0.3m (Stricter Tolerance):                        │
│  ├─ AP:      65.5%                                          │
│  ├─ Peak F1: 69.0%                                          │
│  ├─ EER:     69.0%                                          │
│  └─ TP: 128,879 | FP: 1,418,956 | FN: 24,779               │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─ Macro-Averaged Metrics ───────────────────────────────────┐
│                                                              │
│  mAP:        68.1%  ◆ Average performance across distances  │
│  mPeak F1:   70.3%  ◆ Average F1 score                      │
│  mEER:       70.3%  ◆ Average error rate                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─ Performance Analysis ──────────────────────────────────────┐
│                                                              │
│  Strength:   Strong at 0.5m tolerance (70.6% AP)            │
│  Weakness:   Stricter 0.3m tolerance shows decline (65.5%)  │
│  Precision:  High false positives (1.4M+) indicate          │
│              model is prone to false detections             │
│  Recall:     Good true positive rate despite high FP        │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─ Recommendations ───────────────────────────────────────────┐
│                                                              │
│  1. Reduce false positives via post-processing filters      │
│  2. Consider ensemble methods for better accuracy           │
│  3. Fine-tune threshold based on use case requirements      │
│  4. Investigate 0.3m performance degradation                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
    """
    
    ax.text(0.05, 0.95, report_text, fontsize=10, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='lightyellow', alpha=0.8, pad=1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_comprehensive_report.png"), dpi=150, bbox_inches='tight')
    print("✓ Saved: 05_comprehensive_report.png")
    plt.close()


if __name__ == "__main__":
    print(f"Generating analysis plots in {PLOTS_DIR}/...\n")
    
    plot_metrics_comparison()
    plot_unified_metrics_chart()
    plot_confusion_matrix()
    plot_detection_performance()
    plot_metrics_heatmap()
    plot_summary_report()
    
    print(f"\n✓ All visualizations saved to {PLOTS_DIR}/")
    plt.savefig(os.path.join(PLOTS_DIR, "05_comprehensive_report.png"), dpi=150, bbox_inches='tight')
    print("✓ Saved: 05_comprehensive_report.png")
    plt.close()


if __name__ == "__main__":
    print(f"Generating analysis plots in {PLOTS_DIR}/...\n")
    
    plot_metrics_comparison()
    plot_unified_metrics_chart()
    plot_confusion_matrix()
    plot_detection_performance()
    plot_metrics_heatmap()
    plot_summary_report()
    
    print(f"\n✓ All visualizations saved to {PLOTS_DIR}/")

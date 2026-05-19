import numpy as np

# Load dữ liệu từ file kết quả
data = np.load('/home/s2410433/frog_project/results/test_metrics_V3.npz', allow_pickle=True)

# Dùng vòng lặp để in ra cả 0.5m và 0.3m
for dist in [0.5, 0.3]:
    key = f"dist_{int(dist*10):02d}"  # dist_05 for 0.5, dist_03 for 0.3
    metrics = data[key].item()
    print(f"--- Results for Association Distance {dist}m ---")
    print(f"AP: {metrics['AP']*100:.2f}%")
    print(f"Peak F1: {metrics['F1']*100:.2f}%")
    print(f"EER: {metrics['EER']*100:.2f}%")
    print(f"TP: {metrics['TP']}")
    print(f"FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']}")
    print(f"TN: {metrics['TN']}")
    print("-" * 40)
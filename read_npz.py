import numpy as np

# Load dữ liệu từ file kết quả
data = np.load('results/test_metrics_V1.npz', allow_pickle=True)
metrics = data['metrics'].item()

# Dùng vòng lặp để in ra cả 0.5m và 0.3m
for dist in [0.5, 0.3]:
    print(f"--- Results for Association Distance {dist}m ---")
    print(f"AP: {metrics[dist]['AP']*100:.2f}%")
    print(f"Peak F1: {metrics[dist]['F1']*100:.2f}%")
    print(f"EER: {metrics[dist]['EER']*100:.2f}%")
    print("-" * 40)
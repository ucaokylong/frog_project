import torch
from models.loc_model_temporal import TemporalLocModel1D

def count_parameters(model):
    """Đếm tổng số tham số có thể huấn luyện (Trainable Parameters)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("[INFO] Đang khởi tạo TemporalLocModel1D...")
    model = TemporalLocModel1D()
    
    total_params = count_parameters(model)
    
    # In ra với định dạng có dấu phẩy cho dễ đọc
    print("="*50)
    print(f"Tổng số tham số của mô hình: {total_params:,}")
    print(f"Tương đương: {total_params / 1e6:.4f} Triệu (Million) tham số")
    print("="*50)
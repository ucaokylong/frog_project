import torch
import torch.nn as nn
from .common_temporal import ConvNeXtBlock1D, StridedDWPool1D

class Backbone1D(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem: Giảm kernel_size xuống 3 vì đầu vào (cutout) giờ chỉ dài 56 điểm
        self.stem = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn_stem = nn.BatchNorm1d(32)
        
        # Block 1: Không cần lặp quá nhiều vì dải dữ liệu đã ngắn lại
        self.block1 = nn.Sequential(*[ConvNeXtBlock1D(32) for _ in range(2)])
        self.pool1 = StridedDWPool1D(32, stride=2, kernel_size=3) # Ép 56 -> 28 điểm
        
        # Block 2
        self.block2 = nn.Sequential(*[ConvNeXtBlock1D(32) for _ in range(2)])
        self.pool2 = StridedDWPool1D(32, stride=4, kernel_size=5) # Ép 28 -> 7 điểm
        
        # Block 3
        self.block3 = nn.Sequential(*[ConvNeXtBlock1D(32) for _ in range(2)])
        # Lớp Pool cuối cùng: Gom 7 điểm còn lại thành đúng 1 vector đặc trưng
        self.pool3 = nn.AdaptiveAvgPool1d(1) 

    def forward(self, x_seq):
        # Shape từ Cutout Dataloader: (Batch, Time, Spatial, Channel, Length)
        # Trong đó: S=720 (số cutouts), C=1, L=56 (số điểm mỗi cutout)
        B, T, S, C, L = x_seq.shape
        
        # 1. Ép chung B, T, và S lại thành một Batch khổng lồ để CNN quét song song siêu tốc
        x = x_seq.view(B * T * S, C, L)
        
        # 2. Rút trích đặc trưng (Feature Extraction)
        x = self.bn_stem(self.stem(x))
        
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x)) 
        # Output lúc này có shape: (B*T*S, 32, 1)
        
        # 3. Phục hồi lại cấu trúc không gian (Spatial) 720 tia
        # Tách ra thành: (B, T, S, 32)
        out = x.view(B, T, S, 32)
        
        # Permute để đẩy chiều Channels (32) lên trước chiều Spatial (S=720)
        # Để output cuối cùng có shape chuẩn: (B, T, 32, 720)
        out = out.permute(0, 1, 3, 2).contiguous()
        
        # Trả về một khối feature duy nhất, đã tích hợp thông tin của Cutout
        return out
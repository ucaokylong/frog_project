import torch
import torch.nn as nn
from .backbone_temporal import Backbone1D
from .common_temporal import ConvNeXtBlock1D, TemporalConvGRU1D

class TemporalSegModel1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone1D() # Output: (B, T, 32, 720)
        
        # Xử lý chuỗi thời gian ngay trên feature map 32 kênh
        self.temporal_gru = TemporalConvGRU1D(input_dim=32, hidden_dim=32)
        
        # Vài block tinh chỉnh đặc trưng sau khi đã có thông tin quá khứ
        self.refine_blocks = nn.Sequential(*[ConvNeXtBlock1D(32) for _ in range(2)])
        
        # Output Mask: Ép từ 32 kênh về 1 kênh
        self.out_conv = nn.Conv1d(32, 1, 1)

    def forward(self, x_seq):
        # x_seq: (B, T, 720, 1, 56)
        
        # 1. Trích xuất đặc trưng cho toàn bộ T frames và 720 cutouts
        y = self.backbone(x_seq) # -> (B, T, 32, 720)
        
        # 2. Cuộn thời gian lấy trạng thái của frame hiện tại
        h_current = self.temporal_gru(y) # -> (B, 32, 720)
        
        # 3. Tinh chỉnh mượt mà
        h_refined = self.refine_blocks(h_current)
        
        # 4. Xuất Mask phân đoạn
        out = self.out_conv(h_refined) # -> (B, 1, 720)
        
        return out
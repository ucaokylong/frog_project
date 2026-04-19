import torch
import torch.nn as nn
from .backbone_temporal import Backbone1D
from .common_temporal import ConvNeXtBlock1D, TemporalConvGRU1D

class TemporalSegModel1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone1D()
        
        # Chỉ chạy GRU ở bottleneck (Scale nhỏ nhất) để tiết kiệm VRAM
        self.bottleneck_gru = TemporalConvGRU1D(input_dim=32, hidden_dim=32)
        
        # --- Decoder Block 1 ---
        # Nhận: y2_current (32) + up1(y6_current) (32) = 64 kênh
        self.up1 = nn.Upsample(scale_factor=3, mode='nearest')
        self.dec_block1 = nn.Sequential(*[ConvNeXtBlock1D(64) for _ in range(2)])
        
        # --- Decoder Block 2 ---
        # Nhận: y1_current (32) + up2(dec_block1_out) (64) = 96 kênh (SỬA Ở ĐÂY)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_block2 = nn.Sequential(*[ConvNeXtBlock1D(96) for _ in range(2)])
        
        # Output từ 96 kênh giảm xuống còn 1 kênh (Mask) (SỬA Ở ĐÂY)
        self.out_conv = nn.Conv1d(96, 1, 1)

    def forward(self, x_seq):
        # x_seq: (B, T, 1, L)
        y6, y2, y1 = self.backbone(x_seq) 
        
        # Xử lý Temporal ở Bottleneck
        # y6_current mang bối cảnh của toàn bộ quá khứ
        y6_current = self.bottleneck_gru(y6) # (B, 32, L/6)
        
        # Lấy Skip-connection của frame hiện tại (frame T)
        y2_current = y2[:, -1, :, :] # (B, 32, L/2)
        y1_current = y1[:, -1, :, :] # (B, 32, L)
        
        # Decoder 1: 32 + 32 = 64
        y = torch.cat([y2_current, self.up1(y6_current)], dim=1)
        y = self.dec_block1(y)
        
        # Decoder 2: 32 + 64 = 96
        y = torch.cat([y1_current, self.up2(y)], dim=1)
        y = self.dec_block2(y)
        
        return self.out_conv(y)
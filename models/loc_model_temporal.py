import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone_temporal import Backbone1D
from .common_temporal import CoordAtt1D, TemporalConvGRU1D

class DecoupledHead1D(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, num_anchors: int):
        super().__init__()
        self.num_anchors = num_anchors
        self.cls_branch = nn.Sequential(
            nn.Conv1d(in_ch, mid_ch, 3, padding=1, bias=False), nn.BatchNorm1d(mid_ch), nn.SiLU(inplace=True),
            nn.Conv1d(mid_ch, mid_ch, 3, padding=1, bias=False), nn.BatchNorm1d(mid_ch), nn.SiLU(inplace=True),
            nn.Conv1d(mid_ch, num_anchors, 1, bias=True)
        )
        self.reg_branch = nn.Sequential(
            nn.Conv1d(in_ch, mid_ch, 3, padding=1, bias=False), nn.BatchNorm1d(mid_ch), nn.SiLU(inplace=True),
            nn.Conv1d(mid_ch, mid_ch, 3, padding=1, bias=False), nn.BatchNorm1d(mid_ch), nn.SiLU(inplace=True),
            nn.Conv1d(mid_ch, 2 * num_anchors, 1, bias=True)
        )

    def forward(self, y):
        cls_logits = self.cls_branch(y)
        reg_logits = self.reg_branch(y)
        return torch.cat([cls_logits, reg_logits], dim=1)

class TemporalLocModel1D(nn.Module):
    def __init__(self, num_anchors_per_sector=6):
        super().__init__()
        self.num_anchors = num_anchors_per_sector
        
        self.backbone = Backbone1D()
        
        # Temporal Aggregation: Nén (B, T, 96, S) -> (B, 96, S)
        self.temporal_gru = TemporalConvGRU1D(input_dim=96, hidden_dim=96)
        
        # Spatial Attention áp dụng lên frame hiện tại sau khi đã có bối cảnh quá khứ
        self.coord_att = CoordAtt1D(96)
        
        self.head = DecoupledHead1D(in_ch=96, mid_ch=256, num_anchors=self.num_anchors)

    def forward(self, x_seq):
        # x_seq: (B, T, 1, L)
        B, T, _, L = x_seq.shape
        
        # Trích xuất đặc trưng cho toàn bộ T frames
        y6, y2, y1 = self.backbone(x_seq)

        # Multi-scale fusion cho từng frame
        # Phải view lại để dùng hàm max_pool1d
        y2_flat = y2.view(B*T, 32, -1)
        y1_flat = y1.view(B*T, 32, -1)
        
        y2_pooled = F.max_pool1d(y2_flat, 3).view(B, T, 32, -1)
        y1_pooled = F.max_pool1d(y1_flat, 6).view(B, T, 32, -1)
        
        # Feature tổng hợp: (B, T, 96, L/6)
        y_fusion = torch.cat([y6, y2_pooled, y1_pooled], dim=2) 
        
        # 1. Truyền qua GRU để tổng hợp chuỗi thời gian
        # Nhả ra hidden state của frame cuối (T)
        h_current = self.temporal_gru(y_fusion) # (B, 96, L/6)
        
        # 2. Focus không gian bằng CoordAtt
        h_attended = self.coord_att(h_current)
        
        # 3. Dự đoán mỏ neo
        out = self.head(h_attended) # (B, 3A, L/6)
        
        return out.view(B, self.num_anchors, 3, -1).permute(0, 3, 1, 2)
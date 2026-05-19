import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone_temporal import Backbone1D
from .common_temporal import CoordAtt1D, TemporalConvGRU1D

class DecoupledHead1D(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, num_anchors: int):
        super().__init__()
        self.num_anchors = num_anchors
        
        # Nhánh phân loại (Objectness)
        self.cls_branch = nn.Sequential(
            nn.Conv1d(in_ch, mid_ch, 3, padding=1, bias=False), nn.BatchNorm1d(mid_ch), nn.SiLU(inplace=True),
            nn.Conv1d(mid_ch, mid_ch, 3, padding=1, bias=False), nn.BatchNorm1d(mid_ch), nn.SiLU(inplace=True),
            nn.Conv1d(mid_ch, num_anchors, 1, bias=True) # Ra 1 kênh objectness
        )
        
        # Nhánh hồi quy (dx, dy)
        self.reg_branch = nn.Sequential(
            nn.Conv1d(in_ch, mid_ch, 3, padding=1, bias=False), nn.BatchNorm1d(mid_ch), nn.SiLU(inplace=True),
            nn.Conv1d(mid_ch, mid_ch, 3, padding=1, bias=False), nn.BatchNorm1d(mid_ch), nn.SiLU(inplace=True),
            nn.Conv1d(mid_ch, 2 * num_anchors, 1, bias=True) # Ra 2 kênh dx, dy
        )

    def forward(self, y):
        cls_logits = self.cls_branch(y)
        reg_logits = self.reg_branch(y)
        # Nối lại thành 3 kênh: [Objectness, dx, dy]
        return torch.cat([cls_logits, reg_logits], dim=1) 

class TemporalLocModel1D(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone chia sẻ trọng số với SEG
        self.backbone = Backbone1D()
        
        # Temporal Aggregation
        self.temporal_gru = TemporalConvGRU1D(input_dim=32, hidden_dim=32)
        
        # Spatial Attention: Tập trung vào các cutout quan trọng
        self.coord_att = CoordAtt1D(32)
        
        # Head: in_ch=32. Vì mỗi tia (cutout) tự thân nó đã là 1 anchor, nên num_anchors = 1
        self.head = DecoupledHead1D(in_ch=32, mid_ch=128, num_anchors=1)

    def forward(self, x_seq):
        # x_seq: (B, T, 720, 1, 56)
        
        # 1. Trích xuất đặc trưng
        features = self.backbone(x_seq) # -> (B, T, 32, 720)
        
        # 2. Truyền qua GRU để tổng hợp chuỗi thời gian
        h_current = self.temporal_gru(features) # -> (B, 32, 720)
        
        # 3. Focus không gian bằng CoordAtt
        h_attended = self.coord_att(h_current)
        
        # 4. Dự đoán trực tiếp: (B, 3, 720)
        out = self.head(h_attended) 
        
        # 5. Đổi shape lại cho khớp với Loss function trong DataLoader: (B, 720, 3)
        return out.permute(0, 2, 1).contiguous()
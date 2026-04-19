import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import Backbone1D
from .common import ECA1D

# Thử import mamba
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class DecoupledHead1D(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, num_anchors: int):
        super().__init__()
        self.num_anchors = num_anchors

        self.cls_branch = nn.Sequential(
            nn.Conv1d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_ch, num_anchors, kernel_size=1, bias=True)
        )

        self.reg_branch = nn.Sequential(
            nn.Conv1d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_ch, 2 * num_anchors, kernel_size=1, bias=True)
        )

    def forward(self, y):
        cls_logits = self.cls_branch(y)
        reg_logits = self.reg_branch(y)
        out = torch.cat([cls_logits, reg_logits], dim=1)
        return out

class LocModel1D(nn.Module):
    def __init__(self, num_anchors_per_sector=6, glob=False, d_model=96):
        super().__init__()
        self.backbone = Backbone1D(glob=glob)
        self.num_anchors = num_anchors_per_sector
        self.glob = glob
        
        # Mamba Block cấu hình cao hơn
        if Mamba is not None:
            self.temporal_processor = Mamba(
                d_model=d_model, 
                d_state=64,   # Tăng bộ nhớ trạng thái
                d_conv=4, 
                expand=4      # Tăng không gian chiều ẩn
            )
        else:
            self.temporal_processor = nn.Identity()

        self.head_eca = ECA1D(d_model) if glob else None
        self.head = DecoupledHead1D(in_ch=d_model, mid_ch=256, num_anchors=self.num_anchors)

    def forward(self, x):
        # x: (B, T, C, L) hoặc (B, 1, C, L)
        if x.dim() == 4:
            B, T, C, L = x.shape
        else:
            B, T, C, L = x.shape[0], 1, x.shape[1], x.shape[2]
            
        # 1. Ép phẳng để qua CNN 1D
        x_flat = x.view(B * T, C, L)
        y6, y2, y1 = self.backbone(x_flat)

        # 2. Fusion Không gian: (B*T, 96, S)
        y = torch.cat([
            y6,
            F.max_pool1d(y2, 3),
            F.max_pool1d(y1, 6)
        ], dim=1) 
        
        C_feat, S = y.shape[1], y.shape[2]

        # 3. SPATIO-TEMPORAL FUSION (Kiểu Residual)
        y_spatial = y.view(B, T, C_feat, S)
        
        # Lấy feature KHÔNG GIAN của frame cuối làm lõi chính
        y_base_spatial = y_spatial[:, -1, :, :] # (B, C_feat, S)

        # Xử lý THỜI GIAN với Mamba
        if isinstance(self.temporal_processor, nn.Identity):
            y_fused = y_base_spatial
        else:
            y_temporal = y_spatial.permute(0, 3, 1, 2).contiguous() # (B, S, T, C_feat)
            y_temporal = y_temporal.view(B * S, T, C_feat) 
            
            y_temporal = self.temporal_processor(y_temporal) # (B*S, T, C_feat)
            
            # Dùng Mean Pooling triệt tiêu nhiễu
            y_temporal = y_temporal.mean(dim=1) # (B*S, C_feat)
            
            # Đưa về shape Head (B, C_feat, S)
            y_temporal = y_temporal.view(B, S, C_feat).transpose(1, 2).contiguous()
            
            # CỘNG GỘP: Không gian + Thời gian
            y_fused = y_base_spatial + y_temporal

        # 4. Attention & Head
        if self.head_eca is not None:
            y_fused = self.head_eca(y_fused)

        out = self.head(y_fused)
        return out.view(out.shape[0], self.num_anchors, 3, -1).permute(0, 3, 1, 2)
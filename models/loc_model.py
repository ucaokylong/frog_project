import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import Backbone1D
from .common import ECA1D


class LocModel1D(nn.Module):
    def __init__(self, num_anchors_per_sector=6, glob=False):
        super().__init__()
        self.backbone = Backbone1D(glob=glob)
        self.num_anchors = num_anchors_per_sector
        self.glob = glob

        # Nếu glob=True thì dùng ECA ở head, KHÔNG còn GlobalAggregator1D
        self.head_eca = ECA1D(96) if glob else None

        # Vì ECA không nhân đôi kênh nên luôn là 96
        in_ch = 96

        self.head = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.Conv1d(in_ch, 512, kernel_size=1, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3 * self.num_anchors, kernel_size=1, bias=True)
        )

    def forward(self, x):
        y6, y2, y1 = self.backbone(x)

        # Multi-scale fusion giữ nguyên như cũ
        y = torch.cat([
            y6,
            F.max_pool1d(y2, 3),
            F.max_pool1d(y1, 6)
        ], dim=1)   # (B, 96, L/6)

        # glob=True → ECA scale channel attention, shape vẫn là (B, 96, L/6)
        if self.head_eca is not None:
            y = self.head_eca(y)

        y = self.head(y)  # (B, 3*A, L/6)

        return y.view(y.shape[0], self.num_anchors, 3, -1).permute(0, 3, 1, 2)
        # output: (B, num_sectors, num_anchors, 3)
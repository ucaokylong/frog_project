import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone_basic import Backbone1D
from .common import ECA1D


class DecoupledHead1D(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, num_anchors: int):
        super().__init__()
        self.num_anchors = num_anchors

        # Nhánh classification: (B, in_ch, S) -> (B, A, S)
        self.cls_branch = nn.Sequential(
            nn.Conv1d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_ch, num_anchors, kernel_size=1, bias=True)
        )

        # Nhánh regression: (B, in_ch, S) -> (B, 2A, S)
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
        # y: (B, in_ch, S=L/6)
        cls_logits = self.cls_branch(y)          # (B, A, S)
        reg_logits = self.reg_branch(y)          # (B, 2A, S)
        out = torch.cat([cls_logits, reg_logits], dim=1)  # (B, 3A, S)
        return out


class LocModel1D(nn.Module):
    def __init__(self, num_anchors_per_sector=6, glob=False):
        super().__init__()
        self.backbone = Backbone1D(glob=glob)
        self.num_anchors = num_anchors_per_sector
        self.glob = glob

        # Nếu glob=True thì dùng ECA ở head
        self.head_eca = ECA1D(96) if glob else None

        in_ch = 96
        mid_ch = 256

        # Decoupled head: tách cls / reg nhưng vẫn trả (B, 3A, S)
        self.head = DecoupledHead1D(
            in_ch=in_ch,
            mid_ch=mid_ch,
            num_anchors=self.num_anchors
        )

    def forward(self, x):
        y6, y2, y1 = self.backbone(x)

        # Multi-scale fusion giữ nguyên
        y = torch.cat(
            [
                y6,
                F.max_pool1d(y2, 3),
                F.max_pool1d(y1, 6)
            ],
            dim=1
        )  # (B, 96, L/6)

        # glob=True → ECA scale channel attention, shape giữ nguyên
        if self.head_eca is not None:
            y = self.head_eca(y)

        # Decoupled head
        y = self.head(y)  # (B, 3*A, L/6)

        # Giữ nguyên format output cho loss & postprocess
        return y.view(y.shape[0], self.num_anchors, 3, -1).permute(0, 3, 1, 2)
        # (B, num_sectors, num_anchors, 3)
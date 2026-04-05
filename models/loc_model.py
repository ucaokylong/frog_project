import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import Backbone1D
from .common import GlobalAggregator1D

class LocModel1D(nn.Module):
    def __init__(self, num_anchors_per_sector=6, glob=False):
        super().__init__()
        self.backbone = Backbone1D(glob=glob)
        self.num_anchors = num_anchors_per_sector
        self.glob = glob
        if glob: self.global_agg_head = GlobalAggregator1D()

        in_ch = 96 * (2 if glob else 1)
        self.head = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv1d(in_ch, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3 * self.num_anchors, 1)
        )

    def forward(self, x):
        y6, y2, y1 = self.backbone(x)
        y = torch.cat([y6, F.max_pool1d(y2, 3), F.max_pool1d(y1, 6)], 1)
        if self.glob: y = self.global_agg_head(y)
        y = self.head(y)
        return y.view(y.shape[0], self.num_anchors, 3, -1).permute(0, 3, 1, 2)
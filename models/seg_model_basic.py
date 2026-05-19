import torch
import torch.nn as nn
from .backbone_basic import Backbone1D
from .common_basic import ResConvBlock1D

class SegModel1D(nn.Module):
    def __init__(self, glob=False):
        super().__init__()
        self.backbone = Backbone1D(glob=glob)
        self.up1 = nn.Upsample(scale_factor=3, mode='nearest')
        self.dec_block1 = ResConvBlock1D(64, [32,32,32], [9,7,5], glob=glob)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_block2 = ResConvBlock1D(64, [32,32,32], [9,7,5], glob=glob)
        self.out_conv = nn.Conv1d(32, 1, 1)

    def forward(self, x):
        y6, y2, y1 = self.backbone(x)
        y = self.dec_block1(torch.cat([y2, self.up1(y6)], 1))
        y = self.dec_block2(torch.cat([y1, self.up2(y)], 1))
        return self.out_conv(y)
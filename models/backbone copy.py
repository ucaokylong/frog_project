import torch.nn as nn
from .common import ResConvBlock1D

class Backbone1D(nn.Module):
    def __init__(self, glob=False):
        super().__init__()
        self.block1 = ResConvBlock1D(1, [32, 32, 32], [9, 7, 5], glob=False)
        self.pool1  = nn.MaxPool1d(2)
        self.block2 = ResConvBlock1D(32, [32, 32, 32], [9, 7, 5], glob=glob)
        self.pool2  = nn.MaxPool1d(3)
        self.block3 = ResConvBlock1D(32, [32, 32, 32], [9, 7, 5], glob=glob)

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(self.pool1(y1))
        y6 = self.block3(self.pool2(y2))
        return y6, y2, y1
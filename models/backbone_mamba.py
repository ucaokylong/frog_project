import torch.nn as nn
from .common_mamba import ResConvBlock1D, StridedDWPool1D


class Backbone1D(nn.Module):
    def __init__(self, glob=False):
        super().__init__()

        # Block 1: input (B, 1, L) -> output (B, 32, L)
        self.block1 = ResConvBlock1D(
            in_channels=1,
            filters=[32, 32, 32],
            kernels=[9, 7, 5],
            glob=False
        )

        # Thay MaxPool1d(2) bằng Strided Depthwise Conv Pool
        # (B, 32, L) -> (B, 32, L/2)
        self.pool1 = StridedDWPool1D(
            channels=32,
            stride=2,
            kernel_size=3
        )

        # Block 2: input (B, 32, L/2) -> output (B, 32, L/2)
        self.block2 = ResConvBlock1D(
            in_channels=32,
            filters=[32, 32, 32],
            kernels=[9, 7, 5],
            glob=glob
        )

        # Thay MaxPool1d(3) bằng Strided Depthwise Conv Pool
        # (B, 32, L/2) -> (B, 32, L/6)
        self.pool2 = StridedDWPool1D(
            channels=32,
            stride=3,
            kernel_size=3
        )

        # Block 3: input (B, 32, L/6) -> output (B, 32, L/6)
        self.block3 = ResConvBlock1D(
            in_channels=32,
            filters=[32, 32, 32],
            kernels=[9, 7, 5],
            glob=glob
        )

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(self.pool1(y1))
        y6 = self.block3(self.pool2(y2))
        return y6, y2, y1
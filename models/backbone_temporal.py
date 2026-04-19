import torch.nn as nn
from .common_temporal import ConvNeXtBlock1D, StridedDWPool1D

class Backbone1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn_stem = nn.BatchNorm1d(32)
        
        self.block1 = nn.Sequential(*[ConvNeXtBlock1D(32) for _ in range(3)])
        self.pool1 = StridedDWPool1D(32, stride=2, kernel_size=3)
        
        self.block2 = nn.Sequential(*[ConvNeXtBlock1D(32) for _ in range(3)])
        self.pool2 = StridedDWPool1D(32, stride=3, kernel_size=3)
        
        self.block3 = nn.Sequential(*[ConvNeXtBlock1D(32) for _ in range(3)])

    def forward(self, x_seq):
        # x_seq format từ Dataloader: (B, T, 1, L)
        B, T, C, L = x_seq.shape
        
        # Gom B và T lại để chạy 2D CNN (Time-Distributed)
        x = x_seq.view(B * T, C, L)
        
        x = self.bn_stem(self.stem(x))
        y1 = self.block1(x)
        y2 = self.block2(self.pool1(y1))
        y6 = self.block3(self.pool2(y2))
        
        # Tách trở lại (B, T, C, L)
        y1 = y1.view(B, T, 32, -1)
        y2 = y2.view(B, T, 32, -1)
        y6 = y6.view(B, T, 32, -1)
        
        return y6, y2, y1
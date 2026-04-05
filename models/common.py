import torch
import torch.nn as nn

class GlobalAggregator1D(nn.Module):
    def forward(self, x):
        x_global, _ = torch.max(x, dim=2, keepdim=True)
        x_global = x_global.expand_as(x)
        return torch.cat([x, x_global], dim=1)

class ResConvBlock1D(nn.Module):
    def __init__(self, in_channels, filters, kernels, ac='relu', dp=0.2, bn=True, glob=False):
        super().__init__()
        self.glob = glob
        self.global_agg = GlobalAggregator1D() if glob else None
        curr_in = in_channels * (2 if glob else 1)

        layers = []
        for i, (f, k) in enumerate(zip(filters, kernels)):
            if i < len(filters) - 1:
                layers.append(nn.Conv1d(curr_in, curr_in, k, padding=k//2, groups=curr_in, bias=False))
                layers.append(nn.Conv1d(curr_in, f, 1, bias=True))
                layers.append(nn.ReLU() if ac == 'relu' else nn.ReLU())
                if bn: layers.append(nn.BatchNorm1d(f))
                if dp > 0.0: layers.append(nn.Dropout(dp))
                curr_in = f
            else:
                layers.append(nn.Conv1d(curr_in, f, k, padding=k//2, bias=True))
        
        self.conv_seq = nn.Sequential(*layers)
        self.proj = nn.Conv1d(in_channels, filters[-1], 1, bias=False) if in_channels != filters[-1] else None
        self.act = nn.ReLU()
        self.bn_out = nn.BatchNorm1d(filters[-1]) if bn else None

    def forward(self, x):
        identity = self.proj(x) if self.proj is not None else x
        out = self.global_agg(x) if self.glob else x
        out = self.conv_seq(out)
        return self.act(self.bn_out(out + identity))
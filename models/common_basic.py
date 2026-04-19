import torch
import torch.nn as nn


class ECA1D(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        w = self.gap(x)             # (B, C, 1)
        w = w.transpose(-1, -2)     # (B, 1, C)
        w = self.conv(w)            # (B, 1, C)
        w = w.transpose(-1, -2)     # (B, C, 1)
        w = self.sigmoid(w)         # (B, C, 1)
        return x * w.expand_as(x)   # (B, C, L)


class StridedDWPool1D(nn.Module):
    def __init__(self, channels, stride, kernel_size=3, bn=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=channels,
            bias=False
        )
        self.bn = nn.BatchNorm1d(channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class ResConvBlock1D(nn.Module):
    def __init__(self, in_channels, filters, kernels, ac='relu', dp=0.2, bn=True, glob=False, dilations=None):
        super().__init__()

        # glob=True giờ dùng ECA1D thay vì GlobalAggregator1D
        self.glob = glob
        self.eca = ECA1D(in_channels) if glob else None

        # ECA không concat nên số kênh giữ nguyên
        curr_in = in_channels

        # số stage depthwise = len(filters) - 1
        num_dw_stages = len(filters) - 1
        if dilations is None:
            dilations = [2] + [1] * (num_dw_stages - 1)
        assert len(dilations) == num_dw_stages, \
            f"len(dilations) phải bằng {num_dw_stages}, nhưng nhận {len(dilations)}"

        layers = []
        dil_idx = 0

        for i, (f, k) in enumerate(zip(filters, kernels)):
            if i < len(filters) - 1:
                d = dilations[dil_idx]
                dil_idx += 1
                pad = d * (k // 2)

                layers.append(
                    nn.Conv1d(
                        curr_in,
                        curr_in,
                        kernel_size=k,
                        padding=pad,
                        dilation=d,
                        groups=curr_in,
                        bias=False
                    )
                )
                layers.append(nn.Conv1d(curr_in, f, kernel_size=1, bias=True))
                layers.append(nn.ReLU() if ac == 'relu' else nn.ReLU())
                if bn:
                    layers.append(nn.BatchNorm1d(f))
                if dp > 0.0:
                    layers.append(nn.Dropout(dp))
                curr_in = f
            else:
                layers.append(nn.Conv1d(curr_in, f, kernel_size=k, padding=k // 2, bias=True))

        self.conv_seq = nn.Sequential(*layers)
        self.proj = nn.Conv1d(in_channels, filters[-1], kernel_size=1, bias=False) if in_channels != filters[-1] else None
        self.act = nn.ReLU()
        self.bn_out = nn.BatchNorm1d(filters[-1]) if bn else None

    def forward(self, x):
        identity = self.proj(x) if self.proj is not None else x
        out = self.eca(x) if self.eca is not None else x
        out = self.conv_seq(out)
        out = out + identity
        if self.bn_out is not None:
            out = self.bn_out(out)
        return self.act(out)
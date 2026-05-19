import torch
import torch.nn as nn

class CoordAtt1D(nn.Module):
    """ Modern Spatial-Channel Attention thay thế cho SPAAM/ECA """
    def __init__(self, inp, reduction=8):
        super().__init__()
        mip = max(8, inp // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(mip)
        self.act = nn.SiLU() # SiLU (Swish) tốt hơn ReLU
        self.conv2 = nn.Conv1d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        y = self.pool(x)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        w = self.sigmoid(y)
        return x * w

class ConvNeXtBlock1D(nn.Module):
    """ SOTA CNN Block (2022+) thay thế cho ResConvBlock cũ """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # Depthwise conv kernel lớn (k=7)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Pointwise 1 (Inverted bottleneck expansion 4x)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        # Pointwise 2
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1) # (B, C, L) -> (B, L, C) cho LayerNorm/Linear
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1) # Quay lại (B, C, L)
        return input_x + x

class StridedDWPool1D(nn.Module):
    def __init__(self, channels, stride, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=channels, bias=False)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class LayerNormConvGRU1DCell(nn.Module):
    """ ConvGRU1D có LayerNorm để chống mất mát thông tin chuỗi dài """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv_gates = nn.Conv1d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=padding)
        self.conv_can = nn.Conv1d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.ln_gates = nn.LayerNorm(2 * hidden_dim)
        self.ln_can = nn.LayerNorm(hidden_dim)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        
        # Tính Gates
        gates = self.conv_gates(combined)
        gates = gates.permute(0, 2, 1)
        gates = self.ln_gates(gates).permute(0, 2, 1)
        gates = torch.sigmoid(gates)
        r, z = torch.split(gates, self.hidden_dim, dim=1)
        
        # Tính Candidate
        combined_can = torch.cat([x, r * h_prev], dim=1)
        can = self.conv_can(combined_can)
        can = can.permute(0, 2, 1)
        can = self.ln_can(can).permute(0, 2, 1)
        can = torch.tanh(can)
        
        h_next = (1 - z) * h_prev + z * can
        return h_next

class TemporalConvGRU1D(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = LayerNormConvGRU1DCell(input_dim, hidden_dim)

    def forward(self, x_seq):
        # x_seq: (B, T, C, L)
        B, T, C, L = x_seq.shape
        h = torch.zeros(B, self.hidden_dim, L, device=x_seq.device)
        
        for t in range(T):
            h = self.cell(x_seq[:, t, :, :], h)
        return h # Trả về hidden state của frame cuối cùng (Hiện tại)
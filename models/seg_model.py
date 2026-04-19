import torch
import torch.nn as nn
from .backbone import Backbone1D
from .common import ResConvBlock1D

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
        # x shape ban đầu: (B, T, 1, L)
        B, T, C, L = x.shape
        
        # 1. Ép phẳng B và T để qua CNN 1D: (B*T, 1, L)
        x = x.view(B * T, C, L)
        
        y6, y2, y1 = self.backbone(x) # y6: (B*T, 32, L/6), y2: (B*T, 32, L/2), y1: (B*T, 32, L)

        # 2. Chúng ta chỉ dự đoán Segmentation cho frame cuối cùng (hiện tại)
        # Tách B và T ra lại, sau đó lấy [:, -1] (frame cuối)
        def get_last_frame(feat):
            _, C_f, L_f = feat.shape
            return feat.view(B, T, C_f, L_f)[:, -1, :, :]

        y6_last = get_last_frame(y6) # (B, 32, L/6)
        y2_last = get_last_frame(y2) # (B, 32, L/2)
        y1_last = get_last_frame(y1) # (B, 32, L)

        # 3. Chạy Decoder trên frame cuối
        y = self.dec_block1(torch.cat([y2_last, self.up1(y6_last)], 1))
        y = self.dec_block2(torch.cat([y1_last, self.up2(y)], 1))
        
        return self.out_conv(y) # (B, 1, L)
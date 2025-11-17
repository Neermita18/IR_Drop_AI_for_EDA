import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim



class DoubleConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv2d(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv2d(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class TemporalEncoder3D(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_ch, 8, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d((2,2,2)))
        self.conv2 = nn.Sequential(nn.Conv3d(8,16,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool3d((2,2,2)), nn.AdaptiveAvgPool3d((1, None, None)))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x.squeeze(2)

class IRDropUNet(nn.Module):
    def __init__(self, in_spatial_ch=4, temporal_ch=1, base=32):
        super().__init__()
        # Encoder
        self.inc = DoubleConv2d(in_spatial_ch, base)
        self.down1 = Down2d(base, base*2)
        self.down2 = Down2d(base*2, base*4)

        # Temporal
        self.temporal = TemporalEncoder3D(in_ch=temporal_ch)
        self.temp_proj = nn.Conv2d(16, base*4, kernel_size=1)

        # Decoder
        self.up1 = Up2d(base * 10, base * 2)
        self.up2 = Up2d(base * 3, base)
        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, spatial_x, temporal_x, label_shape=None):
        x1 = self.inc(spatial_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        t = self.temporal(temporal_x)
        t_proj = self.temp_proj(t)

        bottleneck = torch.cat([x3, t_proj], dim=1)
        u1 = self.up1(bottleneck, x2)
        u2 = self.up2(u1, x1)
        out = self.outc(u2)

        if label_shape is not None:
            out = F.interpolate(out, size=label_shape[-2:], mode='bilinear', align_corners=False)
        return out
import torch
from torch import nn


class TimeSeriesEncoder(nn.Module):
    def __init__(self, in_dim, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class ImageSequenceEncoder(nn.Module):
    def __init__(self, in_ch=3, d_model=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(96, d_model)
    def forward(self, x):
        b,t,c,h,w = x.shape
        y = self.cnn(x.reshape(b*t,c,h,w)).flatten(1)
        return self.proj(y).reshape(b,t,-1)

"""CNN vision encoder.

Built from nn.Conv2d, activations, normalization and pooling (no
pretrained backbones). Turns a (B, 3, 64, 64) image into a feature map
(B, C, H', W').

The provided tests expect the last nn.Module defined in this file to be
constructible with no arguments and called as encoder(images).
"""
import torch
from torch import nn

class CNNEncoder(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), #normalization
            nn.GELU(), #activation
            nn.MaxPool2d(2),      # 64 --> 32 #pooling

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),      # 32 --> 16

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.MaxPool2d(2),      # 16 -> 8
        )

    def forward(self, x):
        return self.net(x)
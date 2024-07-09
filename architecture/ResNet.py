import torch
import torch.nn as nn
from util.conv_block import ConvBlock


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.res1 = ResidualBlock(64)
        self.conv2 = ConvBlock(64, 128)
        self.res2 = ResidualBlock(128)
        self.conv3 = ConvBlock(128, 256)
        self.res3 = ResidualBlock(256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.res3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.conv1 = ConvBlock(256, 128)
        self.res1 = ResidualBlock(128)
        self.conv2 = ConvBlock(128, 64)
        self.res2 = ResidualBlock(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.fc(x).view(-1, 256, 4, 4)
        x = self.upsample(self.conv1(x))
        x = self.res1(x)
        x = self.upsample(self.conv2(x))
        x = self.res2(x)
        x = self.upsample(self.conv3(x))
        return torch.tanh(x)

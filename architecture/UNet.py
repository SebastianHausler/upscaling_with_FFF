import torch
import torch.nn as nn
from util.conv_block import ConvBlock


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 2 * 2, latent_dim)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x = self.flatten(self.pool(x4))
        return self.fc(x)


class UNetDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)
        self.conv1 = ConvBlock(512, 256)
        self.conv2 = ConvBlock(256, 128)
        self.conv3 = ConvBlock(128, 64)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.fc(x).view(-1, 512, 2, 2)
        x = self.upsample(self.conv1(x))
        x = self.upsample(self.conv2(x))
        x = self.upsample(self.conv3(x))
        x = self.upsample(self.conv4(x))
        return torch.tanh(x)

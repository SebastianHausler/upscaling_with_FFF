import torch
import torch.nn as nn
from util.conv_block import ConvBlock


class SimpleConvEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.flatten(x)
        return self.fc(x)


class SimpleConvDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.conv1 = ConvBlock(256, 128)
        self.conv2 = ConvBlock(128, 64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.fc(x).view(-1, 256, 4, 4)
        x = self.upsample(self.conv1(x))
        x = self.upsample(self.conv2(x))
        x = self.upsample(self.conv3(x))
        return torch.tanh(x)

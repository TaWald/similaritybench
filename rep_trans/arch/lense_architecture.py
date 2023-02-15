import torch
from torch import nn


class UNetConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downsample_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=(1, 1), bias=False
        )

    def forward(self, x):
        return self.downsample_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2, bias=False)

    def forward(self, x):
        return self.up_conv(x)


class UNetLense(nn.Module):
    def __init__(self, mode: str, in_ch):
        super().__init__()
        if mode == "small":
            scale_factor = 0.5
        elif mode == "medium":
            scale_factor = 1
        elif mode == "large":
            scale_factor = 2
        else:
            raise ValueError
        self.in_conv = nn.Conv2d(in_ch, 64 * scale_factor, 3, stride=1, padding=1)
        self.enc_l0_conv = UNetConv(64 * scale_factor, 64 * scale_factor)
        self.enc_l1_conv = UNetConv(128 * scale_factor, 128 * scale_factor)
        self.enc_l2_conv = UNetConv(256 * scale_factor, 256 * scale_factor)
        self.enc_l3_conv = UNetConv(512 * scale_factor, 512 * scale_factor)
        self.bottleneck = UNetConv(1024 * scale_factor, 1024 * scale_factor)
        self.down1 = Downsample(64 * scale_factor, 128 * scale_factor)
        self.down2 = Downsample(128 * scale_factor, 256 * scale_factor)
        self.down3 = Downsample(256 * scale_factor, 512 * scale_factor)
        self.down4 = Downsample(512 * scale_factor, 1024 * scale_factor)
        self.up4 = UpConv(1024 * scale_factor, 512 * scale_factor)
        self.up3 = UpConv(512 * scale_factor, 256 * scale_factor)
        self.up2 = UpConv(256 * scale_factor, 128 * scale_factor)
        self.up1 = UpConv(128 * scale_factor, 64 * scale_factor)
        self.de_l3_conv = UNetConv(1024 * scale_factor, 512 * scale_factor)
        self.de_l2_conv = UNetConv(512 * scale_factor, 256 * scale_factor)
        self.de_l1_conv = UNetConv(256 * scale_factor, 128 * scale_factor)
        self.de_l0_conv = UNetConv(128 * scale_factor, 64 * scale_factor)
        self.out_conv = nn.Conv2d(64 * scale_factor, in_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        e0 = self.in_conv(x)
        e0 = self.enc_l0_conv(e0)
        e1 = self.enc_l1_conv(self.down1(e0))
        e2 = self.enc_l2_conv(self.down2(e1))
        e3 = self.enc_l3_conv(self.down3(e2))
        bottleneck = self.bottleneck(self.down4(e3))

        u3 = self.up4(bottleneck)
        d3 = self.de_l3_conv(torch.concatenate([u3, e3], dim=1))
        u2 = self.up3(d3)
        d2 = self.de_l2_conv(torch.concatenate([u2, e2], dim=1))
        u1 = self.up2(d2)
        d1 = self.de_l1_conv(torch.concatenate([u1, e1], dim=1))
        u0 = self.up1(d1)
        d0 = self.de_l0_conv(torch.concatenate([u0, e0], dim=1))
        out = self.out_conv(d0)
        return out

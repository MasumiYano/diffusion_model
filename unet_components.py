import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channel
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownScale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownScale, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        return self.down(x)


class UpScale(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(UpScale, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channel, out_channel, in_channel // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channel // 2, in_channel // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x, skip_features):
        x = self.up(x)
        x = torch.cat((x, skip_features), dim=1)
        return self.conv(x)


class OutputConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutputConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

from unet_components import *
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownScale(64, 128)
        self.down2 = DownScale(128, 256)
        self.down3 = DownScale(256, 512)
        self.down4 = DownScale(512, 1024)

        self.up1 = UpScale(1024, 512, bilinear)
        self.up2 = UpScale(512, 256, bilinear)
        self.up3 = UpScale(256, 128, bilinear)
        self.up4 = UpScale(128, 64, bilinear)

        self.outc = OutputConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.outc(x)
        return out

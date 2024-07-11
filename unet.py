from unet_components import *
import torch
import torch.nn as nn
import torch.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channel = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.conv1 = DoubleConv(in_channel=3, out_channel=64)
        self.down1 = DownScale()

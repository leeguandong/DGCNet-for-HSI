import torch.nn.functional as F

from torch import nn


class Residual_2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, batch_normal=False, stride=1):
        super(Residual_2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        if batch_normal:
            self.bn = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.bn = nn.ReLU()

    def forward(self, X):
        Y = F.relu(self.conv1(self.bn(X)))
        Y = self.conv2(Y)
        return F.relu(Y + X)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU())
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class Separable_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, kernel_size=1, stride=1):
        super(Separable_Convolution, self).__init__()
        self.depth_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.point_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

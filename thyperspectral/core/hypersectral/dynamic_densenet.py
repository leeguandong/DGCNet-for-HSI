from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['Dydensenet']


class DynamicMultiHeadConv(nn.Module):
    global_progress = 0.0

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, heads=4,
                 squeeze_rate=16, gate_factor=0.25):
        super(DynamicMultiHeadConv, self).__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.squeeze_rate = squeeze_rate
        self.gate_factor = gate_factor
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.is_pruned = True
        self.register_buffer('_inactive_channels', torch.zeros(1))

        # Check if arguments are valid
        assert self.in_channels % self.heads == 0, "head number can not be divided by input channels"
        assert self.out_channels % self.heads == 0, "head number can not be divided by output channels"
        assert self.gate_factor <= 1.0, "gate factor is greater than 1"

        for i in range(self.heads):
            self.__setattr__('headconv_%1d' % i, HeadConv(in_channels, out_channels // self.heads, squeeze_rate,
                                                          kernel_size, stride, padding, dilation, 1, gate_factor))

    def forward(self, x):
        """
        The code here is just a coarse implementation.
        The forward process can be quite slow and memory consuming, need to be optimized.
        """
        if self.training:
            progress = DynamicMultiHeadConv.global_progress
            # gradually deactivate input channels
            if progress < 3.0 / 4 and progress > 1.0 / 12:
                self.inactive_channels = round(
                    self.in_channels * (1 - self.gate_factor) * 3.0 / 2 * (progress - 1.0 / 12))
            elif progress >= 3.0 / 4:
                self.inactive_channels = round(self.in_channels * (1 - self.gate_factor))

        x = self.norm(x)
        x = self.relu(x)

        x_averaged = self.avg_pool(x)

        x_mask = []
        weight = []
        for i in range(self.heads):
            i_x = self.__getattr__('headconv_%1d' % i)(x, x_averaged, self.inactive_channels)
            x_mask.append(i_x)
            weight.append(self.__getattr__('headconv_%1d' % i).conv.weight)

        x_mask = torch.cat(x_mask, dim=1)  # batch_size, 4 x C_in, H, W
        weight = torch.cat(weight, dim=0)  # 4 x C_out, C_in, k, k

        out = F.conv3d(x_mask, weight, None, self.stride, self.padding, self.dilation, self.heads)
        b, c, s, h, w = out.size()
        out = out.view(b, self.heads, c // self.heads, s, h, w)
        out = out.transpose(1, 2).contiguous().view(b, c, s, h, w)
        return out

    @property
    def inactive_channels(self):
        return int(self._inactive_channels[0])

    @inactive_channels.setter
    def inactive_channels(self, val):
        self._inactive_channels.fill_(val)


class HeadConv(nn.Module):
    def __init__(self, in_channels, out_channels, squeeze_rate, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, gate_factor=0.25):
        super(HeadConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=False)
        self.target_pruning_rate = gate_factor
        if in_channels < 80:
            squeeze_rate = squeeze_rate // 2
        self.fc1 = nn.Linear(in_channels, in_channels // squeeze_rate, bias=False)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // squeeze_rate, in_channels, bias=True)
        self.relu_fc2 = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 1.0)

    def forward(self, x, x_averaged, inactive_channels):
        b, c, _, _, _ = x.size()
        x_averaged = x_averaged.view(b, c)
        y = self.fc1(x_averaged)
        y = self.relu_fc1(y)
        y = self.fc2(y)

        mask = self.relu_fc2(y)  # b, c

        mask_d = mask.detach()
        mask_c = mask

        if inactive_channels > 0:
            mask_c = mask.clone()
            topk_maxmum, _ = mask_d.topk(inactive_channels, dim=1, largest=False, sorted=False)
            clamp_max, _ = topk_maxmum.max(dim=1, keepdim=True)
            mask_index = mask_d.le(clamp_max)
            mask_c[mask_index] = 0

        mask_c = mask_c.view(b, c, 1, 1, 1)
        x = x * mask_c.expand_as(x)
        return x


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(Conv, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=padding, bias=False, groups=groups))


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck, gate_factor, squeeze_rate, group_3x3, heads):
        super(_DenseLayer, self).__init__()
        # 1x1 conv: i --> bottleneck * k
        self.conv_1 = DynamicMultiHeadConv(in_channels, bottleneck * growth_rate, kernel_size=1, heads=heads,
                                           squeeze_rate=squeeze_rate, gate_factor=gate_factor)

        # 3x3 conv: bottleneck * k --> k
        self.conv_2 = Conv(bottleneck * growth_rate, growth_rate, kernel_size=3, padding=1, groups=group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x_)
        x = self.conv_2(x)
        x = torch.cat([x_, x], 1)
        return x


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck, gate_factor, squeeze_rate, group_3x3, heads):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck, gate_factor, squeeze_rate,
                                group_3x3, heads)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class DydenseNet(nn.Module):
    def __init__(self, band, num_classes):
        super(DydenseNet, self).__init__()
        self.name = 'dydensenet'
        self.stages = [14, 14, 14]
        self.growth = [8, 16, 32]
        self.progress = 0.0
        self.init_stride = 2
        self.pool_size = 7
        self.bottleneck = 4
        self.gate_factor = 0.25
        self.squeeze_rate = 16
        self.group_3x3 = 4
        self.heads = 4

        self.features = nn.Sequential()
        # Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        # Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv3d(1, self.num_features, kernel_size=3, stride=self.init_stride,
                                                           padding=1, bias=False))
        for i in range(len(self.stages)):
            # Dense-block i
            self.add_block(i)

        # Linear layer
        self.bn_last = nn.BatchNorm3d(self.num_features)
        self.relu_last = nn.ReLU(inplace=True)
        # self.pool_last = nn.AvgPool3d(self.pool_size)
        self.pool_last = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(self.num_features, num_classes)
        self.classifier.bias.data.zero_()

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return

    def add_block(self, i):
        # Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            bottleneck=self.bottleneck,
            gate_factor=self.gate_factor,
            squeeze_rate=self.squeeze_rate,
            group_3x3=self.group_3x3,
            heads=self.heads
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features)
            self.features.add_module('transition_%d' % (i + 1), trans)

    def forward(self, x, progress=None, threshold=None):
        if progress:
            DynamicMultiHeadConv.global_progress = progress
        features = self.features(x)
        features = self.bn_last(features)
        features = self.relu_last(features)
        features = self.pool_last(features)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    net = DydenseNet(200, 12)

    from torchsummary import summary

    summary(net, input_size=[(1, 200, 7, 7)], batch_size=1)

    from thop import profile

    input = torch.randn(1, 1, 200, 7, 7)
    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))

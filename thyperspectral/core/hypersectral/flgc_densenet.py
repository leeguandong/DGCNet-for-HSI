from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['FLGC_Densenet']


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(context, probs):
        binarized = (probs == torch.max(probs, dim=1, keepdim=True)[0]).float()
        context.save_for_backward(binarized)
        return binarized

    @staticmethod
    def backward(context, gradient_output):
        binarized, = context.saved_tensors
        gradient_output[binarized == 0] = 0
        return gradient_output


class Flag3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  # kernel_size=1, stride=1, padding=0, dilation=1, heads=4
                 stride=1, padding=0, dilation=1, groups=8, bias=True):
        super().__init__()
        self.in_channels_in_group_assignment_map = nn.Parameter(torch.Tensor(in_channels, groups))
        nn.init.normal_(self.in_channels_in_group_assignment_map)
        self.out_channels_in_group_assignment_map = nn.Parameter(torch.Tensor(out_channels, groups))
        nn.init.normal_(self.out_channels_in_group_assignment_map)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, 1, bias)
        self.binarize = Binarize.apply

    def forward(self, x):
        map = torch.mm(self.binarize(torch.softmax(self.out_channels_in_group_assignment_map, dim=1)),
                       torch.t(self.binarize(torch.softmax(self.in_channels_in_group_assignment_map, dim=1))))
        return nn.functional.conv3d(x, self.conv.weight * map[:, :, None, None, None], self.conv.bias,
                                    self.conv.stride, self.conv.padding, self.conv.dilation)


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
            self.__setattr__('headconv_%1d' % i, HeadConv(in_channels,
                                                          out_channels // self.heads,
                                                          squeeze_rate,
                                                          kernel_size,
                                                          stride,
                                                          padding,
                                                          dilation,
                                                          1,
                                                          gate_factor))

    def forward(self, x):
        """
        The code here is just a coarse implementation.
        The forward process can be quite slow and memory consuming, need to be optimized.
        """
        if self.training:  # 训练的三个阶段，第一个阶段前1/12用于warmup，inactive_channels=0，第二阶段是逐步提升裁剪比例，第三阶段用于fine-tuning
            progress = DynamicMultiHeadConv.global_progress
            # gradually deactivate input channels
            if progress < 3.0 / 4 and progress > 1.0 / 12:
                self.inactive_channels = round(
                    self.in_channels * (1 - self.gate_factor) * 3.0 / 2 * (progress - 1.0 / 12))
            elif progress >= 3.0 / 4:
                self.inactive_channels = round(self.in_channels * (1 - self.gate_factor))

        x = self.norm(x)  # 2,16,100,4,4
        x = self.relu(x)

        x_averaged = self.avg_pool(x)  # 2,16,1,1,1

        x_mask = []
        weight = []
        for i in range(self.heads):
            i_x = self.__getattr__('headconv_%1d' % i)(x, x_averaged, self.inactive_channels)
            x_mask.append(i_x)
            weight.append(self.__getattr__('headconv_%1d' % i).conv.weight)

        x_mask = torch.cat(x_mask, dim=1)  # batch_size, 4 x C_in, H, W    2,64,100,4,4
        weight = torch.cat(weight, dim=0)  # 4 x C_out, C_in, k, k   [8,16,1,1,1]/[8,15,1,1,1]/

        # 将选择出来的特征和对应的权值选出来进行常规卷积计算，此处就是重要性的维度和对应的权值进行加权
        out = F.conv3d(x_mask, weight, None, self.stride, self.padding, self.dilation, self.heads)  # weights:32,16,1,1 x_mask:2,64,100,4,4
        b, c, s, h, w = out.size()  # 2,32,100,4,4
        out = out.view(b, self.heads, c // self.heads, s, h, w)
        out = out.transpose(1, 2).contiguous().view(b, c, s, h, w)  #
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
        self.fc1 = nn.Linear(in_channels, in_channels // squeeze_rate, bias=False)  # 16,2  squeeze_rate是压缩比
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // squeeze_rate, in_channels, bias=True)  # 2,16
        self.relu_fc2 = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 1.0)

    def forward(self, x, x_averaged, inactive_channels):
        # 此处遵循se设计，本来x_averaged这条支路后mask_c应该是sigmoid，但是此处不用sigmoid
        b, c, _, _, _ = x.size()
        x_averaged = x_averaged.view(b, c)  # 2,16
        y = self.fc1(x_averaged)
        y = self.relu_fc1(y)
        y = self.fc2(y)
        mask = self.relu_fc2(y)  # b, c  2,16

        mask_d = mask.detach()
        mask_c = mask

        if inactive_channels > 0:  # inactive_channels是维度，每个维度中的最大值，inactive应该是选前几个维度
            mask_c = mask.clone()
            topk_maxmum, _ = mask_d.topk(inactive_channels, dim=1, largest=False, sorted=False)
            clamp_max, _ = topk_maxmum.max(dim=1, keepdim=True)
            mask_index = mask_d.le(clamp_max)  # 1.重要性分数小于阈值的将被去除 2.剩余的维度会使用对应的重要性分数进行加权
            mask_c[mask_index] = 0  # 此处就是把小于阈值筛除

        mask_c = mask_c.view(b, c, 1, 1, 1)  # 2,16,1,1
        x = x * mask_c.expand_as(x)  # 每条支路的最大值
        return x  # 2,16,4,4


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
        # self.conv_1 = DynamicMultiHeadConv(in_channels, bottleneck * growth_rate, kernel_size=1, heads=heads,
        #                                    squeeze_rate=squeeze_rate, gate_factor=gate_factor)
        self.conv_1 = Flag3d(in_channels, bottleneck * growth_rate, kernel_size=1, stride=1, padding=0, dilation=1, groups=4)

        # 3x3 conv: bottleneck * k --> k
        self.conv_2 = Conv(bottleneck * growth_rate, growth_rate, kernel_size=3, padding=1, groups=group_3x3)

    def forward(self, x):
        x_ = x  # 2,16,100,4,4
        x = self.conv_1(x_)
        x = self.conv_2(x)
        x = torch.cat([x_, x], 1)  # 2,32,100,4,4
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


class FlgcdenseNet(nn.Module):
    def __init__(self, band, num_classes):
        super(FlgcdenseNet, self).__init__()
        self.name = 'FlgcdenseNet'
        # self.stages = [14, 14, 14]
        # self.growth = [8, 16, 32]
        # self.stages = [4, 6, 8]
        # self.growth = [8, 16, 32]
        self.stages = [10, 10, 10]
        self.growth = [8, 16, 32]
        self.progress = 0.0
        self.init_stride = 2
        self.pool_size = 7
        self.bottleneck = 4
        self.gate_factor = 0.25
        self.squeeze_rate = 16
        self.group_3x3 = 4
        # self.heads = 4
        self.heads = 2

        self.features = nn.Sequential()
        # Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]  # 2*8=16
        # Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv3d(1,
                                                        self.num_features,  # 16
                                                        kernel_size=3,
                                                        stride=self.init_stride,  # 3
                                                        padding=1,
                                                        bias=False))
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
        # if progress:  # x:2,1,200,7,7
        #     DynamicMultiHeadConv.global_progress = progress
        features = self.features(x)  # 2,800,25,1,1

        features = self.bn_last(features)
        features = self.relu_last(features)
        features = self.pool_last(features)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    # net = DydenseNet(200, 16)
    net = FlgcdenseNet(200, 16)
    print(net)
    from torchsummary import summary

    summary(net, input_size=[(1, 200, 7, 7)], batch_size=1)

    from thop import profile

    input = torch.randn(1, 1, 200, 7, 7)
    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    # print('   Number of params: %.2fM' % (total / 1e6))
    # print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))
    print('   Number of params: %.2f' % (total))
    print('   Number of FLOPs: %.2fFLOPs' % (flops))

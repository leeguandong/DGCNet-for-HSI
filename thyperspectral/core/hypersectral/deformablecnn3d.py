import functools
from math import sqrt
import torch.nn.functional as F
from thyperspectral.core.hypersectral.dcn.modules.deform_conv import *


class Deformablecnn(nn.Module):
    def __init__(self, upscale_factor, in_channel=1, out_channel=1, nf=64):
        super(Deformablecnn, self).__init__()
        self.upscale_factor = upscale_factor
        self.in_channel = in_channel

        self.input = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.residual_layer = self.make_layer(functools.partial(ResBlock_3d, nf), 5)
        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(60, out_channel)
                                # nn.Softmax()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        b, c, n, h, w = x.size()
        residual = F.interpolate(x[:, :, n // 2, :, :], scale_factor=self.upscale_factor, mode='bilinear',
                                 align_corners=False)
        out = self.input(x)
        out = self.residual_layer(out)


        x10 = self.global_pooling(out)
        x10 = x10.view(x10.size(0), -1)

        output = self.full_connection(x10)

        return out


class ResBlock_deformable3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_deformable3d, self).__init__()
        self.dcn0 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.dcn1 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x


class ResBlock_3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_3d, self).__init__()
        self.cnn0 = nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.cnn1 = nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.cnn1(self.lrelu(self.cnn0(x))) + x


if __name__ == "__main__":
    net = Net(4).cuda()
    from thop import profile

    input = torch.randn(1, 1, 7, 320, 180).cuda()
    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))

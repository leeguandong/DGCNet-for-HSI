import torch
import math
from torch import nn


class FDSSC_network(nn.Module):
    def __init__(self, band, classes):
        super(FDSSC_network, self).__init__()

        # spectral branch
        self.name = 'FDSSC'
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm1 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    nn.PReLU()
        )
        self.conv2 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    nn.PReLU()
        )
        self.conv3 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    nn.PReLU()
        )
        self.conv4 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm4 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        kernel_3d = math.ceil((band - 6) / 2)
        # print(kernel_3d)
        self.conv5 = nn.Conv3d(in_channels=60, out_channels=200, padding=(0, 0, 0),
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))

        self.batch_norm5 = nn.Sequential(
                                    nn.BatchNorm3d(1, eps=0.001, momentum=0.1, affine=True),
                                    nn.PReLU()
        )
        self.conv6 = nn.Conv3d(in_channels=1, out_channels=24, padding=(1, 1, 0),
                                kernel_size=(3, 3, 200), stride=(1, 1, 1))
        self.batch_norm6 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.PReLU()
        )
        self.conv7 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm7 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv8 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm8 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv9 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm9 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(60, classes)
                                # nn.Softmax()
        )


    def forward(self, X):
        # spectral
        x1 = self.conv1(X)
        #print('x11', x11.shape)
        x2 = self.batch_norm1(x1)
        x2 = self.conv2(x2)
        #print('x12', x12.shape)

        x3 = torch.cat((x1, x2), dim=1)
        #print('x13', x13.shape)
        x3 = self.batch_norm2(x3)
        x3 = self.conv3(x3)
        #print('x13', x13.shape)

        x4 = torch.cat((x1, x2, x3), dim=1)
        x4 = self.batch_norm3(x4)
        x4 = self.conv4(x4)

        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        # print('x15', x15.shape)

        # print(x5.shape)
        x6 = self.batch_norm4(x5)
        x6 = self.conv5(x6)
        #print('x16', x16.shape)  # 7*7*97, 60

        #print('x16', x16.shape)
        # 光谱注意力通道
        x6 = x6.permute(0, 4, 2, 3, 1)
        # print(x6.shape)

        x7 = self.batch_norm5(x6)
        x7 = self.conv6(x7)

        x8 = self.batch_norm6(x7)
        x8 = self.conv7(x8)

        x9 = torch.cat((x7, x8), dim=1)
        x9 = self.batch_norm7(x9)
        x9 = self.conv8(x9)

        x10 = torch.cat((x7, x8, x9), dim=1)
        x10 = self.batch_norm8(x10)
        x10 = self.conv9(x10)

        x10 = torch.cat((x7, x8, x9, x10), dim=1)
        x10 = self.batch_norm9(x10)
        x10 = self.global_pooling(x10)
        x10 = x10.view(x10.size(0), -1)

        output = self.full_connection(x10)
        # output = self.fc(x_pre)
        return output

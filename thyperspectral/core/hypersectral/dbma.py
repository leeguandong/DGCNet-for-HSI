import torch
import math
from torch import nn


class DBMA_network(nn.Module):
    def __init__(self, band, classes):
        super(DBMA_network, self).__init__()

        # spectral branch
        self.name = 'DBMA'
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Conv3d(in_channels=48, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm3d(72, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Conv3d(in_channels=72, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm3d(96, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=96, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))  # kernel size随数据变化

        # 注意力机制模块

        # self.max_pooling1 = nn.MaxPool3d(kernel_size=(7, 7, 1))
        # self.avg_pooling1 = nn.AvgPool3d(kernel_size=(7, 7, 1))
        self.max_pooling1 = nn.AdaptiveAvgPool3d(1)
        self.avg_pooling1 = nn.AdaptiveAvgPool3d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv3d(in_channels=60, out_channels=30,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(in_channels=30, out_channels=60,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )
        # self.fc11 = Dense(30, activation=None, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
        # self.fc12 = Dense(60, activation=None, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

        self.activation1 = nn.Sigmoid()

        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        # 注意力机制模块

        # self.max_pooling2 = nn.MaxPool3d(kernel_size=(1, 1, 60))
        # self.avg_pooling2 = nn.AvgPool3d(kernel_size=(1, 1, 60))
        # self.max_pooling2 = nn.AdaptiveAvgPool3d(1)
        # self.avg_pooling2 = nn.AdaptiveAvgPool3d(1)

        self.conv25 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                      kernel_size=(3, 3, 2), stride=(1, 1, 1)),
            nn.Sigmoid()
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
            nn.Linear(120, classes)  # ,
            # nn.Softmax()
        )

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        # print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        # print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        # print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        # print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        # print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        x_max1 = self.max_pooling1(x16)
        x_avg1 = self.avg_pooling1(x16)
        # print('x_max1', x_max1.shape)


        # x_max1 = self.fc11(x_max1)
        # x_max1 = self.fc12(x_max1)
        #
        # x1_avg1 = self.fc11(x_avg1)
        # x1_avg1 = self.fc12(x_avg1)
        # print('x_max1', x_max1.shape)
        # x_max1 = x_max1.view(x_max1.size(0), -1)
        # x_avg1 = x_avg1.view(x_avg1.size(0), -1)
        # print('x_max1', x_max1.shape)
        x_max1 = self.shared_mlp(x_max1)
        x_avg1 = self.shared_mlp(x_avg1)
        # print('x_max1', x_max1.shape)
        x1 = torch.add(x_max1, x_avg1)
        x1 = self.activation1(x1)

        # x1 = x1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print('x1', x1.shape)
        # print('x16', x16.shape)

        # x1 = multiply([x1, x16])
        # x1 = self.activation1(x1)
        x1 = torch.mul(x1, x16)
        # print('x1', x1.shape)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        # print('x1', x1.shape)
        # x1 = Reshape(target_shape=(7, 7, 1, 60))(x1)
        # x1 = GlobalAveragePooling3D()(x1)

        # spatial
        # print('x', X.shape)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        # print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        # print('x25', x25.shape)

        # 空间注意力机制
        # x_max2 = self.max_pooling2(x25)
        # x_avg2 = self.avg_pooling2(x25)
        # x_avg2 = x_avg2.permute(0, 4, 2, 3, 1)
        x_avg2 = torch.mean(x25, dim=1, keepdim=True)
        x_max2, _ = torch.max(x25, dim=1, keepdim=True)
        # print('x_avg2', x_avg2.shape)

        x2 = torch.cat((x_max2, x_avg2), dim=-1)
        x2 = self.conv25(x2)
        # print('x2', x2.shape)
        # print('x25', x25.shape)


        x2 = torch.mul(x2, x25)
        # print('x2', x2.shape)
        x2 = self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)
        # x2 = Reshape(target_shape=(7, 7, 1, 60))(x2)
        # x2 = GlobalAveragePooling3D()(x2)

        # print('x1', x1.shape)
        # print('x2', x2.shape)

        x_pre = torch.cat((x1, x2), dim=1)
        # print('x_pre', x_pre.shape)
        x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        return output
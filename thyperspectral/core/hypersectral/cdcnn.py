'''
https://blog.csdn.net/edogawachia/article/details/88674649
Going Deeper with Contextual CNN for Hyperspectral Image Classification
'''
import torch
from torch import nn
from .basicblock import Residual_2D

class CDCNN_network(nn.Module):
    def __init__(self, band, classes):
        super(CDCNN_network, self).__init__()
        self.name = 'CDCNN'

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=band, out_channels=128, kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=(5, 5))
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=band, out_channels=128, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3))
        )
        self.conv13 = nn.Conv2d(in_channels=band, out_channels=128, kernel_size=(5, 5))

        self.batch_normal1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True)
        )
        self.conv2 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=(1, 1))
        self.res_net1 = Residual_2D(128, 128, (1, 1), (0, 0), batch_normal=True)
        self.res_net2 = Residual_2D(128, 128, (1, 1), (0, 0))

        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1))
        )
        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1))
        )
        self.conv5 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1))
        )

        self.full_connection = nn.Sequential(
            nn.Linear(128, classes)
            # nn.Sigmoid()
        )

    def forward(self, X):
        X = X.squeeze(1).permute(0, 3, 1, 2)
        x11 = self.conv11(X)
        x12 = self.conv12(X)
        x13 = self.conv13(X)

        x1 = torch.cat((x11, x12, x13), dim=1)
        x1 = self.conv2(x1)
        x1 = self.res_net1(x1)
        x1 = self.res_net2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        x1 = x1.view(x1.shape[0], -1)
        return self.full_connection(x1)

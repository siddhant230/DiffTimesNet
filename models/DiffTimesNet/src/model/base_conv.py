import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DiffTimesNet.src.model.utilities_model.inception import Inception_Block_V1


class BaseNet(nn.Module):
    def __init__(self, input_channels=50, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, 3, padding=1)
        self.conv2 = nn.Conv2d(
            input_channels*2, input_channels*4, 3, padding=1)
        self.conv3 = nn.Conv2d(input_channels*4, out_channels, 3, padding=1)
        self.projection1 = nn.Conv2d(out_channels, input_channels, 1)
        self.projection2 = nn.Conv2d(out_channels, input_channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        q = F.softmax(self.projection1(x))
        x = self.projection2(x)
        x = x*q
        return x


class LargeConv(nn.Module):
    def __init__(self, input_channels=50, out_channels=64,
                 hidden_channels=64):
        super().__init__()
        self.input_channels = input_channels
        self.conv = nn.Sequential(
            Inception_Block_V1(input_channels, hidden_channels,
                               num_kernels=6),
            nn.GELU(),
            Inception_Block_V1(hidden_channels, hidden_channels*2,
                               num_kernels=6),
            nn.GELU(),
            Inception_Block_V1(hidden_channels*2, hidden_channels*4,
                               num_kernels=6),
            nn.GELU(),
            Inception_Block_V1(hidden_channels*4, out_channels,
                               num_kernels=6),
            nn.GELU(),
            Inception_Block_V1(out_channels, input_channels,
                               num_kernels=6)
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, input_channels=50, out_channels=64):
        super().__init__()
        self.input_channels = input_channels
        self.backbone = BaseNet(input_channels, out_channels)
        self.conv = nn.Sequential(
            Inception_Block_V1(input_channels, out_channels,
                               num_kernels=6),
            nn.GELU(),
            Inception_Block_V1(out_channels, input_channels,
                               num_kernels=6)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.conv(x)


if __name__ == '__main__':
    conv_block = ConvBlock(50)
    net = BaseNet(input_channels=50,
                  output_channels=128)

    x = torch.zeros((16, 50, 64, 132))
    print(x.shape)

    y = net(x)
    print(y.shape)

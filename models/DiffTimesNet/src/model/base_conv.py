import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.utilities_model import Inception_Block_V1


class ConvBlock(nn.Module):
    def __init__(self, input_channels=50, out_channels=64):
        super().__init__()
        self.input_channels = input_channels

        self.conv = nn.Sequential(
            Inception_Block_V1(input_channels, out_channels,
                               num_kernels=6),
            nn.GELU(),
            Inception_Block_V1(out_channels, input_channels,
                               num_kernels=6)
        )

    def forward(self, x):
        return self.conv(x)


class Net(nn.Module):
    def __init__(self, input_channels=50, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, 3, padding=1)
        self.conv2 = nn.Conv2d(
            input_channels*2, input_channels*4, 3, padding=1)
        self.conv3 = nn.Conv2d(input_channels*4, out_channels, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x


if __name__ == '__main__':
    conv_block = ConvBlock(50)
    net = Net(input_channels=50,
              output_channels=128)

    x = torch.zeros((16, 50, 64, 132))
    print(x.shape)

    y = net(x)
    print(y.shape)

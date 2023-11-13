import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels=50):
        super().__init__()
        self.input_channels = input_channels
        self.model = torch.hub.load('pytorch/vision:v0.10.0',
                                    'inception_v3', pretrained=True)
        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                                     stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)


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

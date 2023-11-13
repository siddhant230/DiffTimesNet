import numpy as np
import torch
import torch.nn as nn

from siren_model.siren import SirenBlock
from base_conv import Net
from aggregator import Aggregator


class ResidualBlock(nn.Module):
    def __init__(
        self,
        T=1200,
        input_channels=50,
        max_num_periods=64,
        out_channels=128,
        agg_channels=1,
        use_1x1conv=True
    ):
        super().__init__()
        self.sirenblock = SirenBlock(
            max_num_periods=max_num_periods, window_length=(T//max_num_periods)+1)
        self.conv_net = Net(input_channels, out_channels)
        self.agg = Aggregator(max_num_periods, agg_channels)
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, out_channels,
                                   kernel_size=1)

    def forward(self, x):
        y = self.sirenblock(x)
        y = self.conv_net(y)
        y = self.agg(y)

        if self.conv3:
            x = self.conv3(x)
            y = x + y
        return y


if __name__ == "__main__":
    bs = 16
    ch = 50
    T = 1200

    x = torch.zeros((bs, ch, T))
    print(x.shape)
    residual_block = ResidualBlock(
        T=1200,
        input_channels=50,
        max_num_periods=64,
        out_channels=128,
        agg_channels=1,
        use_1x1conv=True
    )
    y = residual_block(x)
    print(y.shape)

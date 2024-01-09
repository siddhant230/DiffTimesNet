import numpy as np
import torch
import torch.nn as nn

from models.DiffTimesNet.src.model.siren_model.siren import SirenBlock
from models.DiffTimesNet.src.model.base_conv import Net, ConvBlock
from models.DiffTimesNet.src.model.aggregator import Aggregator


class DiffTimesBlock(nn.Module):
    def __init__(
        self,
        input_channels=50,
        max_num_periods=5,
        out_channels=128,
        agg_channels=1,
        use_1x1conv=False,
        num_siren_depth=3
    ):
        super().__init__()
        self.sirenblock = SirenBlock(max_num_periods=max_num_periods,
                                     window_length=max_num_periods,
                                     num_layers=num_siren_depth)

        self.conv_net = ConvBlock(input_channels, out_channels)
        self.agg = Aggregator(max_num_periods, output_dim=agg_channels)
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, out_channels,
                                   kernel_size=1)

    def forward(self, x):
        B, N, T = x.shape

        y = self.sirenblock(x)
        y = self.conv_net(y)
        y = self.agg(y)
        y = y[:, :, :T]
        y = x + y
        return y


if __name__ == "__main__":
    bs = 16
    ch = 50
    T = 1200

    x = torch.zeros((bs, ch, T))
    print(x.shape)
    residual_block = DiffTimesBlock(
        T=1200,
        input_channels=50,
        max_num_periods=64,
        out_channels=128,
        agg_channels=1,
        use_1x1conv=True
    )
    y = residual_block(x)
    print(y.shape)

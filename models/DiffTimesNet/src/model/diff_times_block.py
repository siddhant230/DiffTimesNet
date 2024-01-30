import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DiffTimesNet.src.model.siren_model.siren import SirenBlock
from models.DiffTimesNet.src.model.base_conv import BaseNet, ConvBlock, ConvBlock_exp2
from models.DiffTimesNet.src.model.aggregator import Aggregator


class DiffTimesBlock(nn.Module):
    def __init__(
        self,
        time_period,
        input_channels=50,
        max_num_periods=5,
        out_channels=128,
        agg_channels=1,
        use_1x1conv=False,
        num_siren_depth=3
    ):
        super().__init__()
        self.sirenblock = SirenBlock(time_period=time_period,
                                     max_num_periods=max_num_periods,
                                     window_length=max_num_periods,
                                     num_layers=num_siren_depth)

        self.conv_net = ConvBlock(input_channels, out_channels)
        self.agg = Aggregator(max_num_periods, output_dim=agg_channels)
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, out_channels,
                                   kernel_size=1)

    def pad_sequence_util(self, seq, max_len):
        delta = max_len - seq.shape[-1]
        seq = F.pad(input=seq, pad=(0, delta, 0, 0), mode='constant', value=0)
        return seq

    def forward(self, x):
        B, N, T = x.shape

        y, attn_map = self.sirenblock(x)
        y = self.conv_net(y)
        y = self.agg(y)
        y = self.pad_sequence_util(y, x.shape[-1])
        y = y[:, :, :T]
        y = x + y
        return y, attn_map


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

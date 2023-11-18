import numpy as np
import torch
import torch.nn as nn

from model.diff_times_block import DiffTimesBlock


class DiffTimesNet(nn.Module):
    def __init__(self, in_channels=50,
                 hidden_channels=64,
                 max_num_periods=9,
                 num_res_blocks=4,
                 reduction_factor=1):

        super().__init__()
        reduction_factor = reduction_factor
        diff_time_blocks = []
        for _ in range(num_res_blocks):
            diff_time_blocks.append(
                DiffTimesBlock(
                    input_channels=in_channels,
                    max_num_periods=max_num_periods,
                    out_channels=hidden_channels,
                    agg_channels=max_num_periods,
                    use_1x1conv=False
                )
            )
            max_num_periods = max_num_periods-(reduction_factor*2)

        self.model = nn.Sequential(*diff_time_blocks)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        x = self.model(x)
        x = x.permute(0, 1, 2)  # (B, C, T) -> (B, T, C)
        return x


if __name__ == "__main__":
    bs = 16
    ch = 50
    T = 1200

    x = torch.zeros((bs, ch, T))
    print(x.shape)

    in_channels = 32
    hidden_channels = 64
    num_res_blocks = 5
    reduction_factor = 1
    max_num_periods = (num_res_blocks*(2*reduction_factor))+1

    resnet_obj = DiffTimesNet(in_channels=in_channels,
                              hidden_channels=hidden_channels,
                              max_num_periods=max_num_periods,
                              num_res_blocks=num_res_blocks)

    y = resnet_obj(x)
    print(y.shape)

import numpy as np
import torch
import torch.nn as nn

from residual_block import ResidualBlock


class ResNet(nn.Module):
    def __init__(self, in_channels=50, hidden_channels=64, window_length=5):
        super().__init__()
        self.residual_block1 = ResidualBlock(
            input_channels=in_channels,
            max_num_periods=64,
            out_channels=hidden_channels,
            window_length=window_length,
            agg_channels=1,
            use_1x1conv=True
        )

        self.drop_dim_1d_1 = nn.Conv1d(hidden_channels, hidden_channels,
                                       kernel_size=window_length,
                                       stride=2)

        self.residual_block2 = ResidualBlock(
            input_channels=hidden_channels,
            max_num_periods=64,
            out_channels=hidden_channels*2,
            agg_channels=1,
            use_1x1conv=True
        )

        self.drop_dim_1d_2 = nn.Conv1d(hidden_channels*2, hidden_channels*2,
                                       kernel_size=window_length,
                                       stride=2)

        self.residual_block3 = ResidualBlock(
            input_channels=hidden_channels*2,
            max_num_periods=64,
            out_channels=hidden_channels*4,
            agg_channels=1,
            use_1x1conv=True
        )

    def forward(self, x):
        x = self.residual_block1(x)
        x = self.drop_dim_1d_1(x)
        x = self.residual_block2(x)
        x = self.drop_dim_1d_2(x)
        x = self.residual_block3(x)
        return x


if __name__ == "__main__":
    bs = 16
    ch = 50
    T = 1200

    x = torch.zeros((bs, ch, T))
    print(x.shape)

    resnet_obj = ResNet()
    y = resnet_obj(x)
    print(y.shape)

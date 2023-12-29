import torch
import torch.nn as nn


class Aggregator(nn.Module):
    def __init__(
        self,
        max_num_periods,
        output_dim
    ):
        super().__init__()
        self.pointwise_agg = nn.Conv2d(
            max_num_periods, output_dim, kernel_size=1)

    def forward(self, x):
        bs, ftrs, period, freq = x.shape
        x = x.permute(0, 2, 1, 3)
        x = self.pointwise_agg(x)
        x = x.permute(0, 1, 2, 3)
        x = x.view(bs, ftrs, -1)
        return x


if __name__ == '__main__':
    max_num_periods = 56
    x = torch.zeros((16, 128, max_num_periods, 124))
    print(x.shape)

    merge_dim = 8
    agg = Aggregator(max_num_periods, merge_dim)
    y = agg(x)
    print(y.shape)

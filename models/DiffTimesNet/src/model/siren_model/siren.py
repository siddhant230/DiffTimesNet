import torch
import torch.nn as nn
from models.DiffTimesNet.src.model.siren_model.layer import SirenLayer, BlockAttention


class SirenBlock(nn.Module):
    def __init__(
            self, time_period,
            max_num_periods,
            window_length,
            num_layers=5, max_stride=9
    ):
        super().__init__()
        self.time_period = time_period
        self.num_layers = num_layers
        self.max_num_periods = max_num_periods
        self.window_length = window_length
        net = []
        in_features = 1
        self.padding = 1
        self.stride = min(max_stride, (window_length//2)+1)
        x_prime_out_shape = time_period
        for _ in range(num_layers):
            x_prime_out_shape = self.out_calcualtor(x_prime_out_shape,
                                                    window_length, self.stride,
                                                    self.padding)
            net.append(
                SirenLayer(
                    in_features=in_features,
                    kernel_size=window_length,
                    out_features=max_num_periods,
                    stride=self.stride, padding=self.padding))

            in_features = max_num_periods
            self.stride, self.padding = 1, 1

        self.model = nn.Sequential(*net)
        self.block_attention = BlockAttention(x_dim=(max_num_periods,
                                                     x_prime_out_shape))

    def out_calcualtor(self, x0, kernel, stride, pad):
        x_prime = ((x0 - kernel + 2*pad)/stride)+1
        return int(x_prime)

    def forward(self, x, attn_map=None):
        bs, ch, T = x.shape
        x = x.view(-1, 1, T)
        x = self.model(x)  # (b*x, ch, time)
        x, attn_map = self.block_attention(x)
        out = x.view(bs, ch, self.max_num_periods, x.shape[-1])

        return out, attn_map


if __name__ == '__main__':
    bs = 16
    ch = 50
    T = 1929

    x = torch.zeros((bs, ch, T))
    print(x.shape)

    max_num_periods = 64
    w_len = (T//max_num_periods)+1
    print(w_len)
    num_layers = 5

    siren = SirenBlock(time_period=T, max_num_periods=max_num_periods,
                       window_length=w_len, num_layers=num_layers)
    y = siren(x)
    print(f"SIREN : {x.shape} -> {y.shape}")

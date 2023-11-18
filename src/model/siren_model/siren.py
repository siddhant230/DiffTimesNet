import torch
import torch.nn as nn
from src.model.siren_model.layer import SirenLayer


class SirenBlock(nn.Module):
    def __init__(
            self,
            max_num_periods,
            window_length,
            num_layers=5
    ):
        super().__init__()
        self.num_layers = num_layers
        self.max_num_periods = max_num_periods
        self.window_length = window_length
        net = []
        in_features = 1
        padding = 0
        stride = (window_length//2)+1

        for _ in range(num_layers):

            net.append(
                SirenLayer(
                    in_features=in_features,
                    kernel_size=window_length,
                    out_features=max_num_periods,
                    stride=stride, padding=padding))

            in_features = max_num_periods
            stride, padding = 1, 1

        self.model = nn.Sequential(*net)

    def forward(self, x):
        bs, ch, T = x.shape
        x = x.view(-1, 1, T)
        x = self.model(x)
        out = x.view(bs, ch, self.max_num_periods, x.shape[-1])
        return out


if __name__ == '__main__':
    bs = 16
    ch = 50
    T = 1200

    x = torch.zeros((bs, ch, T))
    print(x.shape)

    max_num_periods = 64
    siren = SirenBlock(max_num_periods=max_num_periods,
                       window_length=(T//max_num_periods)+1)
    y = siren(x)
    print(f"SIREN : {x.shape} -> {y.shape}")

import numpy as np
import torch
import torch.nn as nn


def paper_init_(weight, is_first=False, omega=1):
    in_features = weight.shape[1]

    with torch.no_grad():
        if is_first:
            bound = 1 / in_features
        else:
            bound = np.sqrt(6 / in_features) / omega

        weight.uniform_(-bound, bound)


class SirenLayer(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            is_first=False,
            omega=30, padding=0,
            kernel_size=3, stride=10,
            custom_init_function_=None,
    ):
        super().__init__()
        self.omega = omega
        self.padding = padding
        self.stride = stride
        self.out_features = out_features
        self.linear = nn.Conv1d(in_features, out_features,
                                kernel_size=kernel_size, bias=bias,
                                stride=stride, padding=padding)

        if custom_init_function_ is None:
            paper_init_(self.linear.weight, is_first=is_first, omega=omega)
        else:
            custom_init_function_(self.linear.weight)

    def forward(self, x):
        out = torch.sin(self.omega * self.linear(x))
        return out

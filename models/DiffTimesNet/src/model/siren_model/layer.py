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


class Attention(nn.Module):
    def __init__(self, dims, n_heads, qkv_bias=False,
                 attention_drop_p=0.5, projection_drop_p=0.5):
        super().__init__()

        self.dims = dims
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        self.attention_drop_p = attention_drop_p
        self.projection_drop_p = projection_drop_p

        self.head_dim = dims // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dims, self.dims*3, bias=self.qkv_bias)
        self.attention_dropout = nn.Dropout(attention_drop_p)
        self.projection = nn.Linear(self.dims, self.dims)
        self.projection_dropout = nn.Dropout(projection_drop_p)

    def forward(self, x):
        n_samples, n_tokens, dims = x.shape

        if dims != self.dims:
            raise ValueError("dimensions mismatch")
        qkv = self.qkv(x)
        # for qkv
        qkv = qkv.reshape(n_samples, n_tokens, 3,
                          self.n_heads, self.head_dim
                          ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        dot_prouct_q_k = (query @ key.transpose(-2, -1)) * self.scale
        attention = dot_prouct_q_k.softmax(dim=-1)
        attention = self.attention_dropout(attention)
        context = attention @ value
        context = context.transpose(1, 2).flatten(2)
        out = self.projection(context)
        out = self.projection_dropout(out)
        return out


class BlockAttention(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.attn_block_period = Attention(x_dim[0], n_heads=2)
        self.attn_block_ftrs = Attention(x_dim[1], n_heads=2)
        self.alpha = torch.nn.Parameter(torch.randn((1, 1)))
        print(self.alpha)

    def forward(self, x):
        x_p = self.attn_block_period(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_ftrs = self.attn_block_ftrs(x)
        x_out = self.alpha*x_p + (1-self.alpha)*x_ftrs
        return x_out

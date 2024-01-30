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
    def __init__(self, dim, hidden_dim=16, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.project0 = nn.Linear(dim, hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(hidden_dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        x = self.project0(x)
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        # (n_samples, n_heads, head_dim, n_patches + 1)
        k_t = k.transpose(-2, -1)
        dp = (
            q @ k_t
        ) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)
        # TODO : sparsity loss calc to be here
        # attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(
            1, 2)
        weighted_avg = weighted_avg.flatten(2)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        return x, attn


class BlockAttention(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        # self.attn_block_ftrs = Attention(x_dim[0], n_heads=1)
        self.attn_block_period = Attention(x_dim[1], n_heads=2)
        self.alpha = torch.nn.Parameter(torch.randn((1, 1)))
        # self.beta = torch.nn.Parameter(torch.randn((1, 1)))

    def forward(self, x, attn_map=None):
        # x_ftrs, attn_map = self.attn_block_ftrs(
        #     x.permute(0, 2, 1))
        # x_ftrs = x_ftrs.permute(0, 2, 1)
        x_p, attn_map = self.attn_block_period(x)
        x_out = self.alpha*x_p  # + self.beta*x_ftrs
        return x_out, attn_map.mean(dim=1).squeeze(1)

import numpy as np
import torch
import torch.nn as nn


def paper_init_(weight, is_first=False, omega=1):
    """Initialize the weigth of the Linear layer.

    Parameters
    ----------
    weight : torch.Tensor
        The learnable 2D weight matrix.

    is_first : bool
        If True, this Linear layer is the very first one in the network.

    omega : float
        Hyperparamter.
    """
    in_features = weight.shape[1]

    with torch.no_grad():
        if is_first:
            bound = 1 / in_features
        else:
            bound = np.sqrt(6 / in_features) / omega

        weight.uniform_(-bound, bound)


class SirenLayer(nn.Module):
    """Linear layer followed by the sine activation.

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of output features.

    bias : bool
        If True, the bias is included.

    is_first : bool
        If True, then it represents the first layer of the network. Note that
        it influences the initialization scheme.

    omega : int
        Hyperparameter. Determines scaling.

    custom_init_function_ : None or callable
        If None, then we are going to use the `paper_init_` defined above.
        Otherwise, any callable that modifies the `weight` parameter in place.

    Attributes
    ----------
    linear : nn.Linear
        Linear layer.
    """

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            is_first=False,
            omega=30,
            kernel_size=10,
            custom_init_function_=None,
    ):
        super().__init__()
        self.omega = omega
        self.out_features = out_features
        self.linear = nn.Conv1d(in_features, out_features,
                                kernel_size=kernel_size, bias=bias,
                                stride=kernel_size//2)

        if custom_init_function_ is None:
            paper_init_(self.linear.weight, is_first=is_first, omega=omega)
        else:
            custom_init_function_(self.linear.weight)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, in_features)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, out_features).
        """
        out = torch.sin(self.omega * self.linear(x))
        return out

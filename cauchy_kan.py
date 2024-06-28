import torch
import torch.nn.functional as F
import math
import torch
import torch.nn as nn


class CauchyKANLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(CauchyKANLinear, self).__init__()
        scale_noise = 0.1
        scale_base = 1.0
        scale_cauchy = 1.0
        enable_standalone_scale_cauchy = True
        base_activation = torch.nn.SiLU
        
        self.in_features = in_features
        self.out_features = out_features

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.cauchy_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        if enable_standalone_scale_cauchy:
            self.cauchy_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_cauchy = scale_cauchy
        self.enable_standalone_scale_cauchy = enable_standalone_scale_cauchy
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.out_features, self.in_features) - 0.5)
                * self.scale_noise
            )
            self.cauchy_weight.data.copy_(
                (self.scale_cauchy if not self.enable_standalone_scale_cauchy else 1.0)
                * noise
            )
            if self.enable_standalone_scale_cauchy:
                torch.nn.init.kaiming_uniform_(self.cauchy_scaler, a=math.sqrt(5) * self.scale_cauchy)

    def cauchy_kernel(self, x: torch.Tensor):
        """
        Compute the Cauchy kernel for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Cauchy kernel tensor of shape (batch_size, in_features).
        """
        gamma = 1.0  # You can adjust this parameter for the Cauchy kernel
        return 1 / (1 + gamma * x.pow(2))

    @property
    def scaled_cauchy_weight(self):
        return self.cauchy_weight * (
            self.cauchy_scaler if self.enable_standalone_scale_cauchy else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        cauchy_kernel = self.cauchy_kernel(x)  # (batch_size, in_features)
        scaled_weights = self.scaled_cauchy_weight  # (in_features, out_features)

        cauchy_output = torch.einsum('bi,ji->bj', cauchy_kernel, scaled_weights)

        output = base_output + cauchy_output
        output = output.view(*original_shape[:-1], self.out_features)
        return output

    def regularization_loss(self, regularize_cauchy=1.0, regularize_l1=1.0):
        regularization_loss_cauchy = self.cauchy_weight.abs().sum()
        regularization_loss_l1 = self.base_weight.abs().sum()
        return (
            regularize_cauchy * regularization_loss_cauchy
            + regularize_l1 * regularization_loss_l1
        )

class CauchyKANMNIST(nn.Module):
    def __init__(self):
        super(CauchyKANMNIST, self).__init__()
        self.layer1 = CauchyKANLinear(28*28, 32)
        self.ln1 = nn.LayerNorm(32)
        self.layer2 = CauchyKANLinear(32, 16)
        self.ln2 = nn.LayerNorm(16)
        self.layer3 = CauchyKANLinear(16, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.layer1(x)
        x = self.ln1(x)
        x = self.layer2(x)
        x = self.ln2(x)
        x = self.layer3(x)
        return x

    def regularization_loss(self, regularize_cauchy=1.0, regularize_l1=1.0):
        return sum(
            layer.regularization_loss(regularize_cauchy, regularize_l1)
            for layer in self.children() if isinstance(layer, CauchyKANLinear)
        )

class CauchyKAN(nn.Module):
    def __init__(self, n_features=28*28):
        super(CauchyKAN, self).__init__()
        self.n_features = n_features
        self.layer1 = CauchyKANLinear(self.n_features, 32)
        self.ln1 = nn.LayerNorm(32)
        self.layer2 = CauchyKANLinear(32, 16)
        self.ln2 = nn.LayerNorm(16)
        self.layer3 = CauchyKANLinear(16, 10)

    def forward(self, x):
        x = x.view(-1, self.n_features)  # Flatten the images
        x = self.layer1(x)
        x = self.ln1(x)
        x = self.layer2(x)
        x = self.ln2(x)
        x = self.layer3(x)
        return x

    def regularization_loss(self, regularize_cauchy=1.0, regularize_l1=1.0):
        return sum(
            layer.regularization_loss(regularize_cauchy, regularize_l1)
            for layer in self.children() if isinstance(layer, CauchyKANLinear)
        )
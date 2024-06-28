import torch
import torch.nn.functional as F
import math
import torch
import torch.nn as nn

class ChebyshevKANLinearMNIST(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        chebyshev_degree=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_chebyshev=1.0,
        enable_standalone_scale_chebyshev=True,
        base_activation=torch.nn.SiLU,
    ):
        super(ChebyshevKANLinearMNIST, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chebyshev_degree = chebyshev_degree

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.chebyshev_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, chebyshev_degree + 1)
        )
        if enable_standalone_scale_chebyshev:
            self.chebyshev_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_chebyshev = scale_chebyshev
        self.enable_standalone_scale_chebyshev = enable_standalone_scale_chebyshev
        self.base_activation = base_activation()

        self.reset_parameters()

        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.out_features, self.in_features, self.chebyshev_degree + 1) - 0.5)
                * self.scale_noise
            )
            self.chebyshev_weight.data.copy_(
                (self.scale_chebyshev if not self.enable_standalone_scale_chebyshev else 1.0)
                * noise
            )
            if self.enable_standalone_scale_chebyshev:
                torch.nn.init.kaiming_uniform_(self.chebyshev_scaler, a=math.sqrt(5) * self.scale_chebyshev)


    def chebyshev_polynomials(self, x: torch.Tensor):
        """
        Compute the Chebyshev polynomial bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Chebyshev bases tensor of shape (batch_size, in_features, chebyshev_degree + 1).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        x = torch.tanh(x)
        T = [torch.ones_like(x), x]
        for k in range(2, self.chebyshev_degree + 1):
            T_k = 2 * x * T[-1] - T[-2]
            T.append(T_k)

        bases = torch.stack(T, dim=-1)
        assert bases.size() == (x.size(0), self.in_features, self.chebyshev_degree + 1)
        return bases

    @property
    def scaled_chebyshev_weight(self):
        return self.chebyshev_weight * (
            self.chebyshev_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_chebyshev
            else 1.0
        )
    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        chebyshev_polynomials = self.chebyshev_polynomials(x)
        scaled_weights = self.scaled_chebyshev_weight

        chebyshev_output = torch.einsum('bik,jik->bj', chebyshev_polynomials, scaled_weights)

        output = base_output + chebyshev_output
        output = output.view(*original_shape[:-1], self.out_features)
        return output
    def forward2(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        chebyshev_output = F.linear(
            self.chebyshev_polynomials(x).view(x.size(0), -1),
            self.scaled_chebyshev_weight.view(self.out_features, -1),
        )
        output = base_output + chebyshev_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.chebyshev_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class ChebyshevKANMNIST(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        chebyshev_degree=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_chebyshev=1.0,
        base_activation=torch.nn.SiLU,
    ):
        super(ChebyshevKANMNIST, self).__init__()
        self.chebyshev_degree = chebyshev_degree

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                ChebyshevKANLinearMNIST(
                    in_features,
                    out_features,
                    chebyshev_degree=chebyshev_degree,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_chebyshev=scale_chebyshev,
                    base_activation=base_activation,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
        

class ChebyshevKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        chebyshev_degree=5
    ):
        super(ChebyshevKANLinear, self).__init__()
        scale_noise=0.1
        scale_base=1.0
        scale_chebyshev=1.0
        enable_standalone_scale_chebyshev=True
        base_activation=torch.nn.SiLU
        self.in_features = in_features
        self.out_features = out_features
        self.chebyshev_degree = chebyshev_degree

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.chebyshev_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, chebyshev_degree + 1)
        )
        if enable_standalone_scale_chebyshev:
            self.chebyshev_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_chebyshev = scale_chebyshev
        self.enable_standalone_scale_chebyshev = enable_standalone_scale_chebyshev
        self.base_activation = base_activation()

        self.reset_parameters()

        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.out_features, self.in_features, self.chebyshev_degree + 1) - 0.5)
                * self.scale_noise
            )
            self.chebyshev_weight.data.copy_(
                (self.scale_chebyshev if not self.enable_standalone_scale_chebyshev else 1.0)
                * noise
            )
            if self.enable_standalone_scale_chebyshev:
                torch.nn.init.kaiming_uniform_(self.chebyshev_scaler, a=math.sqrt(5) * self.scale_chebyshev)


    def chebyshev_polynomials(self, x: torch.Tensor):
        """
        Compute the Chebyshev polynomial bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Chebyshev bases tensor of shape (batch_size, in_features, chebyshev_degree + 1).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        x = torch.tanh(x)
        T = [torch.ones_like(x), x]
        for k in range(2, self.chebyshev_degree + 1):
            T_k = 2 * x * T[-1] - T[-2]
            T.append(T_k)

        bases = torch.stack(T, dim=-1)
        assert bases.size() == (x.size(0), self.in_features, self.chebyshev_degree + 1)
        return bases

    @property
    def scaled_chebyshev_weight(self):
        return self.chebyshev_weight * (
            self.chebyshev_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_chebyshev
            else 1.0
        )
    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        chebyshev_polynomials = self.chebyshev_polynomials(x
        scaled_weights = self.scaled_chebyshev_weight

        chebyshev_output = torch.einsum('bik,jik->bj', chebyshev_polynomials, scaled_weights)

        output = base_output + chebyshev_output
        output = output.view(*original_shape[:-1], self.out_features)
        return output
    def forward2(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        chebyshev_output = F.linear(
            self.chebyshev_polynomials(x).view(x.size(0), -1),
            self.scaled_chebyshev_weight.view(self.out_features, -1),
        )
        output = base_output + chebyshev_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.chebyshev_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class ChebyshevKAN(nn.Module):
    def __init__(self,n_features=28*28):
        super(ChebyshevKAN, self).__init__()
        self.n_features = n_features
        self.chebykan1 = ChebyshevKANLinear(n_features, 32, 4)
        self.ln1 = nn.LayerNorm(32)
        self.chebykan2 = ChebyshevKANLinear(32, 16, 4)
        self.ln2 = nn.LayerNorm(16)
        self.chebykan3 = ChebyshevKANLinear(16, 10, 4)

    def forward(self, x):
        x = x.view(-1, self.n_features)
        x = self.chebykan1(x)
        x = self.ln1(x)
        x = self.chebykan2(x)
        x = self.ln2(x)
        x = self.chebykan3(x)
        return x
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
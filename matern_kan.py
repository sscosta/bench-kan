import torch
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
class MaternKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        nu=1.5,
    ):
        super(MaternKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(0, grid_size) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.nu = nu

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T, noise)
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def matern_kernel(self, x: torch.Tensor):
        """
        Compute the Matern kernel for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Matern kernel tensor of shape (batch_size, in_features, grid_size).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        distance = torch.abs(x.unsqueeze(-1) - self.grid)
        if self.nu == 0.5:
            kernel = torch.exp(-distance)
        elif self.nu == 1.5:
            sqrt_3 = math.sqrt(3)
            kernel = (1.0 + sqrt_3 * distance) * torch.exp(-sqrt_3 * distance)
        elif self.nu == 2.5:
            sqrt_5 = math.sqrt(5)
            kernel = (1.0 + sqrt_5 * distance + (5.0 / 3.0) * distance ** 2) * torch.exp(-sqrt_5 * distance)
        else:
            raise ValueError(f"Unsupported value of nu: {self.nu}")

        assert kernel.size() == (x.size(0), self.in_features, self.grid_size)
        return kernel.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points using Laplacian kernel.
    
        Args:
            x (torch.Tensor): Input tensor of shape (grid_size, in_features).
            y (torch.Tensor): Output tensor of shape (grid_size, in_features, out_features).
    
        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (self.grid_size, self.in_features, self.out_features)
    
        #
        distances = torch.cdist(x, self.grid.T)
        weights = (-distances).relu()
    
        A = weights
        B = y.permute(1, 2, 0).reshape(self.in_features * self.out_features, self.grid_size).T 
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.T.reshape(self.out_features, self.in_features, self.grid_size)
    
        assert result.size() == (self.out_features, self.in_features, self.grid_size)
        return result.contiguous()


    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.matern_kernel(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.matern_kernel(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size, dtype=torch.int64, device=x.device)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / (self.grid_size - 1)
        grid_uniform = (
            torch.arange(self.grid_size, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want a memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors' implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class MaternKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        nu=1.5,
    ):
        super(MaternKAN, self).__init__()
        self.grid_size = grid_size

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                MaternKANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    nu=nu,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
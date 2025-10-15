# src/pinn/core.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn as nn

Tensor = torch.Tensor

Activations = Literal[
    "tanh",
    "relu",
    "leaky_relu",
    "sigmoid",
    "selu",
    "softplus",
    "identity",
]


def get_activation(name: Activations) -> nn.Module:
    return {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "selu": nn.SELU(),
        "softplus": nn.Softplus(),
        "identity": nn.Identity(),
    }[name]


class Field(nn.Module):
    """
    A neural field mapping coordinates -> vector of state variables.
    Example (ODE): t -> [S, I, R].
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: list[int],
        activation: Activations,
        output_activation: Activations | None = None,
        encode: Callable[[Tensor], Tensor] | None = None,
        name: str = "u",
    ):
        super().__init__()
        self.name = name
        self.encode = encode
        dims = [in_dim] + hidden_layers + [out_dim]
        act = get_activation(activation)

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)

        if output_activation is not None:
            out_act = get_activation(output_activation)
            layers.append(out_act)

        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.encode is not None:
            x = self.encode(x)
        return self.net(x)  # type: ignore


class Parameter(nn.Module):
    """
    Learnable parameter. Supports scalar or function-valued parameter.
    For β(t), use a small MLP with in_dim=1 -> out_dim=1.
    """

    def __init__(
        self,
        mode: Literal["scalar", "mlp"],
        init_value: float = 0.1,
        in_dim: int = 1,
        hidden_layers: list[int] | None = None,
        activation: Activations | None = None,
        output_activation: Activations | None = None,
        name: str = "param",
    ):
        super().__init__()
        self.name = name
        self.mode = mode
        if mode == "scalar":
            self.value = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))
        else:  # mode == "mlp"
            hl = hidden_layers or [32, 32]
            dims = [in_dim] + hl + [1]
            act = get_activation(activation or "tanh")

            layers: list[nn.Module] = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(act)

            if output_activation is not None:
                out_act = get_activation(output_activation)
                layers.append(out_act)

            self.net = nn.Sequential(*layers)
            self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor | None = None) -> Tensor:
        if self.mode == "scalar":
            return self.value if x is None else self.value.expand_as(x)
        else:
            assert x is not None, "Function-valued parameter requires input"
            return self.net(x)  # type: ignore


@dataclass
class Domain:
    """
    Domain for ODE problems: time interval [t0, t1].
    """

    t0: float
    t1: float


@dataclass
class Observations:
    """
    Observations for ODE problems: time points and values.
    """

    t: Tensor
    i: Tensor


# TODO: deprecated
class Sampler(Protocol):
    def collocation(self, n: int) -> Tensor: ...
    def observed(self) -> Observations: ...


@dataclass
class Loss:
    value: Tensor
    weight: float


class Operator(Protocol):
    """
    Builds residuals given fields and parameters.
    Returns dict of name->Tensor residuals evaluated at provided coordinates.
    """

    def residuals(self, coords: Tensor) -> dict[str, Loss]: ...


class Constraint(Protocol):
    """
    Returns a named loss tensor (or dict of tensors).
    """

    def loss(self) -> dict[str, Loss]: ...


class Problem:
    """
    Aggregates operator residuals and constraints into total loss.
    """

    def __init__(
        self,
        operator: Operator,  # TODO: why not more than one?
        constraints: list[Constraint],
    ):
        self.operator = operator
        self.constraints = constraints

    def total_loss(self, collocation_coords: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        losses: dict[str, Loss] = {}
        # PDE residuals
        res = self.operator.residuals(collocation_coords)
        for k, v in res.items():
            losses[k] = v
        # Constraints
        for c in self.constraints:
            for k, v in c.loss().items():
                losses[k] = v

        device = next(iter(losses.values())).value.device
        total = torch.zeros((), dtype=torch.float32, device=device)
        for loss in losses.values():
            total = total + loss.weight * loss.value

        log_losses = {k: v.value for k, v in losses.items()}
        log_losses["total"] = total
        return total, log_losses

    @staticmethod
    def _default_weight_by_name(name: str) -> float:
        # Default grouping: pde/* -> pde_weight, ic/* -> ic_weight, data/* -> data_weight
        if name.startswith("pde/"):
            return 1.0
        if name.startswith("ic/"):
            return 1.0
        if name.startswith("data/"):
            return 1.0
        if name.startswith("reg/"):
            return 1.0
        return 1.0

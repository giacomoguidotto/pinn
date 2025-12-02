from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, override

import torch
from torch import Tensor
import torch.nn as nn

from pinn.core.dataset import DataBatch, PINNBatch, Transformer

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


class LogFn(Protocol):
    """
    A function that logs a value to a dictionary.

    Args:
        name: The name to log the value under.
        value: The value to log.
        progress_bar: Whether the value should be logged to the progress bar.
    """

    def __call__(self, name: str, value: Tensor, progress_bar: bool = False) -> None: ...


LOSS_KEY: str = "loss"


@dataclass
class MLPConfig:
    in_dim: int
    out_dim: int
    hidden_layers: list[int]
    activation: Activations
    output_activation: Activations | None = None
    encode: Callable[[Tensor], Tensor] | None = None
    name: str = "u"


@dataclass
class ScalarConfig:
    init_value: float = 0.1
    name: str = "p"


class Field(nn.Module):
    """
    A neural field mapping coordinates -> vector of state variables.
    Example (ODE): t -> [S, I, R].
    """

    def __init__(
        self,
        config: MLPConfig,
    ):
        super().__init__()
        self._name = config.name
        self.encode = config.encode
        dims = [config.in_dim] + config.hidden_layers + [config.out_dim]
        act = get_activation(config.activation)

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)

        if config.output_activation is not None:
            out_act = get_activation(config.output_activation)
            layers.append(out_act)

        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.encode is not None:
            x = self.encode(x)
        return self.net(x)  # type: ignore


class Argument:
    def __init__(self, value: float | Callable[[Tensor], float], name: str):
        self._value = value
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, x: Tensor) -> float:
        if callable(self._value):
            return self._value(x)
        else:
            return self._value


class Parameter(nn.Module, Argument):
    """
    Learnable parameter. Supports scalar or function-valued parameter.
    For β(t), use a small MLP with in_dim=1 -> out_dim=1.
    """

    def __init__(
        self,
        config: ScalarConfig | MLPConfig,
    ):
        super().__init__()
        self._name = config.name
        self._mode: Literal["scalar", "mlp"]

        if isinstance(config, ScalarConfig):
            self._mode = "scalar"
            self.value = nn.Parameter(torch.tensor(float(config.init_value), dtype=torch.float32))

        else:  # isinstance(config, MLPConfig)
            self._mode = "mlp"
            dims = [config.in_dim] + config.hidden_layers + [config.out_dim]
            act = get_activation(config.activation)

            layers: list[nn.Module] = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(act)

            if config.output_activation is not None:
                out_act = get_activation(config.output_activation)
                layers.append(out_act)

            self.net = nn.Sequential(*layers)
            self.apply(self._init)

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> Literal["scalar", "mlp"]:
        return self._mode

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    @override
    def forward(self, x: Tensor | None = None) -> Tensor:
        if self.mode == "scalar":
            return self.value if x is None else self.value.expand_as(x)
        else:
            assert x is not None, "Function-valued parameter requires input"
            return self.net(x)  # type: ignore


class Constraint(Protocol):
    """
    Returns a named loss for the given batch.
    Returns dict of name->Loss.
    """

    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        transformer: Transformer,
        log: LogFn | None = None,
    ) -> Tensor: ...


class Problem(nn.Module):
    """
    Aggregates operator residuals and constraints into total loss.
    """

    def __init__(
        self,
        constraints: list[Constraint],
        criterion: nn.Module,
        fields: list[Field],
        params: list[Parameter],
        transformer: Transformer | None = None,
    ):
        super().__init__()
        self.constraints = constraints
        self.criterion = criterion
        self.fields = fields
        self.params = params

        self._fields = nn.ModuleList(fields)
        self._params = nn.ModuleList(params)

        self.transformer = transformer or Transformer()

    def total_loss(self, batch: PINNBatch, log: LogFn | None = None) -> Tensor:
        device = batch[1].device

        total = torch.tensor(0.0, device=device)
        for c in self.constraints:
            total = total + c.loss(batch, self.criterion, self.transformer, log)

        if log is not None:
            for param in self.params:
                if param.mode == "scalar":
                    log(param.name, param.forward(), progress_bar=True)
                # else if param.mode == "mlp":
                #     log euclidean norm of the parameters with the ref function if provided
            log(LOSS_KEY, total, progress_bar=True)

        return total

    def predict(self, batch: DataBatch) -> dict[str, Tensor]:
        x_data, y_data = batch

        results = {
            "x_data": self.transformer.inverse_domain(x_data),
            "y_data": self.transformer.inverse_values(y_data),
        }

        for field in self.fields:
            results[field.name] = self.transformer.inverse_values(field(x_data))

        return results

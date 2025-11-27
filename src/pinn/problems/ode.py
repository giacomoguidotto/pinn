from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchdiffeq import odeint

from pinn.core import Argument, DataBatch

# ODECallable = Callable[..., Tensor]
ODECallable = Callable[[Tensor, Tensor, list[Argument]], Tensor]
"""
ODE function signature:
    ode(x: Tensor, y: Tensor, args: list[Argument]) -> Tensor
"""


@dataclass
class Domain1D:
    """
    One-dimensional domain: time interval [x0, x1] with step size dx.
    """

    x0: float
    x1: float
    dx: float = 1.0


@dataclass
class ODEProperties:
    ode: ODECallable
    domain: Domain1D
    # args: tuple[Any, ...]
    args: list[Argument]
    Y0: list[float]


class ODEDataset(Dataset[DataBatch]):
    def __init__(self, props: ODEProperties):
        x0, x1 = props.domain.x0, props.domain.x1
        dx = props.domain.dx
        steps = int((x1 - x0) / dx) + 1
        self.x = torch.linspace(x0, x1, steps)

        y0 = torch.tensor(props.Y0, dtype=torch.float32)

        self.data: Tensor = odeint(
            lambda x, y: props.ode(x, y, props.args),
            y0,
            self.x,
        )

    @override
    def __getitem__(self, idx: int) -> DataBatch:
        return (self.x[idx], self.data[idx])

    def __len__(self) -> int:
        return self.data.shape[0]

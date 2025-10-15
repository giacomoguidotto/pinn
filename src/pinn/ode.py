from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import torch
from torchdiffeq import odeint

from pinn.core import Domain, PINNDataset, Tensor

ODECallable = Callable[..., Tensor]


@dataclass
class ODEProperties:
    domain: Domain
    args: tuple[Any, ...]
    X0: list[float]


class ODEDataset(PINNDataset):
    def __init__(self, props: ODEProperties, generator: ODECallable):
        t0, t1 = props.domain.t0, props.domain.t1
        steps = int(t1 - t0 + 1)
        self.t = torch.linspace(t0, t1, steps)

        x0 = torch.tensor(props.X0, dtype=torch.float32)

        sol = odeint(generator, x0, self.t, args=props.args)

        self.data = torch.tensor(sol)

    @override
    def __getitem__(self, idx: int) -> Tensor:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

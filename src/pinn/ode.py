from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import torch
from torchdiffeq import odeint

from pinn.core import Domain, PINNDataset, Tensor

ODECallable = Callable[..., Tensor]


@dataclass
class ODEProperties:
    generator: ODECallable
    domain: Domain
    args: tuple[Any, ...]
    Y0: list[float]


class ODEDataset(PINNDataset):
    def __init__(self, props: ODEProperties):
        t0, t1 = props.domain.t0, props.domain.t1
        steps = int(t1 - t0 + 1)
        self.t = torch.linspace(t0, t1, steps)

        y0 = torch.tensor(props.Y0, dtype=torch.float32)

        # TODO: consider a nn.Module for the SIR system
        def ode_fn(t: Tensor, y: Tensor) -> Tensor:
            return props.generator(t, y, *props.args)

        sol = odeint(ode_fn, y0, self.t)

        self.data = torch.tensor(sol)

    @override
    def __getitem__(self, idx: int) -> Tensor:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

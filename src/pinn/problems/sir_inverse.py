from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import cast, override

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset

from pinn.core import (
    Argument,
    Constraint,
    Field,
    MLPConfig,
    Parameter,
    PINNDataModule,
    PINNDataset,
    Problem,
    ScalarConfig,
    Scaler,
)
from pinn.lib.utils import find_or_raise
from pinn.lightning import IngestionConfig, PINNHyperparameters, SchedulerConfig
from pinn.problems.ode import (
    DataConstraint,
    Domain1D,
    ICConstraint,
    ODECallable,
    ODEDataset,
    ODEProperties,
    ResidualsConstraint,
)

BETA_KEY = "params/beta"
DELTA_KEY = "args/delta"
N_KEY = "args/N"


def SIR(x: Tensor, y: Tensor, args: list[Argument]) -> Tensor:
    S, I = y
    b = find_or_raise(args, lambda a: a.name == BETA_KEY)
    d = find_or_raise(args, lambda a: a.name == DELTA_KEY)
    N = find_or_raise(args, lambda a: a.name == N_KEY)

    dS = -b(x) * S * I / N(x)
    dI = b(x) * S * I / N(x) - d(x) * I
    # dR = d(x) * I
    return torch.stack([dS, dI])


def beta_fn(x: Tensor) -> float:
    return 0.6 * (1 + torch.sin(x * 2 * math.pi / 90.0)).item()


@dataclass
class SIRInvProperties(ODEProperties):
    ode: ODECallable = field(default_factory=lambda: SIR)
    domain: Domain1D = field(
        default_factory=lambda: Domain1D(
            x0=0.0,
            x1=90.0,
            dx=1.0,
        )
    )

    N: float = 56e6
    delta: float = 1 / 5
    # beta: float = delta * 3.0
    args: list[Argument] = field(default_factory=list)

    I0: float = 1.0
    Y0: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.args = [
            Argument(self.delta, name=DELTA_KEY),
            Argument(beta_fn, name=BETA_KEY),
            # Argument(self.beta, name=BETA_KEY),
            Argument(self.N, name=N_KEY),
        ]

        S0 = self.N - self.I0
        self.Y0 = [S0, self.I0]


@dataclass
class SIRInvHyperparameters(PINNHyperparameters):
    max_epochs: int = 1000
    batch_size: int = 100
    data_ratio: int | float = 2
    data_noise_level: float = 1.0
    collocations: int = 6000
    lr: float = 5e-4
    gradient_clip_val: float = 0.1
    scheduler: SchedulerConfig | None = field(
        default_factory=lambda: SchedulerConfig(
            mode="min",
            factor=0.5,
            patience=55,
            threshold=5e-3,
            min_lr=1e-6,
        )
    )
    fields_config: MLPConfig = field(
        default_factory=lambda: MLPConfig(
            in_dim=1,
            out_dim=1,
            hidden_layers=[64, 128, 128, 64],
            activation="tanh",
            output_activation="softplus",
        )
    )
    param_config: MLPConfig | ScalarConfig = field(
        default_factory=lambda: ScalarConfig(
            init_value=0.5,
        )
    )
    ingestion: IngestionConfig | None = None
    # TODO: implement adaptive weights
    pde_weight: float = 100.0
    ic_weight: float = 1
    data_weight: float = 1


class SIRInvProblem(Problem):
    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
        scaler: Scaler | None = None,
    ) -> None:
        S_field = Field(config=replace(hp.fields_config, name="S"))
        I_field = Field(config=replace(hp.fields_config, name="I"))
        beta = Parameter(config=replace(hp.param_config, name=BETA_KEY))

        def predict_data(t_data: Tensor, fields: list[Field]) -> Tensor:
            I = find_or_raise(fields, lambda f: f.name == "I")
            return cast(Tensor, I(t_data))

        constraints: list[Constraint] = [
            ResidualsConstraint(
                fields=[S_field, I_field],
                weight=hp.pde_weight,
                props=props,
                params=[beta],
                scaler=scaler,
            ),
            ICConstraint(
                fields=[S_field, I_field],
                weight=hp.ic_weight,
                props=props,
                scaler=scaler,
            ),
            DataConstraint(
                fields=[S_field, I_field],
                predict_data=predict_data,
                weight=hp.data_weight,
            ),
        ]

        criterion = nn.MSELoss()

        super().__init__(
            constraints=constraints,
            criterion=criterion,
            fields=[S_field, I_field],
            params=[beta],
            scaler=scaler,
        )


class SIRInvDataset(ODEDataset):
    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
        scaler: Scaler | None = None,
    ):
        self.data_noise_level = hp.data_noise_level
        super().__init__(props, hp, scaler)

    @override
    def gen_data(self) -> tuple[Tensor, Tensor]:
        x, data = super().gen_data()

        if data.dim() > 2 and data.shape[-1] == 1:
            data = data.squeeze(-1)

        I_scaled = data[:, 1].clamp_min(0.0)
        I_physical = self.scaler.inverse_values(I_scaled)
        I_obs_physical = torch.poisson(I_physical / self.data_noise_level) * self.data_noise_level
        I_obs_scaled = self.scaler.transform_values(I_obs_physical)

        return x, I_obs_scaled.unsqueeze(-1)


class SIRInvCollocationset(Dataset[Tensor]):
    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
        scaler: Scaler | None = None,
    ):
        scaler = scaler or Scaler()
        self.domain = props.domain
        self.collocations = hp.collocations
        t = self.gen_coll()

        self.t = scaler.transform_domain(t)

    def gen_coll(self) -> Tensor:
        t0_s = torch.log1p(torch.tensor(self.domain.x0, dtype=torch.float32))
        t1_s = torch.log1p(torch.tensor(self.domain.x1, dtype=torch.float32))
        t_s = torch.rand((self.collocations, 1)) * (t1_s - t0_s) + t0_s
        t = torch.expm1(t_s)
        return t

    @override
    def __getitem__(self, idx: int) -> Tensor:
        return self.t[idx]

    def __len__(self) -> int:
        return len(self.t)


class SIRInvDataModule(PINNDataModule):
    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
        scaler: Scaler | None = None,
    ):
        super().__init__()
        self.props = props
        self.hp = hp
        self.scaler = scaler

    @override
    def setup(self, stage: str | None = None) -> None:
        self.data_ds = SIRInvDataset(
            self.props,
            self.hp,
            self.scaler,
        )
        self.coll_ds = SIRInvCollocationset(
            self.props,
            self.hp,
            self.scaler,
        )
        self.pinn_ds = PINNDataset(
            data_ds=self.data_ds,
            coll_ds=self.coll_ds,
            batch_size=self.hp.batch_size,
            data_ratio=self.hp.data_ratio,
        )

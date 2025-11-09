# src/pinn/sir.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias, cast, override

import lightning.pytorch as pl
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pinn.core import Batch, Constraint, Field, Operator, Parameter, PINNDataset, Problem
from pinn.core.core import LogFn, MLPConfig, ScalarConfig
from pinn.core.dataset import Scaler
from pinn.lightning.module import PINNHyperparameters, SchedulerConfig
from pinn.problems.ode import Domain1D, ODEDataset, ODEProperties

BETA_KEY = "params/beta"

SIRCallable: TypeAlias = Callable[[Tensor, Tensor, float, float, float], Tensor]


def SIR(_: Tensor, y: Tensor, d: float, b: float, N: float) -> Tensor:
    S, I, _ = y.unbind()

    dS = -b * S * I / N
    dI = b * S * I / N - d * I
    dR = d * I
    return torch.stack([dS, dI, dR])


@dataclass
class SIRInvHyperparameters(PINNHyperparameters):
    max_epochs: int = 2000
    batch_size: int = 100
    data_ratio: int | float = 2
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
    # beta_config: MLPConfig = field(
    #     default_factory=lambda: MLPConfig(
    #         in_dim=1,
    #         out_dim=1,
    #         hidden_layers=[64, 64],
    #         activation="tanh",
    #         output_activation="softplus",
    #         name=BETA_KEY,
    #     )
    # )
    beta_config: ScalarConfig = field(
        default_factory=lambda: ScalarConfig(
            init_value=0.5,
            name=BETA_KEY,
        )
    )
    # losses
    pde_weight: float = 100.0
    ic_weight: float = 1
    data_weight: float = 1
    beta_smoothness_weight: float = 0.0


@dataclass
class SIRInvProperties(ODEProperties):
    ode: SIRCallable = field(default_factory=lambda: SIR)
    domain: Domain1D = field(
        default_factory=lambda: Domain1D(
            t0=0.0,
            t1=90.0,
        )
    )

    N: float = 56e6
    delta: float = 1 / 5
    beta: float = delta * 3.0
    args: tuple[float, float, float] = (delta, beta, N)

    I0: float = 1.0
    Y0: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        S0 = self.N - self.I0
        R0 = self.N - self.I0 - S0  # 0 by definition
        self.Y0 = [S0, self.I0, R0]


class SIRInvOperator(Operator):
    def __init__(
        self,
        field_S: Field,
        field_I: Field,
        beta: Parameter,
        props: SIRInvProperties,
        scaler: SIRInvScaler,
        weight: float,
    ):
        self.SIR = props.ode
        self.delta = props.delta
        self.N = cast(float, scaler.scale_data(props.N))  # type: ignore

        self.S = field_S
        self.I = field_I
        self.beta = beta
        self.weight = weight

    @override
    def residuals(self, t_coll: Tensor, criterion: nn.Module, log: LogFn | None = None) -> Tensor:
        t_coll = t_coll.requires_grad_(True)
        S = self.S(t_coll)
        I = self.I(t_coll)
        R = self.N - S - I
        y = torch.stack([S, I, R])

        beta = self.beta(t_coll)

        dy = self.SIR(t_coll, y, self.delta, beta, self.N)
        dS_pred, dI_pred, _ = dy

        dS = torch.autograd.grad(S, t_coll, torch.ones_like(S), create_graph=True)[0]
        dI = torch.autograd.grad(I, t_coll, torch.ones_like(I), create_graph=True)[0]

        S_res: Tensor = dS - dS_pred
        I_res: Tensor = dI - dI_pred

        S_loss: Tensor = criterion(S_res, torch.zeros_like(S_res))
        I_loss: Tensor = criterion(I_res, torch.zeros_like(I_res))
        loss = self.weight * (S_loss + I_loss)

        if log is not None:
            log("loss/res", loss)
            log("loss/res/S", S_loss)
            log("loss/res/I", I_loss)

        return loss


class DataConstraint(Constraint):
    def __init__(
        self,
        field_S: Field,
        field_I: Field,
        weight: float,
    ):
        self.S = field_S
        self.I = field_I
        self.weight = weight

    @override
    def loss(self, batch: Batch, criterion: nn.Module, log: LogFn | None = None) -> Tensor:
        (t_data, I_data), _ = batch

        I_pred = self.I(t_data)

        data_loss: Tensor = criterion(I_pred, I_data)
        loss = self.weight * data_loss
        if log is not None:
            log("loss/data", loss)
            log("loss/data/I", data_loss)

        return loss


class ICConstraint(Constraint):
    def __init__(
        self,
        field_S: Field,
        field_I: Field,
        weight: float,
        props: SIRInvProperties,
        scaler: SIRInvScaler,
    ):
        Y0 = torch.tensor(props.Y0, dtype=torch.float32).reshape(-1, 1, 1)
        t0 = torch.tensor(props.domain.t0, dtype=torch.float32).reshape(1, 1)

        self.Y0 = scaler.scale_data(Y0)
        self.t0 = t0

        self.S = field_S
        self.I = field_I
        self.weight = weight

    @override
    def loss(self, batch: Batch, criterion: nn.Module, log: LogFn | None = None) -> Tensor:
        device = batch[1].device

        t0 = self.t0.to(device)
        S0, I0, _ = self.Y0.to(device)

        S0_pred = self.S(t0)
        I0_pred = self.I(t0)

        S0_loss: Tensor = criterion(S0_pred, S0)
        I0_loss: Tensor = criterion(I0_pred, I0)

        loss = self.weight * (S0_loss + I0_loss)

        if log is not None:
            log("loss/ic", loss)
            log("loss/ic/S0", S0_loss)
            log("loss/ic/I0", I0_loss)

        return loss


class BetaSmoothness(Constraint):
    """
    Regularizer: penalize beta'(t)^2 for smoothness.
    """

    def __init__(
        self,
        beta: Parameter,
        weight: float,
    ):
        self.beta = beta
        self.weight = weight

    @override
    def loss(self, batch: Batch, criterion: nn.Module, log: LogFn | None = None) -> Tensor:
        _, t_colloc = batch

        t = t_colloc.requires_grad_(True)
        b = self.beta(t)
        db = torch.autograd.grad(b, t, torch.ones_like(b), create_graph=True)[0]

        loss: Tensor = criterion(db)
        if log is not None:
            log("loss/reg/beta_smooth", loss)

        return self.weight * loss


class SIRInvProblem(Problem):
    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
        scaler: SIRInvScaler,
    ) -> None:
        field_S = Field(config=hp.fields_config)
        field_I = Field(config=hp.fields_config)
        beta = Parameter(config=hp.beta_config)

        operator = SIRInvOperator(
            field_S=field_S,
            field_I=field_I,
            weight=hp.pde_weight,
            props=props,
            scaler=scaler,
            beta=beta,
        )

        constraints: list[Constraint] = [
            ICConstraint(
                field_S=field_S,
                field_I=field_I,
                weight=hp.ic_weight,
                props=props,
                scaler=scaler,
            ),
            DataConstraint(
                field_S=field_S,
                field_I=field_I,
                weight=hp.data_weight,
            ),
            # BetaSmoothness(
            #     beta=beta,
            #     weight=hp.beta_smoothness_weight,
            # ),
        ]

        criterion = nn.MSELoss()

        super().__init__(
            operator=operator,
            constraints=constraints,
            criterion=criterion,
            fields=[field_S, field_I],
            parameters=[beta],
        )


class SIRInvDataset(ODEDataset):
    def __init__(self, props: SIRInvProperties):
        # SIR components are generated in self.data
        super().__init__(props)

        I = self.data[:, 1].clamp_min(0.0)

        # noising I
        noise_level = 1  # TODO: make this a parameter
        I_obs = torch.poisson(I / noise_level) * noise_level
        self.obs = torch.stack((self.t, I_obs), dim=1).unsqueeze(-1)

    @override
    def __getitem__(self, idx: int) -> Tensor:
        return self.obs[idx]

    @override
    def __len__(self) -> int:
        return self.obs.shape[0]


class SIRInvCollocationset(Dataset[Tensor]):
    def __init__(self, props: SIRInvProperties, hp: SIRInvHyperparameters):
        t0_s = torch.log1p(torch.tensor(props.domain.t0, dtype=torch.float32))
        t1_s = torch.log1p(torch.tensor(props.domain.t1, dtype=torch.float32))
        t_s = torch.rand((hp.collocations, 1)) * (t1_s - t0_s) + t0_s
        self.t = torch.expm1(t_s)

    @override
    def __getitem__(self, idx: int) -> Tensor:
        return self.t[idx]

    def __len__(self) -> int:
        return len(self.t)


class SIRInvScaler(Scaler):
    def __init__(self, props: SIRInvProperties):
        self.props = props

    @override
    def scale_domain(self, domain: Tensor) -> Tensor:
        return domain

    @override
    def scale_data(self, data: Tensor) -> Tensor:
        return data / self.props.N


class SIRInvDataModule(pl.LightningDataModule):
    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
        scaler: SIRInvScaler,
    ):
        super().__init__()
        self.props = props
        self.hp = hp
        self.scaler = scaler

    @override
    def setup(self, stage: str | None = None) -> None:
        self.dataset = SIRInvDataset(self.props)
        self.collocationset = SIRInvCollocationset(self.props, self.hp)

    @override
    def train_dataloader(self) -> DataLoader[Batch]:
        pinn_dataset = PINNDataset(
            data_ds=self.dataset,
            coll_ds=self.collocationset,
            batch_size=self.hp.batch_size,
            data_ratio=self.hp.data_ratio,
            scaler=self.scaler,
        )

        return DataLoader[Batch](
            pinn_dataset,
            batch_size=None,  # handled internally
            num_workers=7,
            persistent_workers=True,
        )

# src/pinn/sir.py
from __future__ import annotations

from dataclasses import dataclass
from typing import override

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pinn.core import (
    Activations,
    Batch,
    Constraint,
    Domain,
    Field,
    Loss,
    Operator,
    Parameter,
    PINNDataset,
    Problem,
    Tensor,
)
from pinn.module import PINNHyperparameters, SchedulerConfig
from pinn.ode import ODEDataset, ODEProperties


@dataclass
class SIRInvHyperparameters(PINNHyperparameters):
    lr = 1e-3
    batch_size = 256
    max_epochs = 1000
    gradient_clip_val = 0.1
    collocations: int = 4096
    scheduler = SchedulerConfig(
        mode="min",
        factor=0.5,
        patience=65,
        threshold=5e-3,
        min_lr=1e-6,
    )
    # fields networks
    hidden_layers: list[int] = None  # type: ignore
    activation: Activations = "tanh"
    output_activation: Activations = "softplus"
    # beta param network
    beta_hidden: list[int] = None  # type: ignore
    beta_activation: Activations = "tanh"
    beta_output_activation: Activations = "softplus"
    # losses
    pde_weight: float = 10.0
    ic_weight: float = 5.0
    data_weight: float = 1.0
    reg_beta_smooth_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_layers is None:
            self.hidden_layers = [64, 128, 128, 64]
        if self.beta_hidden is None:
            self.beta_hidden = [64, 64]


@dataclass
class SIRInvProperties(ODEProperties):
    domain: Domain = None  # type: ignore

    N: float = 56e6
    delta: float = 1 / 5
    beta: float = delta * 3.0
    args: tuple[float, float, float] = (delta, beta, N)

    I0: float = 1.0
    S0: float = N - I0
    X0: list[float] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.domain is None:
            self.domain = Domain(t0=0.0, t1=90.0)
        if self.X0 is None:
            R0 = self.N - self.S0 - self.I0
            self.X0 = [self.S0, self.I0, R0]


def SIR(x: Tensor, _: Tensor, d: float, b: float, N: float) -> Tensor:
    S, I, _ = x.unbind()

    dR = -b * S * I / N
    dI = b * S * I / N - d * I
    return torch.stack([dR, dI])


class SIRInvDataset(ODEDataset):
    def __init__(self, props: SIRInvProperties):
        # generate SIR components into self.data
        super().__init__(props, SIR)

        I = self.data[:, 1].clamp_min(0.0)
        I_obs = torch.poisson(I)
        self.obs = torch.stack([self.t, I_obs])

    @override
    def __getitem__(self, idx: int) -> Tensor:
        return self.obs[idx]

    @override
    def __len__(self) -> int:
        return len(self.obs)


class SIRInvCollocations(PINNDataset):
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


class SIROperator(Operator):
    def __init__(
        self,
        field_S: Field,
        field_I: Field,
        weight_S: float,
        weight_I: float,
        beta: Parameter,
        delta: float,
    ):
        self.S = field_S
        self.I = field_I
        self.beta = beta
        self.delta = delta
        self.weight_S = weight_S
        self.weight_I = weight_I

    @override
    def residuals(self, t: Tensor) -> dict[str, Loss]:
        # S' = -β S I / N  and I' = β S I / N - δ I, but we assume normalized N=1; scale outside
        t = t.requires_grad_(True)
        S = self.S(t)
        I = self.I(t)
        # R = 1.0 - S - I  # unused in residuals directly; can add R' if desired

        dS_dt = torch.autograd.grad(S, t, torch.ones_like(S), create_graph=True)[0]
        dI_dt = torch.autograd.grad(I, t, torch.ones_like(I), create_graph=True)[0]
        beta_t = self.beta(t)  # shape (B,1)

        res_S = dS_dt + beta_t * S * I
        res_I = dI_dt - beta_t * S * I + self.delta * I

        loss_S = Loss(value=res_S, weight=self.weight_S)
        loss_I = Loss(value=res_I, weight=self.weight_I)
        return {"pde/sir_S": loss_S, "pde/sir_I": loss_I}


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
        self.loss_fn: nn.Module = nn.MSELoss()

    @override
    def set_loss_fn(self, loss_fn: nn.Module) -> None:
        self.loss_fn = loss_fn

    @override
    def loss(self, batch: Batch) -> dict[str, Loss]:
        data, _ = batch
        (t, I) = data.unbind(dim=1)

        I_pred = self.I(t)

        data_loss = Loss(
            value=self.loss_fn(I_pred, I),
            weight=self.weight,
        )
        return {"data/I": data_loss}


class ICConstraint(Constraint):
    def __init__(
        self,
        props: SIRInvProperties,
        field_S: Field,
        field_I: Field,
        weight_S0: float,
        weight_I0: float,
    ):
        # Normalize to N=1
        self.S = field_S
        self.I = field_I
        self.X0 = torch.tensor(props.X0, dtype=torch.float32) / props.N
        self.t0 = torch.tensor(props.domain.t0, dtype=torch.float32)
        self.weight_S0 = weight_S0
        self.weight_I0 = weight_I0

        self.loss_fn: nn.Module = nn.MSELoss()

    @override
    def set_loss_fn(self, loss_fn: nn.Module) -> None:
        self.loss_fn = loss_fn

    @override
    def loss(self, _: Batch) -> dict[str, Loss]:
        S0, I0, _ = self.X0

        S0_pred = self.S(self.t0)
        I0_pred = self.I(self.t0)
        # Optionally enforce S+I<=1 at t0; often implied

        S0_loss = Loss(
            value=self.loss_fn(S0_pred, S0),
            weight=self.weight_S0,
        )
        I0_loss = Loss(
            value=self.loss_fn(I0_pred, I0),
            weight=self.weight_I0,
        )
        return {"ic/S0": S0_loss, "ic/I0": I0_loss}


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
        self.loss_fn: nn.Module = nn.MSELoss()

    @override
    def set_loss_fn(self, loss_fn: nn.Module) -> None:
        self.loss_fn = loss_fn

    @override
    def loss(self, batch: Batch) -> dict[str, Loss]:
        _, collocations = batch

        t = collocations.requires_grad_(True)
        b = self.beta(t)
        db_dt = torch.autograd.grad(b, t, torch.ones_like(b), create_graph=True)[0]

        loss = Loss(value=self.loss_fn(db_dt), weight=self.weight)
        return {"reg/beta_smooth": loss}


class SIRInvDataModule(pl.LightningDataModule):
    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
    ):
        super().__init__()
        self.props = props
        self.hp = hp

    @override
    def setup(self, stage: str | None = None) -> None:
        self.ds_data = SIRInvDataset(self.props)
        self.ds_col = SIRInvCollocations(self.props, self.hp)

    @override
    def train_dataloader(self) -> tuple[DataLoader[Tensor], DataLoader[Tensor]]:
        # In DDP, Lightning will wrap with DistributedSampler automatically
        loader_data = DataLoader[Tensor](
            self.ds_data,
            batch_size=self.hp.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        loader_col = DataLoader[Tensor](
            self.ds_col,
            batch_size=self.hp.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        return (loader_data, loader_col)


class SIRInvProblem(Problem):
    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
    ) -> None:
        in_dim, out_dim = 1, 1
        field_S = Field(
            in_dim, out_dim, hp.hidden_layers, hp.activation, hp.output_activation, name="S"
        )
        field_I = Field(
            in_dim, out_dim, hp.hidden_layers, hp.activation, hp.output_activation, name="I"
        )

        # beta = Parameter(
        #     mode="mlp",
        #     in_dim=in_dim,
        #     hidden_layers=hp.beta_hidden,
        #     activation=hp.beta_activation,
        #     output_activation=hp.beta_output_activation,
        #     name="beta",
        # )
        beta = Parameter(
            mode="scalar",
            init_value=props.beta,
            name="beta",
        )

        operator = SIROperator(
            field_S=field_S,
            field_I=field_I,
            weight_S=hp.pde_weight,
            weight_I=hp.pde_weight,
            beta=beta,
            delta=props.delta,
        )

        constraints: list[Constraint] = [
            ICConstraint(
                props=props,
                field_S=field_S,
                field_I=field_I,
                weight_S0=hp.ic_weight,
                weight_I0=hp.ic_weight,
            ),
            DataConstraint(
                field_S=field_S,
                field_I=field_I,
                weight=hp.data_weight,
            ),
            BetaSmoothness(
                beta=beta,
                weight=hp.reg_beta_smooth_weight,
            ),
        ]

        loss_fn = nn.MSELoss()

        super().__init__(
            operator=operator,
            constraints=constraints,
            loss_fn=loss_fn,
        )

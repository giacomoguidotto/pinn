# src/pinn/sir.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias, override

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pinn.core import (
    Activations,
    Batch,
    Constraint,
    Dataset,
    Domain,
    Field,
    Loss,
    Operator,
    Parameter,
    Problem,
    Tensor,
)
from pinn.module import PINNHyperparameters, SchedulerConfig
from pinn.ode import ODEDataset, ODEProperties

SIRCallable: TypeAlias = Callable[[Tensor, Tensor, float, float, float], Tensor]


def SIR(_: Tensor, y: Tensor, d: float, b: float, N: float) -> Tensor:
    S, I, _ = y.unbind()

    dS = -b * S * I / N
    dI = b * S * I / N - d * I
    dR = d * I
    return torch.stack([dS, dI, dR])


@dataclass
class SIRInvHyperparameters(PINNHyperparameters):
    lr: float = 1e-3
    batch_size: int = 91  # FIXME: dataloader is not batching correctly
    max_epochs: int = 1000
    gradient_clip_val: float = 0.1
    collocations: int = 4096
    scheduler = SchedulerConfig(
        mode="min",
        factor=0.5,
        patience=65,
        threshold=5e-3,
        min_lr=1e-6,
    )
    # fields networks
    hidden_layers: list[int] = field(default_factory=lambda: [64, 128, 128, 64])
    activation: Activations = "tanh"
    output_activation: Activations = "softplus"
    # beta param network
    beta_hidden: list[int] = field(default_factory=lambda: [64, 64])
    beta_activation: Activations = "tanh"
    beta_output_activation: Activations = "softplus"
    # losses
    pde_weight: float = 10.0
    ic_weight: float = 5.0
    data_weight: float = 1.0
    reg_beta_smooth_weight: float = 0.0


@dataclass
class SIRInvProperties(ODEProperties):
    domain: Domain = field(default_factory=lambda: Domain(t0=0.0, t1=90.0))
    generator: SIRCallable = field(default_factory=lambda: SIR)

    N: float = 56e6
    delta: float = 1 / 5
    beta: float = delta * 3.0
    args: tuple[float, float, float] = (delta, beta, N)

    I0: float = 1.0
    Y0: list[float] = None  # type: ignore

    def __post_init__(self) -> None:
        S0 = self.N - self.I0
        R0 = self.N - self.I0 - S0  # 0 by definition
        self.Y0 = [S0, self.I0, R0]


class SIROperator(Operator):
    def __init__(
        self,
        props: SIRInvProperties,
        field_S: Field,
        field_I: Field,
        weight_S: float,
        weight_I: float,
        beta: Parameter,
    ):
        self.SIR = props.generator
        self.delta = props.delta

        self.S = field_S
        self.I = field_I
        self.beta = beta
        self.weight_S = weight_S
        self.weight_I = weight_I

    @override
    def residuals(self, t: Tensor) -> dict[str, Loss]:
        t = t.requires_grad_(True)
        S = self.S(t)
        I = self.I(t)
        y = torch.stack([S, I, torch.zeros_like(S)])

        beta = self.beta(t)

        dy = self.SIR(t, y, self.delta, beta, 1)  # TODO: check scaling everywhere
        dS_pred, dI_pred, _ = dy.unbind(dim=0)

        dS = torch.autograd.grad(S, t, torch.ones_like(S), create_graph=True)[0]
        dI = torch.autograd.grad(I, t, torch.ones_like(I), create_graph=True)[0]

        S_res = dS - dS_pred
        I_res = dI - dI_pred

        loss_S = Loss(value=S_res, weight=self.weight_S)
        loss_I = Loss(value=I_res, weight=self.weight_I)
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
        (t, I_data), _ = batch

        I_pred = self.I(t)

        data_loss = Loss(
            value=self.loss_fn(I_pred, I_data),
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
        self.Y0 = torch.tensor(props.Y0, dtype=torch.float32).reshape(-1, 1, 1)
        self.t0 = torch.tensor(props.domain.t0, dtype=torch.float32).reshape(1, 1)

        self.S = field_S
        self.I = field_I
        self.weight_S0 = weight_S0
        self.weight_I0 = weight_I0

        self.loss_fn: nn.Module = nn.MSELoss()

    @override
    def set_loss_fn(self, loss_fn: nn.Module) -> None:
        self.loss_fn = loss_fn

    @override
    def loss(self, batch: Batch) -> dict[str, Loss]:
        _, collocations = batch
        device = collocations.device

        t0 = self.t0.to(device)
        S0, I0, _ = self.Y0.to(device)

        S0_pred = self.S(t0)
        I0_pred = self.I(t0)

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
        db = torch.autograd.grad(b, t, torch.ones_like(b), create_graph=True)[0]

        loss = Loss(
            value=self.loss_fn(db),
            weight=self.weight,
        )
        return {"reg/beta_smooth": loss}


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
            props=props,
            field_S=field_S,
            field_I=field_I,
            weight_S=hp.pde_weight,
            weight_I=hp.pde_weight,
            beta=beta,
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
            # BetaSmoothness(
            #     beta=beta,
            #     weight=hp.reg_beta_smooth_weight,
            # ),
        ]

        loss_fn = nn.MSELoss()

        super().__init__(
            operator=operator,
            constraints=constraints,
            loss_fn=loss_fn,
        )

        # store modules for device inference
        self.field_S = field_S
        self.field_I = field_I
        self.beta = beta


class SIRInvDataset(ODEDataset):
    def __init__(self, props: SIRInvProperties):
        # generate SIR components into self.data
        super().__init__(props)

        I = self.data[:, 1].clamp_min(0.0)
        I_obs = torch.poisson(I)
        self.obs = torch.stack((self.t, I_obs), dim=1).unsqueeze(-1)

    @override
    def __getitem__(self, idx: int) -> Tensor:
        return self.obs[idx]

    @override
    def __len__(self) -> int:
        return self.obs.shape[1]


class SIRInvCollocationset(Dataset):
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
        self.dataset = SIRInvDataset(self.props)
        self.collocationset = SIRInvCollocationset(self.props, self.hp)

    @override
    def train_dataloader(self) -> tuple[DataLoader[Tensor], DataLoader[Tensor]]:
        data_loader = DataLoader[Tensor](
            self.dataset,
            batch_size=self.hp.batch_size,
            shuffle=True,
            num_workers=5,
            # pin_memory=True, # not supported in MPS yet
            persistent_workers=True,
        )
        collocations_loader = DataLoader[Tensor](
            self.collocationset,
            batch_size=self.hp.batch_size,
            shuffle=True,
            num_workers=5,
            # pin_memory=True, # not supported in MPS yet
            persistent_workers=True,
        )

        return (data_loader, collocations_loader)

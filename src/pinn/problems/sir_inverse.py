from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import TypeVar, override

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset

from pinn.core import (
    Argument,
    Constraint,
    DataBatch,
    Field,
    LogFn,
    MLPConfig,
    Operator,
    Parameter,
    PINNBatch,
    PINNDataModule,
    PINNDataset,
    Problem,
    ScalarConfig,
    Transformer,
)
from pinn.lib.utils import get_arg_or_raise
from pinn.lightning import PINNHyperparameters, SchedulerConfig
from pinn.problems.ode import Domain1D, ODECallable, ODEDataset, ODEProperties

BETA_KEY = "params/beta"
DELTA_KEY = "args/delta"
N_KEY = "args/N"


def SIR(x: Tensor, y: Tensor, args: list[Argument]) -> Tensor:
    S, I = y
    b = get_arg_or_raise(args, BETA_KEY)
    d = get_arg_or_raise(args, DELTA_KEY)
    N = get_arg_or_raise(args, N_KEY)

    dS = -b(x) * S * I / N(x)  # TODO: N should be a constant
    dI = b(x) * S * I / N(x) - d(x) * I
    # dR = d(x) * I
    return torch.stack([dS, dI])


# TODO: should this return a tensor?
def beta_fn(x: Tensor) -> float:
    return 0.6 * (1 + torch.sin(x * 2 * math.pi / 90.0)).item()


@dataclass
class SIRInvHyperparameters(PINNHyperparameters):
    max_epochs: int = 1000
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
    beta_config: MLPConfig | ScalarConfig = field(
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
    ode: ODECallable = field(default_factory=lambda: SIR)
    domain: Domain1D = field(
        default_factory=lambda: Domain1D(
            x0=0.0,
            x1=90.0,
        )
    )

    N: float = 56e6
    delta: float = 1 / 5
    beta: float = delta * 3.0
    args: list[Argument] = field(default_factory=list)

    I0: float = 1.0
    Y0: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.args = [
            Argument(self.delta, name=DELTA_KEY),
            Argument(self.beta, name=BETA_KEY),
            Argument(self.N, name=N_KEY),
        ]

        S0 = self.N - self.I0
        self.Y0 = [S0, self.I0]

class SIRInvTransformer(Transformer):
    T = TypeVar("T", Tensor, float)

    def __init__(self, props: SIRInvProperties):
        self.props = props

    @override
    def transform_values(self, data: T) -> T:
        return data / self.props.N

    @override
    def inverse_transform_values(self, data: T) -> T:
        return data * self.props.N


class SIRInvOperator(Operator):
    def __init__(
        self,
        field_S: Field,
        field_I: Field,
        params: list[Parameter],
        props: SIRInvProperties,
        weight: float,
    ):
        self.SIR = props.ode
        self.S = field_S
        self.I = field_I

        params_names = [p.name for p in params]
        self.args = [a for a in props.args if a.name not in params_names]
        self.args.extend(params)

        self.weight = weight

    @override
    def residuals(
        self,
        t_coll: Tensor,  # already scaled
        criterion: nn.Module,
        transformer: Transformer,
        log: LogFn | None = None,
    ) -> Tensor:
        t_coll = t_coll.requires_grad_(True)

        S = self.S(t_coll)
        I = self.I(t_coll)
        y = torch.stack([S, I])

        # TODO: remove workaround for N
        for a in self.args:
            if a.name == N_KEY:
                a._value = 1.0

        dy = self.SIR(t_coll, y, self.args)
        dS_pred, dI_pred = dy

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
        field_I: Field,
        weight: float,
    ):
        self.I = field_I
        self.weight = weight

    @override
    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        transformer: Transformer,
        log: LogFn | None = None,
    ) -> Tensor:
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
    ):
        self.Y0 = torch.tensor(props.Y0, dtype=torch.float32).reshape(-1, 1, 1)
        self.t0 = torch.tensor(props.domain.x0, dtype=torch.float32).reshape(1, 1)

        self.S = field_S
        self.I = field_I
        self.weight = weight

    @override
    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        transformer: Transformer,
        log: LogFn | None = None,
    ) -> Tensor:
        device = batch[1].device

        t0 = transformer.transform_domain(self.t0.to(device))
        S0, I0 = transformer.transform_values(self.Y0.to(device))

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
    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        transformer: Transformer,
        log: LogFn | None = None,
    ) -> Tensor:
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
        transformer: SIRInvTransformer | None = None,
    ) -> None:
        S_field = Field(config=replace(hp.fields_config, name="S"))
        I_field = Field(config=replace(hp.fields_config, name="I"))
        beta = Parameter(config=hp.beta_config)

        operator = SIRInvOperator(
            field_S=S_field,
            field_I=I_field,
            weight=hp.pde_weight,
            props=props,
            params=[beta],
        )

        constraints: list[Constraint] = [
            ICConstraint(
                field_S=S_field,
                field_I=I_field,
                weight=hp.ic_weight,
                props=props,
            ),
            DataConstraint(
                field_I=I_field,
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
            fields=[S_field, I_field],
            params=[beta],
            transformer=transformer,
        )


class SIRInvDataset(ODEDataset):
    def __init__(self, props: SIRInvProperties):
        # SIR components are generated in self.data
        super().__init__(props)

        I = self.data[:, 1].clamp_min(0.0)

        # noising I
        noise_level = 1  # TODO: make this a parameter
        obs = torch.poisson(I / noise_level) * noise_level
        self.x = self.x.unsqueeze(-1)
        self.obs = obs.unsqueeze(-1)

    @override
    def __getitem__(self, idx: int) -> DataBatch:
        return (self.x[idx], self.obs[idx])

    @override
    def __len__(self) -> int:
        return self.obs.shape[0]


class SIRInvCollocationset(Dataset[Tensor]):
    def __init__(self, props: SIRInvProperties, hp: SIRInvHyperparameters):
        t0_s = torch.log1p(torch.tensor(props.domain.x0, dtype=torch.float32))
        t1_s = torch.log1p(torch.tensor(props.domain.x1, dtype=torch.float32))
        t_s = torch.rand((hp.collocations, 1)) * (t1_s - t0_s) + t0_s
        self.t = torch.expm1(t_s)

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
        transformer: SIRInvTransformer | None = None,
    ):
        super().__init__()
        self.props = props
        self.hp = hp
        self.transformer = transformer

    @override
    def setup(self, stage: str | None = None) -> None:
        self.data_ds = SIRInvDataset(self.props)
        self.coll_ds = SIRInvCollocationset(self.props, self.hp)
        self.pinn_ds = PINNDataset(
            data_ds=self.data_ds,
            coll_ds=self.coll_ds,
            batch_size=self.hp.batch_size,
            data_ratio=self.hp.data_ratio,
            transformer=self.transformer,
        )

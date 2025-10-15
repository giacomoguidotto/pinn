# src/pinn/sir.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from pinn.core import (
    Activations,
    Constraint,
    Domain,
    Field,
    Loss,
    Observations,
    Operator,
    Parameter,
    Problem,
    Sampler,
    Tensor,
)


@dataclass
class SIRProperties:
    domain: Domain = None  # type: ignore
    delta: float = 1 / 5
    N: float = 56e6
    n_collocation: int = 4096
    I0: float = 1.0
    S0: float = N - I0

    def __post_init__(self) -> None:
        if self.domain is None:
            self.domain = Domain(t0=0.0, t1=90.0)


@dataclass
class SIRHyperparameters:
    # network
    hidden_layers: list[int] = None  # type: ignore
    activation: Activations = "tanh"
    output_activation: Activations = "softplus"
    # beta(t) network
    beta_hidden: list[int] = None  # type: ignore
    beta_activation: Activations = "tanh"
    beta_output_activation: Activations = "softplus"
    # losses
    pde_weight: float = 10.0
    ic_weight: float = 5.0
    data_weight: float = 1.0
    reg_beta_smooth_weight: float = 0.0
    # training
    lr: float = 1e-3
    batch_size: int = 256

    def __post_init__(self) -> None:
        if self.hidden_layers is None:
            self.hidden_layers = [64, 128, 128, 64]
        if self.beta_hidden is None:
            self.beta_hidden = [64, 64]


@dataclass
class SIRObservation:
    t: np.ndarray
    i: np.ndarray


class TimeSampler(Sampler):
    def __init__(
        self,
        domain: Domain,
        obs: SIRObservation,
        n_collocation: int,
        N: float,
    ):
        self.domain = domain
        self.t_obs = torch.tensor(obs.t, dtype=torch.float32).reshape(-1, 1)
        self.i_obs = torch.tensor(obs.i / N, dtype=torch.float32).reshape(-1, 1)
        self.n_collocation = n_collocation

    def collocation(self, n: int) -> Tensor:
        t0, t1 = self.domain.t0, self.domain.t1
        # Exponential-ish sampling similar to your dataset
        t = torch.expm1(
            torch.rand((n, 1)) * (torch.log1p(torch.tensor(t1)) - torch.log1p(torch.tensor(t0)))
            + torch.log1p(torch.tensor(t0))
        ).float()
        return t

    def observed(self) -> Observations:
        return Observations(t=self.t_obs, i=self.i_obs)


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


class ICConstraint(Constraint):
    def __init__(
        self,
        field_S: Field,
        field_I: Field,
        weight_S: float,
        weight_I: float,
        S0: float,
        I0: float,
        N: float,
    ):
        # Normalize to N=1
        self.field_S = field_S
        self.field_I = field_I
        self.S0 = torch.tensor(S0 / N, dtype=torch.float32).reshape(1, 1)
        self.I0 = torch.tensor(I0 / N, dtype=torch.float32).reshape(1, 1)
        self.t0 = torch.zeros((1, 1), dtype=torch.float32)
        self.loss_fn = nn.MSELoss()
        self.weight_S = weight_S
        self.weight_I = weight_I

    def loss(self) -> dict[str, Loss]:
        S0_pred = self.field_S(self.t0)
        I0_pred = self.field_I(self.t0)
        ic_S = self.loss_fn(S0_pred, self.S0)
        ic_I = self.loss_fn(I0_pred, self.I0)
        # Optionally enforce S+I<=1 at t0; often implied

        loss_S = Loss(value=ic_S, weight=self.weight_S)
        loss_I = Loss(value=ic_I, weight=self.weight_I)
        return {"ic/S0": loss_S, "ic/I0": loss_I}


class DataConstraint(Constraint):
    def __init__(self, field_S: Field, field_I: Field, sampler: TimeSampler, weight: float):
        self.S = field_S
        self.I = field_I
        self.sampler = sampler
        self.loss_fn = nn.MSELoss()
        self.weight = weight

    def loss(self) -> dict[str, Loss]:
        obs = self.sampler.observed()
        if obs.t.numel() == 0:
            return {"data/I": Loss(value=torch.tensor(0.0), weight=self.weight)}

        I_pred = self.I(obs.t)
        data_loss = self.loss_fn(I_pred, obs.i)

        loss_I = Loss(value=data_loss, weight=self.weight)
        return {"data/I": loss_I}


class BetaSmoothness(Constraint):
    """
    Optional regularizer: penalize beta'(t)^2 for smoothness.
    """

    def __init__(self, beta: Parameter, sampler: TimeSampler, weight: float = 0.0):
        self.beta = beta
        self.sampler = sampler
        self.weight = weight

    def loss(self) -> dict[str, Loss]:
        if self.weight <= 0:
            return {"reg/beta_smooth": Loss(value=torch.tensor(0.0), weight=0.0)}

        t = self.sampler.collocation(self.sampler.n_collocation // 4).requires_grad_(True)
        b = self.beta(t)
        db_dt = torch.autograd.grad(b, t, torch.ones_like(b), create_graph=True)[0]

        loss = Loss(value=(db_dt**2).mean(), weight=self.weight)
        return {"reg/beta_smooth": loss}


# TODO: make it the SIRProblem class
def build_sir_problem(
    props: SIRProperties,
    obs: SIRObservation,
    hp: SIRHyperparameters,
) -> tuple[Problem, Sampler]:
    sampler = TimeSampler(
        props.domain,
        obs,
        n_collocation=props.n_collocation,
        N=props.N,
    )

    in_dim = 1
    out_dim = 1
    field_S = Field(
        in_dim, out_dim, hp.hidden_layers, hp.activation, hp.output_activation, name="S"
    )
    field_I = Field(
        in_dim, out_dim, hp.hidden_layers, hp.activation, hp.output_activation, name="I"
    )

    beta = Parameter(
        mode="mlp",
        in_dim=in_dim,
        hidden_layers=hp.beta_hidden,
        activation=hp.beta_activation,
        output_activation=hp.beta_output_activation,
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
            field_S,
            field_I,
            weight_S=hp.ic_weight,
            weight_I=hp.ic_weight,
            S0=props.S0,
            I0=props.I0,
            N=props.N,
        ),
        DataConstraint(
            field_S,
            field_I,
            sampler,
            weight=hp.data_weight,
        ),
        BetaSmoothness(
            beta,
            sampler,
            weight=hp.reg_beta_smooth_weight,
        ),
    ]

    problem = Problem(
        operator=operator,
        constraints=constraints,
    )

    # TODO: do not instantiate here
    # opt_cfg = {
    #     "lr": hp.lr,
    #     "batch_size": hp.batch_size,
    # }
    return problem, sampler

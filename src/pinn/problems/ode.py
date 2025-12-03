from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, override

import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
from torchdiffeq import odeint

from pinn.core import Argument, Constraint, DataBatch, Field, LogFn, Parameter, PINNBatch, Scaler
from pinn.lightning import IngestionConfig, PINNHyperparameters

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
    dx: float


@dataclass
class ODEProperties:
    ode: ODECallable
    domain: Domain1D
    args: list[Argument]
    Y0: list[float]

class ResidualsConstraint(Constraint):
    def __init__(
        self,
        props: ODEProperties,
        fields: list[Field],
        params: list[Parameter],
        weight: float,
        scaler: Scaler | None = None,
    ):
        self.scaler = scaler or Scaler()
        self.fields = fields
        self.ode = self.scaler.scale_ode(props.ode)

        params_names = [p.name for p in params]
        self.args = [a for a in props.args if a.name not in params_names]
        self.args.extend(params)

        self.weight = weight

    @override
    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        _, t_coll = batch
        t_coll = t_coll.requires_grad_(True)

        preds = [f(t_coll) for f in self.fields]
        y = torch.stack(preds)

        dy_dt_pred = self.ode(t_coll, y, self.args)

        dy_dt_list = []
        for pred in preds:
            grad = torch.autograd.grad(pred, t_coll, torch.ones_like(pred), create_graph=True)[0]
            dy_dt_list.append(grad)

        dy_dt = torch.stack(dy_dt_list)

        residuals = dy_dt - dy_dt_pred

        # normalize residuals by t_scale to match physical time derivative magnitude
        if self.scaler.t_scale != 0:
            residuals = residuals / self.scaler.t_scale

        loss = torch.tensor(0.0, device=t_coll.device)
        for res in residuals:
            loss = loss + criterion(res, torch.zeros_like(res))
        loss = self.weight * loss

        if log is not None:
            log("loss/res", loss)

        return loss


class ICConstraint(Constraint):
    def __init__(
        self,
        fields: list[Field],
        weight: float,
        props: ODEProperties,
        scaler: Scaler | None = None,
    ):
        self.scaler = scaler or Scaler()
        Y0 = torch.tensor(props.Y0, dtype=torch.float32).reshape(-1, 1, 1)
        t0 = torch.tensor(props.domain.x0, dtype=torch.float32).reshape(1, 1)

        self.t0 = self.scaler.transform_domain(t0)
        self.Y0 = self.scaler.transform_values(Y0)

        self.fields = fields
        self.weight = weight

    @override
    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        device = batch[1].device

        t0 = self.t0.to(device)
        Y0 = self.Y0.to(device)

        Y0_preds = [f(t0) for f in self.fields]

        loss = torch.tensor(0.0, device=device)
        for y0_target, y0_pred in zip(Y0, Y0_preds, strict=False):
            loss = loss + criterion(y0_pred, y0_target)
        loss = self.weight * loss

        if log is not None:
            log("loss/ic", loss)

        return loss


PredictDataFn: TypeAlias = Callable[[Tensor, list[Field]], Tensor]


class DataConstraint(Constraint):
    def __init__(
        self,
        fields: list[Field],
        predict_data: PredictDataFn,
        weight: float,
    ):
        self.fields = fields
        self.predict_data = predict_data
        self.weight = weight

    @override
    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        (t_data, data), _ = batch

        data_pred = self.predict_data(t_data, self.fields)

        loss: Tensor = criterion(data_pred, data)
        loss = self.weight * loss

        if log is not None:
            log("loss/data", loss)

        return loss


class ODEDataset(Dataset[DataBatch]):
    def __init__(
        self,
        props: ODEProperties,
        hp: PINNHyperparameters,
        scaler: Scaler | None = None,
    ):
        self.hp = hp
        self.props = props
        self.domain = props.domain
        self.scaler = scaler or Scaler()

        self.x, self.obs = (
            self.load_data(hp.ingestion) if hp.ingestion is not None else self.gen_data()
        )

    def gen_data(self) -> tuple[Tensor, Tensor]:
        x0, x1, dx = self.domain.x0, self.domain.x1, self.domain.dx
        steps = int((x1 - x0) / dx) + 1

        x = torch.linspace(x0, x1, steps)
        y0 = torch.tensor(self.props.Y0, dtype=torch.float32)

        x = self.scaler.transform_domain(x)
        y0 = self.scaler.transform_values(y0)

        ode_fn = self.scaler.scale_ode(self.props.ode)
        data = odeint(
            lambda x, y: ode_fn(x, y, self.props.args),
            y0,
            x,
        )

        return x.unsqueeze(-1), data.unsqueeze(-1)

    def load_data(self, ingestion: IngestionConfig) -> tuple[Tensor, Tensor]:
        df = pd.read_csv(ingestion.df_path)

        x_col, y_cols = ingestion.x_column, ingestion.y_columns
        if not {x_col, *y_cols}.issubset(df.columns):
            raise ValueError(
                f"Expected {', '.join(y_cols)} and {x_col} columns in the dataframe, "
                f"but got {', '.join(df.columns)}"
            )

        x = torch.tensor(df[x_col].values, dtype=torch.float32)
        obs = torch.tensor(df[y_cols].values, dtype=torch.float32)

        # transforming x to the problem domain
        x0, x1 = self.domain.x0, self.domain.x1
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)
        x = x * (x1 - x0) + x0

        x = self.scaler.transform_domain(x)
        obs = self.scaler.transform_values(obs)

        return x.unsqueeze(-1), obs.unsqueeze(-1)

    @override
    def __getitem__(self, idx: int) -> DataBatch:
        return (self.x[idx], self.obs[idx])

    def __len__(self) -> int:
        return self.obs.shape[0]
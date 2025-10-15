# src/pinn/pinn_module.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch

from pinn.core import Problem, Sampler, Tensor

# tensor of collocation points in the time domain
PINNDataset = Tensor


@dataclass
class SchedulerConfig:
    mode: Literal["min", "max"] = "min"
    factor: float = 0.5
    patience: int = 50
    threshold: float = 1e-3
    min_lr: float = 1e-6


# belonging to the "ODE pinn solver" package
class PINNModule(LightningModule):
    """
    Generic PINN Lightning module.
    Expects external Problem + Sampler + optimizer config.
    """

    def __init__(
        self,
        problem: Problem,
        lr: float = 1e-3,
        scheduler_cfg: SchedulerConfig | None = None,
        log_prefix: str = "train",
        gradient_clip_val: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.problem = problem
        self.lr = lr
        self.scheduler_cfg = scheduler_cfg
        self.log_prefix = log_prefix
        self.gradient_clip_val = gradient_clip_val

    def training_step(self, batch: PINNDataset, batch_idx: int) -> Tensor:
        # batch may be ignored if sampler handles observed data internally
        # Get collocation coordinates (time points for ODE)
        t_col = batch
        total, losses = self.problem.total_loss(t_col)
        # log components
        for k, v in losses.items():
            self.log(f"{self.log_prefix}/{k}", v, on_step=False, on_epoch=True)

        return total

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        if not self.scheduler_cfg:
            return opt
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.scheduler_cfg.mode,
            factor=self.scheduler_cfg.factor,
            patience=self.scheduler_cfg.patience,
            threshold=self.scheduler_cfg.threshold,
            min_lr=self.scheduler_cfg.min_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": f"{self.log_prefix}/total_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# belonging to the "ODE pinn solver" package
class CollocationDataset(torch.utils.data.Dataset[PINNDataset]):
    def __init__(self, sampler: Sampler, batch_size: int):
        self.sampler = sampler
        self.batch_size = batch_size

    def __getitem__(self, _: int) -> PINNDataset:
        return self.sampler.collocation(self.batch_size)

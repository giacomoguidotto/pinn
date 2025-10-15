# src/pinn/train_sir_inverse.py
from __future__ import annotations

from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from pinn.module import PINNModule, SchedulerConfig
from pinn.sir_inverse import (
    SIRInvDataModule,
    SIRInvHyperparameters,
    SIRInvProperties,
    build_sir_problem,
)

LOG_DIR = Path("./data/logs")
TENSORBOARD_DIR = LOG_DIR / "tensorboard"
CSV_DIR = LOG_DIR / "csv"
VERSIONS_DIR = Path("./data/versions")


def train_sir_inverse(
    props: SIRInvProperties, hp: SIRInvHyperparameters, run_name: str = "sir_inverse_beta_mlp"
) -> tuple[Path, str]:
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    dm = SIRInvDataModule(props, hp)

    problem = build_sir_problem(props, hp)

    module = PINNModule(
        problem=problem,
        lr=hp.lr,
        scheduler_cfg=SchedulerConfig(
            mode="min",
            factor=0.5,
            patience=65,
            threshold=5e-3,
            min_lr=1e-6,
        ),
        gradient_clip_val=0.1,
    )

    checkpoint = ModelCheckpoint(
        dirpath="./data/checkpoints",
        filename="{epoch:02d}",
        monitor="train/total_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    loggers = [
        TensorBoardLogger(save_dir=TENSORBOARD_DIR, name="sir_beta_mlp", version=run_name),
        CSVLogger(save_dir=CSV_DIR, name="sir_beta_mlp", version=run_name),
    ]
    trainer = Trainer(
        max_epochs=1000,
        logger=loggers,
        callbacks=[
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="train/total_loss", patience=100, mode="min"),
        ],
        log_every_n_steps=1,
        gradient_clip_val=0.1,
    )

    trainer.fit(module, dm)

    version = f"v{len(list(VERSIONS_DIR.iterdir()))}_{run_name}"
    model_path = VERSIONS_DIR / f"{version}.ckpt"
    trainer.save_checkpoint(model_path)
    return model_path, version

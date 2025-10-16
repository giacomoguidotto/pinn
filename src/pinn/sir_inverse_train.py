# src/pinn/train_sir_inverse.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from pinn.module import PINNModule
from pinn.sir_inverse import (
    SIRInvDataModule,
    SIRInvHyperparameters,
    SIRInvProblem,
    SIRInvProperties,
)


def create_temp_dir() -> Path:
    temp_dir = Path("./temp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    else:
        temp_dir.mkdir(exist_ok=True)
    return temp_dir


def clean_dir(dir: Path) -> None:
    if dir.exists():
        shutil.rmtree(dir)


@dataclass
class SIRInvTrainConfig:
    name: str
    version: str
    tensorboard_dir: Path
    csv_dir: Path
    saved_models_dir: Path


def train_sir_inverse(
    props: SIRInvProperties, hp: SIRInvHyperparameters, config: SIRInvTrainConfig
) -> None:
    # prepare
    clean_dir(config.tensorboard_dir)
    clean_dir(config.csv_dir)
    temp_dir = create_temp_dir()

    dm = SIRInvDataModule(
        props=props,
        hp=hp,
    )

    problem = SIRInvProblem(
        props=props,
        hp=hp,
    )

    module = PINNModule(
        problem=problem,
        hp=hp,
    )

    loggers = [
        TensorBoardLogger(
            save_dir=config.tensorboard_dir,
            name=config.name,
            version=config.version,
        ),
        CSVLogger(
            save_dir=config.csv_dir,
            name=config.name,
            version=config.version,
        ),
    ]

    callbacks = [
        ModelCheckpoint(
            dirpath=temp_dir,
            filename="{epoch:02d}",
            monitor="train/total_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(
            logging_interval="epoch",
        ),
    ]

    trainer = Trainer(
        max_epochs=hp.max_epochs,
        gradient_clip_val=hp.gradient_clip_val,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=0,
    )

    # train
    trainer.fit(module, dm)

    # save
    clean_dir(temp_dir)

    trainer.save_checkpoint(
        config.saved_models_dir / f"{config.version}.ckpt",
    )


if __name__ == "__main__":
    log_dir = Path("./data/logs")
    tensorboard_dir = log_dir / "tensorboard"
    csv_dir = log_dir / "csv"
    saved_models_dir = Path("./data/versions")

    props = SIRInvProperties()
    hp = SIRInvHyperparameters()
    config = SIRInvTrainConfig(
        name="sir_inverse_test",
        version="v0",
        tensorboard_dir=tensorboard_dir,
        csv_dir=csv_dir,
        saved_models_dir=saved_models_dir,
    )

    train_sir_inverse(props, hp, config)

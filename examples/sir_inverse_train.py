# src/pinn/train_sir_inverse.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor

from pinn.core import LOSS_KEY
from pinn.lightning import PINNModule, SMMAStopping
from pinn.lightning.callbacks import FormattedProgressBar, Metric, PredictionsWriter
from pinn.problems import SIRInvDataModule, SIRInvHyperparameters, SIRInvProblem, SIRInvProperties
from pinn.problems.sir_inverse import BETA_KEY, SIRInvTransformer


def create_dir(dir: Path) -> Path:
    if dir.exists():
        shutil.rmtree(dir)

    dir.mkdir(exist_ok=True)
    return dir


def clean_dir(dir: Path) -> None:
    if dir.exists():
        shutil.rmtree(dir)


def format(key: str, value: Metric) -> Metric:
    if key == LOSS_KEY:
        return f"{value:.2e}"
    elif key == BETA_KEY:
        return f"{value:.5f} -> {props.beta:.5f}"

    return value


def plot(predictions: dict[str, Tensor]) -> Figure:
    t_data = predictions["x_data"].squeeze()
    I_data = predictions["y_data"].squeeze()
    I_pred = predictions["I"].squeeze()

    fig, ax = plt.subplots()
    sns.lineplot(x=t_data, y=I_pred, label="I_pred", ax=ax)
    sns.lineplot(x=t_data, y=I_data, label="I_data", ax=ax)
    ax.legend()
    fig.tight_layout()

    return fig


@dataclass
class SIRInvTrainConfig:
    name: str
    version: str
    tensorboard_dir: Path
    csv_dir: Path
    saved_models_dir: Path
    predictions_dir: Path


def train_sir_inverse(
    props: SIRInvProperties, hp: SIRInvHyperparameters, config: SIRInvTrainConfig
) -> None:
    # prepare
    clean_dir(config.tensorboard_dir / config.name / config.version)
    clean_dir(config.csv_dir / config.name / config.version)
    temp_dir = Path("./temp")
    create_dir(temp_dir)
    create_dir(config.predictions_dir)
    checkpoint_path = config.saved_models_dir / f"{config.version}.ckpt"

    transformer = SIRInvTransformer(props)

    dm = SIRInvDataModule(
        props=props,
        hp=hp,
        transformer=transformer,
    )

    problem = SIRInvProblem(
        props=props,
        hp=hp,
        transformer=transformer,
    )

    module = PINNModule(
        problem=problem,
        hp=hp,
    )
    # module = PINNModule.load_from_checkpoint(
    #     checkpoint_path,
    #     problem=problem,
    # )

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
            monitor=LOSS_KEY,
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(
            logging_interval="epoch",
        ),
        FormattedProgressBar(
            refresh_rate=10,
            format=format,
        ),
        PredictionsWriter(
            dirpath=config.predictions_dir,
            plot=plot,
        ),
    ]

    if hp.smma_stopping:
        callbacks.append(
            SMMAStopping(
                config=hp.smma_stopping,
                loss_key=LOSS_KEY,
            ),
        )

    trainer = Trainer(
        max_epochs=hp.max_epochs,
        gradient_clip_val=hp.gradient_clip_val,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=0,
    )

    # train
    trainer.fit(module, dm)

    # predict
    trainer.predict(module, dm)

    # save
    trainer.save_checkpoint(checkpoint_path)

    # clean up
    clean_dir(temp_dir)


if __name__ == "__main__":
    root_dir = Path("./data")
    saved_models_dir = root_dir / "versions"
    predictions_dir = root_dir / "predictions"
    log_dir = root_dir / "logs"
    tensorboard_dir = log_dir / "tensorboard"
    csv_dir = log_dir / "csv"

    props = SIRInvProperties()
    hp = SIRInvHyperparameters()
    config = SIRInvTrainConfig(
        name="sir_inverse_test",
        version="v0",
        tensorboard_dir=tensorboard_dir,
        csv_dir=csv_dir,
        saved_models_dir=saved_models_dir,
        predictions_dir=predictions_dir,
    )

    train_sir_inverse(props, hp, config)

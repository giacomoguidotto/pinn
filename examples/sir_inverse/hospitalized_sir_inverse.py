from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
from typing import cast

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor
import torch.nn as nn

from pinn.core import (
    LOSS_KEY,
    ArgsRegistry,
    Argument,
    ColumnRef,
    Constraint,
    Field,
    FieldsRegistry,
    IngestionConfig,
    MLPConfig,
    Parameter,
    Predictions,
    Problem,
    SchedulerConfig,
    ValidationRegistry,
)
from pinn.lightning import PINNModule, SMMAStopping
from pinn.lightning.callbacks import DataScaling, FormattedProgressBar, Metric, PredictionsWriter
from pinn.problems import ODEProperties, SIRInvDataModule, SIRInvHyperparameters
from pinn.problems.ode import DataConstraint, ICConstraint, ResidualsConstraint
from pinn.problems.sir_inverse import DELTA_KEY, I_KEY, Rt_KEY

# ============================================================================
# Constants
# ============================================================================

SIGMA_KEY = "sigma"
H_KEY = "H"


# ============================================================================
# Hospitalized Problem Definition
# ============================================================================


class HospitalizedSIRInvProblem(Problem):
    """
    SIR Inverse Problem using hospitalization data.
    """

    def __init__(
        self,
        props: ODEProperties,
        hp: SIRInvHyperparameters,
        fields: list[Field],
        params: list[Parameter],
        C_H: float,
        C_I: float,
    ) -> None:
        def predict_data(x_data: Tensor, fields: FieldsRegistry) -> Tensor:
            delta = props.args[DELTA_KEY]
            I = fields[I_KEY]
            sigma = next(p for p in params if p.name == SIGMA_KEY)

            H_pred: Tensor = (delta(x_data) * C_I * sigma(x_data) * I(x_data)) / C_H

            return H_pred

        # Build constraints
        constraints: list[Constraint] = [
            ResidualsConstraint(
                props=props,
                fields=fields,
                params=[p for p in params if p.name == Rt_KEY],
                weight=hp.pde_weight,
            ),
            ICConstraint(
                props=props,
                fields=fields,
                weight=hp.ic_weight,
            ),
            DataConstraint(
                fields=fields,
                predict_data=predict_data,
                weight=hp.data_weight,
            ),
        ]

        criterion = nn.MSELoss()

        super().__init__(
            constraints=constraints,
            criterion=criterion,
            fields=fields,
            params=params,
        )


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RunConfig:
    max_epochs: int
    gradient_clip_val: float
    predict: bool

    run_name: str
    tensorboard_dir: Path
    csv_dir: Path
    model_path: Path
    predictions_dir: Path
    checkpoint_dir: Path
    experiment_name: str


# ============================================================================
# Helpers
# ============================================================================


def create_dir(dir: Path) -> Path:
    dir.mkdir(exist_ok=True, parents=True)
    return dir


def clean_dir(dir: Path) -> None:
    if dir.exists():
        shutil.rmtree(dir)


def format_progress_bar(key: str, value: Metric) -> Metric:
    if LOSS_KEY in key:
        return f"{value:.2e}"

    return value


def main(config: RunConfig) -> None:
    # ========================================================================
    # Hyperparameters
    # ========================================================================

    hp = SIRInvHyperparameters(
        lr=5e-4,
        training_data=IngestionConfig(
            batch_size=100,
            data_ratio=2,
            collocations=6000,
            df_path=Path("./data/synt_h_data.csv"),
            y_columns=["H_obs"],
        ),
        fields_config=MLPConfig(
            in_dim=1,
            out_dim=1,
            hidden_layers=[64, 128, 128, 64],
            activation="tanh",
            output_activation="softplus",
        ),
        params_config=MLPConfig(
            in_dim=1,
            out_dim=1,
            hidden_layers=[64, 128, 128, 64],
            activation="tanh",
            output_activation="softplus",
        ),
        scheduler=SchedulerConfig(
            mode="min",
            factor=0.5,
            patience=55,
            threshold=5e-3,
            min_lr=1e-6,
        ),
        # smma_stopping=SMMAStoppingConfig(
        #     window=50,
        #     threshold=0.01,
        #     lookback=50,
        # ),
    )

    # ========================================================================
    # Problem Properties
    # ========================================================================

    C_I = 1e6
    C_H = 1e3
    T = 120
    d = 1 / 5

    def hSIR_s(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        """Reduced SIR ODE with hospitalization constraint: dI/dt = δ(Rt - 1)I"""
        I = y
        d, Rt = args[DELTA_KEY], args[Rt_KEY]

        dI = d(x) * (Rt(x) - 1) * I
        dI = dI * T
        return dI

    props = ODEProperties(
        ode=hSIR_s,
        y0=torch.tensor([1]) / C_I,
        args={
            DELTA_KEY: Argument(d, name=DELTA_KEY),
        },
    )

    # ========================================================================
    # Validation Configuration
    # ========================================================================

    validation: ValidationRegistry = {
        Rt_KEY: ColumnRef(column="Rt"),
        SIGMA_KEY: ColumnRef(column="sigma"),
    }

    # ============================================================================
    # Training / Prediction Execution
    # ============================================================================

    dm = SIRInvDataModule(
        hp=hp,
        validation=validation,
        callbacks=[DataScaling(y_scale=1 / C_H)],
    )

    # define problem
    I_field = Field(config=replace(hp.fields_config, name=I_KEY))
    Rt = Parameter(config=replace(hp.params_config, name=Rt_KEY))
    sigma = Parameter(
        config=replace(
            cast(MLPConfig, hp.params_config),
            name=SIGMA_KEY,
            output_activation="sigmoid",
        )
    )

    problem = HospitalizedSIRInvProblem(
        props=props,
        hp=hp,
        fields=[I_field],
        params=[Rt, sigma],
        C_H=C_H,
        C_I=C_I,
    )

    if config.predict:
        module = PINNModule.load_from_checkpoint(
            config.model_path,
            problem=problem,
            weights_only=False,
        )
    else:
        module = PINNModule(
            problem=problem,
            hp=hp,
        )

    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
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
            format=format_progress_bar,
        ),
        PredictionsWriter(
            predictions_path=config.predictions_dir / "predictions.pt",
            on_prediction=lambda _, __, predictions_list, ___: plot_and_save(
                predictions_list[0], config.predictions_dir, props, C_I, C_H, d
            ),
        ),
    ]

    if hp.smma_stopping:
        callbacks.append(
            SMMAStopping(
                config=hp.smma_stopping,
                loss_key=LOSS_KEY,
            ),
        )

    loggers = [
        TensorBoardLogger(
            save_dir=config.tensorboard_dir,
            name=config.experiment_name,
            version=config.run_name,
        ),
        CSVLogger(
            save_dir=config.csv_dir,
            name=config.experiment_name,
            version=config.run_name,
        ),
    ]

    trainer = Trainer(
        max_epochs=config.max_epochs,
        gradient_clip_val=config.gradient_clip_val,
        logger=loggers if not config.predict else [],
        callbacks=callbacks,
        log_every_n_steps=0,
    )

    if not config.predict:
        trainer.fit(module, dm)
        trainer.save_checkpoint(config.model_path, weights_only=False)

    trainer.predict(module, dm)

    clean_dir(config.checkpoint_dir)


# ============================================================================
# Plotting and Saving
# ============================================================================


def plot_and_save(
    predictions: Predictions,
    predictions_dir: Path,
    props: ODEProperties,
    C_I: float,
    C_H: float,
    delta: float,
) -> None:
    batch, preds, trues = predictions
    t_data, H_obs_data = batch

    # Extract predictions
    I_pred = C_I * preds[I_KEY]  # Unscale infections
    Rt_pred = preds[Rt_KEY]
    sigma_pred = preds[SIGMA_KEY]

    # Compute predicted hospitalizations from I and sigma
    # H_pred = delta * sigma * I (daily new hospitalizations)
    H_pred = delta * sigma_pred * I_pred

    # Unscale observed hospitalizations
    H_obs = C_H * H_obs_data

    # Extract ground truth if available
    Rt_true = trues[Rt_KEY] if trues and Rt_KEY in trues else None
    sigma_true = trues[SIGMA_KEY] if trues and SIGMA_KEY in trues else None

    # Create plots
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Hospitalization data fit
    ax = axes[0, 0]
    sns.lineplot(x=t_data, y=H_pred, label=r"$\Delta H_{pred}$", ax=ax, color="C0")
    sns.scatterplot(
        x=t_data, y=H_obs, label=r"$\Delta H_{obs}$", ax=ax, color="C1", s=20, alpha=0.6
    )
    ax.set_title("Daily Hospitalizations")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Daily Hospitalizations")
    ax.legend()

    # Plot 2: Predicted infections
    ax = axes[0, 1]
    sns.lineplot(x=t_data, y=I_pred, label="$I_{pred}$", ax=ax, color="C2")
    ax.set_title("Predicted Infected Population")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("I (Population)")
    ax.legend()

    # Plot 3: Rt parameter
    ax = axes[1, 0]
    if Rt_true is not None:
        sns.lineplot(x=t_data, y=Rt_true, label=r"$R_{t, true}$", ax=ax, color="C3")
    sns.lineplot(
        x=t_data,
        y=Rt_pred,
        label=r"$R_{t, pred}$",
        ax=ax,
        linestyle="--" if Rt_true is not None else "-",
        color="C4" if Rt_true is not None else "C3",
    )
    ax.axhline(y=1, color="red", linestyle=":", alpha=0.5, label="$R_t = 1$")
    ax.set_title(r"$R_t$ (Reproduction Number)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"$R_t$")
    ax.legend()

    # Plot 4: Sigma parameter (hospitalization rate)
    ax = axes[1, 1]
    if sigma_true is not None:
        sns.lineplot(x=t_data, y=sigma_true, label=r"$\sigma_{true}$", ax=ax, color="C5")
    sns.lineplot(
        x=t_data,
        y=sigma_pred,
        label=r"$\sigma_{pred}$",
        ax=ax,
        linestyle="--" if sigma_true is not None else "-",
        color="C6" if sigma_true is not None else "C5",
    )
    ax.set_title(r"$\sigma$ (Hospitalization Rate)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"$\sigma$ (fraction)")
    ax.legend()

    plt.tight_layout()

    fig.savefig(predictions_dir / "predictions.png", dpi=300)
    plt.close(fig)

    # save
    df = pd.DataFrame(
        {
            "t": t_data,
            "H_obs": H_obs,
            "H_pred": H_pred,
            "I_pred": I_pred,
            "Rt_pred": Rt_pred,
            "sigma_pred": sigma_pred,
        }
    )

    if Rt_true is not None:
        df["Rt_true"] = Rt_true
    if sigma_true is not None:
        df["sigma_true"] = sigma_true

    df.to_csv(predictions_dir / "predictions.csv", index=False, float_format="%.6e")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIR Inverse Problem using Hospitalization Data")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction. Does not train the model.",
    )
    args = parser.parse_args()

    experiment_name = "hospitalized-sir-inverse"
    run_name = "v0"

    log_dir = Path("./logs")
    tensorboard_dir = log_dir / "tensorboard"
    csv_dir = log_dir / "csv"

    models_dir = Path("./models") / experiment_name / run_name
    model_path = models_dir / "model.ckpt"
    predictions_dir = models_dir

    temp_dir = Path("./temp")

    create_dir(log_dir)
    create_dir(models_dir)
    create_dir(predictions_dir)
    create_dir(temp_dir)

    clean_dir(temp_dir)
    if not args.predict:
        clean_dir(csv_dir / experiment_name / run_name)
        clean_dir(tensorboard_dir / experiment_name / run_name)

    config = RunConfig(
        max_epochs=2000,
        gradient_clip_val=0.1,
        predict=args.predict,
        run_name=run_name,
        tensorboard_dir=tensorboard_dir,
        csv_dir=csv_dir,
        model_path=model_path,
        predictions_dir=predictions_dir,
        checkpoint_dir=temp_dir,
        experiment_name=experiment_name,
    )
    main(config)

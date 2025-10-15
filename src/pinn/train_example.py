# src/pinn/train_sir_inverse.py
from __future__ import annotations

from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from pinn.module import CollocationDataset, PINNModule, SchedulerConfig
from pinn.sir import SIRHyperparameters, SIRObservation, SIRProperties, build_sir_problem
from pinn.sir_pinn import SIRConfig, generate_sir_data

LOG_DIR = Path("./data/logs")
TENSORBOARD_DIR = LOG_DIR / "tensorboard"
CSV_DIR = LOG_DIR / "csv"
VERSIONS_DIR = Path("./data/versions")


def train_sir_inverse(
    props: SIRProperties, hp: SIRHyperparameters, run_name: str = "sir_inverse_beta_mlp"
) -> tuple[Path, str]:
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: do it internally
    t_obs, _, i_obs = generate_sir_data(
        SIRConfig(
            N=props.N,
            delta=props.delta,
            time_domain=(int(props.domain.t0), int(props.domain.t1)),
            collocation_points=props.n_collocation,
            initial_conditions=[1.0, 0.0],
            beta_true=props.delta * 3.0,  # dummy for synthetic generation
        )
    )

    problem, sampler = build_sir_problem(props, SIRObservation(t_obs, i_obs), hp)

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

    loader = CollocationDataset(
        sampler,
        hp.batch_size,
    )

    trainer.fit(module, loader)

    version = f"v{len(list(VERSIONS_DIR.iterdir()))}_{run_name}"
    model_path = VERSIONS_DIR / f"{version}.ckpt"
    trainer.save_checkpoint(model_path)
    return model_path, version

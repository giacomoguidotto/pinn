from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from pinn.core import Argument


def get_tensorboard_logger_or_raise(trainer: Trainer) -> TensorBoardLogger:
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            return logger
    raise ValueError("TensorBoard logger not found")


def get_tensorboard_logger(
    trainer: Trainer,
    default: TensorBoardLogger | None = None,
) -> TensorBoardLogger | None:
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            return logger
    return default


def get_arg_or_raise(args: list[Argument], name: str) -> Argument:
    for arg in args:
        if arg.name == name:
            return arg
    raise ValueError(f"Argument {name} not found")


def get_arg(
    args: list[Argument],
    name: str,
    default: Argument | None = None,
) -> Argument | None:
    for arg in args:
        if arg.name == name:
            return arg
    return default
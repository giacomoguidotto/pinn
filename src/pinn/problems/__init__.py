"""Problem templates and implementations."""

from pinn.problems.ode import Domain1D, ODECallable, ODEDataset, ODEProperties
from pinn.problems.sir_inverse import (
    SIR,
    ODEDataset,
    SIRInvCollocationset,
    SIRInvDataModule,
    SIRInvHyperparameters,
    SIRInvProblem,
    SIRInvProperties,
)

__all__ = [
    "SIR",
    "Domain1D",
    "ODECallable",
    "ODEDataset",
    "ODEDataset",
    "ODEProperties",
    "SIRInvCollocationset",
    "SIRInvDataModule",
    "SIRInvHyperparameters",
    "SIRInvProblem",
    "SIRInvProperties",
]

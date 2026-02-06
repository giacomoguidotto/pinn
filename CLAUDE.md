# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PINN is a modular Python library for solving differential equations using Physics-Informed Neural Networks. It separates problem definition from solution architecture, built on PyTorch Lightning.

## Commands

```bash
# Setup
uv sync                      # Install dependencies

# Testing & Quality (via nox)
uv run nox -s test           # Run tests (multi-Python, requires 100% coverage)
uv run nox -s lint           # Check code style
uv run nox -s fmt            # Format code (isort + ruff)
uv run nox -s lint_fix       # Auto-fix linting issues
uv run nox -s type_check     # MyPy strict type checking

# Documentation
uv run nox -s docs           # Build mkdocs
uv run nox -s docs_serve     # Serve docs locally

# Direct invocation
pytest tests/                # Run tests directly
pytest tests/test_foo.py::test_bar  # Run single test
```

## Architecture

### Core Abstractions (`src/pinn/core/`)

- **Problem** (`problem.py`): Aggregates constraints, manages fields and parameters, provides `training_loss()` and `predict()`
- **Constraint** (`problem.py`): Abstract base for loss terms. Subclasses define specific physics/data matching losses
- **Field** (`nn.py`): MLP that maps coordinates to state variables (e.g., time → [S, I, R])
- **Parameter** (`nn.py`): Learnable scalar or function-valued parameters (can be fixed or time-dependent)
- **InferredContext** (`context.py`): Runtime context holding domain bounds and data extracted from training data

### Data Flow

```
Data Source (CSV or generated)
    ↓
PINNDataModule.load_data() / gen_data()
    ↓
InferredContext (domain bounds, validation)
    ↓
PINNDataset (data + collocation points)
    ↓
PINNModule.training_step()
    ↓
Problem.training_loss() → Σ weighted constraint losses
```

### Problem Implementation Pattern (`src/pinn/problems/`)

ODE problems use three constraint types from `ode.py`:
- **ResidualsConstraint**: Enforces ODE satisfaction (||dy/dt - f(t,y)||²)
- **ICConstraint**: Enforces initial conditions
- **DataConstraint**: Matches predictions to observed data

### Lightning Integration (`src/pinn/lightning/`)

- **PINNModule** (`module.py`): Wraps Problem for Lightning training
- **Callbacks** (`callbacks.py`): SMMA-based early stopping, progress bar

### Configuration System

All hyperparameters via dataclasses in `core/config.py`:
- `PINNHyperparameters`: Batch size, LR, scheduler, stopping criteria
- `MLPConfig`: Neural network architecture
- `ScalarConfig`: Learnable parameter initialization

## Extending the Library

To add a new problem:
1. Define hyperparameters: `class MyHyperparameters(PINNHyperparameters)`
2. Create constraints: `class MyConstraint(Constraint)` with `forward()` returning loss
3. Create problem: `class MyProblem(Problem)` aggregating constraints
4. Create data module: `class MyDataModule(PINNDataModule)` implementing `gen_data()` and `gen_coll()`

See `examples/sir_inverse/sir_inverse.py` for a complete training pipeline.

## Code Style

- Line length: 99
- Ruff linter with rules: F, E, I, N, UP, RUF, B, C4, ISC, PIE, PT, PTH, SIM, TID
- Absolute imports only (no relative imports)
- MyPy strict mode with exhaustive-match enabled

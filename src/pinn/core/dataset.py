from collections.abc import Sized
from typing import TYPE_CHECKING, Any, TypeAlias, cast, override

import lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from pinn.problems.ode import ODECallable

DataBatch: TypeAlias = tuple[Tensor, Tensor]

PINNBatch: TypeAlias = tuple[DataBatch, Tensor]
"""
Batch is a tuple of (data, collocations) where: data is another tuple of two tensors 
(t_data y_data) with both having shape (batch_size, 1); collocations is a tensor with 
shape (collocations_size, 1) of collocation points over the domain.
"""


class Scaler:
    """
    Apply a scaling to a batch of data and collocations.
    """

    def __init__(self) -> None:
        self.t_min = 0.0
        self.t_max = 1.0
        self.t_scale = 1.0
        self.y_scale = 1.0

    def fit(self, domain_info: Any, data: Tensor, Y0: list[float] | Tensor) -> None:
        """
        Infer scaling parameters from domain and data.

        Args:
            domain_info: Object with x0 and x1 attributes or similar
            data: Observation data tensor
            Y0: Initial conditions
        """
        # Time scaling
        if hasattr(domain_info, "x0") and hasattr(domain_info, "x1"):
            self.t_min = float(domain_info.x0)
            self.t_max = float(domain_info.x1)
        else:
            # Fallback if domain_info is just bounds tuple or similar
            pass

        self.t_scale = self.t_max - self.t_min if self.t_max != self.t_min else 1.0

        # Value scaling
        # Uses max absolute value of data and Y0 for value scaling
        y0_tensor = torch.tensor(Y0, dtype=torch.float32) if isinstance(Y0, list) else Y0

        max_data = torch.max(torch.abs(data)) if data.numel() > 0 else torch.tensor(0.0)
        max_y0 = torch.max(torch.abs(y0_tensor)) if y0_tensor.numel() > 0 else torch.tensor(0.0)

        max_val = max(float(max_data), float(max_y0))
        self.y_scale = max_val if max_val > 1e-8 else 1.0

    def transform_domain(self, domain: Tensor) -> Tensor:
        return (domain - self.t_min) / self.t_scale

    def inverse_domain(self, domain: Tensor) -> Tensor:
        return domain * self.t_scale + self.t_min

    def transform_values(self, values: Tensor) -> Tensor:
        return values / self.y_scale

    def inverse_values(self, values: Tensor) -> Tensor:
        return values * self.y_scale

    def transform_batch(self, batch: PINNBatch) -> PINNBatch:
        (x_data, y_data), x_coll = batch

        x_data = self.transform_domain(x_data)
        y_data = self.transform_values(y_data)
        x_coll = self.transform_domain(x_coll)

        return ((x_data, y_data), x_coll)

    def scale_ode(self, physical_ode: "ODECallable") -> "ODECallable":
        """
        Wraps the user's ODE function (defined in physical units) to operate in the scaled domain.

        dy_hat/dt_hat = (1/y_scale) * dy/dt * t_scale
        """

        def scaled_ode_fn(t_hat: Tensor, y_hat: Tensor, args: list[Any]) -> Tensor:
            # Transform inputs back to physical domain
            t = self.inverse_domain(t_hat)
            y = self.inverse_values(y_hat)

            # Compute physical derivatives
            dy_dt = physical_ode(t, y, args)

            # Transform derivatives to scaled domain
            # dy_hat/dt_hat = dy/dt * (dt/dt_hat) * (dy_hat/dy)
            # dt/dt_hat = t_scale
            # dy_hat/dy = 1/y_scale

            return dy_dt * (self.t_scale / self.y_scale)

        return scaled_ode_fn


class PINNDataset(Dataset[PINNBatch]):
    """
    Dataset used for PINN training. Combines labeled data and collocation points
    per sample.  Given a data_ratio, the amount of data points `K` is determined
    either by applying `data_ratio * batch_size` if ratio is a float between 0
    and 1 or by an absolute count if ratio is an integer. The remaining `C`
    points are used for collocation.  The data points are sampled without
    replacement per epoch i.e. cycles through all data points and at the last
    batch, wraps around to the first indices to ensure batch size. The collocation
    points are sampled with replacement from the pool.
    The dataset produces a batch of shape ((t_data[K,1], y_data[K,1]), t_coll[C,1]).

    Args:
        data_ds: Dataset of data points.
        coll_ds: Dataset of collocation points.
        batch_size: Size of the batch.
        data_ratio: Ratio of data points to collocation points, either as a ratio [0,1] or absolute
            count [0,batch_size].
        transform: Optional transformation to apply to the batch.
    """

    def __init__(
        self,
        data_ds: Dataset[DataBatch],
        coll_ds: Dataset[Tensor],
        batch_size: int,
        data_ratio: float | int,
    ):
        super().__init__()
        assert batch_size > 0

        if isinstance(data_ratio, float):
            assert 0.0 <= data_ratio <= 1.0
            self.K = round(data_ratio * batch_size)
        else:
            assert 0 <= data_ratio <= batch_size
            self.K = data_ratio

        self.data_ds = data_ds
        self.coll_ds = coll_ds

        self.batch_size = batch_size
        self.C = batch_size - self.K

        self.total_data = len(cast(Sized, data_ds))
        self.total_coll = len(cast(Sized, coll_ds))

    def __len__(self) -> int:
        """Number of steps per epoch to see all data points once. Ceiling division."""
        return (self.total_data + self.K - 1) // self.K

    @override
    def __getitem__(self, idx: int) -> PINNBatch:
        """Return one sample containing K data points and C collocation points."""
        data_idx = self._get_data_indices(idx)
        coll_idx = self._get_coll_indices(idx)

        x_data, y_data = self.data_ds[data_idx]
        x_coll = self.coll_ds[coll_idx]

        return ((x_data, y_data), x_coll)

    def _get_data_indices(self, idx: int) -> Tensor:
        """Get data indices for this step without replacement.
        When getting the last batch, wrap around to the first indices to ensure batch size.
        """
        if self.total_data == 0:
            return torch.empty(0, 1)

        start = idx * self.K
        indices = [(start + i) % self.total_data for i in range(self.K)]
        return torch.tensor(indices)

    def _get_coll_indices(self, idx: int) -> Tensor:
        """Get collocation indices for this step with replacement."""
        if self.total_coll == 0:
            return torch.empty(0, 1)

        temp_generator = torch.Generator()
        temp_generator.manual_seed(idx)
        return torch.randint(0, self.total_coll, (self.C,), generator=temp_generator)


class PINNDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.data_ds: Dataset[DataBatch]
        self.coll_ds: Dataset[Tensor]
        self.pinn_ds: PINNDataset

    @override
    def train_dataloader(self) -> DataLoader[PINNBatch]:
        assert self.pinn_ds is not None
        return DataLoader[PINNBatch](
            self.pinn_ds,
            batch_size=None,  # handled internally
            num_workers=7,
            persistent_workers=True,
        )

    @override
    def predict_dataloader(self) -> DataLoader[DataBatch]:
        assert self.data_ds is not None
        data_size = len(cast(Sized, self.data_ds))
        return DataLoader[DataBatch](
            self.data_ds,
            batch_size=data_size,
            num_workers=7,
            persistent_workers=True,
        )

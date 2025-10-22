from collections.abc import Sized
from typing import TypeAlias, cast, override

import torch
from torch import Tensor
from torch.utils.data import Dataset

Batch: TypeAlias = tuple[tuple[Tensor, Tensor], Tensor]
"""
Batch is a tuple of (data, collocations) where: data is another tuple of two tensors 
(t_data y_data) with both having shape (batch_size, 1); collocations is a tensor with 
shape (collocations_size, 1) of collocation points over the domain.
"""


class PINNDataset(Dataset[Batch]):
    """
    Map-style dataset that mixes labeled data and collocation points per sample.

    - Each sample contains K data points and C collocation points.
    - K is derived from `data_ratio` parameter (ratio [0,1] or absolute count [0,B]).
    - C = batch_size - K.
    - Data points are sampled without replacement per epoch (cycles through all data).
    - Collocation points are sampled with replacement from the pool.
    - One epoch = one pass through all data points.
    - Shapes produced: ((t_data[K,1], y_data[K,1]), t_colloc[C,1])
    """

    def __init__(
        self,
        data_ds: Dataset[Tensor],
        coll_ds: Dataset[Tensor],
        batch_size: int,
        data_ratio: float | int,  # ratio [0,1] or absolute count [0,B]
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
        """Number of steps per epoch to see all data points once."""
        # ceiling division
        return (self.total_data + self.K - 1) // self.K

    @override
    def __getitem__(self, idx: int) -> Batch:
        """Return one sample containing K data points and C collocation points."""
        # Sample K data points without replacement (cycling through epoch)
        data_indices = self._get_data_indices(idx)
        colloc_indices = self._get_colloc_indices(idx)

        if data_indices:
            # Pre-allocate output tensors
            t_data = torch.empty(len(data_indices), 1)
            y_data = torch.empty(len(data_indices), 1)

            # Single loop with direct tensor assignment (no intermediate lists)
            for i, data_idx in enumerate(data_indices):
                sample = self.data_ds[data_idx]  # [2, 1]
                t_data[i, 0] = sample[0, 0]  # Direct assignment
                y_data[i, 0] = sample[1, 0]  # Direct assignment
        else:
            t_data = torch.empty(0, 1)
            y_data = torch.empty(0, 1)

        # Maximum efficiency collocation processing
        if colloc_indices:
            # Pre-allocate output tensor
            t_colloc = torch.empty(len(colloc_indices), 1)

            # Single loop with direct tensor assignment
            for i, colloc_idx in enumerate(colloc_indices):
                sample = self.coll_ds[colloc_idx]  # [1]
                t_colloc[i, 0] = sample[0]  # Direct assignment
        else:
            t_colloc = torch.empty(0, 1)

        return ((t_data, y_data), t_colloc)

    def _get_data_indices(self, idx: int) -> list[int]:
        """Get data indices for this step without replacement."""
        if self.total_data == 0:
            return []

        start_idx = idx * self.K
        end_idx = min(start_idx + self.K, self.total_data)

        if end_idx - start_idx < self.K:
            remaining = self.K - (end_idx - start_idx)
            indices = list(range(start_idx, end_idx)) + list(range(remaining))
        else:
            indices = list(range(start_idx, end_idx))

        return indices

    def _get_colloc_indices(self, idx: int) -> list[int]:
        """Get collocation indices for this step with replacement."""
        if self.total_coll == 0:
            return []

        temp_generator = torch.Generator()
        temp_generator.manual_seed(idx)

        indices = torch.randint(0, self.total_coll, (self.C,), generator=temp_generator).tolist()
        return indices

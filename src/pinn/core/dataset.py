from collections.abc import Callable, Iterator, Sized
from typing import cast, override

import torch
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler

from pinn.core import Batch, Tensor


class MixedPINNIterable(IterableDataset[Batch]):
    """
    Generic iterable dataset that mixes labeled data and collocation points per batch.

    - Total batch size per step is `batch_size` (B).
    - Number of labeled data per batch is derived from `data_per_batch`:
      - If float in [0, 1], it's treated as a ratio of B.
      - If int in [0, B], it's treated as an absolute count.
    - Labeled data can be sampled without replacement (epoch-bounded) or with replacement.
    - Collocation points can be drawn from a dataset (with/without replacement) or
      generated freshly every step via `colloc_fn(C) -> Tensor[C, 1]`.
    - If replacement is enabled for any stream, you can cap epoch length using
      `steps_per_epoch` to keep iteration finite.
    - Shapes produced:
        ((t_data[K,1], y_data[K,1]), t_colloc[C,1])
    """

    def __init__(
        self,
        data_ds: Dataset[Tensor],  # yields either (t, y) or a Tensor that can be split
        colloc_ds: Dataset[Tensor] | None,  # yields t; optional if using `colloc_fn`
        batch_size: int,
        data_ratio: float | int,  # ratio [0,1] or absolute count [0,B]
        drop_last: bool = False,
        data_replacement: bool = False,
        colloc_replacement: bool = False,
        steps_per_epoch: int | None = None,
        generator_data: torch.Generator | None = None,
        generator_colloc: torch.Generator | None = None,
        colloc_fn: Callable[[int], Tensor] | None = None,
        data_transform: Callable[[Tensor], tuple[Tensor, Tensor]] | None = None,
        colloc_transform: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        assert batch_size > 0

        if isinstance(data_ratio, float):
            assert 0.0 <= data_ratio <= 1.0
            self.data_ratio = float(data_ratio)
        else:
            assert 0 <= data_ratio <= batch_size
            self.data_ratio = data_ratio / float(batch_size)

        self.data_ds = data_ds
        self.colloc_ds = colloc_ds
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_replacement = data_replacement
        self.colloc_replacement = colloc_replacement
        self.steps_per_epoch = steps_per_epoch
        self.generator_data = generator_data
        self.generator_colloc = generator_colloc
        self.colloc_fn = colloc_fn

        # Transforms
        self.data_transform = data_transform or self._default_data_transform
        self.colloc_transform = colloc_transform or self._default_colloc_transform

    @staticmethod
    def _default_data_transform(sample: Tensor | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        # Allow (t, y) tuples directly
        if isinstance(sample, tuple):
            t, y = sample
            # Flatten singletons to 1D then we will stack to [K, 1]
            return t.reshape(-1), y.reshape(-1)

        # If a Tensor, try to split first dimension (e.g., [2, 1])
        if isinstance(sample, torch.Tensor):
            if sample.ndim == 2 and sample.shape[0] == 2:
                t = sample[0].reshape(-1)
                y = sample[1].reshape(-1)
                return t, y
            if sample.ndim == 1 and sample.shape[0] == 2:
                t = sample[0].reshape(-1)
                y = sample[1].reshape(-1)
                return t, y

        raise ValueError("Unsupported data sample format for default transform.")

    @staticmethod
    def _default_colloc_transform(sample: Tensor) -> Tensor:
        # Flatten any singleton dims to 1D (length 1 for each sample), later stacked to [C, 1]
        return sample.reshape(-1)

    def _make_sampler(
        self, ds: Sized, *, replacement: bool, generator: torch.Generator | None
    ) -> Sampler[int]:
        if replacement:
            # Infinite-like sampling; caller should still cap via steps_per_epoch
            return RandomSampler(
                ds,
                replacement=True,
                num_samples=10**12,
                generator=generator,
            )
        return RandomSampler(ds, generator=generator)

    @override
    def __iter__(self) -> Iterator[Batch]:
        # Build samplers (colloc may be generated via function instead of dataset)
        data_ds_sized: Sized = cast(Sized, self.data_ds)
        data_sampler = iter(
            self._make_sampler(
                data_ds_sized,
                replacement=self.data_replacement,
                generator=self.generator_data,
            )
        )
        colloc_sampler: Iterator[int] | None = None
        if self.colloc_fn is None and self.colloc_ds is not None:
            colloc_ds_sized: Sized = cast(Sized, self.colloc_ds)
            colloc_sampler = iter(
                self._make_sampler(
                    colloc_ds_sized,
                    replacement=self.colloc_replacement,
                    generator=self.generator_colloc,
                )
            )

        data_len = len(data_ds_sized)
        colloc_len = len(cast(Sized, self.colloc_ds)) if self.colloc_ds is not None else 0
        seen_data = 0
        seen_colloc = 0

        def fetch(ds: Dataset[Tensor], sampler_iter: Iterator[int], n: int) -> list[Tensor]:
            out: list[Tensor] = []
            for _ in range(n):
                try:
                    idx = next(sampler_iter)
                except StopIteration:
                    return out
                out.append(ds[idx])
            return out

        produced_steps = 0
        # Precompute desired counts per batch from ratio
        desired_K = min(
            self.batch_size,
            max(0, round(self.data_ratio * self.batch_size)),
        )
        desired_C = self.batch_size - desired_K

        while True:
            if self.steps_per_epoch is not None and produced_steps >= self.steps_per_epoch:
                break

            if self.data_replacement:
                K = desired_K
            else:
                rem_data = data_len - seen_data
                K = min(desired_K, max(0, rem_data))

            if self.colloc_fn is not None:
                C = desired_C
            else:
                if self.colloc_replacement:
                    C = desired_C
                else:
                    rem_colloc = colloc_len - seen_colloc
                    C = min(desired_C, max(0, rem_colloc))

            if K == 0 and C == 0:
                break  # epoch over

            # If drop_last is set, avoid yielding incomplete final batch
            if self.drop_last and self.batch_size > (K + C):
                break

            # Fetch labeled data
            data_samples: list[Tensor] = fetch(self.data_ds, data_sampler, K) if K > 0 else []

            # Fetch or generate collocation points
            if C > 0:
                if self.colloc_fn is not None:
                    colloc_tensor = self.colloc_fn(C)
                    # Normalize to [C, 1]
                    t_colloc = colloc_tensor.reshape(C, -1)
                    if t_colloc.shape[1] != 1:
                        # If more dims were generated, keep first column as the time coordinate
                        t_colloc = t_colloc[:, :1]
                    fetched_colloc = C
                    colloc_samples_list: list[Tensor] = []
                else:
                    if colloc_sampler is None or self.colloc_ds is None:
                        raise RuntimeError(
                            "colloc_ds or colloc_sampler is not available while colloc_fn is None"
                        )
                    colloc_samples_list = fetch(self.colloc_ds, colloc_sampler, C)
                    fetched_colloc = len(colloc_samples_list)
                    if fetched_colloc > 0:
                        transformed = [self.colloc_transform(s) for s in colloc_samples_list]
                        t_colloc = torch.stack(transformed).reshape(fetched_colloc, -1)
                    else:
                        t_colloc = torch.empty(0, 1)
            else:
                fetched_colloc = 0
                t_colloc = torch.empty(0, 1)

            # Update seen counters for non-replacement, dataset-backed streams
            if not self.data_replacement:
                seen_data += len(data_samples)
            if self.colloc_fn is None and not self.colloc_replacement:
                seen_colloc += fetched_colloc

            # If either stream under-delivered and drop_last requested, end epoch
            if self.drop_last and (len(data_samples) + fetched_colloc < self.batch_size):
                break
            if len(data_samples) == 0 and fetched_colloc == 0:
                break

            # Collate labeled data -> ((t[K,1], y[K,1]), ...)
            if len(data_samples) > 0:
                t_list: list[Tensor] = []
                y_list: list[Tensor] = []
                for s in data_samples:
                    t_i, y_i = self.data_transform(s)
                    t_list.append(t_i.reshape(-1))  # [1]
                    y_list.append(y_i.reshape(-1))  # [1]
                t_data = torch.stack(t_list, dim=0).reshape(len(t_list), 1)
                y_data = torch.stack(y_list, dim=0).reshape(len(y_list), 1)
            else:
                t_data = torch.empty(0, 1)
                y_data = torch.empty(0, 1)

            # Ensure colloc tensor has shape [C, 1]
            if t_colloc.numel() > 0:
                t_colloc = t_colloc.reshape(t_colloc.shape[0], 1)

            yield ((t_data, y_data), t_colloc)
            produced_steps += 1

# -*- coding: utf-8 -*-
"""Distributed calibration context for sample-parallel calibration."""

from __future__ import annotations

import typing as tp

import torch
import torch.distributed as dist


__all__ = ["DistributedCalibContext"]


class DistributedCalibContext:
    """Manages torch.distributed state for sample-parallel calibration.

    Distributes calibration sample evaluation across multiple GPUs while
    preserving the strict layer-by-layer sequential dependency. Only the
    sample dimension within a single layer's calibration is parallelized.

    When world_size=1 or distributed is not initialized, all operations
    gracefully degrade to no-ops.

    Args:
        group: The process group to use. If None, uses the default group.
    """

    def __init__(self, group: dist.ProcessGroup | None = None) -> None:
        if dist.is_initialized():
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
            self.device = torch.device("cuda", self.rank)
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        self.group = group

    @property
    def is_distributed(self) -> bool:
        """Return True if running in a multi-GPU distributed context."""
        return self.world_size > 1

    def shard_range(self, total: int) -> tuple[int, int]:
        """Compute the (start, end) index range for this rank.

        Divides `total` items as evenly as possible across ranks.
        The first ``total % world_size`` ranks each get one extra item.

        Args:
            total: Total number of items to shard.

        Returns:
            A (start, end) tuple where end is exclusive.
        """
        if not self.is_distributed:
            return 0, total
        per_rank = total // self.world_size
        remainder = total % self.world_size
        # Distribute remainder across first `remainder` ranks
        if self.rank < remainder:
            start = self.rank * (per_rank + 1)
            end = start + per_rank + 1
        else:
            start = remainder * (per_rank + 1) + (self.rank - remainder) * per_rank
            end = start + per_rank
        return start, end

    def shard_data(self, data_list: tp.Sequence) -> tp.Sequence:
        """Return the local slice of data_list for this rank.

        Args:
            data_list: A sequence of items to shard across ranks.

        Returns:
            The subsequence assigned to this rank.
        """
        if not self.is_distributed:
            return data_list
        start, end = self.shard_range(len(data_list))
        return data_list[start:end]

    def shard_indices(self, total: int) -> range:
        """Return the range of indices assigned to this rank.

        Args:
            total: Total number of items.

        Returns:
            A range object for the local indices.
        """
        start, end = self.shard_range(total)
        return range(start, end)

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce a tensor by summing across all ranks.

        When not distributed, returns the tensor unchanged.

        Args:
            tensor: The tensor to reduce.

        Returns:
            The reduced tensor (in-place).
        """
        if not self.is_distributed:
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        return tensor

    def all_reduce_error_list(self, errors: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
        """All-reduce a list of error tensors by summing across all ranks.

        All ranks MUST call this method together (it is a collective
        operation). Every entry in ``errors`` must be a non-None tensor
        of identical shape on all ranks -- otherwise ``dist.all_reduce``
        will deadlock or produce undefined behaviour.

        To guarantee this, callers must ensure that every rank processes
        at least one sample so that all error slots are populated.  Use
        :meth:`validate_sample_count` before the calibration loop to
        check this precondition.

        Handles None entries gracefully (they remain None).

        Args:
            errors: A list of error tensors (some may be None).

        Returns:
            The list with each non-None tensor all-reduced in-place.
        """
        if not self.is_distributed:
            return errors
        for idx, e in enumerate(errors):
            if e is not None:
                dist.all_reduce(e, op=dist.ReduceOp.SUM, group=self.group)
        return errors

    def validate_sample_count(self, num_samples: int) -> bool:
        """Check if there are enough samples for distributed sharding.

        When ``num_samples < world_size`` some ranks would receive zero
        samples. In this case, return False to signal the caller should
        fall back to non-distributed mode (all ranks process all samples).

        Args:
            num_samples: The total number of calibration samples.

        Returns:
            True if sharding is safe, False if caller should skip sharding.
        """
        if self.is_distributed and num_samples < self.world_size:
            return False
        return True

    def _validate_sample_count_strict(self, num_samples: int) -> None:
        """Strict version that raises an error. Kept for reference."""
        if self.is_distributed and num_samples < self.world_size:
            raise ValueError(
                f"Sample-parallel calibration requires at least as many "
                f"samples as GPUs. Got num_samples={num_samples} but "
                f"world_size={self.world_size}. Either increase the number "
                f"of calibration samples or reduce the number of GPUs."
            )

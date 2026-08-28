# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport-independent sparse runtime weight patch operations."""

from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from collections.abc import Iterator

import torch


@dataclass
class SparseWeightPatch:
    """A sparse in-place patch for one existing runtime parameter."""

    name: str
    indices: torch.Tensor
    values: torch.Tensor


def validate_sparse_patch(
    model: torch.nn.Module,
    patch: SparseWeightPatch,
    expected_shape: list[int] | None = None,
) -> torch.nn.Parameter:
    """Validate a flat-index patch and return its target parameter."""
    param = model.get_parameter(patch.name)
    if expected_shape is not None and list(param.shape) != expected_shape:
        raise ValueError(
            f"Sparse parameter shape {list(param.shape)} does not match "
            f"declared shape {expected_shape} for {patch.name}"
        )
    if not param.data.is_contiguous():
        raise NotImplementedError(
            "Sparse weight updates currently require contiguous params: "
            f"{patch.name}"
        )
    if patch.indices.dtype != torch.int32:
        raise ValueError(
            "Sparse weight updates currently require int32 indices: "
            f"{patch.name}"
        )
    if patch.indices.ndim != 1 or patch.values.ndim != 1:
        raise ValueError(
            f"Sparse weight patches must be 1D flattened updates: {patch.name}"
        )
    if patch.indices.numel() != patch.values.numel():
        raise ValueError(
            "`indices` and `values` must have matching lengths for "
            f"{patch.name}"
        )
    if patch.values.dtype != param.dtype:
        raise ValueError(
            f"Sparse values dtype {patch.values.dtype} does not match "
            f"parameter dtype {param.dtype} for {patch.name}"
        )
    return param


def apply_sparse_patch(
    model: torch.nn.Module,
    patch: SparseWeightPatch,
    expected_shape: list[int] | None = None,
    *,
    verify: bool = True,
) -> None:
    """Apply a validated patch and optionally require exact selected values."""
    param = validate_sparse_patch(model, patch, expected_shape)
    flat_param = param.data.view(-1)
    indices = patch.indices.to(device=flat_param.device, dtype=torch.long)
    values = patch.values.to(device=flat_param.device)
    flat_param.index_copy_(0, indices, values)
    if verify and not torch.equal(flat_param.index_select(0, indices), values):
        raise RuntimeError(
            f"Sparse weight update verification failed for {patch.name}"
        )


def apply_sparse_hf_patch(
    model: torch.nn.Module,
    patch: SparseWeightPatch,
    expected_shape: list[int],
) -> None:
    """Apply a global HF-coordinate patch through vLLM's TP-aware loader.

    Every inference TP worker receives the same compact patch. The patch is
    expanded one parameter at a time to a NaN-masked HF tensor; the model's
    normal ``load_weights`` implementation performs fused-name mapping and TP
    slicing, while the temporary masked-copy context preserves untouched local
    elements.
    """
    if patch.indices.dtype != torch.int32:
        raise ValueError(f"Sparse HF indices must use int32: {patch.name}")
    if patch.indices.ndim != 1 or patch.values.ndim != 1:
        raise ValueError(f"Sparse HF patches must be flattened: {patch.name}")
    if patch.indices.numel() != patch.values.numel():
        raise ValueError(f"Sparse HF indices/value length mismatch: {patch.name}")
    numel = prod(expected_shape)
    if numel >= 2**31:
        raise OverflowError(
            f"Sparse HF int32 index limit exceeded for {patch.name}: {numel}"
        )
    indices = patch.indices.to(device=patch.values.device, dtype=torch.long)
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= numel):
        raise IndexError(f"Sparse HF index out of bounds for {patch.name}")
    dense = torch.full(
        (numel,),
        float("nan"),
        dtype=patch.values.dtype,
        device=patch.values.device,
    )
    dense.index_copy_(0, indices, patch.values)
    load_weights = getattr(model, "load_weights", None)
    if load_weights is None:
        raise TypeError(f"Model {type(model).__name__} does not expose load_weights")
    with _masked_copy():
        load_weights([(patch.name, dense.view(expected_shape))])


@contextmanager
def _masked_copy() -> Iterator[None]:
    original_copy = torch.Tensor.copy_

    def masked_copy(self: torch.Tensor, src: torch.Tensor, *args, **kwargs):
        if (
            isinstance(src, torch.Tensor)
            and src.is_floating_point()
            and self.shape == src.shape
        ):
            cast = src.to(self.dtype)
            return original_copy(self, torch.where(torch.isnan(cast), self, cast))
        return original_copy(self, src, *args, **kwargs)

    torch.Tensor.copy_ = masked_copy
    try:
        yield
    finally:
        torch.Tensor.copy_ = original_copy

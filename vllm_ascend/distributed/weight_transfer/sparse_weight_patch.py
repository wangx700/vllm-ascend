# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport-independent sparse runtime weight patch operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from typing import Any

import torch


@dataclass
class SparseWeightPatch:
    """A sparse in-place patch for one existing runtime parameter."""

    name: str
    indices: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class SparseLoadPlan:
    """Cached mapping from one HF parameter to one runtime parameter."""

    runtime_name: str
    param: torch.nn.Parameter
    expected_shape: tuple[int, ...]
    shard_dim: int | None = None
    source_start: int = 0
    source_size: int = 0
    target_offset: int = 0

    def map_indices(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return runtime-local flat indices and the contributing mask."""
        if self.shard_dim is None:
            selected = torch.ones_like(indices, dtype=torch.bool)
            return indices.to(device=self.param.device), selected

        stride = prod(self.expected_shape[self.shard_dim + 1 :])
        coordinate = torch.div(indices, stride, rounding_mode="floor")
        coordinate = torch.remainder(
            coordinate, self.expected_shape[self.shard_dim]
        )
        selected = (coordinate >= self.source_start) & (
            coordinate < self.source_start + self.source_size
        )
        selected_indices = indices[selected]
        coordinate = coordinate[selected]
        outer = torch.div(
            selected_indices,
            self.expected_shape[self.shard_dim] * stride,
            rounding_mode="floor",
        )
        inner = torch.remainder(selected_indices, stride)
        local_coordinate = (
            coordinate - self.source_start + self.target_offset
        )
        local_indices = (
            (
                outer * self.param.shape[self.shard_dim]
                + local_coordinate
            )
            * stride
            + inner
        )
        return local_indices.to(device=self.param.device), selected


SparseLoadPlanCache = dict[
    tuple[str, tuple[int, ...]], SparseLoadPlan | None
]


def partition_qwen3_sparse_patches(
    patches: list[tuple[SparseWeightPatch, list[int]]],
    rollout_tp_sizes: list[int],
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    vocab_padding_size: int = 64,
) -> list[list[SparseWeightPatch]]:
    """Filter global Qwen3 patches for every rollout communicator rank.

    Returned rows follow communicator ranks 1..N and retain HF-global indices;
    the receiver's cached ``SparseLoadPlan`` performs the final local mapping.
    """
    rank_patches: list[list[SparseWeightPatch]] = []
    for tp_size in rollout_tp_sizes:
        for tp_rank in range(tp_size):
            filtered = []
            for patch, shape in patches:
                mask = _qwen3_tp_contribution_mask(
                    patch.name,
                    patch.indices.to(torch.long),
                    shape,
                    tp_rank,
                    tp_size,
                    num_attention_heads,
                    num_key_value_heads,
                    vocab_padding_size,
                )
                filtered.append(
                    SparseWeightPatch(
                        patch.name,
                        patch.indices[mask],
                        patch.values[mask],
                    )
                )
            rank_patches.append(filtered)
    return rank_patches


def _qwen3_tp_contribution_mask(
    name: str,
    indices: torch.Tensor,
    shape: list[int],
    tp_rank: int,
    tp_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    vocab_padding_size: int,
) -> torch.Tensor:
    shard_dim: int | None = None
    unique_shards = tp_size
    shard_rank = tp_rank
    padded_dim = shape[0]
    if any(token in name for token in ("q_proj.", "gate_proj.", "up_proj.")):
        shard_dim = 0
    elif any(token in name for token in ("k_proj.", "v_proj.")):
        shard_dim = 0
        unique_shards = min(tp_size, num_key_value_heads)
        replicas = tp_size // unique_shards
        shard_rank = tp_rank // replicas
    elif any(token in name for token in ("o_proj.", "down_proj.")):
        # Row-parallel weight shards input_dim=1; its optional 1D bias is
        # replicated by vLLM's loader.
        shard_dim = 1 if len(shape) > 1 else None
    elif "embed_tokens." in name or name.startswith("lm_head."):
        shard_dim = 0
        padded_dim = (
            (shape[0] + vocab_padding_size - 1) // vocab_padding_size
        ) * vocab_padding_size
    if shard_dim is None or tp_size == 1:
        return torch.ones_like(indices, dtype=torch.bool)
    if shard_dim >= len(shape):
        raise ValueError(f"Invalid Qwen3 sparse shape for {name}: {shape}")
    if "q_proj." in name and shape[0] % num_attention_heads != 0:
        raise ValueError(f"Qwen3 q_proj shape is inconsistent for {name}")
    global_dim = padded_dim if shard_dim == 0 else shape[shard_dim]
    if global_dim % unique_shards != 0:
        raise ValueError(
            f"Qwen3 TP dimension {global_dim} is not divisible by "
            f"{unique_shards} for {name}"
        )
    shard_size = global_dim // unique_shards
    source_start = shard_rank * shard_size
    source_end = min(source_start + shard_size, shape[shard_dim])
    stride = prod(shape[shard_dim + 1 :])
    coordinate = torch.remainder(
        torch.div(indices, stride, rounding_mode="floor"), shape[shard_dim]
    )
    return (coordinate >= source_start) & (coordinate < source_end)


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
    """Apply one global HF-coordinate patch."""
    apply_sparse_hf_patches(model, [(patch, expected_shape)])


def apply_sparse_hf_patches(
    model: torch.nn.Module,
    patches: list[tuple[SparseWeightPatch, list[int]]],
    *,
    plan_cache: SparseLoadPlanCache | None = None,
    force_legacy: bool = False,
) -> None:
    """Apply and coalesce HF patches by their runtime parameter.

    Known unquantized Qwen3 layouts are filtered for the local TP rank and all
    constituent HF patches targeting the same fused runtime parameter are
    concatenated into one ``index_copy_``. Unknown layouts retain the normal
    NaN-masked vLLM loader path.
    """
    cache = plan_cache if plan_cache is not None else {}
    direct: dict[str, tuple[torch.nn.Parameter, list[torch.Tensor], list[torch.Tensor]]] = {}
    legacy: list[tuple[SparseWeightPatch, list[int], torch.Tensor]] = []
    for patch, expected_shape in patches:
        indices = _validate_sparse_hf_patch(patch, expected_shape)
        key = (patch.name, tuple(expected_shape))
        if key not in cache:
            cache[key] = _build_sparse_load_plan(
                model, patch.name, expected_shape
            )
        plan = None if force_legacy else cache[key]
        if plan is None or patch.values.dtype != plan.param.dtype:
            legacy.append((patch, expected_shape, indices))
            continue
        local_indices, selected = plan.map_indices(indices)
        values = patch.values.to(device=plan.param.device)[selected]
        group = direct.setdefault(plan.runtime_name, (plan.param, [], []))
        group[1].append(local_indices)
        group[2].append(values)

    for param, index_parts, value_parts in direct.values():
        param.data.view(-1).index_copy_(
            0, torch.cat(index_parts), torch.cat(value_parts)
        )
    for patch, expected_shape, indices in legacy:
        _apply_sparse_hf_patch_legacy(
            model, patch, expected_shape, indices
        )


def _validate_sparse_hf_patch(
    patch: SparseWeightPatch, expected_shape: list[int]
) -> torch.Tensor:
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
    return indices


def _apply_sparse_hf_patch_legacy(
    model: torch.nn.Module,
    patch: SparseWeightPatch,
    expected_shape: list[int],
    indices: torch.Tensor,
) -> None:
    numel = prod(expected_shape)
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


def _build_sparse_load_plan(
    model: torch.nn.Module,
    hf_name: str,
    expected_shape: list[int],
) -> SparseLoadPlan | None:
    """Build a direct unquantized vLLM TP mapping when layout is known.

    vLLM's linear loaders expose the sharded dimension and the owning layer's
    TP metadata on the runtime parameter/module.  Reusing that metadata keeps
    this path compatible with Ascend pluggable layers without importing a
    model or device-specific linear implementation.  ``False`` means that the
    layout cannot be proven safe and the caller must use the normal loader.
    """
    mapped_name, shard_id = _map_qwen3_stacked_name(hf_name)
    try:
        param = model.get_parameter(mapped_name)
    except (AttributeError, KeyError):
        return None

    if (
        not param.data.is_contiguous()
        or not _supports_logical_flat_index_copy(param.data)
        or getattr(param, "packed_dim", None) is not None
        or getattr(param, "is_sharded_weight", False)
    ):
        return None

    module_name, _, _ = mapped_name.rpartition(".")
    try:
        module = model.get_submodule(module_name) if module_name else model
    except AttributeError:
        return None
    return _get_sparse_load_plan(
        mapped_name, module, param, expected_shape, shard_id
    )


def _supports_logical_flat_index_copy(tensor: torch.Tensor) -> bool:
    """Reject transformed Ascend layouts until physical mapping is defined."""
    if tensor.device.type != "npu":
        return True
    try:
        import torch_npu
    except ImportError:
        return False
    npu_format = str(torch_npu.get_npu_format(tensor)).upper()
    return npu_format in {"ND", "2"}


def _map_qwen3_stacked_name(name: str) -> tuple[str, str | int | None]:
    stacked = {
        "q_proj.": ("qkv_proj.", "q"),
        "k_proj.": ("qkv_proj.", "k"),
        "v_proj.": ("qkv_proj.", "v"),
        "gate_proj.": ("gate_up_proj.", 0),
        "up_proj.": ("gate_up_proj.", 1),
    }
    for source, (target, shard_id) in stacked.items():
        if source in name:
            return name.replace(source, target, 1), shard_id
    return name, None


def _get_sparse_load_plan(
    runtime_name: str,
    module: torch.nn.Module,
    param: torch.nn.Parameter,
    expected_shape: list[int],
    shard_id: str | int | None,
) -> SparseLoadPlan | None:
    if list(param.shape) == expected_shape and shard_id is None:
        return SparseLoadPlan(runtime_name, param, tuple(expected_shape))

    output_dim = getattr(param, "output_dim", None)
    input_dim = getattr(param, "input_dim", None)
    tp_rank = getattr(module, "tp_rank", None)
    tp_size = getattr(module, "tp_size", None)
    if tp_rank is None or tp_size is None:
        return None

    source_start: int
    source_size: int
    target_offset = 0
    shard_dim: int

    if isinstance(shard_id, str):
        get_offset = getattr(module, "_get_shard_offset_mapping", None)
        get_size = getattr(module, "_get_shard_size_mapping", None)
        replicas = getattr(module, "num_kv_head_replicas", None)
        if output_dim is None or get_offset is None or get_size is None or replicas is None:
            return None
        shard_dim = output_dim
        target_offset = get_offset(shard_id)
        source_size = get_size(shard_id)
        shard_rank = tp_rank if shard_id == "q" else tp_rank // replicas
        source_start = shard_rank * source_size
    elif isinstance(shard_id, int):
        output_sizes = getattr(module, "output_sizes", None)
        if output_dim is None or output_sizes is None or shard_id >= len(output_sizes):
            return None
        shard_dim = output_dim
        source_size = output_sizes[shard_id] // tp_size
        source_start = tp_rank * source_size
        target_offset = sum(output_sizes[:shard_id]) // tp_size
    elif input_dim is not None:
        shard_dim = input_dim
        source_size = param.shape[shard_dim]
        source_start = tp_rank * source_size
    elif output_dim is not None and hasattr(module, "shard_indices"):
        shard_dim = output_dim
        shard_indices: Any = module.shard_indices
        source_start = shard_indices.org_vocab_start_index
        source_size = shard_indices.org_vocab_end_index - source_start
    elif output_dim is not None:
        shard_dim = output_dim
        source_size = param.shape[shard_dim]
        source_start = tp_rank * source_size
    else:
        return None

    if shard_dim >= len(expected_shape):
        return None
    return SparseLoadPlan(
        runtime_name=runtime_name,
        param=param,
        expected_shape=tuple(expected_shape),
        shard_dim=shard_dim,
        source_start=source_start,
        source_size=source_size,
        target_offset=target_offset,
    )


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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse HCCL weight transfer engine."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferUpdateInfo,
)

from vllm_ascend.distributed.weight_transfer.hccl_common import (
    HCCLWeightTransferInitInfo,
    trainer_init,
    worker_init_process_group,
)
from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerSendWeightsArgs,
)
from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import (
    SparseWeightPatch,
    apply_sparse_hf_patch,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.distributed.device_communicators.pyhccl import (
        PyHcclCommunicator,
    )

__all__ = [
    "SparseWeightPatch",
    "SparseHCCLWeightTransferUpdateInfo",
    "SparseHCCLWeightTransferEngine",
]


@dataclass
class SparseHCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the sparse HCCL weight transfer backend."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    num_updates_list: list[int]
    """Number of sparse entries to receive for each parameter in ``names``."""

    def __post_init__(self) -> None:
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {len(self.names)}"
            )
        if len(self.num_updates_list) == 0:
            raise ValueError("`num_updates_list` cannot be empty for sparse updates")
        if len(self.num_updates_list) != num_params:
            raise ValueError(
                f"`num_updates_list` should be of the same size as `names`: "
                f"got {len(self.num_updates_list)} and {len(self.names)}"
            )
        if any(num_updates < 0 for num_updates in self.num_updates_list):
            raise ValueError("Sparse `num_updates_list` entries must be non-negative")


class SparseHCCLWeightTransferEngine(
    WeightTransferEngine[
        HCCLWeightTransferInitInfo,
        SparseHCCLWeightTransferUpdateInfo,
    ]
):
    """Apply sparse, flat-index weight patches received through HCCL."""

    init_info_cls = HCCLWeightTransferInitInfo
    update_info_cls = SparseHCCLWeightTransferUpdateInfo
    supports_draft_weight_update = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self.model_update_group: PyHcclCommunicator | None = None

    def init_transfer_engine(self, init_info: HCCLWeightTransferInitInfo) -> None:
        """Initialize the HCCL process group with the trainer."""
        self.model_update_group = worker_init_process_group(
            init_info,
            self.parallel_config,
        )

    def start_weight_update(self) -> None:
        """Sparse HF patches are applied through the model's TP-aware loader."""
        if self.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError(
                "Sparse HCCL weight updates currently require PP=1"
            )

    def finish_weight_update(self) -> None:
        """Sparse runtime-format patches need no layerwise finalization."""
        pass

    def receive_weights(
        self,
        update_info: SparseHCCLWeightTransferUpdateInfo,
    ) -> None:
        """Receive sparse flat-index patches and apply them in place."""
        if self.model_update_group is None:
            raise RuntimeError(
                "HCCL weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

        total_updates = sum(update_info.num_updates_list)
        packed_indices = torch.empty(
            total_updates,
            dtype=torch.int32,
            device=self.device,
        )
        if packed_indices.numel():
            self.model_update_group.broadcast(
                packed_indices,
                src=0,
                stream=torch.npu.current_stream(),
            )

        # Values cannot be concatenated across different dtypes.  Keep one
        # packed buffer per dtype instead, preserving first-seen dtype order so
        # trainer and workers issue collectives in exactly the same sequence.
        dtype_order = list(dict.fromkeys(update_info.dtype_names))
        packed_values: dict[str, torch.Tensor] = {}
        for dtype_name in dtype_order:
            dtype_updates = sum(
                num_updates
                for patch_dtype, num_updates in zip(
                    update_info.dtype_names,
                    update_info.num_updates_list,
                    strict=True,
                )
                if patch_dtype == dtype_name
            )
            values = torch.empty(
                dtype_updates,
                dtype=getattr(torch, dtype_name),
                device=self.device,
            )
            if values.numel():
                self.model_update_group.broadcast(
                    values,
                    src=0,
                    stream=torch.npu.current_stream(),
                )
            packed_values[dtype_name] = values

        index_offset = 0
        value_offsets = dict.fromkeys(dtype_order, 0)
        for name, dtype_name, expected_shape, num_updates in zip(
            update_info.names,
            update_info.dtype_names,
            update_info.shapes,
            update_info.num_updates_list,
            strict=True,
        ):
            indices = packed_indices[index_offset : index_offset + num_updates]
            value_offset = value_offsets[dtype_name]
            values = packed_values[dtype_name][
                value_offset : value_offset + num_updates
            ]
            index_offset += num_updates
            value_offsets[dtype_name] = value_offset + num_updates
            patch = SparseWeightPatch(
                name=name,
                indices=indices,
                values=values,
            )
            apply_sparse_hf_patch(
                self.model,
                patch,
                expected_shape,
            )

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[SparseWeightPatch],
        trainer_args: dict[str, Any] | HCCLTrainerSendWeightsArgs,
    ) -> None:
        """Broadcast sparse flat-index patches from trainer to workers."""
        if isinstance(trainer_args, dict):
            args = HCCLTrainerSendWeightsArgs(**trainer_args)
        else:
            args = trainer_args
        if args.packed:
            raise ValueError(
                "Sparse HCCL updates cannot be combined with `packed=True`"
            )

        patches = list(iterator)
        if not patches:
            return

        stream = args.stream or torch.npu.current_stream()
        packed_indices = torch.cat([patch.indices for patch in patches])
        if packed_indices.numel():
            args.group.broadcast(packed_indices, src=args.src, stream=stream)

        # One values collective per dtype amortizes HCCL launch latency while
        # retaining native parameter dtypes and avoiding byte-buffer alignment
        # constraints.
        dtype_order = list(dict.fromkeys(patch.values.dtype for patch in patches))
        for dtype in dtype_order:
            packed_values = torch.cat(
                [patch.values for patch in patches if patch.values.dtype == dtype]
            )
            if packed_values.numel():
                args.group.broadcast(packed_values, src=args.src, stream=stream)

    trainer_init = staticmethod(trainer_init)

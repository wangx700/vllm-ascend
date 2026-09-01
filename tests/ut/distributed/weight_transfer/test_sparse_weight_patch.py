# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import (
    SparseWeightPatch,
    apply_sparse_hf_patch,
    apply_sparse_hf_patches,
    apply_sparse_patch,
    partition_qwen3_sparse_patches,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([10.0, 20.0, 30.0, 40.0])
        )


def test_apply_sparse_patch_updates_only_selected_values():
    model = DummyModel()
    apply_sparse_patch(
        model,
        SparseWeightPatch(
            "weight",
            torch.tensor([1, 3], dtype=torch.int32),
            torch.tensor([200.0, 400.0]),
        ),
        expected_shape=[4],
    )
    assert torch.equal(
        model.weight,
        torch.tensor([10.0, 200.0, 30.0, 400.0]),
    )


@pytest.mark.parametrize("indices", [[-1], [4]])
def test_apply_sparse_patch_rejects_out_of_bounds_indices(indices):
    with pytest.raises((IndexError, RuntimeError)):
        apply_sparse_patch(
            DummyModel(),
            SparseWeightPatch(
                "weight",
                torch.tensor(indices, dtype=torch.int32),
                torch.tensor([1.0]),
            ),
        )


def test_apply_sparse_patch_rejects_declared_shape_mismatch():
    with pytest.raises(ValueError, match="declared shape"):
        apply_sparse_patch(
            DummyModel(),
            SparseWeightPatch(
                "weight",
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1.0]),
            ),
            expected_shape=[2, 2],
        )


def test_apply_sparse_patch_empty_update_is_noop():
    model = DummyModel()
    before = model.weight.detach().clone()
    apply_sparse_patch(
        model,
        SparseWeightPatch(
            "weight",
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.float32),
        ),
    )
    assert torch.equal(model.weight, before)


@pytest.mark.parametrize(
    ("indices", "values", "message"),
    [
        (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([1.0]),
            "require int32 indices",
        ),
        (
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([1.0]),
            "must be 1D flattened updates",
        ),
        (
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([[1.0]]),
            "must be 1D flattened updates",
        ),
        (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([1.0]),
            "matching lengths",
        ),
        (
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            "does not match parameter dtype",
        ),
    ],
)
def test_apply_sparse_patch_rejects_invalid_patch(indices, values, message):
    with pytest.raises(ValueError, match=message):
        apply_sparse_patch(
            DummyModel(),
            SparseWeightPatch("weight", indices, values),
        )


def test_apply_sparse_patch_rejects_noncontiguous_parameter():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(
        torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    )
    assert not model.weight.is_contiguous()

    with pytest.raises(NotImplementedError, match="require contiguous params"):
        apply_sparse_patch(
            model,
            SparseWeightPatch(
                "weight",
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1.0]),
            ),
        )


class DummyTPModel(torch.nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
        self.weight = torch.nn.Parameter(torch.tensor([10.0, 20.0]))

    def load_weights(self, weights):
        for name, full_weight in weights:
            self.weight.data.copy_(full_weight[self.rank * 2 : (self.rank + 1) * 2])
            return {name}


class DummyStackedTPModel(DummyTPModel):
    def load_weights(self, weights):
        for _, full_weight in weights:
            self.weight.data.copy_(full_weight[self.rank * 2 : (self.rank + 1) * 2])
            return {"gate_up_proj.weight"}


class DummyTPLayer(torch.nn.Module):
    def __init__(self, shape, rank=0, size=2):
        super().__init__()
        self.tp_rank = rank
        self.tp_size = size
        self.weight = torch.nn.Parameter(torch.zeros(shape))


class DummyDirectModel(torch.nn.Module):
    def __init__(self, layer_name, layer):
        super().__init__()
        setattr(self, layer_name, layer)

    def load_weights(self, weights):
        raise AssertionError("direct sparse patch unexpectedly used load_weights")


class DummyQKVLayer(DummyTPLayer):
    def __init__(self, shape, rank=0, size=2):
        super().__init__(shape, rank, size)
        self.num_kv_head_replicas = 2

    def _get_shard_offset_mapping(self, shard):
        return {"q": 0, "k": 2, "v": 3}[shard]

    def _get_shard_size_mapping(self, shard):
        return {"q": 2, "k": 1, "v": 1}[shard]


class DummyMergedModel(torch.nn.Module):
    def __init__(self, rank=1):
        super().__init__()
        self.gate_up_proj = DummyTPLayer((4, 2), rank=rank)
        self.gate_up_proj.output_sizes = [4, 4]
        self.gate_up_proj.weight.output_dim = 0
        self.gate_up_proj.weight.data.copy_(
            torch.arange(8, dtype=torch.float32).reshape(4, 2)
        )

    def load_weights(self, weights):
        for name, full_weight in weights:
            shard_id = 0 if "gate_proj" in name else 1
            shard_size = 2
            source = full_weight.narrow(
                0, self.gate_up_proj.tp_rank * shard_size, shard_size
            )
            target = self.gate_up_proj.weight.data.narrow(
                0, shard_id * shard_size, shard_size
            )
            target.copy_(source)
        return {name}


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(0, [10.0, 200.0]), (1, [300.0, 20.0])],
)
def test_apply_sparse_hf_patch_uses_tp_loader_and_preserves_untouched(rank, expected):
    model = DummyTPModel(rank)
    apply_sparse_hf_patch(
        model,
        SparseWeightPatch(
            "weight",
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([200.0, 300.0]),
        ),
        [4],
    )
    assert torch.equal(model.weight, torch.tensor(expected))


def test_apply_sparse_hf_patch_allows_stacked_parameter_name_mapping():
    model = DummyStackedTPModel(rank=0)
    apply_sparse_hf_patch(
        model,
        SparseWeightPatch(
            "gate_proj.weight",
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([200.0]),
        ),
        [4],
    )
    assert torch.equal(model.weight, torch.tensor([10.0, 200.0]))


def test_apply_sparse_hf_patch_direct_row_parallel_index_copy():
    layer = DummyTPLayer((2, 2), rank=1)
    layer.weight.input_dim = 1
    model = DummyDirectModel("down_proj", layer)

    apply_sparse_hf_patch(
        model,
        SparseWeightPatch(
            "down_proj.weight",
            torch.tensor([1, 2, 3, 6], dtype=torch.int32),
            torch.tensor([11.0, 12.0, 13.0, 16.0]),
        ),
        [2, 4],
    )

    assert torch.equal(layer.weight, torch.tensor([[12.0, 13.0], [16.0, 0.0]]))


def test_apply_sparse_hf_patch_direct_merged_column_index_copy():
    layer = DummyTPLayer((4, 2), rank=1)
    layer.output_sizes = [4, 4]
    layer.weight.output_dim = 0
    model = DummyDirectModel("gate_up_proj", layer)

    apply_sparse_hf_patch(
        model,
        SparseWeightPatch(
            "gate_proj.weight",
            torch.tensor([1, 4, 6], dtype=torch.int32),
            torch.tensor([11.0, 14.0, 16.0]),
        ),
        [4, 2],
    )
    apply_sparse_hf_patch(
        model,
        SparseWeightPatch(
            "up_proj.weight",
            torch.tensor([4, 7], dtype=torch.int32),
            torch.tensor([24.0, 27.0]),
        ),
        [4, 2],
    )

    assert torch.equal(
        layer.weight,
        torch.tensor([[14.0, 0.0], [16.0, 0.0], [24.0, 0.0], [0.0, 27.0]]),
    )


def test_apply_sparse_hf_patch_direct_qkv_honors_kv_replication():
    layer = DummyQKVLayer((4, 2), rank=3, size=4)
    layer.weight.output_dim = 0
    model = DummyDirectModel("qkv_proj", layer)

    apply_sparse_hf_patch(
        model,
        SparseWeightPatch(
            "k_proj.weight",
            torch.tensor([0, 2, 3], dtype=torch.int32),
            torch.tensor([10.0, 12.0, 13.0]),
        ),
        [2, 2],
    )

    assert torch.equal(
        layer.weight,
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [12.0, 13.0], [0.0, 0.0]]),
    )


def test_sparse_load_plan_coalesces_fused_patches_into_one_index_copy(monkeypatch):
    model = DummyMergedModel()
    patches = [
        (
            SparseWeightPatch(
                "gate_proj.weight",
                torch.tensor([4, 7], dtype=torch.int32),
                torch.tensor([14.0, 17.0]),
            ),
            [4, 2],
        ),
        (
            SparseWeightPatch(
                "up_proj.weight",
                torch.tensor([5, 6], dtype=torch.int32),
                torch.tensor([25.0, 26.0]),
            ),
            [4, 2],
        ),
    ]
    original_index_copy = torch.Tensor.index_copy_
    target_data_ptr = model.gate_up_proj.weight.data_ptr()
    calls = 0

    def counted_index_copy(self, dim, index, source):
        nonlocal calls
        if self.data_ptr() == target_data_ptr:
            calls += 1
        return original_index_copy(self, dim, index, source)

    monkeypatch.setattr(torch.Tensor, "index_copy_", counted_index_copy)
    apply_sparse_hf_patches(model, patches)

    assert calls == 1
    assert torch.equal(
        model.gate_up_proj.weight,
        torch.tensor([[14.0, 1.0], [2.0, 17.0], [4.0, 25.0], [26.0, 7.0]]),
    )


def test_sparse_load_plan_direct_and_legacy_are_bit_exact():
    direct_model = DummyMergedModel()
    legacy_model = DummyMergedModel()
    patches = [
        (
            SparseWeightPatch(
                "gate_proj.weight",
                torch.tensor([1, 4, 7], dtype=torch.int32),
                torch.tensor([11.25, 14.5, 17.75]),
            ),
            [4, 2],
        ),
        (
            SparseWeightPatch(
                "up_proj.weight",
                torch.tensor([0, 5, 6], dtype=torch.int32),
                torch.tensor([20.25, 25.5, 26.75]),
            ),
            [4, 2],
        ),
    ]

    apply_sparse_hf_patches(direct_model, patches)
    apply_sparse_hf_patches(legacy_model, patches, force_legacy=True)

    assert torch.equal(
        direct_model.gate_up_proj.weight.view(torch.int32),
        legacy_model.gate_up_proj.weight.view(torch.int32),
    )


def test_sparse_load_plan_cache_reuses_parameter_mapping():
    model = DummyMergedModel()
    cache = {}
    patch = SparseWeightPatch(
        "gate_proj.weight",
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([14.0]),
    )

    apply_sparse_hf_patches(model, [(patch, [4, 2])], plan_cache=cache)
    cached_plan = cache[("gate_proj.weight", (4, 2))]
    apply_sparse_hf_patches(model, [(patch, [4, 2])], plan_cache=cache)

    assert len(cache) == 1
    assert cache[("gate_proj.weight", (4, 2))] is cached_plan


def test_partition_qwen3_sparse_patches_filters_qkv_and_row_by_rollout_rank():
    q_patch = SparseWeightPatch(
        "model.layers.0.self_attn.q_proj.weight",
        torch.arange(0, 16, 2, dtype=torch.int32),
        torch.arange(8, dtype=torch.float32),
    )
    k_patch = SparseWeightPatch(
        "model.layers.0.self_attn.k_proj.weight",
        torch.arange(0, 8, 2, dtype=torch.int32),
        torch.arange(4, dtype=torch.float32),
    )
    row_patch = SparseWeightPatch(
        "model.layers.0.self_attn.o_proj.weight",
        torch.arange(16, dtype=torch.int32),
        torch.arange(16, dtype=torch.float32),
    )

    per_rank = partition_qwen3_sparse_patches(
        [(q_patch, [8, 2]), (k_patch, [4, 2]), (row_patch, [2, 8])],
        [4],
        num_attention_heads=4,
        num_key_value_heads=2,
    )

    assert per_rank[2][0].indices.tolist() == [8, 10]
    assert per_rank[0][1].indices.tolist() == [0, 2]
    assert per_rank[1][1].indices.tolist() == [0, 2]
    assert per_rank[2][1].indices.tolist() == [4, 6]
    assert per_rank[3][1].indices.tolist() == [4, 6]
    assert per_rank[1][2].indices.tolist() == [2, 3, 10, 11]

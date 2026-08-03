#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ascend_config import SparseKVOffloadConfig
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata
from vllm_ascend.attention.sparse_kv_offload import (
    SparseKVOffloadWorkspace,
    SparseKVSelection,
    get_gather_selection_kv_cache_op,
    make_sparse_kv_offload_memory_plan,
)

BLOCK_SIZE = 128
INDEX_TOPK = 2048


def test_host_mode_memory_plan_accounts_for_persistent_npu_state():
    plan = make_sparse_kv_offload_memory_plan(
        num_blocks=5,
        block_size=BLOCK_SIZE,
        index_topk=INDEX_TOPK,
        full_kv_head_dim=576,
        index_head_dim=128,
        kv_dtype_size=2,
        num_sparse_layers=2,
        num_indexer_layers=1,
        indexer_alignment_bytes_per_layer=2 * 1024 * 1024,
        shared_prefill_alignment_bytes=2 * 1024 * 1024,
    )

    assert plan.indexer_cache_bytes == 5 * BLOCK_SIZE * 128 * 2
    assert plan.indexer_alignment_bytes == 2 * 1024 * 1024
    assert plan.shared_prefill_bytes == 5 * BLOCK_SIZE * 576 * 2
    assert plan.shared_prefill_alignment_bytes == 2 * 1024 * 1024
    assert plan.selected_cache_bytes == 2 * INDEX_TOPK * 576 * 2
    assert plan.selection_metadata_bytes == 2 * (
        2 * (INDEX_TOPK // BLOCK_SIZE) * 4 + (INDEX_TOPK + 1) * 4 + INDEX_TOPK * 4
    )
    assert plan.total_bytes == (
        plan.indexer_cache_bytes
        + plan.indexer_alignment_bytes
        + plan.shared_prefill_bytes
        + plan.shared_prefill_alignment_bytes
        + plan.selected_cache_bytes
        + plan.selection_metadata_bytes
    )


def _cpu_swapped_allocator(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def _unexpected_swapped_allocator(*args, **kwargs):
    raise AssertionError("host mode must not allocate a second Full KV mirror")


def _fake_gather_selection_kv_cache(**kwargs) -> torch.Tensor:
    topk_indices = kwargs["selection_topk_indices"][0, 0]
    full_actual_seq = int(kwargs["full_kv_actual_seq"][0])
    full_block_table = kwargs["full_kv_block_table"][0]
    selected_nope = kwargs["selection_kv_cache"].view(-1, kwargs["selection_kv_cache"].shape[-1])
    selected_rope = kwargs["selection_k_rope"].view(-1, kwargs["selection_k_rope"].shape[-1])
    full_nope = kwargs["full_kv_cache"]
    full_rope = kwargs["full_k_rope"]
    status = kwargs["selection_kv_block_status"]

    assert torch.count_nonzero(status != -1) == 0
    valid_count = 0
    for logical_token in topk_indices.tolist():
        if logical_token < 0:
            break
        if logical_token >= full_actual_seq:
            continue
        logical_block = logical_token // BLOCK_SIZE
        offset = logical_token % BLOCK_SIZE
        physical_block = int(full_block_table[logical_block])
        selected_nope[valid_count].copy_(full_nope[physical_block, offset])
        selected_rope[valid_count].copy_(full_rope[physical_block, offset])
        status[0, 0, valid_count] = logical_token
        valid_count += 1
    status[0, 0, INDEX_TOPK] = valid_count
    return torch.tensor([valid_count], dtype=torch.int32)


def test_gather_op_loader_reports_missing_external_package():
    with (
        patch(
            "vllm_ascend.attention.sparse_kv_offload.importlib.import_module",
            side_effect=ImportError,
        ),
        pytest.raises(RuntimeError, match="requires the cann-recipes-infer custom_ops"),
    ):
        get_gather_selection_kv_cache_op()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_workspace_mirrors_touched_blocks_and_gathers_selected_kv(dtype):
    full_nope = torch.zeros(4, BLOCK_SIZE, 1, 2, dtype=dtype)
    full_rope = torch.zeros(4, BLOCK_SIZE, 1, 1, dtype=dtype)
    for block_id in range(4):
        for offset in range(BLOCK_SIZE):
            value = block_id * BLOCK_SIZE + offset
            full_nope[block_id, offset, 0] = torch.tensor(
                [value, value + 1],
                dtype=dtype,
            )
            full_rope[block_id, offset, 0, 0] = value

    workspace = SparseKVOffloadWorkspace(
        (full_nope, full_rope),
        index_topk=INDEX_TOPK,
        block_size=BLOCK_SIZE,
        swapped_allocator=_cpu_swapped_allocator,
        gather_op=_fake_gather_selection_kv_cache,
        validate_device=False,
    )

    updated_blocks = workspace.sync_updated_blocks(
        (full_nope, full_rope),
        torch.tensor(
            [
                3 * BLOCK_SIZE + 5,
                1 * BLOCK_SIZE + 2,
            ],
            dtype=torch.int64,
        ),
        num_actual_tokens=2,
    )
    assert updated_blocks == (1, 3)

    topk_indices = torch.full((1, 1, INDEX_TOPK), -1, dtype=torch.int32)
    topk_indices[0, 0, :2] = torch.tensor([130, 5], dtype=torch.int32)
    workspace.reset_selection_state = MagicMock()
    selection = workspace.gather(
        topk_indices=topk_indices,
        full_block_table=torch.tensor([[3, 1]], dtype=torch.int32),
        full_actual_seq_lengths=torch.tensor([256], dtype=torch.int32),
        full_query_actual_seq_lengths=torch.tensor([1], dtype=torch.int32),
    )
    workspace.reset_selection_state.assert_not_called()

    assert selection.kv_cache[0].shape == (16, BLOCK_SIZE, 1, 2)
    assert selection.kv_cache[1].shape == (16, BLOCK_SIZE, 1, 1)
    torch.testing.assert_close(
        selection.kv_cache[0].view(-1, 2)[:2],
        torch.tensor([[130, 131], [389, 390]], dtype=dtype),
    )
    torch.testing.assert_close(
        selection.kv_cache[1].view(-1, 1)[:2],
        torch.tensor([[130], [389]], dtype=dtype),
    )
    assert selection.actual_seq_lengths_kv.tolist() == [2]
    assert selection.sparse_indices[0, 0, :3].tolist() == [0, 1, -1]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_host_workspace_uses_shared_prefill_cache_and_persists_only_touched_slots(
    dtype,
):
    full_nope = torch.full((2, BLOCK_SIZE, 1, 2), -1, dtype=dtype)
    full_rope = torch.full((2, BLOCK_SIZE, 1, 1), -1, dtype=dtype)
    indexer_cache = torch.zeros(2, BLOCK_SIZE, 1, 4, dtype=dtype)
    prefill_nope = torch.full_like(full_nope, 99)
    prefill_rope = torch.full_like(full_rope, 77)
    full_kv_cache = (full_nope, full_rope, indexer_cache)
    prefill_kv_cache = (prefill_nope, prefill_rope)

    workspace = SparseKVOffloadWorkspace(
        full_kv_cache,
        index_topk=INDEX_TOPK,
        block_size=BLOCK_SIZE,
        mode="host",
        prefill_kv_cache=prefill_kv_cache,
        swapped_allocator=_unexpected_swapped_allocator,
        gather_op=_fake_gather_selection_kv_cache,
        validate_device=False,
    )

    assert workspace.full_nope_source is full_nope
    assert workspace.full_rope_source is full_rope
    forward_cache = workspace.get_forward_kv_cache(
        full_kv_cache,
        is_decode=False,
    )
    assert forward_cache[0] is prefill_nope
    assert forward_cache[1] is prefill_rope
    assert forward_cache[2] is indexer_cache
    assert (
        workspace.get_forward_kv_cache(
            full_kv_cache,
            is_decode=True,
        )
        is full_kv_cache
    )

    updated_blocks = workspace.persist_prefill_blocks(
        full_kv_cache,
        torch.tensor(
            [
                3,
                BLOCK_SIZE + 5,
                -1,
            ],
            dtype=torch.int64,
        ),
        num_actual_tokens=3,
    )

    assert updated_blocks == (0, 1)
    torch.testing.assert_close(full_nope[0, 3], prefill_nope[0, 3])
    torch.testing.assert_close(full_rope[1, 5], prefill_rope[1, 5])
    # A whole-block copy would incorrectly replace these untouched rows.
    torch.testing.assert_close(
        full_nope[0, 4],
        torch.full_like(full_nope[0, 4], -1),
    )
    torch.testing.assert_close(
        full_rope[1, 6],
        torch.full_like(full_rope[1, 6], -1),
    )


def test_host_workspace_requires_matching_shared_prefill_cache():
    full_kv_cache = (
        torch.empty(2, BLOCK_SIZE, 1, 2, dtype=torch.bfloat16),
        torch.empty(2, BLOCK_SIZE, 1, 1, dtype=torch.bfloat16),
        torch.empty(2, BLOCK_SIZE, 1, 4, dtype=torch.bfloat16),
    )
    with pytest.raises(ValueError, match="requires the shared prefill"):
        SparseKVOffloadWorkspace(
            full_kv_cache,
            index_topk=INDEX_TOPK,
            block_size=BLOCK_SIZE,
            mode="host",
            gather_op=_fake_gather_selection_kv_cache,
            validate_device=False,
        )

    with pytest.raises(ValueError, match="does not match Full KV shape"):
        SparseKVOffloadWorkspace(
            full_kv_cache,
            index_topk=INDEX_TOPK,
            block_size=BLOCK_SIZE,
            mode="host",
            prefill_kv_cache=(
                torch.empty(1, BLOCK_SIZE, 1, 2, dtype=torch.bfloat16),
                torch.empty(2, BLOCK_SIZE, 1, 1, dtype=torch.bfloat16),
            ),
            gather_op=_fake_gather_selection_kv_cache,
            validate_device=False,
        )


def test_host_workspace_restores_chunked_prefill_context_from_full_kv():
    full_nope = torch.full((4, BLOCK_SIZE, 1, 2), -1, dtype=torch.bfloat16)
    full_rope = torch.full((4, BLOCK_SIZE, 1, 1), -1, dtype=torch.bfloat16)
    full_nope[2].fill_(20)
    full_rope[2].fill_(21)
    full_nope[1].fill_(10)
    full_rope[1].fill_(11)
    prefill_nope = torch.full_like(full_nope, 99)
    prefill_rope = torch.full_like(full_rope, 77)
    full_kv_cache = (
        full_nope,
        full_rope,
        torch.empty(4, BLOCK_SIZE, 1, 4, dtype=torch.bfloat16),
    )
    workspace = SparseKVOffloadWorkspace(
        full_kv_cache,
        index_topk=INDEX_TOPK,
        block_size=BLOCK_SIZE,
        mode="host",
        prefill_kv_cache=(prefill_nope, prefill_rope),
        gather_op=_fake_gather_selection_kv_cache,
        validate_device=False,
    )

    restored_blocks = workspace.restore_prefill_context(
        full_kv_cache,
        torch.tensor([[2, 1, 3]], dtype=torch.int32),
        context_len=BLOCK_SIZE + 1,
    )

    assert restored_blocks == (1, 2)
    torch.testing.assert_close(prefill_nope[2], full_nope[2])
    torch.testing.assert_close(prefill_rope[2], full_rope[2])
    torch.testing.assert_close(prefill_nope[1], full_nope[1])
    torch.testing.assert_close(prefill_rope[1], full_rope[1])
    # Blocks outside the logical prefix must not be copied.
    torch.testing.assert_close(
        prefill_nope[3],
        torch.full_like(prefill_nope[3], 99),
    )
    torch.testing.assert_close(
        prefill_rope[3],
        torch.full_like(prefill_rope[3], 77),
    )


def _make_sfa_metadata(attn_state: AscendAttentionState) -> AscendSFAMetadata:
    return AscendSFAMetadata(
        num_actual_tokens=1,
        slot_mapping=torch.tensor([128], dtype=torch.int64),
        slot_mapping_cpu=torch.tensor([128], dtype=torch.int64),
        block_table_cpu=torch.tensor([[3, 1]], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([128], dtype=torch.int32),
        seq_lens=torch.tensor([129], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([129], dtype=torch.int32),
        cum_query_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.tensor([[3, 1]], dtype=torch.int32),
        sin=torch.zeros(1, 1),
        cos=torch.zeros(1, 1),
        attn_state=attn_state,
    )


def test_sfa_host_chunked_prefill_restores_prior_context():
    full_kv_cache = (
        torch.empty(4, BLOCK_SIZE, 1, 2),
        torch.empty(4, BLOCK_SIZE, 1, 1),
        torch.empty(4, BLOCK_SIZE, 1, 4),
    )
    metadata = _make_sfa_metadata(AscendAttentionState.ChunkedPrefill)
    workspace = MagicMock()
    fake_impl = MagicMock()
    fake_impl.sparse_kv_offload_config = SparseKVOffloadConfig(
        enabled=True,
        mode="host",
    )
    fake_impl._get_sparse_kv_offload_workspace.return_value = workspace

    AscendSFAImpl._restore_sparse_kv_offload_prefill_context(
        fake_impl,
        full_kv_cache,
        metadata,
    )

    workspace.restore_prefill_context.assert_called_once_with(
        full_kv_cache,
        metadata.block_table_cpu,
        128,
    )


def test_sfa_host_initial_prefill_does_not_restore_context():
    full_kv_cache = (
        torch.empty(4, BLOCK_SIZE, 1, 2),
        torch.empty(4, BLOCK_SIZE, 1, 1),
        torch.empty(4, BLOCK_SIZE, 1, 4),
    )
    metadata = _make_sfa_metadata(AscendAttentionState.ChunkedPrefill)
    metadata.num_computed_tokens_cpu.zero_()
    fake_impl = MagicMock()
    fake_impl.sparse_kv_offload_config = SparseKVOffloadConfig(
        enabled=True,
        mode="host",
    )

    AscendSFAImpl._restore_sparse_kv_offload_prefill_context(
        fake_impl,
        full_kv_cache,
        metadata,
    )

    fake_impl._get_sparse_kv_offload_workspace.assert_not_called()


def test_sfa_decode_switches_to_selected_kv_and_metadata():
    full_kv_cache = (
        torch.empty(4, BLOCK_SIZE, 1, 2),
        torch.empty(4, BLOCK_SIZE, 1, 1),
        torch.empty(4, BLOCK_SIZE, 1, 4),
    )
    topk_indices = torch.full((1, 1, INDEX_TOPK), -1, dtype=torch.int32)
    actual_query = torch.tensor([1], dtype=torch.int32)
    actual_key = torch.tensor([129], dtype=torch.int32)
    metadata = _make_sfa_metadata(AscendAttentionState.DecodeOnly)

    selection = SparseKVSelection(
        kv_cache=(
            torch.empty(16, BLOCK_SIZE, 1, 2),
            torch.empty(16, BLOCK_SIZE, 1, 1),
        ),
        block_table=torch.arange(16, dtype=torch.int32).view(1, 16),
        sparse_indices=torch.full((1, 1, INDEX_TOPK), -1, dtype=torch.int32),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([128], dtype=torch.int32),
    )
    workspace = MagicMock()
    workspace.gather.return_value = selection
    fake_impl = MagicMock()
    fake_impl.sparse_kv_offload_config = SparseKVOffloadConfig(enabled=True)
    fake_impl._get_sparse_kv_offload_workspace.return_value = workspace

    result = AscendSFAImpl._prepare_sparse_kv_offload_attention(
        fake_impl,
        full_kv_cache,
        full_kv_cache,
        topk_indices,
        metadata,
        actual_query,
        actual_key,
    )

    workspace.sync_updated_blocks.assert_called_once_with(
        full_kv_cache,
        metadata.slot_mapping_cpu,
        metadata.num_actual_tokens,
    )
    workspace.reset_selection_state.assert_not_called()
    workspace.gather.assert_called_once_with(
        topk_indices=topk_indices,
        full_block_table=metadata.block_table,
        full_actual_seq_lengths=actual_key,
        full_query_actual_seq_lengths=actual_query,
    )
    assert result[0] is selection.kv_cache
    assert result[1] is selection.sparse_indices
    assert result[2].block_table is selection.block_table
    assert metadata.block_table is not selection.block_table
    assert result[3] is selection.actual_seq_lengths_query
    assert result[4] is selection.actual_seq_lengths_kv


def test_sfa_request_boundary_resets_existing_selection_workspace():
    workspace = MagicMock()
    fake_impl = MagicMock()
    fake_impl.sparse_kv_offload_workspace = workspace

    AscendSFAImpl.reset_sparse_kv_offload_selection_state(fake_impl)

    workspace.reset_selection_state.assert_called_once_with()


def test_sfa_request_boundary_before_workspace_initialization_is_noop():
    fake_impl = MagicMock()
    fake_impl.sparse_kv_offload_workspace = None

    AscendSFAImpl.reset_sparse_kv_offload_selection_state(fake_impl)


def test_sfa_prefill_only_updates_host_mirror():
    full_kv_cache = (
        torch.empty(4, BLOCK_SIZE, 1, 2),
        torch.empty(4, BLOCK_SIZE, 1, 1),
        torch.empty(4, BLOCK_SIZE, 1, 4),
    )
    topk_indices = torch.full((1, 1, INDEX_TOPK), -1, dtype=torch.int32)
    actual_query = torch.tensor([1], dtype=torch.int32)
    actual_key = torch.tensor([1], dtype=torch.int32)
    metadata = _make_sfa_metadata(AscendAttentionState.ChunkedPrefill)

    workspace = MagicMock()
    fake_impl = MagicMock()
    fake_impl.sparse_kv_offload_config = SparseKVOffloadConfig(enabled=True)
    fake_impl._get_sparse_kv_offload_workspace.return_value = workspace

    result = AscendSFAImpl._prepare_sparse_kv_offload_attention(
        fake_impl,
        full_kv_cache,
        full_kv_cache,
        topk_indices,
        metadata,
        actual_query,
        actual_key,
    )

    workspace.sync_updated_blocks.assert_called_once()
    workspace.reset_selection_state.assert_called_once_with()
    workspace.gather.assert_not_called()
    assert result[0] is full_kv_cache
    assert result[1] is topk_indices
    assert result[2] is metadata
    assert result[3] is actual_query
    assert result[4] is actual_key


def test_sfa_host_prefill_persists_workspace_without_gathering():
    full_kv_cache = (
        torch.empty(4, BLOCK_SIZE, 1, 2),
        torch.empty(4, BLOCK_SIZE, 1, 1),
        torch.empty(4, BLOCK_SIZE, 1, 4),
    )
    forward_kv_cache = (
        torch.empty_like(full_kv_cache[0]),
        torch.empty_like(full_kv_cache[1]),
        full_kv_cache[2],
    )
    topk_indices = torch.full((1, 1, INDEX_TOPK), -1, dtype=torch.int32)
    actual_query = torch.tensor([1], dtype=torch.int32)
    actual_key = torch.tensor([1], dtype=torch.int32)
    metadata = _make_sfa_metadata(AscendAttentionState.ChunkedPrefill)

    workspace = MagicMock()
    fake_impl = MagicMock()
    fake_impl.sparse_kv_offload_config = SparseKVOffloadConfig(
        enabled=True,
        mode="host",
    )
    fake_impl._get_sparse_kv_offload_workspace.return_value = workspace

    result = AscendSFAImpl._prepare_sparse_kv_offload_attention(
        fake_impl,
        full_kv_cache,
        forward_kv_cache,
        topk_indices,
        metadata,
        actual_query,
        actual_key,
    )

    workspace.sync_updated_blocks.assert_not_called()
    workspace.persist_prefill_blocks.assert_called_once_with(
        full_kv_cache,
        metadata.slot_mapping_cpu,
        metadata.num_actual_tokens,
    )
    workspace.reset_selection_state.assert_called_once_with()
    workspace.gather.assert_not_called()
    assert result[0] is forward_kv_cache

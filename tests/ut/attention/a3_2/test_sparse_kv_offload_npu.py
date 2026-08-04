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

import math
from unittest.mock import MagicMock

import pytest
import torch
import torch_npu

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.attention.sparse_kv_offload import SparseKVOffloadWorkspace
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

BLOCK_SIZE = 128
INDEX_TOPK = 2048


def _allocate_framework_swapped_cache(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Match the model runner's raw-int8 swapped allocation and dtype view."""
    dtype_size = torch.empty((), dtype=dtype).element_size()
    raw_storage = torch_npu.empty_with_swapped_memory(
        (math.prod(shape) * dtype_size,),
        dtype=torch.int8,
        device=device,
    )
    return raw_storage.view(dtype).view(shape)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gather_reads_full_kv_from_swapped_memory(dtype):
    device = torch.device("npu")
    full_nope = torch.zeros(4, BLOCK_SIZE, 1, 2, dtype=dtype, device=device)
    full_rope = torch.zeros(4, BLOCK_SIZE, 1, 1, dtype=dtype, device=device)
    logical_tokens = torch.arange(256, dtype=torch.float32, device=device).to(dtype)
    rope_values = logical_tokens + 4096

    # Use a non-identity logical-to-physical block mapping.
    full_nope[2, :, 0, 0] = logical_tokens[:BLOCK_SIZE]
    full_nope[2, :, 0, 1] = logical_tokens[:BLOCK_SIZE] + 1
    full_rope[2, :, 0, 0] = rope_values[:BLOCK_SIZE]
    full_nope[1, :, 0, 0] = logical_tokens[BLOCK_SIZE:]
    full_nope[1, :, 0, 1] = logical_tokens[BLOCK_SIZE:] + 1
    full_rope[1, :, 0, 0] = rope_values[BLOCK_SIZE:]

    workspace = SparseKVOffloadWorkspace(
        (full_nope, full_rope),
        index_topk=INDEX_TOPK,
        block_size=BLOCK_SIZE,
    )
    updated_blocks = workspace.sync_updated_blocks(
        (full_nope, full_rope),
        torch.tensor(
            [
                2 * BLOCK_SIZE,
                1 * BLOCK_SIZE,
            ],
            dtype=torch.int64,
        ),
        num_actual_tokens=2,
    )
    assert updated_blocks == (1, 2)

    topk_indices = torch.full(
        (1, 1, INDEX_TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_indices[0, 0, :256] = torch.arange(
        256,
        dtype=torch.int32,
        device=device,
    )
    selection = workspace.gather(
        topk_indices=topk_indices,
        full_block_table=torch.tensor(
            [[2, 1]],
            dtype=torch.int32,
            device=device,
        ),
        full_actual_seq_lengths=torch.tensor(
            [256],
            dtype=torch.int32,
            device=device,
        ),
        full_query_actual_seq_lengths=torch.tensor(
            [1],
            dtype=torch.int32,
            device=device,
        ),
    )
    torch.npu.synchronize()

    assert selection.actual_seq_lengths_kv.cpu().tolist() == [256]
    assert selection.sparse_indices[0, 0, :257].cpu().tolist() == [
        *range(256),
        -1,
    ]

    selected_nope = selection.kv_cache[0].view(-1, 2)[:256, 0]
    selected_rope = selection.kv_cache[1].view(-1, 1)[:256, 0]
    torch.testing.assert_close(
        torch.sort(selected_nope.float()).values.cpu(),
        torch.arange(256, dtype=torch.float32),
    )
    torch.testing.assert_close(
        torch.sort(selected_rope.float()).values.cpu(),
        torch.sort(rope_values.float()).values.cpu(),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_host_mode_persists_prefill_rows_and_gathers_from_framework_swapped_kv(
    dtype,
):
    device = torch.device("npu")
    full_nope = _allocate_framework_swapped_cache(
        (4, BLOCK_SIZE, 1, 2),
        dtype,
        device,
    )
    full_rope = _allocate_framework_swapped_cache(
        (4, BLOCK_SIZE, 1, 1),
        dtype,
        device,
    )
    full_nope.fill_(-1)
    full_rope.fill_(-1)
    prefill_nope = torch.zeros_like(full_nope)
    prefill_rope = torch.zeros_like(full_rope)
    logical_tokens = torch.arange(256, dtype=torch.float32, device=device).to(dtype)
    rope_values = logical_tokens + 4096

    prefill_nope[2, :, 0, 0] = logical_tokens[:BLOCK_SIZE]
    prefill_nope[2, :, 0, 1] = logical_tokens[:BLOCK_SIZE] + 1
    prefill_rope[2, :, 0, 0] = rope_values[:BLOCK_SIZE]
    prefill_nope[1, :, 0, 0] = logical_tokens[BLOCK_SIZE:]
    prefill_nope[1, :, 0, 1] = logical_tokens[BLOCK_SIZE:] + 1
    prefill_rope[1, :, 0, 0] = rope_values[BLOCK_SIZE:]

    workspace = SparseKVOffloadWorkspace(
        (full_nope, full_rope),
        index_topk=INDEX_TOPK,
        block_size=BLOCK_SIZE,
        mode="host",
        prefill_kv_cache=(prefill_nope, prefill_rope),
    )
    physical_slots = torch.cat(
        (
            torch.arange(
                2 * BLOCK_SIZE,
                3 * BLOCK_SIZE,
                dtype=torch.int64,
            ),
            torch.arange(
                BLOCK_SIZE,
                2 * BLOCK_SIZE,
                dtype=torch.int64,
            ),
        )
    )
    updated_blocks = workspace.persist_updated_slots(
        (full_nope, full_rope),
        physical_slots,
        num_actual_tokens=physical_slots.numel(),
    )
    assert updated_blocks == (1, 2)

    topk_indices = torch.full(
        (1, 1, INDEX_TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_indices[0, 0, :256] = torch.arange(
        256,
        dtype=torch.int32,
        device=device,
    )
    selection = workspace.gather(
        topk_indices=topk_indices,
        full_block_table=torch.tensor(
            [[2, 1]],
            dtype=torch.int32,
            device=device,
        ),
        full_actual_seq_lengths=torch.tensor(
            [256],
            dtype=torch.int32,
            device=device,
        ),
        full_query_actual_seq_lengths=torch.tensor(
            [1],
            dtype=torch.int32,
            device=device,
        ),
    )
    torch.npu.synchronize()

    selected_nope = selection.kv_cache[0].view(-1, 2)[:256, 0]
    selected_rope = selection.kv_cache[1].view(-1, 1)[:256, 0]
    torch.testing.assert_close(
        selected_nope.float().cpu(),
        torch.arange(256, dtype=torch.float32),
    )
    torch.testing.assert_close(
        selected_rope.float().cpu(),
        rope_values.float().cpu(),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_host_mode_persists_decode_staging_row_before_gather(dtype):
    device = torch.device("npu")
    full_nope = _allocate_framework_swapped_cache(
        (2, BLOCK_SIZE, 1, 2),
        dtype,
        device,
    )
    full_rope = _allocate_framework_swapped_cache(
        (2, BLOCK_SIZE, 1, 1),
        dtype,
        device,
    )
    full_nope.fill_(-1)
    full_rope.fill_(-1)
    staging_nope = torch.full(
        tuple(full_nope.shape),
        99,
        dtype=dtype,
        device=device,
    )
    staging_rope = torch.full(
        tuple(full_rope.shape),
        77,
        dtype=dtype,
        device=device,
    )
    staging_nope[1, 0, 0] = torch.tensor([12, 13], dtype=dtype, device=device)
    staging_rope[1, 0, 0, 0] = 14

    workspace = SparseKVOffloadWorkspace(
        (full_nope, full_rope),
        index_topk=INDEX_TOPK,
        block_size=BLOCK_SIZE,
        mode="host",
        prefill_kv_cache=(staging_nope, staging_rope),
    )
    forward_cache = workspace.get_forward_kv_cache((full_nope, full_rope))
    assert forward_cache[0].data_ptr() == staging_nope.data_ptr()
    assert forward_cache[1].data_ptr() == staging_rope.data_ptr()

    workspace.persist_updated_slots(
        (full_nope, full_rope),
        torch.tensor([BLOCK_SIZE], dtype=torch.int64),
        num_actual_tokens=1,
    )
    topk_indices = torch.full(
        (1, 1, INDEX_TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_indices[0, 0, 0] = 0
    selection = workspace.gather(
        topk_indices=topk_indices,
        full_block_table=torch.tensor([[1]], dtype=torch.int32, device=device),
        full_actual_seq_lengths=torch.tensor([1], dtype=torch.int32, device=device),
        full_query_actual_seq_lengths=torch.tensor([1], dtype=torch.int32, device=device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        selection.kv_cache[0].view(-1, 2)[0].float().cpu(),
        torch.tensor([12, 13], dtype=torch.float32),
    )
    torch.testing.assert_close(
        selection.kv_cache[1].view(-1, 1)[0].float().cpu(),
        torch.tensor([14], dtype=torch.float32),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_host_mode_restores_chunked_prefill_context(dtype):
    device = torch.device("npu")
    full_nope = _allocate_framework_swapped_cache(
        (4, BLOCK_SIZE, 1, 2),
        dtype,
        device,
    )
    full_rope = _allocate_framework_swapped_cache(
        (4, BLOCK_SIZE, 1, 1),
        dtype,
        device,
    )
    full_nope.fill_(-1)
    full_rope.fill_(-1)
    full_nope[2].fill_(20)
    full_rope[2].fill_(21)
    full_nope[1].fill_(10)
    full_rope[1].fill_(11)
    prefill_nope = torch.full(
        tuple(full_nope.shape),
        99,
        dtype=dtype,
        device=device,
    )
    prefill_rope = torch.full(
        tuple(full_rope.shape),
        77,
        dtype=dtype,
        device=device,
    )
    workspace = SparseKVOffloadWorkspace(
        (full_nope, full_rope),
        index_topk=INDEX_TOPK,
        block_size=BLOCK_SIZE,
        mode="host",
        prefill_kv_cache=(prefill_nope, prefill_rope),
    )

    restored_blocks = workspace.restore_prefill_context(
        (full_nope, full_rope),
        torch.tensor([[2, 1, 3]], dtype=torch.int32),
        context_len=BLOCK_SIZE + 1,
    )
    torch.npu.synchronize()

    assert restored_blocks == (1, 2)
    torch.testing.assert_close(prefill_nope[2], full_nope[2])
    torch.testing.assert_close(prefill_rope[2], full_rope[2])
    torch.testing.assert_close(prefill_nope[1], full_nope[1])
    torch.testing.assert_close(prefill_rope[1], full_rope[1])
    torch.testing.assert_close(
        prefill_nope[3].float().cpu(),
        torch.full(tuple(prefill_nope[3].shape), 99, dtype=torch.float32),
    )
    torch.testing.assert_close(
        prefill_rope[3].float().cpu(),
        torch.full(tuple(prefill_rope[3].shape), 77, dtype=torch.float32),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_len", [18, 256])
def test_host_gather_selected_sfa_matches_full_npu_kv_sfa(dtype, seq_len):
    torch.manual_seed(2026)
    device = torch.device("npu")
    num_blocks = 4
    kv_lora_rank = 512
    rope_head_dim = 64
    num_query_heads = 16
    full_nope = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        1,
        kv_lora_rank,
        dtype=dtype,
        device=device,
    )
    full_rope = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        1,
        rope_head_dim,
        dtype=dtype,
        device=device,
    )
    full_block_table = torch.tensor(
        [[2, 1]],
        dtype=torch.int32,
        device=device,
    )
    topk_indices = torch.full(
        (1, 1, INDEX_TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_indices[0, 0, :seq_len] = torch.arange(
        seq_len,
        dtype=torch.int32,
        device=device,
    )
    actual_query = torch.tensor([1], dtype=torch.int32, device=device)
    actual_key = torch.tensor([seq_len], dtype=torch.int32, device=device)
    ql_nope = torch.randn(
        1,
        num_query_heads,
        kv_lora_rank,
        dtype=dtype,
        device=device,
    )
    q_pe = torch.randn(
        1,
        num_query_heads,
        rope_head_dim,
        dtype=dtype,
        device=device,
    )

    fake_impl = MagicMock()
    fake_impl.scale = 1.0 / math.sqrt(kv_lora_rank + rope_head_dim)
    full_metadata = MagicMock()
    full_metadata.block_table = full_block_table
    full_output = AscendSFAImpl._execute_sparse_flash_attention_process(
        fake_impl,
        ql_nope,
        q_pe,
        (full_nope, full_rope),
        topk_indices,
        full_metadata,
        actual_query,
        actual_key,
    )

    host_nope = _allocate_framework_swapped_cache(
        tuple(full_nope.shape),
        dtype,
        device,
    )
    host_rope = _allocate_framework_swapped_cache(
        tuple(full_rope.shape),
        dtype,
        device,
    )
    host_nope.fill_(0)
    host_rope.fill_(0)
    workspace = SparseKVOffloadWorkspace(
        (host_nope, host_rope),
        index_topk=INDEX_TOPK,
        block_size=BLOCK_SIZE,
        mode="host",
        prefill_kv_cache=(full_nope, full_rope),
    )
    physical_slots = torch.cat(
        (
            torch.arange(
                2 * BLOCK_SIZE,
                3 * BLOCK_SIZE,
                dtype=torch.int64,
            ),
            torch.arange(
                BLOCK_SIZE,
                2 * BLOCK_SIZE,
                dtype=torch.int64,
            ),
        )
    )
    workspace.persist_updated_slots(
        (host_nope, host_rope),
        physical_slots,
        num_actual_tokens=physical_slots.numel(),
    )
    selection = workspace.gather(
        topk_indices=topk_indices,
        full_block_table=full_block_table,
        full_actual_seq_lengths=actual_key,
        full_query_actual_seq_lengths=actual_query,
    )
    selected_metadata = MagicMock()
    selected_metadata.block_table = selection.block_table
    selected_output = AscendSFAImpl._execute_sparse_flash_attention_process(
        fake_impl,
        ql_nope,
        q_pe,
        selection.kv_cache,
        selection.sparse_indices,
        selected_metadata,
        selection.actual_seq_lengths_query,
        selection.actual_seq_lengths_kv,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        selected_output,
        full_output,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mirror_gather_selected_sfa_matches_full_npu_kv_across_decode_updates(
    dtype,
):
    """Exercise selected-KV reuse across two non-block-aligned decode steps."""
    torch.manual_seed(2026)
    device = torch.device("npu")
    num_blocks = 4
    kv_lora_rank = 512
    rope_head_dim = 64
    num_query_heads = 16
    full_nope = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        1,
        kv_lora_rank,
        dtype=dtype,
        device=device,
    )
    full_rope = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        1,
        rope_head_dim,
        dtype=dtype,
        device=device,
    )
    full_block_table = torch.tensor(
        [[2, 1]],
        dtype=torch.int32,
        device=device,
    )
    workspace = SparseKVOffloadWorkspace(
        (full_nope, full_rope),
        index_topk=INDEX_TOPK,
        block_size=BLOCK_SIZE,
        mode="mirror",
    )
    fake_impl = MagicMock()
    fake_impl.scale = 1.0 / math.sqrt(kv_lora_rank + rope_head_dim)
    full_metadata = MagicMock()
    full_metadata.block_table = full_block_table

    for step, seq_len in enumerate((18, 19)):
        # The newly generated token extends the same partially filled physical
        # block. Mirroring the block must make the new row visible without
        # invalidating selected-KV reuse from the previous decode step.
        updated_slot = 2 * BLOCK_SIZE + seq_len - 1
        workspace.sync_updated_blocks(
            (full_nope, full_rope),
            torch.tensor([updated_slot], dtype=torch.int64),
            num_actual_tokens=1,
        )

        topk_indices = torch.full(
            (1, 1, INDEX_TOPK),
            -1,
            dtype=torch.int32,
            device=device,
        )
        logical_indices = torch.arange(
            seq_len,
            dtype=torch.int32,
            device=device,
        )
        topk_indices[0, 0, :seq_len] = torch.roll(
            logical_indices,
            shifts=step,
        )
        actual_query = torch.tensor([1], dtype=torch.int32, device=device)
        actual_key = torch.tensor([seq_len], dtype=torch.int32, device=device)
        ql_nope = torch.randn(
            1,
            num_query_heads,
            kv_lora_rank,
            dtype=dtype,
            device=device,
        )
        q_pe = torch.randn(
            1,
            num_query_heads,
            rope_head_dim,
            dtype=dtype,
            device=device,
        )

        full_output = AscendSFAImpl._execute_sparse_flash_attention_process(
            fake_impl,
            ql_nope,
            q_pe,
            (full_nope, full_rope),
            topk_indices,
            full_metadata,
            actual_query,
            actual_key,
        )
        selection = workspace.gather(
            topk_indices=topk_indices,
            full_block_table=full_block_table,
            full_actual_seq_lengths=actual_key,
            full_query_actual_seq_lengths=actual_query,
        )
        selected_metadata = MagicMock()
        selected_metadata.block_table = selection.block_table
        selected_output = AscendSFAImpl._execute_sparse_flash_attention_process(
            fake_impl,
            ql_nope,
            q_pe,
            selection.kv_cache,
            selection.sparse_indices,
            selected_metadata,
            selection.actual_seq_lengths_query,
            selection.actual_seq_lengths_kv,
        )
        torch.npu.synchronize()

        assert selection.actual_seq_lengths_kv.cpu().tolist() == [seq_len]
        torch.testing.assert_close(
            selected_output,
            full_output,
            rtol=1e-2,
            atol=1e-2,
        )

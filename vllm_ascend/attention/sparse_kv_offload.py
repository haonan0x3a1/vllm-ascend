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

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch_npu

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

SELECTION_TOPK_BLOCK_SIZE = 1
SUPPORTED_KV_DTYPES = (torch.bfloat16, torch.float16)

SwappedAllocator = Callable[..., torch.Tensor]
GatherSelectionOp = Callable[..., torch.Tensor]


def get_gather_selection_kv_cache_op() -> GatherSelectionOp:
    """Load the external GatherSelectionKvCache operator on demand."""
    try:
        importlib.import_module("custom_ops")
    except ImportError as exc:
        raise RuntimeError(
            "sparse_kv_offload requires the cann-recipes-infer custom_ops "
            "package. Install the matching custom operator wheel before "
            "starting vLLM."
        ) from exc

    gather_op = getattr(torch_npu, "npu_gather_selection_kv_cache", None)
    if not callable(gather_op):
        raise RuntimeError(
            "custom_ops was imported, but "
            "torch_npu.npu_gather_selection_kv_cache is unavailable. Check "
            "that the custom operator wheel matches the installed CANN and "
            "torch-npu versions."
        )
    return gather_op


def _allocate_swapped_memory(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    allocator = getattr(torch_npu, "empty_with_swapped_memory", None)
    if not callable(allocator):
        raise RuntimeError("torch_npu.empty_with_swapped_memory is required by sparse_kv_offload mirror mode.")
    return allocator(shape, dtype=dtype, device=device)


@dataclass(frozen=True)
class SparseKVSelection:
    kv_cache: tuple[torch.Tensor, torch.Tensor]
    block_table: torch.Tensor
    sparse_indices: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor


class SparseKVOffloadWorkspace:
    """Per-layer Host-mirror and selected-KV workspace for the mirror PoC."""

    def __init__(
        self,
        full_kv_cache: tuple[torch.Tensor, ...],
        *,
        index_topk: int,
        block_size: int,
        swapped_allocator: SwappedAllocator | None = None,
        gather_op: GatherSelectionOp | None = None,
        validate_device: bool = True,
    ) -> None:
        self._validate_full_kv_cache(full_kv_cache, block_size)
        if validate_device and get_ascend_device_type() != AscendDeviceType.A3:
            raise RuntimeError(
                "sparse_kv_offload mirror mode currently requires an Ascend "
                f"A3 device, got {get_ascend_device_type().name}."
            )
        if index_topk != 2048:
            raise ValueError(f"sparse_kv_offload mirror mode currently requires index_topk=2048, got {index_topk}.")

        self.index_topk = index_topk
        self.block_size = block_size
        self.num_full_blocks = full_kv_cache[0].shape[0]
        self.device = full_kv_cache[0].device
        self.dtype = full_kv_cache[0].dtype
        self.gather_op = gather_op or get_gather_selection_kv_cache_op()
        allocator = swapped_allocator or _allocate_swapped_memory

        self.full_nope_mirror = allocator(
            tuple(full_kv_cache[0].shape),
            dtype=self.dtype,
            device=self.device,
        )
        self.full_rope_mirror = allocator(
            tuple(full_kv_cache[1].shape),
            dtype=self.dtype,
            device=self.device,
        )

        selection_num_blocks = (index_topk + block_size - 1) // block_size
        self.selected_nope = torch.empty(
            (
                selection_num_blocks,
                block_size,
                full_kv_cache[0].shape[-1],
            ),
            dtype=self.dtype,
            device=self.device,
        )
        self.selected_rope = torch.empty(
            (
                selection_num_blocks,
                block_size,
                full_kv_cache[1].shape[-1],
            ),
            dtype=self.dtype,
            device=self.device,
        )
        self._selection_block_table_template = torch.arange(
            selection_num_blocks,
            dtype=torch.int32,
            device=self.device,
        ).view(1, selection_num_blocks)
        self.selection_block_table = self._selection_block_table_template.clone()
        self.selection_block_status = torch.full(
            (1, 1, index_topk + 1),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.default_topk_indices = torch.arange(
            index_topk,
            dtype=torch.int32,
            device=self.device,
        ).view(1, index_topk)

    @staticmethod
    def _validate_full_kv_cache(
        full_kv_cache: tuple[torch.Tensor, ...],
        block_size: int,
    ) -> None:
        if len(full_kv_cache) < 2:
            raise ValueError("sparse_kv_offload requires at least the MLA nope and rope cache tensors.")
        full_nope, full_rope = full_kv_cache[:2]
        if full_nope.ndim != 4 or full_rope.ndim != 4:
            raise ValueError(
                "sparse_kv_offload expects paged MLA caches with shape [num_blocks, block_size, 1, head_dim]."
            )
        if full_nope.shape[:3] != full_rope.shape[:3]:
            raise ValueError("MLA nope and rope caches must share num_blocks, block_size, and head dimensions.")
        if full_nope.shape[1] != block_size:
            raise ValueError(f"KV cache block size mismatch: expected {block_size}, got {full_nope.shape[1]}.")
        if full_nope.shape[2] != 1:
            raise ValueError("GatherSelectionKvCache currently supports exactly one KV head.")
        if full_nope.dtype != full_rope.dtype:
            raise ValueError("MLA nope and rope cache dtypes must match.")
        if full_nope.dtype not in SUPPORTED_KV_DTYPES:
            raise ValueError(f"sparse_kv_offload mirror mode only supports BF16/FP16 KV cache, got {full_nope.dtype}.")
        if full_nope.device != full_rope.device:
            raise ValueError("MLA nope and rope caches must be on the same device.")

    def reset_selection_state(self) -> None:
        """Disable cross-step reuse for the first single-request PoC."""
        self.selection_block_table.copy_(self._selection_block_table_template)
        self.selection_block_status.fill_(-1)

    def sync_updated_blocks(
        self,
        full_kv_cache: tuple[torch.Tensor, ...],
        slot_mapping_cpu: torch.Tensor,
        num_actual_tokens: int,
    ) -> tuple[int, ...]:
        """Synchronously mirror physical KV blocks touched by this forward."""
        self._validate_full_kv_cache(full_kv_cache, self.block_size)
        if slot_mapping_cpu is None:
            raise RuntimeError(
                "sparse_kv_offload requires CPU slot_mapping metadata to avoid "
                "an NPU-to-CPU synchronization in the attention hot path."
            )
        if slot_mapping_cpu.device.type != "cpu":
            raise ValueError("slot_mapping_cpu must reside on CPU.")
        if slot_mapping_cpu.ndim != 1:
            raise ValueError(
                f"slot_mapping_cpu must be a flat physical-slot tensor, got shape {tuple(slot_mapping_cpu.shape)}."
            )
        if num_actual_tokens < 0 or num_actual_tokens > slot_mapping_cpu.numel():
            raise ValueError(
                f"Invalid num_actual_tokens={num_actual_tokens} for "
                f"slot_mapping_cpu with {slot_mapping_cpu.numel()} entries."
            )

        valid_slots = slot_mapping_cpu[:num_actual_tokens]
        valid_slots = valid_slots[valid_slots >= 0]
        if valid_slots.numel() == 0:
            return ()

        physical_block_ids = torch.unique(
            torch.div(
                valid_slots,
                self.block_size,
                rounding_mode="floor",
            )
        ).tolist()
        for block_id in physical_block_ids:
            block_id = int(block_id)
            if block_id >= self.num_full_blocks:
                raise ValueError(
                    f"slot_mapping references physical block {block_id}, but "
                    f"the KV cache has only {self.num_full_blocks} blocks."
                )
            self.full_nope_mirror[block_id].copy_(
                full_kv_cache[0][block_id],
                non_blocking=False,
            )
            self.full_rope_mirror[block_id].copy_(
                full_kv_cache[1][block_id],
                non_blocking=False,
            )
        return tuple(int(block_id) for block_id in physical_block_ids)

    def gather(
        self,
        *,
        topk_indices: torch.Tensor,
        full_block_table: torch.Tensor,
        full_actual_seq_lengths: torch.Tensor,
        full_query_actual_seq_lengths: torch.Tensor,
    ) -> SparseKVSelection:
        if topk_indices.ndim == 2:
            topk_indices = topk_indices.unsqueeze(1)
        if topk_indices.shape != (1, 1, self.index_topk):
            raise ValueError(
                "sparse_kv_offload mirror mode expects Top-K shape "
                f"[1, 1, {self.index_topk}], got {tuple(topk_indices.shape)}."
            )
        if full_block_table.shape[0] != 1:
            raise ValueError(
                "sparse_kv_offload mirror mode currently supports one request, "
                f"got block table shape {tuple(full_block_table.shape)}."
            )

        self.reset_selection_state()
        selected_actual_seq_lengths = self.gather_op(
            selection_k_rope=self.selected_rope,
            selection_kv_cache=self.selected_nope,
            selection_kv_block_table=self.selection_block_table,
            selection_kv_block_status=self.selection_block_status,
            selection_topk_indices=topk_indices.to(torch.int32).contiguous(),
            full_k_rope=self.full_rope_mirror.squeeze(2),
            full_kv_cache=self.full_nope_mirror.squeeze(2),
            full_kv_block_table=full_block_table.to(torch.int32).contiguous(),
            full_kv_actual_seq=full_actual_seq_lengths.to(torch.int32).contiguous(),
            full_q_actual_seq=full_query_actual_seq_lengths.to(torch.int32).contiguous(),
            selection_topk_block_size=SELECTION_TOPK_BLOCK_SIZE,
        )

        local_sparse_indices = torch.where(
            self.default_topk_indices < selected_actual_seq_lengths.unsqueeze(1),
            self.default_topk_indices,
            -1,
        ).view(1, 1, self.index_topk)
        selected_query_actual_seq_lengths = torch.ones(
            1,
            dtype=torch.int32,
            device=self.device,
        )
        return SparseKVSelection(
            kv_cache=(
                self.selected_nope.unsqueeze(2),
                self.selected_rope.unsqueeze(2),
            ),
            block_table=self.selection_block_table,
            sparse_indices=local_sparse_indices,
            actual_seq_lengths_query=selected_query_actual_seq_lengths,
            actual_seq_lengths_kv=selected_actual_seq_lengths,
        )

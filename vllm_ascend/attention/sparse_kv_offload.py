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
        raise RuntimeError(
            "torch_npu.empty_with_swapped_memory is required by "
            "sparse_kv_offload."
        )
    return allocator(shape, dtype=dtype, device=device)


@dataclass(frozen=True)
class SparseKVSelection:
    kv_cache: tuple[torch.Tensor, torch.Tensor]
    block_table: torch.Tensor
    sparse_indices: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor


class SparseKVOffloadWorkspace:
    """Per-layer sparse KV offload runtime workspace.

    Mirror mode owns a swapped-memory mirror of the framework NPU KV cache.
    Host mode treats the framework KV cache itself as the swapped-memory Full
    KV store and uses a model-runner-owned, cross-layer NPU cache for prefill.
    Both modes own a per-layer selected-KV NPU workspace for decode.
    """

    def __init__(
        self,
        full_kv_cache: tuple[torch.Tensor, ...],
        *,
        index_topk: int,
        block_size: int,
        mode: str = "mirror",
        prefill_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        swapped_allocator: SwappedAllocator | None = None,
        gather_op: GatherSelectionOp | None = None,
        validate_device: bool = True,
    ) -> None:
        self._validate_full_kv_cache(full_kv_cache, block_size)
        if mode not in {"host", "mirror"}:
            raise ValueError(f"Unsupported sparse KV offload mode: {mode!r}.")
        if validate_device and get_ascend_device_type() != AscendDeviceType.A3:
            raise RuntimeError(
                f"sparse_kv_offload {mode} mode currently requires an Ascend "
                f"A3 device, got {get_ascend_device_type().name}."
            )
        if index_topk != 2048:
            raise ValueError(
                f"sparse_kv_offload {mode} mode currently requires "
                f"index_topk=2048, got {index_topk}."
            )

        self.mode = mode
        self.index_topk = index_topk
        self.block_size = block_size
        self.num_full_blocks = full_kv_cache[0].shape[0]
        self.device = full_kv_cache[0].device
        self.dtype = full_kv_cache[0].dtype
        self.gather_op = gather_op or get_gather_selection_kv_cache_op()
        allocator = swapped_allocator or _allocate_swapped_memory

        if mode == "mirror":
            self.full_nope_source = allocator(
                tuple(full_kv_cache[0].shape),
                dtype=self.dtype,
                device=self.device,
            )
            self.full_rope_source = allocator(
                tuple(full_kv_cache[1].shape),
                dtype=self.dtype,
                device=self.device,
            )
            self.prefill_kv_cache = None
        else:
            if prefill_kv_cache is None:
                raise ValueError(
                    "sparse_kv_offload host mode requires the shared prefill "
                    "NPU KV cache allocated by the model runner."
                )
            self._validate_prefill_kv_cache(prefill_kv_cache, full_kv_cache)
            self.full_nope_source = full_kv_cache[0]
            self.full_rope_source = full_kv_cache[1]
            self.prefill_kv_cache = prefill_kv_cache

        # Backwards-compatible aliases retained for the mirror-mode unit tests
        # and for code that inspected the first PoC workspace.
        self.full_nope_mirror = self.full_nope_source
        self.full_rope_mirror = self.full_rope_source

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
    def _validate_prefill_kv_cache(
        prefill_kv_cache: tuple[torch.Tensor, torch.Tensor],
        full_kv_cache: tuple[torch.Tensor, ...],
    ) -> None:
        if len(prefill_kv_cache) != 2:
            raise ValueError("The shared prefill KV cache must contain exactly nope and rope tensors.")
        for name, prefill_tensor, full_tensor in zip(
            ("nope", "rope"),
            prefill_kv_cache,
            full_kv_cache[:2],
        ):
            if prefill_tensor.shape != full_tensor.shape:
                raise ValueError(
                    f"Shared prefill {name} shape {tuple(prefill_tensor.shape)} "
                    f"does not match Full KV shape {tuple(full_tensor.shape)}."
                )
            if prefill_tensor.dtype != full_tensor.dtype:
                raise ValueError(
                    f"Shared prefill {name} dtype {prefill_tensor.dtype} "
                    f"does not match Full KV dtype {full_tensor.dtype}."
                )
            if prefill_tensor.device != full_tensor.device:
                raise ValueError(
                    f"Shared prefill {name} device {prefill_tensor.device} "
                    f"does not match Full KV device {full_tensor.device}."
                )

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
            raise ValueError(
                "sparse_kv_offload only supports BF16/FP16 KV cache, got "
                f"{full_nope.dtype}."
            )
        if full_nope.device != full_rope.device:
            raise ValueError("MLA nope and rope caches must be on the same device.")

    def reset_selection_state(self) -> None:
        """Drop selected-KV reuse state at a request/prefill boundary."""
        self.selection_block_table.copy_(self._selection_block_table_template)
        self.selection_block_status.fill_(-1)

    def get_forward_kv_cache(
        self,
        full_kv_cache: tuple[torch.Tensor, ...],
        *,
        is_decode: bool,
    ) -> tuple[torch.Tensor, ...]:
        """Return the cache that MLA Prolog/Prefill Attention should access."""
        self._validate_full_kv_cache(full_kv_cache, self.block_size)
        if self.mode == "host" and not is_decode:
            assert self.prefill_kv_cache is not None
            return (*self.prefill_kv_cache, *full_kv_cache[2:])
        return full_kv_cache

    def _copy_updated_blocks(
        self,
        source_kv_cache: tuple[torch.Tensor, ...],
        target_kv_cache: tuple[torch.Tensor, torch.Tensor],
        slot_mapping_cpu: torch.Tensor,
        num_actual_tokens: int,
    ) -> tuple[int, ...]:
        self._validate_full_kv_cache(source_kv_cache, self.block_size)
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
            target_kv_cache[0][block_id].copy_(
                source_kv_cache[0][block_id],
                non_blocking=False,
            )
            target_kv_cache[1][block_id].copy_(
                source_kv_cache[1][block_id],
                non_blocking=False,
            )
        return tuple(int(block_id) for block_id in physical_block_ids)

    def _copy_updated_slots(
        self,
        source_kv_cache: tuple[torch.Tensor, torch.Tensor],
        target_kv_cache: tuple[torch.Tensor, torch.Tensor],
        slot_mapping_cpu: torch.Tensor,
        num_actual_tokens: int,
    ) -> tuple[int, ...]:
        """Copy only touched token rows from the shared prefill workspace.

        The prefill workspace is reused by every layer. Copying whole physical
        blocks would therefore corrupt earlier tokens when chunked prefill
        revisits a partially filled block after another layer has reused the
        workspace.
        """
        if slot_mapping_cpu is None:
            raise RuntimeError(
                "sparse_kv_offload requires CPU slot_mapping metadata to avoid "
                "an NPU-to-CPU synchronization in the attention hot path."
            )
        if slot_mapping_cpu.device.type != "cpu":
            raise ValueError("slot_mapping_cpu must reside on CPU.")
        if slot_mapping_cpu.ndim != 1:
            raise ValueError(
                "slot_mapping_cpu must be a flat physical-slot tensor, got "
                f"shape {tuple(slot_mapping_cpu.shape)}."
            )
        if num_actual_tokens < 0 or num_actual_tokens > slot_mapping_cpu.numel():
            raise ValueError(
                f"Invalid num_actual_tokens={num_actual_tokens} for "
                f"slot_mapping_cpu with {slot_mapping_cpu.numel()} entries."
            )

        valid_slots = slot_mapping_cpu[:num_actual_tokens]
        valid_slots = torch.unique(valid_slots[valid_slots >= 0])
        if valid_slots.numel() == 0:
            return ()

        total_slots = self.num_full_blocks * self.block_size
        max_slot = int(valid_slots.max())
        if max_slot >= total_slots:
            raise ValueError(
                f"slot_mapping references physical slot {max_slot}, but the "
                f"KV cache has only {total_slots} slots."
            )

        device_slots = valid_slots.to(
            device=self.device,
            dtype=torch.int64,
            non_blocking=True,
        )
        for source, target in zip(source_kv_cache, target_kv_cache):
            source_rows = source.view(total_slots, source.shape[-1])
            target_rows = target.view(total_slots, target.shape[-1])
            target_rows.index_copy_(
                0,
                device_slots,
                source_rows.index_select(0, device_slots),
            )

        physical_block_ids = torch.unique(
            torch.div(
                valid_slots,
                self.block_size,
                rounding_mode="floor",
            )
        )
        return tuple(int(block_id) for block_id in physical_block_ids.tolist())

    def sync_updated_blocks(
        self,
        full_kv_cache: tuple[torch.Tensor, ...],
        slot_mapping_cpu: torch.Tensor,
        num_actual_tokens: int,
    ) -> tuple[int, ...]:
        """Synchronously update the swapped mirror in mirror mode."""
        if self.mode != "mirror":
            raise RuntimeError("sync_updated_blocks is only valid in sparse KV offload mirror mode.")
        return self._copy_updated_blocks(
            full_kv_cache,
            (self.full_nope_source, self.full_rope_source),
            slot_mapping_cpu,
            num_actual_tokens,
        )

    def persist_prefill_blocks(
        self,
        full_kv_cache: tuple[torch.Tensor, ...],
        slot_mapping_cpu: torch.Tensor,
        num_actual_tokens: int,
    ) -> tuple[int, ...]:
        """Copy the current layer's shared NPU prefill cache into Full Host KV."""
        if self.mode != "host":
            raise RuntimeError(
                "persist_prefill_blocks is only valid in sparse KV offload "
                "host mode."
            )
        assert self.prefill_kv_cache is not None
        return self._copy_updated_slots(
            self.prefill_kv_cache,
            (full_kv_cache[0], full_kv_cache[1]),
            slot_mapping_cpu,
            num_actual_tokens,
        )

    def restore_prefill_context(
        self,
        full_kv_cache: tuple[torch.Tensor, ...],
        full_block_table_cpu: torch.Tensor,
        context_len: int,
    ) -> tuple[int, ...]:
        """Restore prior prompt blocks before reusing the Prefill workspace.

        A single NPU Prefill cache is shared by every sparse layer. After one
        chunk has traversed all layers, that cache contains the final layer's
        data. A later chunk must therefore reload each current layer's already
        computed blocks from its Host Full-KV before Attention can read them.
        """
        if self.mode != "host":
            raise RuntimeError(
                "restore_prefill_context is only valid in sparse KV offload "
                "host mode."
            )
        if context_len < 0:
            raise ValueError(f"context_len must be non-negative, got {context_len}.")
        if context_len == 0:
            return ()
        if full_block_table_cpu.device.type != "cpu":
            raise ValueError("full_block_table_cpu must reside on CPU.")
        if full_block_table_cpu.ndim != 2 or full_block_table_cpu.shape[0] != 1:
            raise ValueError(
                "sparse_kv_offload host mode expects a CPU block table with "
                f"shape [1, max_blocks], got {tuple(full_block_table_cpu.shape)}."
            )
        if context_len > self.num_full_blocks * self.block_size:
            raise ValueError(
                f"context_len={context_len} exceeds Full KV capacity "
                f"{self.num_full_blocks * self.block_size}."
            )

        num_context_blocks = (
            context_len + self.block_size - 1
        ) // self.block_size
        if num_context_blocks > full_block_table_cpu.shape[1]:
            raise ValueError(
                f"Context requires {num_context_blocks} blocks, but the CPU "
                f"block table has only {full_block_table_cpu.shape[1]} entries."
            )

        physical_block_ids = torch.unique(
            full_block_table_cpu[0, :num_context_blocks].to(torch.int64)
        )
        if physical_block_ids.numel() == 0:
            return ()
        min_block = int(physical_block_ids.min())
        max_block = int(physical_block_ids.max())
        if min_block < 0 or max_block >= self.num_full_blocks:
            raise ValueError(
                "CPU block table references physical blocks outside Full KV: "
                f"min={min_block}, max={max_block}, "
                f"num_full_blocks={self.num_full_blocks}."
            )

        assert self.prefill_kv_cache is not None
        block_ids = tuple(int(block_id) for block_id in physical_block_ids.tolist())
        for source, target in zip(
            (full_kv_cache[0], full_kv_cache[1]),
            self.prefill_kv_cache,
        ):
            for block_id in block_ids:
                target[block_id].copy_(
                    source[block_id],
                    non_blocking=False,
                )
        return block_ids

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
                f"sparse_kv_offload {self.mode} mode expects Top-K shape "
                f"[1, 1, {self.index_topk}], got {tuple(topk_indices.shape)}."
            )
        if full_block_table.shape[0] != 1:
            raise ValueError(
                f"sparse_kv_offload {self.mode} mode currently supports one "
                "request, "
                f"got block table shape {tuple(full_block_table.shape)}."
            )

        selected_actual_seq_lengths = self.gather_op(
            selection_k_rope=self.selected_rope,
            selection_kv_cache=self.selected_nope,
            selection_kv_block_table=self.selection_block_table,
            selection_kv_block_status=self.selection_block_status,
            selection_topk_indices=topk_indices.to(torch.int32).contiguous(),
            full_k_rope=self.full_rope_source.squeeze(2),
            full_kv_cache=self.full_nope_source.squeeze(2),
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

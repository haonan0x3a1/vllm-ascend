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
INT32_BYTES = 4

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
        raise RuntimeError("torch_npu.empty_with_swapped_memory is required by sparse_kv_offload.")
    return allocator(shape, dtype=dtype, device=device)


@dataclass(frozen=True)
class SparseKVSelection:
    kv_cache: tuple[torch.Tensor, torch.Tensor]
    block_table: torch.Tensor
    sparse_indices: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor


@dataclass(frozen=True)
class SparseKVOffloadMemoryPlan:
    """Persistent NPU memory owned by sparse KV offload Host mode."""

    indexer_cache_bytes: int
    indexer_alignment_bytes: int
    shared_prefill_bytes: int
    shared_prefill_alignment_bytes: int
    selected_cache_bytes: int
    selection_metadata_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.indexer_cache_bytes
            + self.indexer_alignment_bytes
            + self.shared_prefill_bytes
            + self.shared_prefill_alignment_bytes
            + self.selected_cache_bytes
            + self.selection_metadata_bytes
        )


def make_sparse_kv_offload_memory_plan(
    *,
    num_blocks: int,
    block_size: int,
    index_topk: int,
    full_kv_head_dim: int,
    index_head_dim: int,
    kv_dtype_size: int,
    num_sparse_layers: int,
    num_indexer_layers: int,
    indexer_alignment_bytes_per_layer: int = 0,
    shared_prefill_alignment_bytes: int = 0,
) -> SparseKVOffloadMemoryPlan:
    """Plan Host-mode persistent NPU allocations before creating tensors."""
    positive_values = {
        "num_blocks": num_blocks,
        "block_size": block_size,
        "index_topk": index_topk,
        "full_kv_head_dim": full_kv_head_dim,
        "index_head_dim": index_head_dim,
        "kv_dtype_size": kv_dtype_size,
        "num_sparse_layers": num_sparse_layers,
    }
    invalid_values = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid_values:
        raise ValueError(f"sparse KV offload memory planning requires positive dimensions, got {invalid_values}.")
    if not 0 <= num_indexer_layers <= num_sparse_layers:
        raise ValueError(
            f"num_indexer_layers must be in [0, num_sparse_layers], got {num_indexer_layers} and {num_sparse_layers}."
        )
    if indexer_alignment_bytes_per_layer < 0:
        raise ValueError(
            f"indexer_alignment_bytes_per_layer must be non-negative, got {indexer_alignment_bytes_per_layer}."
        )
    if shared_prefill_alignment_bytes < 0:
        raise ValueError(f"shared_prefill_alignment_bytes must be non-negative, got {shared_prefill_alignment_bytes}.")

    full_tokens = num_blocks * block_size
    selection_num_blocks = (index_topk + block_size - 1) // block_size
    selected_tokens = selection_num_blocks * block_size

    indexer_cache_bytes = num_indexer_layers * full_tokens * index_head_dim * kv_dtype_size
    shared_prefill_bytes = full_tokens * full_kv_head_dim * kv_dtype_size
    selected_cache_bytes = num_sparse_layers * selected_tokens * full_kv_head_dim * kv_dtype_size
    # Each layer owns a block-table template and its mutable copy, a status
    # table with one sentinel entry, and the default Top-K index vector.
    selection_metadata_bytes = num_sparse_layers * (
        2 * selection_num_blocks * INT32_BYTES + (index_topk + 1) * INT32_BYTES + index_topk * INT32_BYTES
    )

    return SparseKVOffloadMemoryPlan(
        indexer_cache_bytes=indexer_cache_bytes,
        indexer_alignment_bytes=(num_indexer_layers * indexer_alignment_bytes_per_layer),
        shared_prefill_bytes=shared_prefill_bytes,
        shared_prefill_alignment_bytes=shared_prefill_alignment_bytes,
        selected_cache_bytes=selected_cache_bytes,
        selection_metadata_bytes=selection_metadata_bytes,
    )


class SparseKVOffloadWorkspace:
    """Per-layer sparse KV offload runtime workspace.

    Mirror mode owns a swapped-memory mirror of the framework NPU KV cache.
    Host mode treats the framework KV cache itself as the swapped-memory Full
    KV store and uses a model-runner-owned, cross-layer NPU cache as the write
    staging area for both prefill and decode. Both modes own a per-layer
    selected-KV NPU workspace for decode.
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
            raise ValueError(f"sparse_kv_offload {mode} mode currently requires index_topk=2048, got {index_topk}.")

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
            raise ValueError(f"sparse_kv_offload only supports BF16/FP16 KV cache, got {full_nope.dtype}.")
        if full_nope.device != full_rope.device:
            raise ValueError("MLA nope and rope caches must be on the same device.")

    def reset_selection_state(self) -> None:
        """Drop selected-KV reuse state at a request/prefill boundary."""
        self.selection_block_table.copy_(self._selection_block_table_template)
        self.selection_block_status.fill_(-1)

    def get_forward_kv_cache(
        self,
        full_kv_cache: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Return the cache that MLA Prolog/Prefill Attention should access.

        Host-backed swapped memory is readable by Gather/copy operations but
        is not a valid output for every native MLA preprocessing kernel. Route
        both prefill and decode writes through the shared NPU workspace; the
        touched rows are persisted into Full Host KV before sparse Gather.
        """
        self._validate_full_kv_cache(full_kv_cache, self.block_size)
        if self.mode == "host":
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
                f"slot_mapping_cpu must be a flat physical-slot tensor, got shape {tuple(slot_mapping_cpu.shape)}."
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
                f"slot_mapping references physical slot {max_slot}, but the KV cache has only {total_slots} slots."
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

    @staticmethod
    def _mirror_tensors_match(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> bool:
        """Compare diagnostic tensors without copying swapped storage to CPU."""
        if torch.equal(left, right):
            return True
        equal_or_paired_nan = torch.eq(left, right) | (
            torch.isnan(left) & torch.isnan(right)
        )
        return bool(torch.all(equal_or_paired_nan).item())

    @staticmethod
    def _mirror_difference_summary(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> str:
        left_float = left.detach().float()
        right_float = right.detach().float()
        finite_pairs = torch.isfinite(left_float) & torch.isfinite(right_float)
        if finite_pairs.any():
            finite_difference = torch.where(
                finite_pairs,
                (left_float - right_float).abs(),
                0,
            )
            max_abs_diff = finite_difference.max().item()
        else:
            max_abs_diff = float("nan")
        return (
            f"max_finite_abs_diff={max_abs_diff:.6g}, "
            f"left_nan_count={torch.isnan(left_float).sum().item()}, "
            f"right_nan_count={torch.isnan(right_float).sum().item()}"
        )

    def _materialize_mirror_block(
        self,
        tensor: torch.Tensor,
        block_id: int,
    ) -> torch.Tensor:
        """Read one paged block into ordinary NPU storage via index_select."""
        block_start = block_id * self.block_size
        block_slots = torch.arange(
            block_start,
            block_start + self.block_size,
            dtype=torch.int64,
            device=self.device,
        )
        return tensor.view(-1, tensor.shape[-1]).index_select(
            0,
            block_slots,
        )

    def validate_mirror_slot_mapping(
        self,
        slot_mapping: torch.Tensor,
        slot_mapping_cpu: torch.Tensor,
        num_actual_tokens: int,
    ) -> None:
        """Verify the CPU mapping used for mirroring matches the MLA mapping."""
        if self.mode != "mirror":
            raise RuntimeError("Mirror slot validation is only valid in mirror mode.")
        device_slots = slot_mapping[:num_actual_tokens].detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        cpu_slots = slot_mapping_cpu[:num_actual_tokens].detach().to(dtype=torch.int64)
        if torch.equal(device_slots, cpu_slots):
            return

        mismatches = torch.nonzero(device_slots != cpu_slots).flatten()
        first_mismatch = int(mismatches[0])
        raise RuntimeError(
            "sparse_kv_offload mirror slot_mapping mismatch before Full-KV sync: "
            f"mismatch_count={mismatches.numel()}, "
            f"first_index={first_mismatch}, "
            f"device_slot={int(device_slots[first_mismatch])}, "
            f"cpu_slot={int(cpu_slots[first_mismatch])}."
        )

    def validate_mirror_sync(
        self,
        full_kv_cache: tuple[torch.Tensor, ...],
        slot_mapping_cpu: torch.Tensor,
        num_actual_tokens: int,
        updated_blocks: tuple[int, ...],
    ) -> None:
        """Classify Mirror copy failures before Gather can touch any buffer.

        Mirror mode is a correctness diagnostic, so the device synchronizations
        and D2H comparisons here are intentional. If a synchronized retry fixes
        the copy, the first copy raced with the MLA cache producer. If it does
        not, the swapped-memory copy/layout path itself is inconsistent.
        """
        if self.mode != "mirror":
            raise RuntimeError("Mirror sync validation is only valid in mirror mode.")
        if not updated_blocks:
            return

        if self.device.type == "npu":
            torch.npu.synchronize()

        mirror_cache = (self.full_nope_source, self.full_rope_source)
        for cache_name, framework_tensor, mirror_tensor in zip(
            ("nope", "rope"),
            full_kv_cache[:2],
            mirror_cache,
        ):
            for block_id in updated_blocks:
                framework_block = self._materialize_mirror_block(
                    framework_tensor,
                    block_id,
                )
                mirror_block = self._materialize_mirror_block(
                    mirror_tensor,
                    block_id,
                )
                if self._mirror_tensors_match(mirror_block, framework_block):
                    continue

                initial_summary = self._mirror_difference_summary(
                    mirror_block,
                    framework_block,
                )
                self._copy_updated_blocks(
                    full_kv_cache,
                    mirror_cache,
                    slot_mapping_cpu,
                    num_actual_tokens,
                )
                if self.device.type == "npu":
                    torch.npu.synchronize()
                retry_framework_block = self._materialize_mirror_block(
                    framework_tensor,
                    block_id,
                )
                retry_mirror_block = self._materialize_mirror_block(
                    mirror_tensor,
                    block_id,
                )
                retry_matches = self._mirror_tensors_match(
                    retry_mirror_block,
                    retry_framework_block,
                )
                retry_summary = self._mirror_difference_summary(
                    retry_mirror_block,
                    retry_framework_block,
                )
                if retry_matches:
                    raise RuntimeError(
                        "sparse_kv_offload mirror producer visibility race before Gather: "
                        f"cache={cache_name}, block_id={block_id}, "
                        f"initial=({initial_summary}), "
                        "synchronized_retry=matched."
                    )
                raise RuntimeError(
                    "sparse_kv_offload mirror swapped-copy mismatch before Gather "
                    "after synchronized retry: "
                    f"cache={cache_name}, block_id={block_id}, "
                    f"initial=({initial_summary}), "
                    f"retry=({retry_summary})."
                )

    def persist_updated_slots(
        self,
        full_kv_cache: tuple[torch.Tensor, ...],
        slot_mapping_cpu: torch.Tensor,
        num_actual_tokens: int,
    ) -> tuple[int, ...]:
        """Copy touched rows from the shared NPU staging cache to Full Host KV."""
        if self.mode != "host":
            raise RuntimeError("persist_updated_slots is only valid in sparse KV offload host mode.")
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
            raise RuntimeError("restore_prefill_context is only valid in sparse KV offload host mode.")
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
                f"context_len={context_len} exceeds Full KV capacity {self.num_full_blocks * self.block_size}."
            )

        num_context_blocks = (context_len + self.block_size - 1) // self.block_size
        if num_context_blocks > full_block_table_cpu.shape[1]:
            raise ValueError(
                f"Context requires {num_context_blocks} blocks, but the CPU "
                f"block table has only {full_block_table_cpu.shape[1]} entries."
            )

        physical_block_ids = torch.unique(full_block_table_cpu[0, :num_context_blocks].to(torch.int64))
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

    def validate_mirror_selection(
        self,
        *,
        full_kv_cache: tuple[torch.Tensor, ...],
        selection: SparseKVSelection,
        topk_indices: torch.Tensor,
        full_block_table: torch.Tensor,
    ) -> None:
        """Distinguish mirror-copy, Gather, and selected-SFA failures."""
        if self.mode != "mirror":
            raise RuntimeError("Mirror selection validation is only valid in mirror mode.")

        topk_cpu = topk_indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        valid_topk = topk_cpu[topk_cpu >= 0]
        selected_length = int(selection.actual_seq_lengths_kv.detach().cpu()[0])
        if valid_topk.numel() != selected_length:
            raise RuntimeError(
                "sparse_kv_offload mirror Top-K length mismatch: "
                f"valid_topk={valid_topk.numel()}, selected_length={selected_length}."
            )
        if selected_length == 0:
            return

        logical_blocks = torch.div(valid_topk, self.block_size, rounding_mode="floor")
        logical_offsets = valid_topk.remainder(self.block_size)
        full_block_table_cpu = full_block_table.detach().to(device="cpu", dtype=torch.int64)[0]
        full_slots_cpu = full_block_table_cpu.index_select(0, logical_blocks) * self.block_size + logical_offsets

        selected_positions = torch.arange(selected_length, dtype=torch.int64)
        selected_logical_blocks = torch.div(selected_positions, self.block_size, rounding_mode="floor")
        selected_offsets = selected_positions.remainder(self.block_size)
        selected_block_table_cpu = selection.block_table.detach().to(device="cpu", dtype=torch.int64)[0]
        selected_slots_cpu = (
            selected_block_table_cpu.index_select(0, selected_logical_blocks) * self.block_size + selected_offsets
        )

        full_slots = full_slots_cpu.to(device=self.device, non_blocking=False)
        selected_slots = selected_slots_cpu.to(device=self.device, non_blocking=False)
        mirror_cache = (self.full_nope_source, self.full_rope_source)
        for cache_name, framework_tensor, mirror_tensor, selected_tensor in zip(
            ("nope", "rope"),
            full_kv_cache[:2],
            mirror_cache,
            selection.kv_cache,
        ):
            framework_rows = framework_tensor.view(-1, framework_tensor.shape[-1]).index_select(0, full_slots)
            mirror_rows = mirror_tensor.view(-1, mirror_tensor.shape[-1]).index_select(0, full_slots)
            selected_rows = selected_tensor.view(-1, selected_tensor.shape[-1]).index_select(0, selected_slots)

            if not self._mirror_tensors_match(mirror_rows, framework_rows):
                summary = self._mirror_difference_summary(
                    mirror_rows,
                    framework_rows,
                )
                raise RuntimeError(
                    "sparse_kv_offload mirror Full-KV changed during Gather: "
                    f"cache={cache_name}, {summary}."
                )
            if not self._mirror_tensors_match(selected_rows, mirror_rows):
                summary = self._mirror_difference_summary(
                    selected_rows,
                    mirror_rows,
                )
                raise RuntimeError(
                    "sparse_kv_offload mirror Gather selection mismatch: "
                    f"cache={cache_name}, {summary}."
                )

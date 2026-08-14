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
# This file is a part of the vllm-ascend project.
#
"""Real NPU gate for the proposed Mooncake Host Full-KV relay path."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys

from .test_mooncake_transfer_engine_npu import (
    ASCEND_RUNTIME_MARKER,
    BUFFER_BYTES,
    HOST_DESTINATION_SENTINEL,
    HOST_TCP_FORCE_ENV,
    HOST_TCP_RUNTIME_MARKER,
    READY_MARKER,
    RESULT_MARKER,
    _allocate_aligned_host_buffer,
    _extract_json_marker,
    _initialize_coexisting_engines,
    _ProcessOutput,
    _register_host_buffer,
    _require_native_mooncake,
    _stop_process,
)

BLOCK_SIZE = 128
INDEX_TOPK = 2048
NUM_BLOCKS = 4
KV_LORA_RANK = 512
ROPE_HEAD_DIM = 64
KV_DTYPE_BYTES = 2
FULL_NOPE_SHAPE = (NUM_BLOCKS, BLOCK_SIZE, 1, KV_LORA_RANK)
FULL_ROPE_SHAPE = (NUM_BLOCKS, BLOCK_SIZE, 1, ROPE_HEAD_DIM)
NOPE_BLOCK_BYTES = BLOCK_SIZE * KV_LORA_RANK * KV_DTYPE_BYTES
ROPE_BLOCK_BYTES = BLOCK_SIZE * ROPE_HEAD_DIM * KV_DTYPE_BYTES
NOPE_BYTES = NUM_BLOCKS * NOPE_BLOCK_BYTES
ROPE_BYTES = NUM_BLOCKS * ROPE_BLOCK_BYTES
RELAY_PREFIX_BYTES = 4096
RELAY_TENSOR_GUARD_BYTES = 4096
NOPE_OFFSET_BYTES = RELAY_PREFIX_BYTES
NOPE_END_BYTES = NOPE_OFFSET_BYTES + NOPE_BYTES
ROPE_OFFSET_BYTES = NOPE_END_BYTES + RELAY_TENSOR_GUARD_BYTES
ROPE_END_BYTES = ROPE_OFFSET_BYTES + ROPE_BYTES
TOUCHED_BLOCKS = (2, 1)
LOGICAL_BLOCK_START = {2: 0, 1: BLOCK_SIZE}
RELAY_TRANSFER_BYTES = len(TOUCHED_BLOCKS) * (NOPE_BLOCK_BYTES + ROPE_BLOCK_BYTES)
HOST_RELAY_PROCESS_TIMEOUT_SECONDS = 180
HOST_RELAY_STAGE_MARKER = "MOONCAKE_HOST_RELAY_STAGE="
DIRECT_HOST_GATHER_PROCESS_TIMEOUT_SECONDS = 120


def _validate_relay_layout() -> None:
    if ROPE_END_BYTES > BUFFER_BYTES:
        raise RuntimeError(
            "DeepSeek-V3.2 Host relay tensors exceed the registered buffer: "
            f"required={ROPE_END_BYTES}, available={BUFFER_BYTES}."
        )


def _relay_views(torch_module, relay_buffer):
    """Return BF16 Full-KV views backed by one registered Host buffer."""
    _validate_relay_layout()
    full_nope = relay_buffer[NOPE_OFFSET_BYTES:NOPE_END_BYTES].view(torch_module.bfloat16).view(FULL_NOPE_SHAPE)
    full_rope = relay_buffer[ROPE_OFFSET_BYTES:ROPE_END_BYTES].view(torch_module.bfloat16).view(FULL_ROPE_SHAPE)
    return full_nope, full_rope


def _expected_block(torch_module, *, block_id: int, width: int, rope: bool):
    logical_start = LOGICAL_BLOCK_START[block_id]
    values = torch_module.arange(
        logical_start,
        logical_start + BLOCK_SIZE,
        dtype=torch_module.float32,
    )
    if rope:
        values.mul_(0.5)
    return values.to(torch_module.bfloat16).view(BLOCK_SIZE, 1, 1).expand(BLOCK_SIZE, 1, width)


def _allocate_source_staging(torch_module, *, device_index: int):
    device = torch_module.device(f"npu:{device_index}")
    source_nope = torch_module.full(
        FULL_NOPE_SHAPE,
        -1,
        dtype=torch_module.bfloat16,
        device=device,
    )
    source_rope = torch_module.full(
        FULL_ROPE_SHAPE,
        -1,
        dtype=torch_module.bfloat16,
        device=device,
    )
    for block_id in TOUCHED_BLOCKS:
        source_nope[block_id].copy_(
            _expected_block(
                torch_module,
                block_id=block_id,
                width=KV_LORA_RANK,
                rope=False,
            ).to(device),
        )
        source_rope[block_id].copy_(
            _expected_block(
                torch_module,
                block_id=block_id,
                width=ROPE_HEAD_DIM,
                rope=True,
            ).to(device),
        )
    return source_nope, source_rope


def _copy_source_blocks_to_relay(
    torch_module,
    source_kv,
    relay_kv,
) -> dict[str, bool]:
    source_nope, source_rope = source_kv
    relay_nope, relay_rope = relay_kv
    for block_id in TOUCHED_BLOCKS:
        relay_nope[block_id].copy_(source_nope[block_id], non_blocking=False)
        relay_rope[block_id].copy_(source_rope[block_id], non_blocking=False)
    torch_module.npu.synchronize()

    verification = {}
    for block_id in TOUCHED_BLOCKS:
        verification[f"nope_block_{block_id}"] = bool(
            torch_module.equal(relay_nope[block_id], source_nope[block_id].cpu())
        )
        verification[f"rope_block_{block_id}"] = bool(
            torch_module.equal(relay_rope[block_id], source_rope[block_id].cpu())
        )
    return verification


def _relay_guard_verification(torch_module, relay_buffer) -> dict[str, bool]:
    guard_regions = {
        "prefix": (0, NOPE_OFFSET_BYTES),
        "nope_block_0": (NOPE_OFFSET_BYTES, NOPE_OFFSET_BYTES + NOPE_BLOCK_BYTES),
        "nope_block_3": (
            NOPE_OFFSET_BYTES + 3 * NOPE_BLOCK_BYTES,
            NOPE_END_BYTES,
        ),
        "between_tensors": (NOPE_END_BYTES, ROPE_OFFSET_BYTES),
        "rope_block_0": (ROPE_OFFSET_BYTES, ROPE_OFFSET_BYTES + ROPE_BLOCK_BYTES),
        "rope_block_3": (
            ROPE_OFFSET_BYTES + 3 * ROPE_BLOCK_BYTES,
            ROPE_END_BYTES,
        ),
        "suffix": (ROPE_END_BYTES, BUFFER_BYTES),
    }
    return {
        name: bool(
            torch_module.all(
                relay_buffer[start:end] == HOST_DESTINATION_SENTINEL
            ).item()
        )
        for name, (start, end) in guard_regions.items()
    }


def _relay_payload_verification(torch_module, relay_kv) -> dict[str, bool]:
    relay_nope, relay_rope = relay_kv
    verification = {}
    for block_id in TOUCHED_BLOCKS:
        verification[f"nope_block_{block_id}"] = bool(
            torch_module.equal(
                relay_nope[block_id],
                _expected_block(
                    torch_module,
                    block_id=block_id,
                    width=KV_LORA_RANK,
                    rope=False,
                ),
            )
        )
        verification[f"rope_block_{block_id}"] = bool(
            torch_module.equal(
                relay_rope[block_id],
                _expected_block(
                    torch_module,
                    block_id=block_id,
                    width=ROPE_HEAD_DIM,
                    rope=True,
                ),
            )
        )
    return verification


def _relay_transfer_arrays(base_address: int) -> tuple[list[int], list[int]]:
    offsets = []
    lengths = []
    for tensor_offset, block_bytes in (
        (NOPE_OFFSET_BYTES, NOPE_BLOCK_BYTES),
        (ROPE_OFFSET_BYTES, ROPE_BLOCK_BYTES),
    ):
        for block_id in TOUCHED_BLOCKS:
            offsets.append(tensor_offset + block_id * block_bytes)
            lengths.append(block_bytes)
    return [base_address + offset for offset in offsets], lengths


def _allocate_framework_swapped_cache(
    torch_module,
    torch_npu_module,
    shape: tuple[int, ...],
    *,
    device_index: int,
):
    raw_storage = torch_npu_module.empty_with_swapped_memory(
        (math.prod(shape) * KV_DTYPE_BYTES,),
        dtype=torch_module.int8,
        device=torch_module.device(f"npu:{device_index}"),
    )
    tensor = raw_storage.view(torch_module.bfloat16).view(shape)
    return raw_storage, tensor


def _persist_relay_blocks_via_npu_staging(
    torch_module,
    relay_kv,
    staging_kv,
    swapped_kv,
) -> dict[str, bool]:
    """Bridge Host relay data into swapped Full KV via ordinary NPU memory.

    ``empty_with_swapped_memory`` exposes an NPU/SVM alias, not the original
    Host allocation address.  The production connector therefore persists
    ordinary NPU staging into swapped Full KV with basic-slice copies.  Keep
    that supported boundary here instead of issuing a Host-to-swapped copy.
    """
    relay_nope, relay_rope = relay_kv
    staging_nope, staging_rope = staging_kv
    swapped_nope, swapped_rope = swapped_kv
    for block_id in TOUCHED_BLOCKS:
        staging_nope[block_id].copy_(relay_nope[block_id], non_blocking=False)
        staging_rope[block_id].copy_(relay_rope[block_id], non_blocking=False)
    torch_module.npu.synchronize()

    verification = {}
    for block_id in TOUCHED_BLOCKS:
        verification[f"nope_block_{block_id}"] = bool(
            torch_module.equal(
                staging_nope[block_id].cpu(),
                _expected_block(
                    torch_module,
                    block_id=block_id,
                    width=KV_LORA_RANK,
                    rope=False,
                ),
            )
        )
        verification[f"rope_block_{block_id}"] = bool(
            torch_module.equal(
                staging_rope[block_id].cpu(),
                _expected_block(
                    torch_module,
                    block_id=block_id,
                    width=ROPE_HEAD_DIM,
                    rope=True,
                ),
            )
        )
    if not all(verification.values()):
        raise RuntimeError(
            "Decode pinned Host to ordinary NPU staging verification failed: "
            f"{verification}"
        )

    for block_id in TOUCHED_BLOCKS:
        swapped_nope[block_id].copy_(
            staging_nope[block_id],
            non_blocking=False,
        )
        swapped_rope[block_id].copy_(
            staging_rope[block_id],
            non_blocking=False,
        )
    torch_module.npu.synchronize()
    return verification


def _swapped_verification(torch_module, swapped_kv) -> dict[str, bool]:
    materialized = tuple(
        torch_module.empty(
            tuple(tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for tensor in swapped_kv
    )
    for target, source in zip(materialized, swapped_kv):
        target.copy_(source, non_blocking=False)
    torch_module.npu.synchronize()
    materialized_nope = materialized[0].cpu()
    materialized_rope = materialized[1].cpu()

    verification = {}
    for block_id in TOUCHED_BLOCKS:
        verification[f"nope_block_{block_id}"] = bool(
            torch_module.equal(
                materialized_nope[block_id],
                _expected_block(
                    torch_module,
                    block_id=block_id,
                    width=KV_LORA_RANK,
                    rope=False,
                ),
            )
        )
        verification[f"rope_block_{block_id}"] = bool(
            torch_module.equal(
                materialized_rope[block_id],
                _expected_block(
                    torch_module,
                    block_id=block_id,
                    width=ROPE_HEAD_DIM,
                    rope=True,
                ),
            )
        )
    for block_id in (0, 3):
        verification[f"nope_untouched_block_{block_id}"] = bool(
            torch_module.all(materialized_nope[block_id] == -1).item()
        )
        verification[f"rope_untouched_block_{block_id}"] = bool(
            torch_module.all(materialized_rope[block_id] == -1).item()
        )
    return verification


def _gather_verification(torch_module, workspace) -> dict[str, bool]:
    device = workspace.device
    topk_indices = torch_module.full(
        (1, 1, INDEX_TOPK),
        -1,
        dtype=torch_module.int32,
        device=device,
    )
    topk_indices[0, 0, : 2 * BLOCK_SIZE] = torch_module.arange(
        2 * BLOCK_SIZE,
        dtype=torch_module.int32,
        device=device,
    )
    selection = workspace.gather(
        topk_indices=topk_indices,
        full_block_table=torch_module.tensor(
            [[2, 1]],
            dtype=torch_module.int32,
            device=device,
        ),
        full_actual_seq_lengths=torch_module.tensor(
            [2 * BLOCK_SIZE],
            dtype=torch_module.int32,
            device=device,
        ),
        full_query_actual_seq_lengths=torch_module.tensor(
            [1],
            dtype=torch_module.int32,
            device=device,
        ),
    )
    torch_module.npu.synchronize()

    selected_nope = selection.kv_cache[0].view(-1, KV_LORA_RANK)[: 2 * BLOCK_SIZE].float().cpu()
    selected_rope = selection.kv_cache[1].view(-1, ROPE_HEAD_DIM)[: 2 * BLOCK_SIZE].float().cpu()
    logical_tokens = torch_module.arange(2 * BLOCK_SIZE, dtype=torch_module.float32)
    expected_nope = logical_tokens.view(-1, 1).expand(2 * BLOCK_SIZE, KV_LORA_RANK)
    expected_rope = logical_tokens.mul(0.5).view(-1, 1).expand(2 * BLOCK_SIZE, ROPE_HEAD_DIM)
    return {
        "selected_nope": bool(torch_module.equal(selected_nope, expected_nope)),
        "selected_rope": bool(torch_module.equal(selected_rope, expected_rope)),
        "actual_seq_lengths": selection.actual_seq_lengths_kv.cpu().tolist() == [2 * BLOCK_SIZE],
        "sparse_indices": selection.sparse_indices[0, 0, : 2 * BLOCK_SIZE + 1].cpu().tolist()
        == [*range(2 * BLOCK_SIZE), -1],
    }


def _run_direct_pinned_host_gather() -> int:
    """Run Gather with Full KV backed directly by pinned Host memory."""
    import torch
    import torch_npu

    # Importing the extension mounts the custom op on torch_npu.
    import custom_ops  # noqa: F401

    torch.npu.set_device(0)
    host_raw, host_buffer = _allocate_aligned_host_buffer(
        torch,
        fill_value=HOST_DESTINATION_SENTINEL,
        pinned=True,
    )
    full_nope, full_rope = _relay_views(torch, host_buffer)
    for block_id in TOUCHED_BLOCKS:
        full_nope[block_id].copy_(
            _expected_block(
                torch,
                block_id=block_id,
                width=KV_LORA_RANK,
                rope=False,
            )
        )
        full_rope[block_id].copy_(
            _expected_block(
                torch,
                block_id=block_id,
                width=ROPE_HEAD_DIM,
                rope=True,
            )
        )

    device = torch.device("npu:0")
    selection_num_blocks = math.ceil(INDEX_TOPK / BLOCK_SIZE)
    selected_nope = torch.empty(
        (selection_num_blocks, BLOCK_SIZE, KV_LORA_RANK),
        dtype=torch.bfloat16,
        device=device,
    )
    selected_rope = torch.empty(
        (selection_num_blocks, BLOCK_SIZE, ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    selection_block_table = torch.arange(
        selection_num_blocks,
        dtype=torch.int32,
        device=device,
    ).view(1, selection_num_blocks)
    selection_block_status = torch.full(
        (1, 1, INDEX_TOPK + 1),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_indices = torch.full(
        (1, 1, INDEX_TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_indices[0, 0, : 2 * BLOCK_SIZE] = torch.arange(
        2 * BLOCK_SIZE,
        dtype=torch.int32,
        device=device,
    )
    full_block_table = torch.tensor(
        [[2, 1]],
        dtype=torch.int32,
        device=device,
    )
    full_actual_seq = torch.tensor(
        [2 * BLOCK_SIZE],
        dtype=torch.int32,
        device=device,
    )
    full_query_actual_seq = torch.tensor(
        [1],
        dtype=torch.int32,
        device=device,
    )

    gather_op = getattr(torch_npu, "npu_gather_selection_kv_cache", None)
    if not callable(gather_op):
        raise RuntimeError(
            "custom_ops did not register "
            "torch_npu.npu_gather_selection_kv_cache"
        )
    selected_actual_seq = gather_op(
        selection_k_rope=selected_rope,
        selection_kv_cache=selected_nope,
        selection_kv_block_table=selection_block_table,
        selection_kv_block_status=selection_block_status,
        selection_topk_indices=topk_indices,
        full_k_rope=full_rope.squeeze(2),
        full_kv_cache=full_nope.squeeze(2),
        full_kv_block_table=full_block_table,
        full_kv_actual_seq=full_actual_seq,
        full_q_actual_seq=full_query_actual_seq,
        selection_topk_block_size=1,
    )
    torch.npu.synchronize()

    selected_nope_cpu = (
        selected_nope.view(-1, KV_LORA_RANK)[: 2 * BLOCK_SIZE]
        .float()
        .cpu()
    )
    selected_rope_cpu = (
        selected_rope.view(-1, ROPE_HEAD_DIM)[: 2 * BLOCK_SIZE]
        .float()
        .cpu()
    )
    logical_tokens = torch.arange(2 * BLOCK_SIZE, dtype=torch.float32)
    gather = {
        "selected_nope": bool(
            torch.equal(
                selected_nope_cpu,
                logical_tokens.view(-1, 1).expand(
                    2 * BLOCK_SIZE,
                    KV_LORA_RANK,
                ),
            )
        ),
        "selected_rope": bool(
            torch.equal(
                selected_rope_cpu,
                logical_tokens.mul(0.5).view(-1, 1).expand(
                    2 * BLOCK_SIZE,
                    ROPE_HEAD_DIM,
                ),
            )
        ),
        "actual_seq_lengths": selected_actual_seq.cpu().tolist()
        == [2 * BLOCK_SIZE],
    }
    source_guards = _relay_guard_verification(torch, host_buffer)
    source_payload = _relay_payload_verification(
        torch,
        (full_nope, full_rope),
    )
    result = {
        "matches": all(
            (
                *gather.values(),
                *source_guards.values(),
                *source_payload.values(),
            )
        ),
        "host_buffer_pinned": host_buffer.is_pinned(),
        "host_buffer_alignment_mod": host_buffer.data_ptr() % BUFFER_BYTES,
        "full_nope_device": str(full_nope.device),
        "full_rope_device": str(full_rope.device),
        "gather": gather,
        "source_guards": source_guards,
        "source_payload": source_payload,
        "touched_blocks": TOUCHED_BLOCKS,
    }
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
    if not result["matches"]:
        raise RuntimeError(
            "Gather did not read pinned Host Full KV correctly: "
            f"{result}"
        )
    del selected_actual_seq
    del selected_rope
    del selected_nope
    del full_rope
    del full_nope
    del host_buffer
    del host_raw
    gc.collect()
    return 0


def _run_host_relay_receiver() -> int:
    import torch_npu

    from vllm_ascend.attention.sparse_kv_offload import SparseKVOffloadWorkspace

    torch, ascend_engine, host_engine, host = _initialize_coexisting_engines(device_index=1)
    host_raw = host_buffer = None
    full_nope_raw = full_rope_raw = None
    full_nope = full_rope = None
    prefill_nope = prefill_rope = None
    workspace = None
    host_registered = False
    host_unregister_result = 0
    try:
        host_raw, host_buffer = _allocate_aligned_host_buffer(
            torch,
            fill_value=HOST_DESTINATION_SENTINEL,
            pinned=True,
        )
        relay_kv = _relay_views(torch, host_buffer)
        full_nope_raw, full_nope = _allocate_framework_swapped_cache(
            torch,
            torch_npu,
            FULL_NOPE_SHAPE,
            device_index=1,
        )
        full_rope_raw, full_rope = _allocate_framework_swapped_cache(
            torch,
            torch_npu,
            FULL_ROPE_SHAPE,
            device_index=1,
        )
        full_nope.fill_(-1)
        full_rope.fill_(-1)
        prefill_nope = torch.empty(
            FULL_NOPE_SHAPE,
            dtype=torch.bfloat16,
            device="npu:1",
        )
        prefill_rope = torch.empty(
            FULL_ROPE_SHAPE,
            dtype=torch.bfloat16,
            device="npu:1",
        )
        workspace = SparseKVOffloadWorkspace(
            (full_nope, full_rope),
            index_topk=INDEX_TOPK,
            block_size=BLOCK_SIZE,
            mode="host",
            prefill_kv_cache=(prefill_nope, prefill_rope),
        )
        torch.npu.synchronize()
        _register_host_buffer(host_engine, host_buffer)
        host_registered = True
        print(
            READY_MARKER
            + json.dumps(
                {
                    "host": host,
                    "host_port": host_engine.get_rpc_port(),
                    "host_address": host_buffer.data_ptr(),
                    "registration_size": BUFFER_BYTES,
                }
            ),
            flush=True,
        )

        command = sys.stdin.readline().strip()
        if command != "VERIFY":
            raise RuntimeError(f"Host relay receiver expected VERIFY, received {command!r}.")

        relay_guards = _relay_guard_verification(torch, host_buffer)
        relay_payload = _relay_payload_verification(torch, relay_kv)
        print(HOST_RELAY_STAGE_MARKER + "host-received", flush=True)
        npu_staging = _persist_relay_blocks_via_npu_staging(
            torch,
            relay_kv,
            (prefill_nope, prefill_rope),
            (full_nope, full_rope),
        )
        print(HOST_RELAY_STAGE_MARKER + "swapped-persisted", flush=True)
        swapped = _swapped_verification(torch, (full_nope, full_rope))
        gather = _gather_verification(torch, workspace)
        print(HOST_RELAY_STAGE_MARKER + "gather-completed", flush=True)
        matches = all(
            (
                *relay_guards.values(),
                *relay_payload.values(),
                *npu_staging.values(),
                *swapped.values(),
                *gather.values(),
            )
        )
        result = {
            "matches": matches,
            "relay_guards": relay_guards,
            "relay_payload": relay_payload,
            "npu_staging": npu_staging,
            "swapped": swapped,
            "gather": gather,
            "touched_blocks": TOUCHED_BLOCKS,
            "full_nope_shape": FULL_NOPE_SHAPE,
            "full_rope_shape": FULL_ROPE_SHAPE,
        }
        print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
        if not matches:
            raise RuntimeError("Mooncake Host relay to swapped/Gather verification failed.")
        return 0
    finally:
        torch.npu.synchronize()
        if host_registered and host_buffer is not None:
            host_unregister_result = host_engine.unregister_memory(host_buffer.data_ptr())
        del workspace
        del host_engine
        del ascend_engine
        del prefill_rope
        del prefill_nope
        del full_rope
        del full_nope
        del full_rope_raw
        del full_nope_raw
        del host_buffer
        del host_raw
        gc.collect()
        if host_unregister_result != 0:
            raise RuntimeError(
                "Host relay receiver memory unregistration failed: "
                f"result={host_unregister_result}"
            )


def _run_host_relay_sender(
    *,
    target_host: str,
    target_host_port: int,
    target_host_address: int,
) -> int:
    torch, ascend_engine, host_engine, _ = _initialize_coexisting_engines(device_index=0)
    host_raw = host_buffer = None
    source_nope = source_rope = None
    host_registered = False
    host_unregister_result = 0
    try:
        source_nope, source_rope = _allocate_source_staging(
            torch,
            device_index=0,
        )
        host_raw, host_buffer = _allocate_aligned_host_buffer(
            torch,
            fill_value=HOST_DESTINATION_SENTINEL,
            pinned=True,
        )
        relay_kv = _relay_views(torch, host_buffer)
        local_copy = _copy_source_blocks_to_relay(
            torch,
            (source_nope, source_rope),
            relay_kv,
        )
        source_guards = _relay_guard_verification(torch, host_buffer)
        if not all((*local_copy.values(), *source_guards.values())):
            raise RuntimeError(
                "Prefill NPU to pinned Host relay copy failed: "
                f"local_copy={local_copy}, guards={source_guards}"
            )

        _register_host_buffer(host_engine, host_buffer)
        host_registered = True
        source_addresses, lengths = _relay_transfer_arrays(host_buffer.data_ptr())
        target_addresses, target_lengths = _relay_transfer_arrays(target_host_address)
        if target_lengths != lengths:
            raise RuntimeError("Host relay source and target transfer layouts differ.")
        session = f"{target_host}:{target_host_port}"
        transfer_result = host_engine.batch_transfer_sync_write(
            session,
            source_addresses,
            target_addresses,
            lengths,
        )
        if transfer_result < 0:
            raise RuntimeError(
                "Mooncake Host relay transfer failed: "
                f"session={session}, result={transfer_result}"
            )
        result = {
            "transfer_result": transfer_result,
            "local_copy": local_copy,
            "source_guards": source_guards,
            "transfer_ranges": len(lengths),
            "transfer_bytes": sum(lengths),
        }
        print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
        return 0
    finally:
        torch.npu.synchronize()
        if host_registered and host_buffer is not None:
            host_unregister_result = host_engine.unregister_memory(host_buffer.data_ptr())
        del host_engine
        del ascend_engine
        del source_rope
        del source_nope
        del host_buffer
        del host_raw
        gc.collect()
        if host_unregister_result != 0:
            raise RuntimeError(
                "Host relay sender memory unregistration failed: "
                f"result={host_unregister_result}"
            )


def test_mooncake_host_relay_to_swapped_gather():
    """Verify the complete BF16 Full-KV Host relay through real Gather."""
    _require_native_mooncake()
    module_name = "tests.ut.distributed.kv_transfer.a3_2.test_mooncake_host_relay_npu"
    command = [sys.executable, "-m", module_name]
    child_environment = os.environ.copy()
    child_environment.pop(HOST_TCP_FORCE_ENV, None)
    receiver = subprocess.Popen(
        [*command, "--role", "receiver"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_environment,
    )
    if receiver.stdin is None or receiver.stdout is None:
        receiver.terminate()
        receiver.wait(timeout=5)
        raise RuntimeError("Failed to create Host relay receiver control pipes.")

    receiver_output = _ProcessOutput(receiver.stdout)
    try:
        ready = receiver_output.wait_for_marker(
            READY_MARKER,
            timeout=HOST_RELAY_PROCESS_TIMEOUT_SECONDS,
        )
        sender = subprocess.run(
            [
                *command,
                "--role",
                "sender",
                "--target-host",
                str(ready["host"]),
                "--target-host-port",
                str(ready["host_port"]),
                "--target-host-address",
                str(ready["host_address"]),
            ],
            capture_output=True,
            text=True,
            timeout=HOST_RELAY_PROCESS_TIMEOUT_SECONDS,
            env=child_environment,
        )
        sender_output = sender.stdout + sender.stderr
        assert sender.returncode == 0, (
            "Mooncake Host relay sender failed.\n"
            f"stdout:\n{sender.stdout}\n"
            f"stderr:\n{sender.stderr}"
        )
        sender_result = _extract_json_marker(sender.stdout, RESULT_MARKER)
        assert sender_result["transfer_result"] == 0
        assert sender_result["transfer_ranges"] == 4
        assert sender_result["transfer_bytes"] == RELAY_TRANSFER_BYTES
        assert all(sender_result["local_copy"].values())
        assert all(sender_result["source_guards"].values())

        receiver.stdin.write("VERIFY\n")
        receiver.stdin.flush()
        try:
            result = receiver_output.wait_for_marker(
                RESULT_MARKER,
                timeout=HOST_RELAY_PROCESS_TIMEOUT_SECONDS,
            )
        except RuntimeError as exc:
            receiver.wait(timeout=5)
            receiver_output.join()
            raise RuntimeError(
                "Mooncake Host relay receiver exited before verification "
                f"completed: returncode={receiver.returncode}.\n"
                f"output:\n{receiver_output.output}"
            ) from exc
        receiver.wait(timeout=HOST_RELAY_PROCESS_TIMEOUT_SECONDS)
        receiver_output.join()
        assert receiver.returncode == 0, (
            "Mooncake Host relay receiver failed.\n"
            f"output:\n{receiver_output.output}"
        )
        assert result["matches"], (
            "Mooncake Host relay receiver observed corrupted data.\n"
            f"result={result}\noutput:\n{receiver_output.output}"
        )
        assert ASCEND_RUNTIME_MARKER in receiver_output.output, (
            "Host relay receiver did not initialize an Ascend transport.\n"
            f"output:\n{receiver_output.output}"
        )
        assert HOST_TCP_RUNTIME_MARKER in receiver_output.output, (
            "Host relay receiver did not initialize a TCP-only transport.\n"
            f"output:\n{receiver_output.output}"
        )
        assert ASCEND_RUNTIME_MARKER in sender_output, (
            "Host relay sender did not initialize an Ascend transport.\n"
            f"output:\n{sender_output}"
        )
        assert HOST_TCP_RUNTIME_MARKER in sender_output, (
            "Host relay sender did not initialize a TCP-only transport.\n"
            f"output:\n{sender_output}"
        )
        print(
            "Mooncake BF16 Host relay to swapped Full-KV/Gather verified: "
            + json.dumps(
                {
                    "sender": sender_result,
                    "receiver": result,
                },
                sort_keys=True,
            )
        )
    finally:
        if receiver.poll() is None:
            _stop_process(receiver, receiver_output)
        else:
            receiver_output.join()


def test_gather_reads_pinned_host_full_kv_directly():
    """Decide whether Decode can Gather without a Full-KV NPU bridge."""
    _require_native_mooncake()
    module_name = (
        "tests.ut.distributed.kv_transfer.a3_2."
        "test_mooncake_host_relay_npu"
    )
    child_environment = os.environ.copy()
    child_environment.pop(HOST_TCP_FORCE_ENV, None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            module_name,
            "--role",
            "direct_host_gather",
        ],
        capture_output=True,
        text=True,
        timeout=DIRECT_HOST_GATHER_PROCESS_TIMEOUT_SECONDS,
        env=child_environment,
    )
    assert completed.returncode == 0, (
        "Pinned Host Full-KV direct Gather failed. This means the current "
        "runtime cannot remove the Decode Full-KV NPU bridge through this "
        "Tensor interface.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    result = _extract_json_marker(completed.stdout, RESULT_MARKER)
    assert result["matches"], result
    assert result["host_buffer_pinned"]
    assert result["host_buffer_alignment_mod"] == 0
    assert result["full_nope_device"] == "cpu"
    assert result["full_rope_device"] == "cpu"
    assert all(result["gather"].values())
    assert all(result["source_guards"].values())
    assert all(result["source_payload"].values())
    print(
        "Pinned Host Full-KV direct Gather verified: "
        + json.dumps(result, sort_keys=True)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        choices=("receiver", "sender", "direct_host_gather"),
        required=True,
    )
    parser.add_argument("--target-host")
    parser.add_argument("--target-host-port", type=int)
    parser.add_argument("--target-host-address", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.role == "receiver":
        raise SystemExit(_run_host_relay_receiver())
    if args.role == "direct_host_gather":
        raise SystemExit(_run_direct_pinned_host_gather())
    if (
        args.target_host is None
        or args.target_host_port is None
        or args.target_host_address is None
    ):
        raise SystemExit(
            "Host relay sender requires --target-host, --target-host-port, "
            "and --target-host-address"
        )
    raise SystemExit(
        _run_host_relay_sender(
            target_host=args.target_host,
            target_host_port=args.target_host_port,
            target_host_address=args.target_host_address,
        )
    )

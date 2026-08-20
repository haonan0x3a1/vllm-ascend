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
#
"""G1 hardware gate for MemFabric BM Host Full-KV -> real Gather.

This probe intentionally runs in a child process.  Constructing a torch NPU
Tensor from an externally owned pointer crosses a private runtime boundary;
an invalid address or ownership contract may abort the process instead of
raising a Python exception.  The parent test turns any such failure into a
normal, evidence-rich pytest failure.

The gate covers the local contracts needed before implementing a remote
Host-to-Host connector:

* one MemFabric BM DRAM allocation exposes a Host GVA plus LOCAL_HOST and
  LOCAL_DEVICE address translations;
* the LOCAL_DEVICE view can be wrapped as raw int8 torch NPU storage and
  reshaped like model_runner_v1.py;
* SparseKVOffloadWorkspace.persist_updated_slots can perform copy 1 from an
  ordinary NPU staging cache into the Host-backed allocation;
* a MemFabric H2G write to the same Host allocation is visible to the real
  GatherSelectionKvCache operator through the NPU Tensor alias;
* CPU-side initialization and verification use MemFabric H2G/G2H instead of
  directly dereferencing an externally managed LOCAL_HOST VA;
* guard regions and teardown remain valid.

This is G1 only.  It uses a single-rank SDMA BM pool and does not prove remote
HOST_RDMA, cross-host completion, or production stream ordering (G2).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

import pytest

ENABLE_ENV = "VLLM_ASCEND_RUN_MEMFABRIC_BM_GATHER_GATE"
RESULT_MARKER = "MEMFABRIC_BM_GATHER_GATE_RESULT="
STAGE_MARKER = "MEMFABRIC_BM_GATHER_GATE_STAGE="
PROCESS_TIMEOUT_SECONDS = 180

BLOCK_SIZE = 128
INDEX_TOPK = 2048
NUM_BLOCKS = 4
KV_LORA_RANK = 512
ROPE_HEAD_DIM = 64
KV_DTYPE_BYTES = 2
FULL_NOPE_SHAPE = (NUM_BLOCKS, BLOCK_SIZE, 1, KV_LORA_RANK)
FULL_ROPE_SHAPE = (NUM_BLOCKS, BLOCK_SIZE, 1, ROPE_HEAD_DIM)
NOPE_BYTES = math.prod(FULL_NOPE_SHAPE) * KV_DTYPE_BYTES
ROPE_BYTES = math.prod(FULL_ROPE_SHAPE) * KV_DTYPE_BYTES

POOL_ALIGNMENT_BYTES = 2 * 1024 * 1024
GUARD_SAMPLE_BYTES = 4096
GUARD_SENTINEL = 0xA5
NOPE_OFFSET_BYTES = POOL_ALIGNMENT_BYTES
ROPE_OFFSET_BYTES = (
    (NOPE_OFFSET_BYTES + NOPE_BYTES + POOL_ALIGNMENT_BYTES - 1) // POOL_ALIGNMENT_BYTES * POOL_ALIGNMENT_BYTES
)
REQUIRED_POOL_BYTES = (
    (ROPE_OFFSET_BYTES + ROPE_BYTES + POOL_ALIGNMENT_BYTES - 1) // POOL_ALIGNMENT_BYTES * POOL_ALIGNMENT_BYTES
)
DEFAULT_POOL_BYTES = 1024 * 1024 * 1024

TOUCHED_BLOCKS = (2, 1)
LOGICAL_BLOCK_START = {2: 0, 1: BLOCK_SIZE}


def _distribution_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "unknown"


def _find_loopback_store_url() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _extract_result(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line.removeprefix(RESULT_MARKER))
    raise AssertionError(f"Missing {RESULT_MARKER!r} in child output:\n{output}")


def _require_callable(owner: Any, name: str):
    value = getattr(owner, name, None)
    if not callable(value):
        raise RuntimeError(f"Required callable {type(owner).__name__}.{name} is unavailable")
    return value


def _construct_raw_npu_alias(
    torch_module,
    torch_npu_module,
    *,
    data_ptr: int,
    nbytes: int,
    device,
):
    """Wrap externally owned BM memory as a non-owning-looking raw NPU Tensor.

    Whether the runtime truly treats this storage as externally owned is part
    of the gate: all views are destroyed before the BM handle is released, and
    a teardown abort/double-free therefore fails the parent test.
    """
    if data_ptr <= 0:
        raise ValueError(f"data_ptr must be positive, got {data_ptr}")
    construct_storage = _require_callable(
        torch_npu_module._C,
        "_construct_storage_from_data_pointer",
    )
    construct_tensor = _require_callable(
        torch_npu_module._C,
        "_construct_NPU_Tensor_From_Storage_And_Metadata",
    )
    metadata = {
        "data_ptr": data_ptr,
        "device": device,
        "nbytes": nbytes,
        "dtype": torch_module.int8,
        "size": (nbytes,),
        "stride": (1,),
        "storage_offset": 0,
    }
    storage = construct_storage(data_ptr, device, nbytes)
    return construct_tensor(metadata, storage)


def _reshape_raw_cache(torch_module, raw_tensor, shape, dtype):
    """Match the raw-int8 -> dtype/as_strided model-runner contract."""
    typed = raw_tensor.view(dtype)
    stride = torch_module.empty(shape, dtype=dtype).stride()
    return torch_module.as_strided(
        typed,
        size=shape,
        stride=stride,
        storage_offset=0,
    )


def _expected_block(
    torch_module,
    *,
    block_id: int,
    width: int,
    rope: bool,
    generation: int,
):
    logical_start = LOGICAL_BLOCK_START[block_id]
    values = torch_module.arange(
        logical_start,
        logical_start + BLOCK_SIZE,
        dtype=torch_module.float32,
    )
    values.add_(generation * 1024)
    if rope:
        values.add_(4096)
    return values.to(torch_module.bfloat16).view(BLOCK_SIZE, 1, 1).expand(BLOCK_SIZE, 1, width).contiguous()


def _fill_npu_staging(torch_module, staging_kv, *, generation: int) -> None:
    staging_nope, staging_rope = staging_kv
    staging_nope.fill_(-1)
    staging_rope.fill_(-1)
    for block_id in TOUCHED_BLOCKS:
        staging_nope[block_id].copy_(
            _expected_block(
                torch_module,
                block_id=block_id,
                width=KV_LORA_RANK,
                rope=False,
                generation=generation,
            ).to(staging_nope.device),
        )
        staging_rope[block_id].copy_(
            _expected_block(
                torch_module,
                block_id=block_id,
                width=ROPE_HEAD_DIM,
                rope=True,
                generation=generation,
            ).to(staging_rope.device),
        )


def _physical_slots(torch_module):
    return torch_module.cat(
        tuple(
            torch_module.arange(
                block_id * BLOCK_SIZE,
                (block_id + 1) * BLOCK_SIZE,
                dtype=torch_module.int64,
            )
            for block_id in TOUCHED_BLOCKS
        )
    )


def _copy_type(bm_module, name: str):
    value = getattr(bm_module.BmCopyType, name, None)
    if value is None:
        raise RuntimeError(f"MemFabric BM copy direction {name} is unavailable")
    return value


def _wait_bm(handle) -> None:
    wait = _require_callable(handle, "wait")
    result = wait()
    if result != 0:
        raise RuntimeError(f"MemFabric BM wait failed: result={result}")


def _copy_global_to_cpu(
    torch_module,
    bm_module,
    handle,
    *,
    source_gva: int,
    shape: tuple[int, ...],
):
    target = torch_module.empty(shape, dtype=torch_module.bfloat16)
    nbytes = target.numel() * target.element_size()
    result = handle.copy_data(
        source_gva,
        target.data_ptr(),
        nbytes,
        _copy_type(bm_module, "G2H"),
        0,
    )
    if result != 0:
        raise RuntimeError(f"MemFabric BM G2H failed: result={result}")
    _wait_bm(handle)
    return target


def _initialize_guard_region(
    torch_module,
    bm_module,
    handle,
    *,
    host_gva: int,
):
    source = torch_module.full(
        (REQUIRED_POOL_BYTES,),
        GUARD_SENTINEL,
        dtype=torch_module.uint8,
    )
    result = handle.copy_data(
        source.data_ptr(),
        host_gva,
        source.numel(),
        _copy_type(bm_module, "H2G"),
        0,
    )
    if result != 0:
        raise RuntimeError(f"MemFabric BM guard H2G initialization failed: result={result}")
    _wait_bm(handle)
    return source


def _verify_copy_one(
    torch_module,
    bm_module,
    handle,
    *,
    host_gva: int,
    generation: int,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for block_id in TOUCHED_BLOCKS:
        actual_nope = _copy_global_to_cpu(
            torch_module,
            bm_module,
            handle,
            source_gva=host_gva + NOPE_OFFSET_BYTES + block_id * NOPE_BYTES // NUM_BLOCKS,
            shape=(BLOCK_SIZE, 1, KV_LORA_RANK),
        )
        actual_rope = _copy_global_to_cpu(
            torch_module,
            bm_module,
            handle,
            source_gva=host_gva + ROPE_OFFSET_BYTES + block_id * ROPE_BYTES // NUM_BLOCKS,
            shape=(BLOCK_SIZE, 1, ROPE_HEAD_DIM),
        )
        checks[f"nope_block_{block_id}"] = torch_module.equal(
            actual_nope,
            _expected_block(
                torch_module,
                block_id=block_id,
                width=KV_LORA_RANK,
                rope=False,
                generation=generation,
            ),
        )
        checks[f"rope_block_{block_id}"] = torch_module.equal(
            actual_rope,
            _expected_block(
                torch_module,
                block_id=block_id,
                width=ROPE_HEAD_DIM,
                rope=True,
                generation=generation,
            ),
        )
    return checks


def _populate_via_memfabric_h2g(
    torch_module,
    bm_module,
    handle,
    *,
    host_gva: int,
    generation: int,
) -> list[Any]:
    """Simulate the target Host arrival using BM-owned Host pool writes."""
    owners: list[Any] = []
    for block_id in TOUCHED_BLOCKS:
        for tensor_offset, width, rope, tensor_bytes in (
            (NOPE_OFFSET_BYTES, KV_LORA_RANK, False, NOPE_BYTES),
            (ROPE_OFFSET_BYTES, ROPE_HEAD_DIM, True, ROPE_BYTES),
        ):
            source = _expected_block(
                torch_module,
                block_id=block_id,
                width=width,
                rope=rope,
                generation=generation,
            )
            owners.append(source)
            result = handle.copy_data(
                source.data_ptr(),
                host_gva + tensor_offset + block_id * tensor_bytes // NUM_BLOCKS,
                source.numel() * source.element_size(),
                _copy_type(bm_module, "H2G"),
                0,
            )
            if result != 0:
                raise RuntimeError(
                    f"MemFabric BM H2G population failed: block={block_id}, rope={rope}, result={result}"
                )
    _wait_bm(handle)
    return owners


def _guard_samples(host_gva: int) -> dict[str, tuple[int, int]]:
    nope_block_bytes = NOPE_BYTES // NUM_BLOCKS
    rope_block_bytes = ROPE_BYTES // NUM_BLOCKS
    return {
        "prefix": (host_gva, GUARD_SAMPLE_BYTES),
        "before_nope": (
            host_gva + NOPE_OFFSET_BYTES - GUARD_SAMPLE_BYTES,
            GUARD_SAMPLE_BYTES,
        ),
        "between_nope_rope": (
            host_gva + NOPE_OFFSET_BYTES + NOPE_BYTES,
            GUARD_SAMPLE_BYTES,
        ),
        "nope_block_0": (
            host_gva + NOPE_OFFSET_BYTES,
            nope_block_bytes,
        ),
        "nope_block_3": (
            host_gva + NOPE_OFFSET_BYTES + 3 * nope_block_bytes,
            nope_block_bytes,
        ),
        "rope_block_0": (
            host_gva + ROPE_OFFSET_BYTES,
            rope_block_bytes,
        ),
        "rope_block_3": (
            host_gva + ROPE_OFFSET_BYTES + 3 * rope_block_bytes,
            rope_block_bytes,
        ),
        "suffix": (
            host_gva + ROPE_OFFSET_BYTES + ROPE_BYTES,
            GUARD_SAMPLE_BYTES,
        ),
    }


def _verify_guards(torch_module, bm_module, handle, *, host_gva: int) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, (address, size) in _guard_samples(host_gva).items():
        target = torch_module.empty((size,), dtype=torch_module.uint8)
        result = handle.copy_data(
            address,
            target.data_ptr(),
            size,
            _copy_type(bm_module, "G2H"),
            0,
        )
        if result != 0:
            raise RuntimeError(f"MemFabric BM guard G2H verification failed: name={name}, result={result}")
        _wait_bm(handle)
        checks[name] = bool(torch_module.all(target == GUARD_SENTINEL).item())
    return checks


def _gather_and_verify(
    torch_module,
    workspace,
    *,
    device,
    generation: int,
) -> dict[str, bool]:
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

    expected_nope = torch_module.cat(
        tuple(
            _expected_block(
                torch_module,
                block_id=block_id,
                width=KV_LORA_RANK,
                rope=False,
                generation=generation,
            )
            for block_id in TOUCHED_BLOCKS
        )
    ).view(2 * BLOCK_SIZE, KV_LORA_RANK)
    expected_rope = torch_module.cat(
        tuple(
            _expected_block(
                torch_module,
                block_id=block_id,
                width=ROPE_HEAD_DIM,
                rope=True,
                generation=generation,
            )
            for block_id in TOUCHED_BLOCKS
        )
    ).view(2 * BLOCK_SIZE, ROPE_HEAD_DIM)

    actual_nope = selection.kv_cache[0].view(-1, KV_LORA_RANK)[: 2 * BLOCK_SIZE].cpu()
    actual_rope = selection.kv_cache[1].view(-1, ROPE_HEAD_DIM)[: 2 * BLOCK_SIZE].cpu()
    actual_seq = selection.actual_seq_lengths_kv.cpu().tolist()
    return {
        "actual_seq": actual_seq == [2 * BLOCK_SIZE],
        "nope": torch_module.equal(actual_nope, expected_nope),
        "rope": torch_module.equal(actual_rope, expected_rope),
    }


def _run_swapped_control(
    torch_module,
    torch_npu_module,
    workspace_cls,
    *,
    device,
) -> dict[str, bool]:
    """Prove the installed Gather stack works before testing the BM alias."""

    def allocate(shape):
        nbytes = math.prod(shape) * KV_DTYPE_BYTES
        raw = torch_npu_module.empty_with_swapped_memory(
            (nbytes,),
            dtype=torch_module.int8,
            device=device,
        )
        return raw, _reshape_raw_cache(
            torch_module,
            raw,
            shape,
            torch_module.bfloat16,
        )

    raw_nope, full_nope = allocate(FULL_NOPE_SHAPE)
    raw_rope, full_rope = allocate(FULL_ROPE_SHAPE)
    staging_nope = torch_module.empty(
        FULL_NOPE_SHAPE,
        dtype=torch_module.bfloat16,
        device=device,
    )
    staging_rope = torch_module.empty(
        FULL_ROPE_SHAPE,
        dtype=torch_module.bfloat16,
        device=device,
    )
    workspace = workspace_cls(
        (full_nope, full_rope),
        index_topk=INDEX_TOPK,
        block_size=BLOCK_SIZE,
        mode="host",
        prefill_kv_cache=(staging_nope, staging_rope),
    )
    generation = 0
    _fill_npu_staging(
        torch_module,
        (staging_nope, staging_rope),
        generation=generation,
    )
    updated_blocks = workspace.persist_updated_slots(
        (full_nope, full_rope),
        _physical_slots(torch_module),
        num_actual_tokens=2 * BLOCK_SIZE,
    )
    torch_module.npu.synchronize()
    checks = _gather_and_verify(
        torch_module,
        workspace,
        device=device,
        generation=generation,
    )
    checks["updated_blocks"] = set(updated_blocks) == set(TOUCHED_BLOCKS)

    # Keep explicit references until all queued work is complete, then exercise
    # the normal framework-owned swapped teardown before the candidate gate.
    del workspace
    del staging_nope, staging_rope
    del full_nope, full_rope
    del raw_nope, raw_rope
    gc.collect()
    torch_module.npu.synchronize()
    return checks


def _run_child(args: argparse.Namespace) -> int:
    # Import order matters for the custom operator and Ascend runtime.
    # isort: off
    import torch
    import torch_npu

    import memfabric_hybrid as mf
    from memfabric_hybrid import bm

    from vllm_ascend.attention.sparse_kv_offload import SparseKVOffloadWorkspace
    from vllm_ascend.utils import enable_custom_op
    # isort: on

    enable_custom_op()

    if args.pool_bytes < REQUIRED_POOL_BYTES:
        raise ValueError(f"pool-bytes must be at least {REQUIRED_POOL_BYTES}, got {args.pool_bytes}")

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch.npu.synchronize()

    mf_initialized = False
    bm_initialized = False
    handle = None
    joined = False
    raw_nope = None
    raw_rope = None
    full_nope = None
    full_rope = None
    staging_nope = None
    staging_rope = None
    workspace = None
    owners: dict[str, Any] = {}
    result: dict[str, Any] = {
        "status": "failed",
        "gate": "G1",
        "transport_scope": "single-rank SDMA; HOST_RDMA is not tested",
        "memfabric_version": _distribution_version("memfabric_hybrid"),
        "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
    }
    store_url = args.store_url or _find_loopback_store_url()
    cleanup_error: BaseException | None = None

    try:
        control_checks = _run_swapped_control(
            torch,
            torch_npu,
            SparseKVOffloadWorkspace,
            device=device,
        )
        result["swapped_control"] = control_checks
        if not all(control_checks.values()):
            raise RuntimeError(
                "Framework swapped-memory Gather control failed; the runtime "
                f"cannot provide a valid comparison for G1: {control_checks}"
            )
        print(
            STAGE_MARKER
            + json.dumps(
                {"stage": "swapped_control", "status": "pass"},
                sort_keys=True,
            ),
            flush=True,
        )

        mf.set_log_level(args.memfabric_log_level)
        mf_result = mf.initialize()
        if mf_result != 0:
            raise RuntimeError(f"memfabric_hybrid.initialize failed: result={mf_result}")
        mf_initialized = True

        config = bm.BmConfig()
        config.rank_id = 0
        config.start_store = True
        config.unified_address_space = True
        bm_result = bm.initialize(store_url, 1, args.device, config)
        if bm_result != 0:
            raise RuntimeError(f"MemFabric BM initialize failed: result={bm_result}")
        bm_initialized = True

        handle = bm.create2(
            id=args.bm_id,
            local_dram_size=args.pool_bytes,
            max_dram_size=args.pool_bytes,
            data_op_type=bm.BmDataOpType.SDMA,
        )
        if handle is None:
            raise RuntimeError("MemFabric BM create2 returned None")
        join_result = handle.join()
        if join_result != 0:
            raise RuntimeError(f"MemFabric BM join failed: result={join_result}")
        joined = True

        host_gva = handle.peer_rank_ptr(0, bm.BmMemType.HOST)
        host_va = handle.gva_to_va(host_gva, bm.BmMemType.LOCAL_HOST)
        device_va = handle.gva_to_va(host_gva, bm.BmMemType.LOCAL_DEVICE)
        if not host_gva or not host_va or not device_va:
            raise RuntimeError(
                "MemFabric BM did not expose all required addresses: "
                f"gva=0x{host_gva:x}, host_va=0x{host_va:x}, "
                f"device_va=0x{device_va:x}"
            )
        result["addresses"] = {
            "host_gva": hex(host_gva),
            "local_host_va": hex(host_va),
            "local_device_va": hex(device_va),
            "host_va_equals_device_va": host_va == device_va,
            "nope_device_alignment_mod": (device_va + NOPE_OFFSET_BYTES) % POOL_ALIGNMENT_BYTES,
            "rope_device_alignment_mod": (device_va + ROPE_OFFSET_BYTES) % POOL_ALIGNMENT_BYTES,
        }

        owners["guard_source"] = _initialize_guard_region(
            torch,
            bm,
            handle,
            host_gva=host_gva,
        )

        raw_nope = _construct_raw_npu_alias(
            torch,
            torch_npu,
            data_ptr=device_va + NOPE_OFFSET_BYTES,
            nbytes=NOPE_BYTES,
            device=device,
        )
        raw_rope = _construct_raw_npu_alias(
            torch,
            torch_npu,
            data_ptr=device_va + ROPE_OFFSET_BYTES,
            nbytes=ROPE_BYTES,
            device=device,
        )
        full_nope = _reshape_raw_cache(
            torch,
            raw_nope,
            FULL_NOPE_SHAPE,
            torch.bfloat16,
        )
        full_rope = _reshape_raw_cache(
            torch,
            raw_rope,
            FULL_ROPE_SHAPE,
            torch.bfloat16,
        )
        owners.update(
            raw_nope=raw_nope,
            raw_rope=raw_rope,
            full_nope=full_nope,
            full_rope=full_rope,
        )
        result["tensor_contract"] = {
            "nope_device": str(full_nope.device),
            "rope_device": str(full_rope.device),
            "nope_shape": list(full_nope.shape),
            "rope_shape": list(full_rope.shape),
            "nope_contiguous": full_nope.is_contiguous(),
            "rope_contiguous": full_rope.is_contiguous(),
            "nope_squeezed_contiguous": full_nope.squeeze(2).is_contiguous(),
            "rope_squeezed_contiguous": full_rope.squeeze(2).is_contiguous(),
            "nope_data_ptr_matches": full_nope.data_ptr() == device_va + NOPE_OFFSET_BYTES,
            "rope_data_ptr_matches": full_rope.data_ptr() == device_va + ROPE_OFFSET_BYTES,
        }
        if not all(
            value
            for key, value in result["tensor_contract"].items()
            if key.endswith("contiguous") or key.endswith("matches")
        ):
            raise RuntimeError(
                f"Constructed MemFabric NPU Tensor alias violates the layout contract: {result['tensor_contract']}"
            )

        staging_nope = torch.empty(FULL_NOPE_SHAPE, dtype=torch.bfloat16, device=device)
        staging_rope = torch.empty(FULL_ROPE_SHAPE, dtype=torch.bfloat16, device=device)
        owners.update(staging_nope=staging_nope, staging_rope=staging_rope)

        workspace = SparseKVOffloadWorkspace(
            (full_nope, full_rope),
            index_topk=INDEX_TOPK,
            block_size=BLOCK_SIZE,
            mode="host",
            prefill_kv_cache=(staging_nope, staging_rope),
        )
        owners["workspace"] = workspace

        copy_one_generation = 1
        _fill_npu_staging(
            torch,
            (staging_nope, staging_rope),
            generation=copy_one_generation,
        )
        updated_blocks = workspace.persist_updated_slots(
            (full_nope, full_rope),
            _physical_slots(torch),
            num_actual_tokens=2 * BLOCK_SIZE,
        )
        torch.npu.synchronize()
        copy_one_checks = _verify_copy_one(
            torch,
            bm,
            handle,
            host_gva=host_gva,
            generation=copy_one_generation,
        )
        result["copy_one"] = {
            "updated_blocks": list(updated_blocks),
            "payload": copy_one_checks,
        }
        if set(updated_blocks) != set(TOUCHED_BLOCKS) or not all(copy_one_checks.values()):
            raise RuntimeError(f"copy 1 NPU-to-Host verification failed: {result['copy_one']}")

        # Use a distinct generation so Gather cannot pass by observing stale
        # data from copy 1.  This is a local BM H2G population, not remote G2G.
        host_generation = 2
        owners["host_payloads"] = _populate_via_memfabric_h2g(
            torch,
            bm,
            handle,
            host_gva=host_gva,
            generation=host_generation,
        )
        torch.npu.synchronize()

        host_population_checks = _verify_copy_one(
            torch,
            bm,
            handle,
            host_gva=host_gva,
            generation=host_generation,
        )
        gather_checks = _gather_and_verify(
            torch,
            workspace,
            device=device,
            generation=host_generation,
        )
        guard_checks = _verify_guards(
            torch,
            bm,
            handle,
            host_gva=host_gva,
        )
        result["host_population"] = host_population_checks
        result["gather"] = gather_checks
        result["guards"] = guard_checks
        if not all(host_population_checks.values()):
            raise RuntimeError(f"MemFabric H2G population verification failed: {host_population_checks}")
        if not all(gather_checks.values()):
            raise RuntimeError(f"Gather did not observe the MemFabric Host population: {gather_checks}")
        if not all(guard_checks.values()):
            raise RuntimeError(f"MemFabric BM guard verification failed: {guard_checks}")

        result["status"] = "pass"
    finally:
        # Drop every Tensor view while BM still owns the underlying allocation.
        # A runtime abort or double-free here is intentionally a G1 failure.
        owners.clear()
        workspace = None
        staging_nope = None
        staging_rope = None
        full_nope = None
        full_rope = None
        raw_nope = None
        raw_rope = None
        gc.collect()
        try:
            if "torch" in locals():
                torch.npu.synchronize()
            if handle is not None and joined:
                leave_result = handle.leave()
                if leave_result != 0:
                    raise RuntimeError(f"MemFabric BM leave failed: result={leave_result}")
                joined = False
            if handle is not None:
                handle.destroy()
                handle = None
            if bm_initialized:
                bm.uninitialize(0)
                bm_initialized = False
            if mf_initialized:
                last_error = mf.get_last_err_msg()
                if last_error:
                    raise RuntimeError(f"MemFabric reported an error before uninitialize: {last_error}")
                mf.uninitialize()
                mf_initialized = False
        except BaseException as exc:
            cleanup_error = exc

    if cleanup_error is not None:
        raise cleanup_error
    result["cleanup"] = True
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
    return 0


@pytest.mark.skipif(
    os.getenv(ENABLE_ENV) != "1",
    reason=(f"MemFabric BM dual-view Gather is an opt-in fatal hardware gate; set {ENABLE_ENV}=1 to run it."),
)
def test_memfabric_bm_dual_view_host_full_kv_gather_gate():
    module_name = "tests.ut.distributed.kv_transfer.a3_2.test_memfabric_bm_dual_view_gather_npu"
    completed = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-m", module_name, "--child"],
        capture_output=True,
        text=True,
        timeout=PROCESS_TIMEOUT_SECONDS,
        env=os.environ.copy(),
    )
    combined_output = completed.stdout + completed.stderr
    assert completed.returncode == 0, (
        "MemFabric BM dual-view Full-KV -> Gather G1 failed.  The child may "
        "have raised an exception, hit an invalid GM address, or aborted "
        "during external-storage teardown.\n"
        f"returncode={completed.returncode}\noutput:\n{combined_output}"
    )
    result = _extract_result(completed.stdout)
    assert result["status"] == "pass", result
    assert result["cleanup"], result
    assert all(result["swapped_control"].values()), result
    assert all(result["copy_one"]["payload"].values()), result
    assert all(result["host_population"].values()), result
    assert all(result["gather"].values()), result
    assert all(result["guards"].values()), result
    print("MemFabric BM dual-view Host Full-KV -> Gather G1 PASSED: " + json.dumps(result, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--store-url")
    parser.add_argument("--pool-bytes", type=int, default=DEFAULT_POOL_BYTES)
    parser.add_argument("--bm-id", type=int, default=73)
    parser.add_argument("--memfabric-log-level", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    if not parsed.child:
        raise SystemExit("Direct execution requires --child")
    raise SystemExit(_run_child(parsed))

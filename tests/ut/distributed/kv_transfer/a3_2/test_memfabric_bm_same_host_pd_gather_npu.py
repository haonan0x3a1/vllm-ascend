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
"""G2a same-host Prefill/Decode MemFabric BM G2G -> real Gather gate.

This manual hardware gate launches two independent Python processes with one
physical NPU each. Rank 0 acts as Prefill and rank 1 acts as Decode:

P NPU staging -> P BM Host allocation -> BM G2G -> D BM Host allocation
              -> D LOCAL_DEVICE alias -> real GatherSelectionKvCache

The default HOST_TCP protocol validates the two-rank BM Host transport and
completion semantics available on a single physical server without routing the
Full KV through a remote NPU allocation. HOST_RDMA can be selected as an
additional same-host smoke, but it does not prove cross-host registration,
RNIC traffic, remote visibility, or performance. Those remain G2b.

The BM allocation explicitly enables SMEM_BM_FLAG_DRAM_MAP_HOST_VA. Without
that create flag, MemFabric Hybrid 1.1.2 can report a non-zero LOCAL_HOST value
for the VMM DRAM segment even though the address is not CPU-dereferenceable.
The flag maps the Host VA and grants the local NPU read/write access to the same
pages, which is the dual-view contract this gate is meant to prove.

HOST_SHM is intentionally not offered here. MemFabric Hybrid 1.1.2 maps its
DRAM pool into CPU processes, but that segment does not publish the
LOCAL_DEVICE alias required by GatherSelectionKvCache.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from tests.ut.distributed.kv_transfer.a3_2.test_memfabric_bm_dual_view_gather_npu import (
    BLOCK_SIZE,
    DEFAULT_POOL_BYTES,
    FULL_NOPE_SHAPE,
    FULL_ROPE_SHAPE,
    GUARD_SENTINEL,
    INDEX_TOPK,
    KV_LORA_RANK,
    NOPE_BYTES,
    NOPE_OFFSET_BYTES,
    NUM_BLOCKS,
    REQUIRED_POOL_BYTES,
    ROPE_BYTES,
    ROPE_HEAD_DIM,
    ROPE_OFFSET_BYTES,
    TOUCHED_BLOCKS,
    _construct_raw_npu_alias,
    _distribution_version,
    _expected_block,
    _fill_npu_staging,
    _gather_and_verify,
    _guard_samples,
    _physical_slots,
    _reshape_raw_cache,
)

MODULE_NAME = "tests.ut.distributed.kv_transfer.a3_2.test_memfabric_bm_same_host_pd_gather_npu"
RESULT_MARKER = "MEMFABRIC_BM_G2A_RESULT="
DEFAULT_TIMEOUT_SECONDS = 240
DEFAULT_BM_ID = 74
TRANSFER_GENERATION = 3
PREFILL_RANK = 0
DECODE_RANK = 1
WORLD_SIZE = 2
PROCESS_TERMINATE_GRACE_SECONDS = 5
SMEM_BM_FLAG_DRAM_MAP_HOST_VA = 1 << 9


def _find_loopback_store_url() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _wait_json(path: Path, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for control marker {path}")


def _protocol_value(bm_module, name: str):
    values = {
        "host_tcp": bm_module.BmDataOpType.HOST_TCP,
        "host_rdma": bm_module.BmDataOpType.HOST_RDMA,
    }
    return values[name]


def _construct_workspace(
    torch_module,
    torch_npu_module,
    workspace_cls,
    *,
    local_device_va: int,
    device,
) -> tuple[Any, dict[str, Any]]:
    raw_nope = _construct_raw_npu_alias(
        torch_module,
        torch_npu_module,
        data_ptr=local_device_va + NOPE_OFFSET_BYTES,
        nbytes=NOPE_BYTES,
        device=device,
    )
    raw_rope = _construct_raw_npu_alias(
        torch_module,
        torch_npu_module,
        data_ptr=local_device_va + ROPE_OFFSET_BYTES,
        nbytes=ROPE_BYTES,
        device=device,
    )
    full_nope = _reshape_raw_cache(
        torch_module,
        raw_nope,
        FULL_NOPE_SHAPE,
        torch_module.bfloat16,
    )
    full_rope = _reshape_raw_cache(
        torch_module,
        raw_rope,
        FULL_ROPE_SHAPE,
        torch_module.bfloat16,
    )
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
    owners = {
        "raw_nope": raw_nope,
        "raw_rope": raw_rope,
        "full_nope": full_nope,
        "full_rope": full_rope,
        "staging_nope": staging_nope,
        "staging_rope": staging_rope,
        "workspace": workspace,
    }
    return workspace, owners


def _assert_cpu_mapping(*, address: int, size: int) -> str:
    """Return mapping permissions or fail before dereferencing a bad Host VA."""
    end_address = address + size
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        address_range, permissions, *_ = line.split(maxsplit=2)
        start_text, end_text = address_range.split("-", maxsplit=1)
        mapping_start = int(start_text, 16)
        mapping_end = int(end_text, 16)
        if mapping_start <= address and end_address <= mapping_end:
            if "r" not in permissions or "w" not in permissions:
                raise RuntimeError(
                    "MemFabric BM LOCAL_HOST mapping is not CPU read/write: "
                    f"address=0x{address:x}, size={size}, permissions={permissions}"
                )
            return permissions
    raise RuntimeError(
        "MemFabric BM LOCAL_HOST is not present in /proc/self/maps; "
        "SMEM_BM_FLAG_DRAM_MAP_HOST_VA may be unsupported or ineffective: "
        f"address=0x{address:x}, size={size}"
    )


def _initialize_local_guard(*, local_host_va: int) -> str:
    """Initialize BM-owned local Host pages without a BM transport copy."""
    permissions = _assert_cpu_mapping(address=local_host_va, size=REQUIRED_POOL_BYTES)
    ctypes.memset(local_host_va, GUARD_SENTINEL, REQUIRED_POOL_BYTES)
    return permissions


def _read_local_bfloat16(
    torch_module,
    *,
    local_host_va: int,
    offset: int,
    shape: tuple[int, ...],
):
    nbytes = math.prod(shape) * torch_module.empty((), dtype=torch_module.bfloat16).element_size()
    payload = bytearray(ctypes.string_at(local_host_va + offset, nbytes))
    return torch_module.frombuffer(payload, dtype=torch_module.bfloat16).clone().reshape(shape)


def _verify_local_payload(
    torch_module,
    *,
    local_host_va: int,
    generation: int,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for block_id in TOUCHED_BLOCKS:
        nope_offset = NOPE_OFFSET_BYTES + block_id * NOPE_BYTES // NUM_BLOCKS
        rope_offset = ROPE_OFFSET_BYTES + block_id * ROPE_BYTES // NUM_BLOCKS
        actual_nope = _read_local_bfloat16(
            torch_module,
            local_host_va=local_host_va,
            offset=nope_offset,
            shape=(BLOCK_SIZE, 1, KV_LORA_RANK),
        )
        actual_rope = _read_local_bfloat16(
            torch_module,
            local_host_va=local_host_va,
            offset=rope_offset,
            shape=(BLOCK_SIZE, 1, ROPE_HEAD_DIM),
        )
        expected_nope = _expected_block(
            torch_module,
            block_id=block_id,
            width=KV_LORA_RANK,
            rope=False,
            generation=generation,
        )
        expected_rope = _expected_block(
            torch_module,
            block_id=block_id,
            width=ROPE_HEAD_DIM,
            rope=True,
            generation=generation,
        )
        checks[f"nope_block_{block_id}"] = bool(torch_module.equal(actual_nope, expected_nope))
        checks[f"rope_block_{block_id}"] = bool(torch_module.equal(actual_rope, expected_rope))
    return checks


def _verify_local_guards(*, local_host_va: int, local_gva: int) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, (gva, size) in _guard_samples(local_gva).items():
        payload = ctypes.string_at(local_host_va + gva - local_gva, size)
        checks[name] = payload == bytes((GUARD_SENTINEL,)) * size
    return checks


def _copy_touched_blocks_g2g(
    bm_module,
    handle,
    *,
    prefill_gva: int,
    decode_gva: int,
) -> list[dict[str, int | str]]:
    copies: list[dict[str, int | str]] = []
    for block_id in TOUCHED_BLOCKS:
        for name, tensor_offset, tensor_bytes in (
            ("nope", NOPE_OFFSET_BYTES, NOPE_BYTES),
            ("rope", ROPE_OFFSET_BYTES, ROPE_BYTES),
        ):
            block_bytes = tensor_bytes // NUM_BLOCKS
            offset = tensor_offset + block_id * block_bytes
            result = handle.copy_data(
                prefill_gva + offset,
                decode_gva + offset,
                block_bytes,
                bm_module.BmCopyType.G2G,
                0,
            )
            if result != 0:
                raise RuntimeError(f"MemFabric BM G2G failed: block={block_id}, tensor={name}, result={result}")
            copies.append(
                {
                    "block": block_id,
                    "tensor": name,
                    "offset": offset,
                    "bytes": block_bytes,
                }
            )
    return copies


def _run_rank(args: argparse.Namespace) -> int:
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

    rank = args.child_rank
    role = "prefill" if rank == PREFILL_RANK else "decode"
    starts_store = role == args.start_store_role
    bm_rank_id = PREFILL_RANK if starts_store else DECODE_RANK
    sync_dir = Path(args.sync_dir)
    peer_rank = DECODE_RANK if rank == PREFILL_RANK else PREFILL_RANK
    own_ready = sync_dir / f"rank-{rank}.ready.json"
    peer_ready = sync_dir / f"rank-{peer_rank}.ready.json"
    own_joined = sync_dir / f"rank-{rank}.joined.json"
    peer_joined = sync_dir / f"rank-{peer_rank}.joined.json"
    transfer_done = sync_dir / "transfer.done.json"
    decode_result_path = sync_dir / "decode.result.json"
    prefill_ack = sync_dir / "prefill.ack.json"

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.npu.synchronize()

    mf_initialized = False
    bm_initialized = False
    joined = False
    handle = None
    workspace = None
    owners: dict[str, Any] = {}
    tensor_owners: dict[str, Any] = {}
    full_kv = None
    staging_kv = None
    result: dict[str, Any] = {
        "status": "failed",
        "gate": "G2a",
        "rank": rank,
        "role": role,
        "bm_rank_id": bm_rank_id,
        "start_store_role": args.start_store_role,
        "protocol": args.protocol,
        "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "unknown"),
        "logical_device": str(device),
        "memfabric_version": _distribution_version("memfabric_hybrid"),
        "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
        "scope": "same physical host; cross-host HOST_RDMA is not proven",
        "bm_create_flags": {
            "dram_map_host_va": SMEM_BM_FLAG_DRAM_MAP_HOST_VA,
        },
    }

    try:
        mf.set_log_level(args.memfabric_log_level)
        mf_result = mf.initialize()
        if mf_result != 0:
            raise RuntimeError(f"memfabric_hybrid.initialize failed: result={mf_result}")
        mf_initialized = True

        config = bm.BmConfig()
        config.rank_id = bm_rank_id
        config.start_store = starts_store
        config.unified_address_space = True
        if args.protocol in ("host_tcp", "host_rdma"):
            config.set_nic(f"tcp://{args.nic_ip}:{args.nic_port_base + rank}")
        bm_result = bm.initialize(args.store_url, WORLD_SIZE, 0, config)
        if bm_result != 0:
            raise RuntimeError(f"MemFabric BM initialize failed: result={bm_result}")
        bm_initialized = True

        handle = bm.create2(
            id=args.bm_id,
            local_dram_size=args.pool_bytes,
            max_dram_size=args.pool_bytes,
            data_op_type=_protocol_value(bm, args.protocol),
            flags=SMEM_BM_FLAG_DRAM_MAP_HOST_VA,
        )
        if handle is None:
            raise RuntimeError("MemFabric BM create2 returned None")
        join_result = handle.join()
        if join_result != 0:
            raise RuntimeError(f"MemFabric BM join failed: result={join_result}")
        joined = True

        local_gva = handle.peer_rank_ptr(bm_rank_id, bm.BmMemType.HOST)
        if not local_gva:
            raise RuntimeError("MemFabric BM did not expose the local Host GVA")
        _write_json(
            own_joined,
            {"rank": rank, "role": role, "local_gva": local_gva, "status": "joined"},
        )
        _wait_json(peer_joined, args.timeout_seconds)

        local_host_va = handle.gva_to_va(local_gva, bm.BmMemType.LOCAL_HOST)
        local_device_va = handle.gva_to_va(local_gva, bm.BmMemType.LOCAL_DEVICE)
        if not local_gva or not local_host_va or not local_device_va:
            raise RuntimeError(
                "MemFabric BM local address translation failed: "
                f"gva=0x{local_gva:x}, host=0x{local_host_va:x}, "
                f"device=0x{local_device_va:x}"
            )
        result["addresses"] = {
            "local_gva": hex(local_gva),
            "local_host_va": hex(local_host_va),
            "local_device_va": hex(local_device_va),
        }

        result["local_host_mapping_permissions"] = _initialize_local_guard(
            local_host_va=local_host_va,
        )
        workspace, tensor_owners = _construct_workspace(
            torch,
            torch_npu,
            SparseKVOffloadWorkspace,
            local_device_va=local_device_va,
            device=device,
        )
        owners.update(tensor_owners)
        full_kv = (owners["full_nope"], owners["full_rope"])
        staging_kv = (owners["staging_nope"], owners["staging_rope"])

        _write_json(
            own_ready,
            {"rank": rank, "role": role, "local_gva": local_gva, "status": "ready"},
        )
        peer = _wait_json(peer_ready, args.timeout_seconds)
        peer_gva = int(peer["local_gva"])

        if rank == PREFILL_RANK:
            _fill_npu_staging(torch, staging_kv, generation=TRANSFER_GENERATION)
            updated_blocks = workspace.persist_updated_slots(
                full_kv,
                _physical_slots(torch),
                num_actual_tokens=2 * BLOCK_SIZE,
            )
            torch.npu.synchronize()
            copy_one_checks = _verify_local_payload(
                torch,
                local_host_va=local_host_va,
                generation=TRANSFER_GENERATION,
            )
            if set(updated_blocks) != set(TOUCHED_BLOCKS) or not all(copy_one_checks.values()):
                raise RuntimeError(f"Prefill copy 1 failed: updated={updated_blocks}, checks={copy_one_checks}")
            result["copy_one"] = {
                "updated_blocks": list(updated_blocks),
                "payload": copy_one_checks,
            }
            result["g2g"] = _copy_touched_blocks_g2g(
                bm,
                handle,
                prefill_gva=local_gva,
                decode_gva=peer_gva,
            )
            _write_json(
                transfer_done,
                {
                    "status": "complete",
                    "generation": TRANSFER_GENERATION,
                    "prefill_gva": local_gva,
                    "decode_gva": peer_gva,
                },
            )
            decode_result = _wait_json(decode_result_path, args.timeout_seconds)
            result["decode_result"] = decode_result
            if decode_result.get("status") != "pass":
                raise RuntimeError(f"Decode rank reported G2a failure: {decode_result}")
            _write_json(prefill_ack, {"status": "ack"})
        else:
            transfer = _wait_json(transfer_done, args.timeout_seconds)
            torch.npu.synchronize()
            gather_checks = _gather_and_verify(
                torch,
                workspace,
                device=device,
                generation=int(transfer["generation"]),
            )
            host_checks = _verify_local_payload(
                torch,
                local_host_va=local_host_va,
                generation=int(transfer["generation"]),
            )
            guard_checks = _verify_local_guards(
                local_host_va=local_host_va,
                local_gva=local_gva,
            )
            result["gather"] = gather_checks
            result["host_payload"] = host_checks
            result["guards"] = guard_checks
            passed = all(gather_checks.values()) and all(host_checks.values()) and all(guard_checks.values())
            decode_result = {
                "status": "pass" if passed else "failed",
                "gather": gather_checks,
                "host_payload": host_checks,
                "guards": guard_checks,
            }
            _write_json(decode_result_path, decode_result)
            if not passed:
                raise RuntimeError(f"Decode G2a verification failed: {decode_result}")
            _wait_json(prefill_ack, args.timeout_seconds)

        result["status"] = "pass"
    except BaseException as exc:
        if rank == DECODE_RANK:
            _write_json(decode_result_path, {"status": "failed", "error": repr(exc)})
        raise
    finally:
        owners.clear()
        tensor_owners.clear()
        workspace = None
        full_kv = None
        staging_kv = None
        gc.collect()
        try:
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
                mf.uninitialize()
                mf_initialized = False
        finally:
            result["cleanup"] = True

    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
    return 0


def _extract_result(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line.removeprefix(RESULT_MARKER))
    raise RuntimeError(f"Missing {RESULT_MARKER!r} in child output:\n{output}")


def _terminate(processes: list[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=PROCESS_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()


def _run_parent(args: argparse.Namespace) -> int:
    if args.prefill_physical_device == args.decode_physical_device:
        raise ValueError("Prefill and Decode must use different physical NPUs")
    store_url = args.store_url or _find_loopback_store_url()
    with tempfile.TemporaryDirectory(prefix="memfabric-g2a-") as directory:
        sync_dir = Path(directory)
        processes: list[subprocess.Popen[Any]] = []
        log_files = []
        log_paths: list[Path] = []
        timed_out = False
        try:
            for rank, physical_device in (
                (PREFILL_RANK, args.prefill_physical_device),
                (DECODE_RANK, args.decode_physical_device),
            ):
                log_path = sync_dir / f"rank-{rank}.log"
                log_file = log_path.open("w", encoding="utf-8")
                log_files.append(log_file)
                log_paths.append(log_path)
                command = [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-m",
                    MODULE_NAME,
                    "--child-rank",
                    str(rank),
                    "--sync-dir",
                    str(sync_dir),
                    "--store-url",
                    store_url,
                    "--protocol",
                    args.protocol,
                    "--start-store-role",
                    args.start_store_role,
                    "--pool-bytes",
                    str(args.pool_bytes),
                    "--bm-id",
                    str(args.bm_id),
                    "--timeout-seconds",
                    str(args.timeout_seconds),
                    "--memfabric-log-level",
                    str(args.memfabric_log_level),
                    "--nic-ip",
                    args.nic_ip,
                    "--nic-port-base",
                    str(args.nic_port_base),
                ]
                child_env = os.environ.copy()
                child_env["ASCEND_RT_VISIBLE_DEVICES"] = str(physical_device)
                processes.append(
                    subprocess.Popen(
                        command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=child_env,
                    )
                )
                if rank == PREFILL_RANK:
                    time.sleep(args.rank_start_delay_seconds)

            deadline = time.monotonic() + args.timeout_seconds
            while time.monotonic() < deadline:
                if all(process.poll() is not None for process in processes):
                    break
                if any(process.poll() not in (None, 0) for process in processes):
                    break
                time.sleep(0.1)
            else:
                timed_out = True
        finally:
            _terminate(processes)
            for log_file in log_files:
                log_file.close()

        outputs = [log_path.read_text(encoding="utf-8", errors="replace") for log_path in log_paths]
        if timed_out:
            details = "\n".join(f"rank={rank}\n{output}" for rank, output in enumerate(outputs))
            raise TimeoutError(f"G2a ranks did not finish within {args.timeout_seconds}s:\n{details}")
        failures = [
            f"rank={rank} returncode={process.returncode}\n{outputs[rank]}"
            for rank, process in enumerate(processes)
            if process.returncode != 0
        ]
        if failures:
            raise RuntimeError("MemFabric BM same-host G2a failed:\n" + "\n".join(failures))
        results = [_extract_result(output) for output in outputs]
        if not all(result.get("status") == "pass" for result in results):
            raise RuntimeError(f"G2a child result failed: {results}")
        print(
            "MemFabric BM same-host P -> D Host Full-KV -> Gather G2a PASSED: " + json.dumps(results, sort_keys=True),
            flush=True,
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-rank", type=int, choices=(0, 1))
    parser.add_argument("--sync-dir")
    parser.add_argument("--store-url")
    parser.add_argument("--protocol", choices=("host_tcp", "host_rdma"), default="host_tcp")
    parser.add_argument(
        "--start-store-role",
        choices=("prefill", "decode"),
        default="prefill",
    )
    parser.add_argument("--prefill-physical-device", type=int, default=0)
    parser.add_argument("--decode-physical-device", type=int, default=8)
    parser.add_argument("--pool-bytes", type=int, default=DEFAULT_POOL_BYTES)
    parser.add_argument("--bm-id", type=int, default=DEFAULT_BM_ID)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--rank-start-delay-seconds", type=float, default=1.0)
    parser.add_argument("--memfabric-log-level", type=int, default=1)
    parser.add_argument("--nic-ip", default="127.0.0.1")
    parser.add_argument("--nic-port-base", type=int, default=12400)
    args = parser.parse_args()
    if args.child_rank is not None and not args.sync_dir:
        parser.error("--sync-dir is required with --child-rank")
    if args.child_rank is not None and not args.store_url:
        parser.error("--store-url is required with --child-rank")
    return args


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.child_rank is None:
        raise SystemExit(_run_parent(parsed))
    raise SystemExit(_run_rank(parsed))

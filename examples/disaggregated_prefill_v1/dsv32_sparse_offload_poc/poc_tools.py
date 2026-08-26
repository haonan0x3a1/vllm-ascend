#!/usr/bin/env python3
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

"""Preflight, validation, and evidence helpers for the DSV3.2 sparse P/D PoC."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from urllib.request import ProxyHandler, Request, build_opener

FATAL_PATTERN = re.compile(
    r"NACK|Find available port failed|Failed to persist|"
    r"producer visibility race|swapped-copy mismatch|"
    r"Full-KV changed during Gather|Gather selection mismatch|"
    r"OutOfMemoryError|Segmentation fault|double free|"
    r"Worker failed|RuntimeError:|Traceback"
)

RUNTIME_IMPORT_ORDER = (
    "torch",
    "torch_npu",
    "custom_ops",
    "mooncake",
    "vllm_ascend",
)
DEVICE_VISIBILITY_ENV = "ASCEND_RT_VISIBLE_DEVICES"
LINUX_EPHEMERAL_PORT_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")
LINUX_RESERVED_PORTS = Path("/proc/sys/net/ipv4/ip_local_reserved_ports")
MAX_TCP_PORT = 65535
MEMFABRIC_BM_ALLOCATOR_INIT_MARKER = "Initialized MemFabric BM Full-KV allocator:"
MEMFABRIC_BM_DATA_PLANE_MARKER = "MemFabric BM data plane active:"
MEMFABRIC_BM_VISIBILITY_FENCE_MARKER = "Decode MemFabric BM Full-KV visibility fence active:"
MEMFABRIC_BM_ALLOCATOR_RELEASE_MARKER = "Released MemFabric BM Full-KV allocator:"
PD_TRANSFER_STAGE_METRICS_MARKER = "PD transfer stage metrics:"
EXPECTED_PD_TRANSFER_STAGES = {
    "prefill": {
        "wait_prefill_full_kv_visible",
        "memfabric_full_kv_host_to_host",
        "mooncake_indexer_npu_to_npu",
        "layer_ack_round_trip",
    },
    "decode": {"decode_full_kv_visibility_fence"},
}


@dataclass(frozen=True)
class ValidationCase:
    name: str
    text: str
    expected: str
    expect_above_topk: bool


def build_validation_cases() -> tuple[ValidationCase, ...]:
    return (
        ValidationCase(
            name="below-topk-alpha",
            text="alpha beta gamma delta " * 450,
            expected="alpha",
            expect_above_topk=False,
        ),
        ValidationCase(
            name="above-topk-beta",
            text="beta alpha gamma delta " * 550,
            expected="beta",
            expect_above_topk=True,
        ),
        ValidationCase(
            name="true-sparse-gamma-3k",
            text="gamma alpha beta delta " * 750,
            expected="gamma",
            expect_above_topk=True,
        ),
        ValidationCase(
            name="near-4k-delta",
            text="delta alpha beta gamma " * 900,
            expected="delta",
            expect_above_topk=True,
        ),
        ValidationCase(
            name="request-reset-alpha",
            text="alpha beta gamma delta " * 600,
            expected="alpha",
            expect_above_topk=True,
        ),
    )


def parse_devices(value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid device list: {value!r}") from exc
    if not devices or len(set(devices)) != len(devices):
        raise ValueError(f"Device list must be non-empty and unique: {value!r}")
    return devices


def can_bind(port: int) -> tuple[bool, str | None]:
    sock = socket.socket()
    try:
        # Match the rendezvous listeners used by the runtime.  Without this,
        # a completed TCP connection left in TIME_WAIT can make preflight
        # report a port as busy even though the runtime can safely reuse it.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        return False, str(exc)
    finally:
        sock.close()
    return True, None


def parse_port_range(value: str) -> tuple[int, int]:
    fields = value.split()
    if len(fields) != 2:
        raise ValueError(f"Expected two port-range bounds, got: {value!r}")
    start, end = (int(field) for field in fields)
    if not 0 < start <= end <= MAX_TCP_PORT:
        raise ValueError(f"Invalid TCP port range: {value!r}")
    return start, end


def parse_reserved_port_ranges(value: str) -> tuple[tuple[int, int], ...]:
    ranges = []
    for item in (item.strip() for item in value.split(",")):
        if not item:
            continue
        bounds = item.split("-", maxsplit=1)
        ranges.append(parse_port_range(" ".join((bounds[0], bounds[-1]))))
    return tuple(ranges)


def find_unreserved_ephemeral_ports(
    ports: list[int],
    ephemeral_range: tuple[int, int],
    reserved_ranges: tuple[tuple[int, int], ...],
) -> list[int]:
    ephemeral_start, ephemeral_end = ephemeral_range
    return sorted(
        {
            port
            for port in ports
            if ephemeral_start <= port <= ephemeral_end
            and not any(start <= port <= end for start, end in reserved_ranges)
        }
    )


def count_lifecycle_records(
    request_id: str,
    prefill_log: str,
    decode_log: str,
) -> tuple[int, int]:
    prefill_count = sum(request_id in line and "done_sending_msg" in line for line in prefill_log.splitlines())
    decode_count = sum(
        request_id in line and "Number of completed KV cache recv requests" in line for line in decode_log.splitlines()
    )
    return prefill_count, decode_count


def find_fatal_lines(named_logs: tuple[tuple[str, str], ...]) -> list[str]:
    return [f"{name}: {line}" for name, text in named_logs for line in text.splitlines() if FATAL_PATTERN.search(line)]


def remove_expected_async_llm_shutdown_cascade(text: str) -> str:
    """Remove the API-side EngineDead cascade after MPClient already stopped."""
    lines = text.splitlines()
    manager_stopped = next(
        (index for index, line in enumerate(lines) if "[shutdown] MPClient: engine manager stopped" in line),
        None,
    )
    if manager_stopped is None:
        return text
    async_failure = next(
        (index for index in range(manager_stopped + 1, len(lines)) if "AsyncLLM output_handler failed" in lines[index]),
        None,
    )
    if async_failure is None:
        return text
    client_stopped = next(
        (
            index
            for index in range(async_failure + 1, len(lines))
            if "[shutdown] API server: engine client stopped" in lines[index]
        ),
        None,
    )
    if client_stopped is None or not any("EngineDeadError" in line for line in lines[async_failure:client_stopped]):
        return text
    return "\n".join(lines[:async_failure] + lines[client_stopped:])


def require_memfabric_bm_runtime_markers(
    prefill_log: str,
    decode_log: str,
    tp_size: int,
) -> dict[str, int]:
    """Require every TP worker to initialize and exercise the M2 data plane."""
    counts = {
        "prefill_allocators": prefill_log.count(MEMFABRIC_BM_ALLOCATOR_INIT_MARKER),
        "decode_allocators": decode_log.count(MEMFABRIC_BM_ALLOCATOR_INIT_MARKER),
        "prefill_data_plane": prefill_log.count(MEMFABRIC_BM_DATA_PLANE_MARKER),
        "decode_visibility_fence": decode_log.count(MEMFABRIC_BM_VISIBILITY_FENCE_MARKER),
    }
    mismatches = {name: count for name, count in counts.items() if count != tp_size}
    if mismatches:
        raise RuntimeError(
            f"MemFabric BM M2 expected one runtime marker per TP worker, got {counts}, expected_each={tp_size}."
        )
    return counts


def require_memfabric_bm_shutdown_markers(
    prefill_log: str,
    decode_log: str,
    tp_size: int,
) -> dict[str, int]:
    """Require every TP worker to explicitly release its BM allocator."""
    counts = {
        "prefill_releases": prefill_log.count(MEMFABRIC_BM_ALLOCATOR_RELEASE_MARKER),
        "decode_releases": decode_log.count(MEMFABRIC_BM_ALLOCATOR_RELEASE_MARKER),
    }
    mismatches = {name: count for name, count in counts.items() if count != tp_size}
    if mismatches:
        raise RuntimeError(
            f"MemFabric BM expected one allocator release per TP worker, got {counts}, expected_each={tp_size}."
        )
    return counts


def summarize_pd_transfer_stage_metrics(
    prefill_log: str,
    decode_log: str,
    tp_size: int,
) -> dict[str, object]:
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    records = []
    for expected_role, text in (
        ("prefill", prefill_log),
        ("decode", decode_log),
    ):
        role_records = []
        for line in text.splitlines():
            if PD_TRANSFER_STAGE_METRICS_MARKER not in line:
                continue
            payload = json.loads(line.split(PD_TRANSFER_STAGE_METRICS_MARKER, 1)[1].strip())
            if payload.get("role") == expected_role:
                role_records.append(payload)
        if len(role_records) != tp_size:
            raise RuntimeError(
                f"PD transfer stage metrics expected {tp_size} {expected_role} worker records, got {len(role_records)}."
            )
        observed_ranks = sorted(int(record["tp_rank"]) for record in role_records)
        expected_ranks = list(range(tp_size))
        if observed_ranks != expected_ranks:
            raise RuntimeError(
                f"PD transfer stage metrics expected {expected_role} TP ranks {expected_ranks}, got {observed_ranks}."
            )
        if any(record.get("status") != "completed" for record in role_records):
            raise RuntimeError(f"PD transfer stage metrics contain failed {expected_role} records.")
        records.extend(role_records)

    summary_rows = []
    for role, expected_stages in EXPECTED_PD_TRANSFER_STAGES.items():
        role_records = [record for record in records if record["role"] == role]
        observed_stages = {stage for record in role_records for stage in record.get("stages", {})}
        missing = sorted(expected_stages - observed_stages)
        if missing:
            raise RuntimeError(f"PD transfer stage metrics missing {role} stages: {missing}")
        for stage in sorted(expected_stages):
            samples = [record["stages"][stage] for record in role_records]
            total_events = sum(int(sample["count"]) for sample in samples)
            if total_events <= 0:
                raise RuntimeError(f"PD transfer stage metrics have no {role} events for {stage}.")
            total_ms = sum(float(sample["total_ms"]) for sample in samples)
            summary_rows.append(
                {
                    "role": role,
                    "stage": stage,
                    "workers": len(samples),
                    "events": total_events,
                    "mean_per_worker_total_ms": total_ms / len(samples),
                    "mean_event_ms": total_ms / total_events,
                    "max_event_ms": max(float(sample["max_ms"]) for sample in samples),
                }
            )
    return {"tp_size": tp_size, "records": records, "stages": summary_rows}


def stage_summary(args: argparse.Namespace) -> int:
    summary = summarize_pd_transfer_stage_metrics(
        Path(args.prefill_log).read_text(errors="replace"),
        Path(args.decode_log).read_text(errors="replace"),
        args.tp_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("role     workers  events  mean/worker total  mean/event  max/event  stage")
    for item in summary["stages"]:
        print(
            f"{item['role']:<8} {item['workers']:>7}  {item['events']:>6}  "
            f"{item['mean_per_worker_total_ms']:>17.3f} ms  "
            f"{item['mean_event_ms']:>10.3f} ms  "
            f"{item['max_event_ms']:>9.3f} ms  {item['stage']}"
        )
    print(f"PD transfer stage summary saved to: {output}")
    return 0


def run_command(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def import_runtime_modules(
    importer: Callable[[str], ModuleType] = import_module,
) -> dict[str, ModuleType]:
    """Import runtime extensions after PyTorch has loaded its shared libraries."""
    return {name: importer(name) for name in RUNTIME_IMPORT_ORDER}


def clear_inherited_device_visibility(
    environment: MutableMapping[str, str],
) -> str | None:
    """Expose the container's full NPU topology to the preflight process."""
    return environment.pop(DEVICE_VISIBILITY_ENV, None)


def memfabric_bm_required_ports(
    tp_size: int,
    store_port_base: int,
    hcom_port_base: int,
) -> tuple[int, ...]:
    """Return BM store, HCOM, and peer-rendezvous listener ports."""
    store_ports = tuple(store_port_base + tp_rank for tp_rank in range(tp_size))
    hcom_ports = tuple(hcom_port_base + tp_rank * 4 + rank_id for tp_rank in range(tp_size) for rank_id in (0, 1))
    rendezvous_ports = tuple(hcom_port_base + tp_rank * 4 + offset for tp_rank in range(tp_size) for offset in (2, 3))
    return (*store_ports, *hcom_ports, *rendezvous_ports)


def resolve_memfabric_bm_api(memfabric_hybrid: ModuleType) -> object:
    """Resolve the BM API exported by memfabric_hybrid 1.1.x."""
    memfabric_bm = getattr(memfabric_hybrid, "bm", None)
    if memfabric_bm is None:
        raise ImportError("memfabric_hybrid does not export the bm API")
    return memfabric_bm


def preflight(args: argparse.Namespace) -> int:
    errors: list[str] = []
    prefill_devices = parse_devices(args.prefill_devices)
    decode_devices = parse_devices(args.decode_devices)

    try:
        ipaddress.ip_address(args.host_ip)
    except ValueError:
        errors.append(f"HOST_IP is not a valid IP address: {args.host_ip!r}")

    if len(prefill_devices) != args.tp_size:
        errors.append(f"Prefill device count {len(prefill_devices)} does not match TP size {args.tp_size}.")
    if len(decode_devices) != args.tp_size:
        errors.append(f"Decode device count {len(decode_devices)} does not match TP size {args.tp_size}.")
    overlap = sorted(set(prefill_devices) & set(decode_devices))
    if overlap:
        errors.append(f"Prefill and Decode devices overlap: {overlap}")
    if args.max_model_len <= args.index_topk:
        errors.append(
            f"max_model_len={args.max_model_len} must exceed index_topk={args.index_topk} "
            "to exercise a true sparse subset."
        )

    required_files = (
        Path(args.model_path) / "config.json",
        Path(args.model_path) / "model.safetensors.index.json",
        Path(args.repo_dir) / "vllm_ascend" / "__init__.py",
        Path(args.proxy_script),
    )
    for path in required_files:
        if not path.is_file():
            errors.append(f"Required file is missing: {path}")

    print("Runtime package versions:")
    for distribution in (
        "vllm",
        "vllm-ascend",
        "mooncake-transfer-engine-npu",
        "torch",
        "torch-npu",
    ):
        print(f"  {distribution}: {package_version(distribution)}")
    if args.transfer_mode == "memfabric_bm":
        print(f"  memfabric-hybrid: {package_version('memfabric_hybrid')}")

    try:
        inherited_visibility = clear_inherited_device_visibility(os.environ)
        if inherited_visibility is not None:
            print(
                f"Ignoring inherited {DEVICE_VISIBILITY_ENV}={inherited_visibility!r} for the physical topology check."
            )
        runtime_modules = import_runtime_modules()
        torch = runtime_modules["torch"]
        torch_npu = runtime_modules["torch_npu"]
        vllm_ascend = runtime_modules["vllm_ascend"]

        print(f"vllm-ascend source: {Path(vllm_ascend.__file__).resolve()}")
        print(f"NPU available: {torch.npu.is_available()}")
        print(f"NPU count: {torch.npu.device_count()}")
        if not torch.npu.is_available():
            errors.append("torch.npu.is_available() is false.")
        max_device = max((*prefill_devices, *decode_devices))
        if torch.npu.device_count() <= max_device:
            errors.append(
                f"Visible NPU count {torch.npu.device_count()} does not include physical device {max_device}."
            )
        if not hasattr(torch_npu, "npu_gather_selection_kv_cache"):
            errors.append("torch_npu.npu_gather_selection_kv_cache is unavailable.")
        if not hasattr(torch.ops.custom, "npu_swiglu_clip_quant"):
            errors.append("torch.ops.custom.npu_swiglu_clip_quant is unavailable.")
        if args.transfer_mode == "memfabric_bm":
            memfabric_hybrid = import_module("memfabric_hybrid")
            memfabric_bm = resolve_memfabric_bm_api(memfabric_hybrid)
            if not hasattr(memfabric_hybrid, "initialize"):
                errors.append("memfabric_hybrid.initialize is unavailable.")
            if not hasattr(memfabric_bm, "create2"):
                errors.append("memfabric_hybrid.bm.create2 is unavailable.")
    except Exception as exc:  # pragma: no cover - hardware/runtime specific
        errors.append(f"Runtime import probe failed: {type(exc).__name__}: {exc}")

    print("Application and control ports:")
    required_ports = [
        args.proxy_port,
        args.prefill_api_port,
        args.decode_api_port,
        *(args.prefill_kv_port_base + rank for rank in range(args.tp_size)),
        *(args.decode_kv_port_base + rank for rank in range(args.tp_size)),
    ]
    if args.transfer_mode == "memfabric_bm":
        required_ports.extend(
            memfabric_bm_required_ports(
                args.tp_size,
                args.memfabric_bm_store_port_base,
                args.memfabric_bm_hcom_port_base,
            )
        )
    try:
        ephemeral_range = parse_port_range(LINUX_EPHEMERAL_PORT_RANGE.read_text(encoding="utf-8"))
        reserved_ranges = parse_reserved_port_ranges(LINUX_RESERVED_PORTS.read_text(encoding="utf-8"))
        unsafe_ports = find_unreserved_ephemeral_ports(
            required_ports,
            ephemeral_range,
            reserved_ranges,
        )
        print(f"Linux ephemeral TCP port range: {ephemeral_range[0]}-{ephemeral_range[1]}")
        if unsafe_ports:
            errors.append(
                "Fixed application/control ports overlap the unreserved Linux "
                f"ephemeral range {ephemeral_range[0]}-{ephemeral_range[1]}: "
                f"{unsafe_ports}. Choose ports outside that range or reserve "
                "them with net.ipv4.ip_local_reserved_ports."
            )
    except (OSError, ValueError) as exc:
        errors.append(f"Could not validate the Linux ephemeral TCP port policy: {exc}")
    for port in required_ports:
        free, reason = can_bind(port)
        print(f"  {port}: {'FREE' if free else f'BUSY ({reason})'}")
        if not free:
            errors.append(f"Required port {port} is busy: {reason}")

    print("Mooncake ADXL port ranges:")
    for device in (*prefill_devices, *decode_devices):
        start = 20000 + device * 100
        free_count = sum(can_bind(port)[0] for port in range(start, start + 100))
        status = "READY" if free_count >= args.min_adxl_free_ports else "BLOCKED"
        print(f"  device {device:2d}: {start}-{start + 99}, free={free_count:3d}, status={status}")
        if free_count < args.min_adxl_free_ports:
            errors.append(
                f"Device {device} ADXL range has only {free_count} free ports; requires {args.min_adxl_free_ports}."
            )

    try:
        revision = run_command(["git", "-C", args.repo_dir, "rev-parse", "HEAD"]).strip()
        print(f"Repository revision: {revision}")
    except Exception as exc:
        errors.append(f"Could not read repository revision: {exc}")

    print(
        "NPU ownership is dynamic and is not inferred from device visibility. "
        "Confirm npu-smi shows no foreign processes on the configured cards."
    )
    if errors:
        print("Preflight FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Preflight PASSED")
    return 0


def validate(args: argparse.Namespace) -> int:
    opener = build_opener(ProxyHandler({}))
    results: list[dict[str, object]] = []

    for case in build_validation_cases():
        prompt = case.text + "\nAfter reading the text, answer beginning with only the word " + f"{case.expected}."
        payload = json.dumps(
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 8,
                "ignore_eos": True,
            }
        ).encode()
        request = Request(
            args.url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        with opener.open(request, timeout=args.timeout) as response:
            body = json.load(response)
            status = response.status
        elapsed = time.perf_counter() - started

        content = body["choices"][0]["message"]["content"]
        usage = body["usage"]
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        if status != 200:
            raise RuntimeError(f"{case.name}: HTTP status {status}")
        if not content.strip().startswith(case.expected):
            raise RuntimeError(f"{case.name}: unexpected content {content!r}")
        if completion_tokens != 8:
            raise RuntimeError(f"{case.name}: expected 8 completion tokens, got {completion_tokens}")
        if case.expect_above_topk and prompt_tokens <= args.index_topk:
            raise RuntimeError(f"{case.name}: prompt_tokens={prompt_tokens} did not exceed Top-K")
        if not case.expect_above_topk and prompt_tokens >= args.index_topk:
            raise RuntimeError(f"{case.name}: prompt_tokens={prompt_tokens} did not stay below Top-K")

        item = {
            "name": case.name,
            "request_id": body["id"],
            "status": status,
            "content": content,
            "usage": usage,
            "elapsed_seconds": elapsed,
        }
        results.append(item)
        print(
            f"{case.name}: request_id={body['id']}, prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, content={content!r}, elapsed={elapsed:.3f}s"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    time.sleep(args.log_settle_seconds)
    prefill_log = Path(args.prefill_log).read_text(errors="replace")
    decode_log = Path(args.decode_log).read_text(errors="replace")
    proxy_log = Path(args.proxy_log).read_text(errors="replace")
    for item in results:
        prefill_count, decode_count = count_lifecycle_records(
            str(item["request_id"]),
            prefill_log,
            decode_log,
        )
        print(f"{item['name']}: Prefill={prefill_count}, Decode={decode_count}")
        if prefill_count != args.tp_size or decode_count != args.tp_size:
            raise RuntimeError(
                f"{item['name']}: expected {args.tp_size} rank records, "
                f"got Prefill={prefill_count}, Decode={decode_count}"
            )

    fatal_lines = find_fatal_lines(
        (
            ("prefill", prefill_log),
            ("decode", decode_log),
            ("proxy", proxy_log),
        )
    )
    if fatal_lines:
        print("Fatal log matches:")
        print("\n".join(fatal_lines[-200:]))
        raise RuntimeError(f"Found {len(fatal_lines)} fatal log matches")

    if args.transfer_mode == "memfabric_bm":
        marker_counts = require_memfabric_bm_runtime_markers(
            prefill_log,
            decode_log,
            args.tp_size,
        )
        print(f"MemFabric BM M2 runtime markers: {marker_counts}")

    print(f"Results saved to: {output_path}")
    print("FINAL 4K ONLINE PD SUITE: PASSED")
    return 0


def verify_shutdown(args: argparse.Namespace) -> int:
    prefill_log = Path(args.prefill_log).read_text(errors="replace")
    decode_log = Path(args.decode_log).read_text(errors="replace")
    marker_counts = require_memfabric_bm_shutdown_markers(
        prefill_log,
        decode_log,
        args.tp_size,
    )
    fatal_lines = find_fatal_lines(
        (
            (
                "prefill",
                remove_expected_async_llm_shutdown_cascade(prefill_log),
            ),
            (
                "decode",
                remove_expected_async_llm_shutdown_cascade(decode_log),
            ),
        )
    )
    if fatal_lines:
        print("Fatal log matches:")
        print("\n".join(fatal_lines[-200:]))
        raise RuntimeError(f"Found {len(fatal_lines)} fatal log matches")

    print(f"MemFabric BM shutdown markers: {marker_counts}")
    print("MEMFABRIC BM SHUTDOWN: PASSED")
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    evidence_dir = Path(args.output_dir) / f"dsv32-pd-real61-4k-{args.transfer_mode}-{timestamp}"
    evidence_dir.mkdir(parents=True, exist_ok=False)

    evidence_files = [
        args.validation_output,
        args.prefill_log,
        args.decode_log,
        args.proxy_log,
    ]
    if args.stage_metrics_output:
        evidence_files.append(args.stage_metrics_output)
    for value in evidence_files:
        source = Path(value)
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = evidence_dir / source.name
        shutil.copy2(source, destination)

    (evidence_dir / "git-revision.txt").write_text(run_command(["git", "-C", args.repo_dir, "rev-parse", "HEAD"]))
    (evidence_dir / "git-status.txt").write_text(
        run_command(["git", "-C", args.repo_dir, "status", "--short", "--branch"])
    )
    runtime_versions = "\n".join(
        f"{distribution}: {package_version(distribution)}"
        for distribution in (
            "vllm",
            "vllm-ascend",
            "mooncake-transfer-engine-npu",
            "torch",
            "torch-npu",
        )
    )
    (evidence_dir / "runtime-versions.txt").write_text(runtime_versions + "\n")

    model_hashes: list[str] = []
    for name in ("config.json", "model.safetensors.index.json"):
        path = Path(args.model_path) / name
        if path.is_file():
            model_hashes.append(f"{sha256_file(path)}  {path}")
    (evidence_dir / "model-metadata-sha256.txt").write_text("\n".join(model_hashes) + "\n")

    try:
        npu_info = run_command(["npu-smi", "info"])
    except Exception as exc:  # pragma: no cover - hardware/runtime specific
        npu_info = f"npu-smi info failed: {exc}\n"
    (evidence_dir / "npu-smi.txt").write_text(npu_info)

    prefill_text = Path(args.prefill_log).read_text(errors="replace")
    decode_text = Path(args.decode_log).read_text(errors="replace")
    lifecycle_markers = (
        "done_sending_msg",
        "Number of completed KV cache recv requests",
        MEMFABRIC_BM_ALLOCATOR_INIT_MARKER,
        MEMFABRIC_BM_DATA_PLANE_MARKER,
        MEMFABRIC_BM_VISIBILITY_FENCE_MARKER,
        MEMFABRIC_BM_ALLOCATOR_RELEASE_MARKER,
        PD_TRANSFER_STAGE_METRICS_MARKER,
    )
    lifecycle_lines = [
        line
        for line in (*prefill_text.splitlines(), *decode_text.splitlines())
        if any(marker in line for marker in lifecycle_markers)
    ]
    (evidence_dir / "transport-lifecycle.txt").write_text("\n".join(lifecycle_lines) + "\n")

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "model_path": args.model_path,
        "served_model_name": args.model,
        "max_model_len": args.max_model_len,
        "index_topk": args.index_topk,
        "tp_size": args.tp_size,
        "prefill_devices": args.prefill_devices,
        "decode_devices": args.decode_devices,
        "prefill_kv_port_base": args.prefill_kv_port_base,
        "decode_kv_port_base": args.decode_kv_port_base,
        "sparse_kv_transfer_mode": args.transfer_mode,
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(evidence_dir.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (evidence_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    print(f"Evidence saved to: {evidence_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--model-path", required=True)
    preflight_parser.add_argument("--repo-dir", required=True)
    preflight_parser.add_argument("--proxy-script", required=True)
    preflight_parser.add_argument("--host-ip", required=True)
    preflight_parser.add_argument("--prefill-devices", required=True)
    preflight_parser.add_argument("--decode-devices", required=True)
    preflight_parser.add_argument("--tp-size", type=int, required=True)
    preflight_parser.add_argument("--max-model-len", type=int, required=True)
    preflight_parser.add_argument("--index-topk", type=int, required=True)
    preflight_parser.add_argument("--proxy-port", type=int, required=True)
    preflight_parser.add_argument("--prefill-api-port", type=int, required=True)
    preflight_parser.add_argument("--decode-api-port", type=int, required=True)
    preflight_parser.add_argument("--prefill-kv-port-base", type=int, required=True)
    preflight_parser.add_argument("--decode-kv-port-base", type=int, required=True)
    preflight_parser.add_argument(
        "--transfer-mode",
        choices=("npu_staging", "host_relay", "memfabric_bm"),
        required=True,
    )
    preflight_parser.add_argument(
        "--memfabric-bm-store-port-base",
        type=int,
        required=True,
    )
    preflight_parser.add_argument(
        "--memfabric-bm-hcom-port-base",
        type=int,
        required=True,
    )
    preflight_parser.add_argument("--min-adxl-free-ports", type=int, default=2)
    preflight_parser.set_defaults(func=preflight)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--url", required=True)
    validate_parser.add_argument("--model", required=True)
    validate_parser.add_argument("--index-topk", type=int, required=True)
    validate_parser.add_argument("--tp-size", type=int, required=True)
    validate_parser.add_argument(
        "--transfer-mode",
        choices=("npu_staging", "host_relay", "memfabric_bm"),
        required=True,
    )
    validate_parser.add_argument("--timeout", type=int, default=1200)
    validate_parser.add_argument("--log-settle-seconds", type=int, default=5)
    validate_parser.add_argument("--output", required=True)
    validate_parser.add_argument("--prefill-log", required=True)
    validate_parser.add_argument("--decode-log", required=True)
    validate_parser.add_argument("--proxy-log", required=True)
    validate_parser.set_defaults(func=validate)

    shutdown_parser = subparsers.add_parser("verify-shutdown")
    shutdown_parser.add_argument("--tp-size", type=int, required=True)
    shutdown_parser.add_argument("--prefill-log", required=True)
    shutdown_parser.add_argument("--decode-log", required=True)
    shutdown_parser.set_defaults(func=verify_shutdown)

    stage_summary_parser = subparsers.add_parser("stage-summary")
    stage_summary_parser.add_argument("--tp-size", type=int, required=True)
    stage_summary_parser.add_argument("--prefill-log", required=True)
    stage_summary_parser.add_argument("--decode-log", required=True)
    stage_summary_parser.add_argument("--output", required=True)
    stage_summary_parser.set_defaults(func=stage_summary)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--output-dir", required=True)
    collect_parser.add_argument("--repo-dir", required=True)
    collect_parser.add_argument("--model-path", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--max-model-len", type=int, required=True)
    collect_parser.add_argument("--index-topk", type=int, required=True)
    collect_parser.add_argument("--tp-size", type=int, required=True)
    collect_parser.add_argument("--prefill-devices", required=True)
    collect_parser.add_argument("--decode-devices", required=True)
    collect_parser.add_argument("--prefill-kv-port-base", type=int, required=True)
    collect_parser.add_argument("--decode-kv-port-base", type=int, required=True)
    collect_parser.add_argument(
        "--transfer-mode",
        choices=("npu_staging", "host_relay", "memfabric_bm"),
        required=True,
    )
    collect_parser.add_argument("--validation-output", required=True)
    collect_parser.add_argument("--stage-metrics-output")
    collect_parser.add_argument("--prefill-log", required=True)
    collect_parser.add_argument("--decode-log", required=True)
    collect_parser.add_argument("--proxy-log", required=True)
    collect_parser.set_defaults(func=collect)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

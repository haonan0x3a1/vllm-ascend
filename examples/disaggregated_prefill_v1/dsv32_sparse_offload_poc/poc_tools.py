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
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

FATAL_PATTERN = re.compile(
    r"NACK|Find available port failed|Failed to persist|"
    r"producer visibility race|swapped-copy mismatch|"
    r"Full-KV changed during Gather|Gather selection mismatch|"
    r"OutOfMemoryError|Segmentation fault|double free|"
    r"Worker failed|RuntimeError:|Traceback"
)


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
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        return False, str(exc)
    finally:
        sock.close()
    return True, None


def count_lifecycle_records(
    request_id: str,
    prefill_log: str,
    decode_log: str,
) -> tuple[int, int]:
    prefill_count = sum(
        request_id in line and "done_sending_msg" in line for line in prefill_log.splitlines()
    )
    decode_count = sum(
        request_id in line and "Number of completed KV cache recv requests" in line
        for line in decode_log.splitlines()
    )
    return prefill_count, decode_count


def find_fatal_lines(named_logs: tuple[tuple[str, str], ...]) -> list[str]:
    return [
        f"{name}: {line}"
        for name, text in named_logs
        for line in text.splitlines()
        if FATAL_PATTERN.search(line)
    ]


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


def preflight(args: argparse.Namespace) -> int:
    errors: list[str] = []
    prefill_devices = parse_devices(args.prefill_devices)
    decode_devices = parse_devices(args.decode_devices)

    try:
        ipaddress.ip_address(args.host_ip)
    except ValueError:
        errors.append(f"HOST_IP is not a valid IP address: {args.host_ip!r}")

    if len(prefill_devices) != args.tp_size:
        errors.append(
            f"Prefill device count {len(prefill_devices)} does not match TP size {args.tp_size}."
        )
    if len(decode_devices) != args.tp_size:
        errors.append(
            f"Decode device count {len(decode_devices)} does not match TP size {args.tp_size}."
        )
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

    try:
        import custom_ops  # noqa: F401
        import mooncake  # noqa: F401
        import torch
        import torch_npu
        import vllm_ascend

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
                f"Device {device} ADXL range has only {free_count} free ports; "
                f"requires {args.min_adxl_free_ports}."
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
        prompt = (
            case.text
            + "\nAfter reading the text, answer beginning with only the word "
            + f"{case.expected}."
        )
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

    print(f"Results saved to: {output_path}")
    print("FINAL 4K ONLINE PD SUITE: PASSED")
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    evidence_dir = Path(args.output_dir) / f"dsv32-pd-real61-4k-{timestamp}"
    evidence_dir.mkdir(parents=True, exist_ok=False)

    for value in (
        args.validation_output,
        args.prefill_log,
        args.decode_log,
        args.proxy_log,
    ):
        source = Path(value)
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = evidence_dir / source.name
        shutil.copy2(source, destination)

    (evidence_dir / "git-revision.txt").write_text(
        run_command(["git", "-C", args.repo_dir, "rev-parse", "HEAD"])
    )
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
    lifecycle_lines = [
        line
        for line in (*prefill_text.splitlines(), *decode_text.splitlines())
        if "done_sending_msg" in line or "Number of completed KV cache recv requests" in line
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
    preflight_parser.add_argument("--min-adxl-free-ports", type=int, default=2)
    preflight_parser.set_defaults(func=preflight)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--url", required=True)
    validate_parser.add_argument("--model", required=True)
    validate_parser.add_argument("--index-topk", type=int, required=True)
    validate_parser.add_argument("--tp-size", type=int, required=True)
    validate_parser.add_argument("--timeout", type=int, default=1200)
    validate_parser.add_argument("--log-settle-seconds", type=int, default=5)
    validate_parser.add_argument("--output", required=True)
    validate_parser.add_argument("--prefill-log", required=True)
    validate_parser.add_argument("--decode-log", required=True)
    validate_parser.add_argument("--proxy-log", required=True)
    validate_parser.set_defaults(func=validate)

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
    collect_parser.add_argument("--validation-output", required=True)
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

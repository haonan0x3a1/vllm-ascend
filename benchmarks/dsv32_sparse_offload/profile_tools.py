#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import socket
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

MODES = ("baseline", "host")
LATENCY_METRICS = (
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
)
THROUGHPUT_METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
)
SUMMARY_METRICS = LATENCY_METRICS + THROUGHPUT_METRICS
PAIR_KEYS = (
    "campaign",
    "revision",
    "vllm_version",
    "vllm_ascend_version",
    "torch_npu_version",
    "model_path",
    "model_config_sha256",
    "devices",
    "input_len",
    "output_len",
    "num_prompts",
    "max_model_len",
    "tp_size",
    "max_num_batched_tokens",
    "block_size",
    "gpu_memory_utilization",
    "engine_seed",
    "bench_seed",
    "request_rate",
    "max_concurrency",
    "enable_prefill_mc2",
    "enable_mlapo",
    "enable_flashcomm1",
)


def parse_devices(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid device list: {value!r}") from exc
    if not devices:
        raise ValueError("At least one NPU device is required.")
    if len(set(devices)) != len(devices):
        raise ValueError("NPU device IDs must be unique.")
    if any(device < 0 for device in devices):
        raise ValueError("NPU device IDs must be non-negative.")
    return devices


def parse_input_lengths(value: str) -> list[int]:
    try:
        lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid input length list: {value!r}") from exc
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("Input lengths must be positive integers.")
    if len(set(lengths)) != len(lengths):
        raise ValueError("Input lengths must be unique.")
    return lengths


def check_port(host: str, port: int) -> None:
    sock = socket.socket()
    try:
        sock.bind((host, port))
    except OSError as exc:
        raise ValueError(f"Port {host}:{port} is busy: {exc}") from exc
    finally:
        sock.close()


def validate_profile_config(
    devices: str,
    tp_size: int,
    input_lengths: str,
    output_len: int,
    max_model_len: int,
    max_num_seqs: int,
    max_concurrency: int,
) -> tuple[list[int], list[int]]:
    parsed_devices = parse_devices(devices)
    parsed_lengths = parse_input_lengths(input_lengths)
    if len(parsed_devices) != tp_size:
        raise ValueError(f"TP size {tp_size} requires {tp_size} devices, got {len(parsed_devices)}.")
    if max_num_seqs != 1:
        raise ValueError("Host Sparse KV Offload currently requires MAX_NUM_SEQS=1.")
    if max_concurrency != 1:
        raise ValueError("The fair baseline/Host comparison requires BENCH_MAX_CONCURRENCY=1.")
    if output_len <= 1:
        raise ValueError("BENCH_OUTPUT_LEN must exceed 1 so TPOT is meaningful.")
    invalid = [length for length in parsed_lengths if length + output_len > max_model_len]
    if invalid:
        raise ValueError(
            "Input plus output length exceeds MAX_MODEL_LEN for input lengths: " + ", ".join(map(str, invalid))
        )
    return parsed_devices, parsed_lengths


def _as_int(result: dict[str, Any], key: str) -> int:
    try:
        return int(result[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Result has invalid {key!r}: {result.get(key)!r}") from exc


def validate_result(result: dict[str, Any], source: Path) -> None:
    mode = result.get("offload_mode")
    if mode not in MODES:
        raise ValueError(f"{source}: invalid offload_mode {mode!r}")
    completed = _as_int(result, "completed")
    failed = _as_int(result, "failed")
    expected = _as_int(result, "num_prompts")
    if failed or completed != expected:
        raise ValueError(
            f"{source}: benchmark did not fully succeed (completed={completed}, failed={failed}, expected={expected})"
        )
    for key in PAIR_KEYS:
        if key not in result:
            raise ValueError(f"{source}: missing pairing metadata {key!r}")
    for metric in SUMMARY_METRICS:
        value = result.get(metric)
        if not isinstance(value, (int, float)):
            raise ValueError(f"{source}: missing numeric metric {metric!r}")
    for field in ("ttfts", "itls", "output_lens"):
        values = result.get(field)
        if not isinstance(values, list) or len(values) != completed:
            raise ValueError(f"{source}: detailed field {field!r} must contain {completed} request samples")


def load_results(result_dir: Path) -> list[dict[str, Any]]:
    results = []
    for source in sorted(result_dir.glob("*.json")):
        result = json.loads(source.read_text(encoding="utf-8"))
        validate_result(result, source)
        result["_source"] = str(source)
        results.append(result)
    if not results:
        raise ValueError(f"No benchmark JSON files found in {result_dir}")
    return results


def _comparison_key(result: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(result[key]) for key in PAIR_KEYS)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile without samples.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution_metrics(prefix: str, values: list[float]) -> dict[str, float]:
    return {
        f"mean_{prefix}_ms": statistics.fmean(values),
        f"median_{prefix}_ms": statistics.median(values),
        f"p99_{prefix}_ms": _percentile(values, 99.0),
    }


def _pooled_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    ttfts_ms = []
    tpots_ms = []
    itls_ms = []
    e2els_ms = []
    total_duration = 0.0
    total_completed = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for result in results:
        total_duration += float(result["duration"])
        total_completed += int(result["completed"])
        total_input_tokens += int(result["total_input_tokens"])
        total_output_tokens += int(result["total_output_tokens"])
        for ttft, request_itls, output_len in zip(result["ttfts"], result["itls"], result["output_lens"]):
            ttft = float(ttft)
            request_itls = [float(value) for value in request_itls]
            output_len = int(output_len)
            ttfts_ms.append(ttft * 1000.0)
            itls_ms.extend(value * 1000.0 for value in request_itls)
            e2els_ms.append((ttft + sum(request_itls)) * 1000.0)
            if output_len > 1:
                tpots_ms.append(sum(request_itls) / (output_len - 1) * 1000.0)

    if total_duration <= 0:
        raise ValueError("Benchmark duration must be positive.")
    metrics = {}
    for prefix, values in (
        ("ttft", ttfts_ms),
        ("tpot", tpots_ms),
        ("itl", itls_ms),
        ("e2el", e2els_ms),
    ):
        metrics.update(_distribution_metrics(prefix, values))
    metrics.update(
        {
            "request_throughput": total_completed / total_duration,
            "output_throughput": total_output_tokens / total_duration,
            "total_token_throughput": (total_input_tokens + total_output_tokens) / total_duration,
        }
    )
    return metrics


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        grouped[_comparison_key(result)][str(result["offload_mode"])].append(result)

    comparisons = []
    for key, by_mode in sorted(grouped.items()):
        missing = sorted(set(MODES) - set(by_mode))
        if missing:
            metadata = dict(zip(PAIR_KEYS, key))
            raise ValueError(f"Missing modes {missing} for comparison {metadata}")
        baseline_runs = by_mode["baseline"]
        host_runs = by_mode["host"]
        if len(baseline_runs) != len(host_runs):
            raise ValueError(
                "Baseline and Host repetition counts differ for "
                f"input_len={dict(zip(PAIR_KEYS, key))['input_len']}: "
                f"{len(baseline_runs)} != {len(host_runs)}"
            )

        baseline = _pooled_metrics(baseline_runs)
        host = _pooled_metrics(host_runs)
        delta = {
            metric: ((host[metric] - baseline[metric]) / baseline[metric] * 100.0) if baseline[metric] else None
            for metric in SUMMARY_METRICS
        }
        comparisons.append(
            {
                "configuration": dict(zip(PAIR_KEYS, key)),
                "repetitions": len(baseline_runs),
                "baseline_mean": baseline,
                "host_mean": host,
                "host_vs_baseline_pct": delta,
                "baseline_sources": [result["_source"] for result in baseline_runs],
                "host_sources": [result["_source"] for result in host_runs],
            }
        )
    return {"comparisons": comparisons}


def print_summary(summary: dict[str, Any]) -> None:
    header = (
        "input  reps  baseline TTFT  host TTFT  delta TTFT  baseline TPOT  host TPOT  delta TPOT  output tok/s delta"
    )
    print(header)
    print("-" * len(header))
    for comparison in summary["comparisons"]:
        config = comparison["configuration"]
        baseline = comparison["baseline_mean"]
        host = comparison["host_mean"]
        delta = comparison["host_vs_baseline_pct"]
        print(
            f"{config['input_len']:>5}  "
            f"{comparison['repetitions']:>4}  "
            f"{baseline['mean_ttft_ms']:>13.2f}  "
            f"{host['mean_ttft_ms']:>9.2f}  "
            f"{delta['mean_ttft_ms']:>10.2f}%  "
            f"{baseline['mean_tpot_ms']:>13.2f}  "
            f"{host['mean_tpot_ms']:>9.2f}  "
            f"{delta['mean_tpot_ms']:>10.2f}%  "
            f"{delta['output_throughput']:>17.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--devices", required=True)
    preflight.add_argument("--tp-size", type=int, required=True)
    preflight.add_argument("--input-lengths", required=True)
    preflight.add_argument("--output-len", type=int, required=True)
    preflight.add_argument("--max-model-len", type=int, required=True)
    preflight.add_argument("--max-num-seqs", type=int, required=True)
    preflight.add_argument("--max-concurrency", type=int, required=True)
    preflight.add_argument("--host", required=True)
    preflight.add_argument("--port", type=int, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "preflight":
        devices, input_lengths = validate_profile_config(
            args.devices,
            args.tp_size,
            args.input_lengths,
            args.output_len,
            args.max_model_len,
            args.max_num_seqs,
            args.max_concurrency,
        )
        check_port(args.host, args.port)
        print(f"Devices: {devices}")
        print(f"Input lengths: {input_lengths}")
        print(f"Application port {args.host}:{args.port}: FREE")
        print("Profiling configuration: PASSED")
    else:
        results = load_results(args.result_dir)
        summary = summarize_results(results)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print_summary(summary)
        print(f"Summary saved to: {args.output}")


if __name__ == "__main__":
    main()

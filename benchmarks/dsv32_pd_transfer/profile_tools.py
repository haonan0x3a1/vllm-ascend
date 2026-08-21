# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

MODES = ("npu_staging", "memfabric_bm")
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
    "hf_overrides",
    "prefill_devices",
    "decode_devices",
    "input_len",
    "output_len",
    "num_prompts",
    "num_warmups",
    "max_model_len",
    "max_num_seqs",
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
    "sparse_kv_offload_mode",
    "memfabric_bm_protocol",
)


def parse_devices(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid device list: {value!r}") from exc
    if not devices or len(devices) != len(set(devices)) or any(device < 0 for device in devices):
        raise ValueError(f"Device IDs must be a non-empty list of unique non-negative integers: {value!r}")
    return devices


def parse_input_lengths(value: str) -> list[int]:
    try:
        lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid input length list: {value!r}") from exc
    if not lengths or len(lengths) != len(set(lengths)) or any(length <= 0 for length in lengths):
        raise ValueError(f"Input lengths must be unique positive integers: {value!r}")
    return lengths


def validate_profile_config(
    *,
    prefill_devices: str,
    decode_devices: str,
    tp_size: int,
    input_lengths: str,
    output_len: int,
    max_model_len: int,
    max_num_seqs: int,
    max_concurrency: int,
) -> tuple[list[int], list[int], list[int]]:
    prefill = parse_devices(prefill_devices)
    decode = parse_devices(decode_devices)
    inputs = parse_input_lengths(input_lengths)
    if len(prefill) != tp_size or len(decode) != tp_size:
        raise ValueError(
            f"TP size {tp_size} requires {tp_size} Prefill and Decode devices, got {len(prefill)} and {len(decode)}."
        )
    overlap = sorted(set(prefill) & set(decode))
    if overlap:
        raise ValueError(f"Prefill and Decode device sets overlap: {overlap}")
    if max_num_seqs != 1 or max_concurrency != 1:
        raise ValueError("Current sparse Host Full-KV profiling requires max_num_seqs=1 and max_concurrency=1.")
    if output_len <= 1:
        raise ValueError("BENCH_OUTPUT_LEN must exceed 1 so TPOT is meaningful.")
    invalid = [length for length in inputs if length + output_len > max_model_len]
    if invalid:
        raise ValueError(f"Input plus output exceeds MAX_MODEL_LEN for: {invalid}")
    return prefill, decode, inputs


def _as_int(result: dict[str, Any], key: str) -> int:
    try:
        return int(result[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Result has invalid {key!r}: {result.get(key)!r}") from exc


def validate_result(result: dict[str, Any], source: Path) -> None:
    mode = result.get("transfer_mode")
    if mode not in MODES:
        raise ValueError(f"{source}: invalid transfer_mode {mode!r}")
    completed = _as_int(result, "completed")
    failed = _as_int(result, "failed")
    expected = _as_int(result, "num_prompts")
    if failed or completed != expected:
        raise ValueError(
            f"{source}: benchmark incomplete (completed={completed}, failed={failed}, expected={expected})"
        )
    for key in PAIR_KEYS:
        if key not in result:
            raise ValueError(f"{source}: missing pairing metadata {key!r}")
    for metric in SUMMARY_METRICS:
        if not isinstance(result.get(metric), (int, float)):
            raise ValueError(f"{source}: missing numeric metric {metric!r}")


def load_result(source: Path) -> dict[str, Any]:
    result = json.loads(source.read_text(encoding="utf-8"))
    validate_result(result, source)
    result["_source"] = str(source)
    return result


def load_results(result_dir: Path) -> list[dict[str, Any]]:
    results = [load_result(source) for source in sorted(result_dir.glob("*.json"))]
    if not results:
        raise ValueError(f"No benchmark JSON files found in {result_dir}")
    return results


def _comparison_key(result: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(result[key]) for key in PAIR_KEYS)


def _mean_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    return {metric: statistics.fmean(float(result[metric]) for result in results) for metric in SUMMARY_METRICS}


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        grouped[_comparison_key(result)][str(result["transfer_mode"])].append(result)

    comparisons = []
    for key, by_mode in sorted(grouped.items()):
        missing = sorted(set(MODES) - set(by_mode))
        metadata = dict(zip(PAIR_KEYS, key))
        if missing:
            raise ValueError(f"Missing modes {missing} for comparison {metadata}")
        baseline_runs = by_mode["npu_staging"]
        memfabric_runs = by_mode["memfabric_bm"]
        if len(baseline_runs) != len(memfabric_runs):
            raise ValueError(
                "npu_staging and memfabric_bm repetition counts differ for "
                f"input_len={metadata['input_len']}: {len(baseline_runs)} != {len(memfabric_runs)}"
            )
        baseline = _mean_metrics(baseline_runs)
        memfabric = _mean_metrics(memfabric_runs)
        delta = {
            metric: ((memfabric[metric] - baseline[metric]) / baseline[metric] * 100.0) if baseline[metric] else None
            for metric in SUMMARY_METRICS
        }
        comparisons.append(
            {
                "configuration": metadata,
                "repetitions": len(baseline_runs),
                "npu_staging_mean": baseline,
                "memfabric_bm_mean": memfabric,
                "memfabric_vs_npu_staging_pct": delta,
                "npu_staging_sources": [result["_source"] for result in baseline_runs],
                "memfabric_bm_sources": [result["_source"] for result in memfabric_runs],
            }
        )
    return {"comparisons": comparisons}


def print_summary(summary: dict[str, Any]) -> None:
    print("input  reps  NPU TTFT  BM TTFT  delta TTFT  NPU TPOT  BM TPOT  delta TPOT  output tok/s delta")
    for item in summary["comparisons"]:
        config = item["configuration"]
        baseline = item["npu_staging_mean"]
        memfabric = item["memfabric_bm_mean"]
        delta = item["memfabric_vs_npu_staging_pct"]
        print(
            f"{config['input_len']:>5}  {item['repetitions']:>4}  "
            f"{baseline['mean_ttft_ms']:>8.2f}  {memfabric['mean_ttft_ms']:>7.2f}  "
            f"{delta['mean_ttft_ms']:>10.2f}%  {baseline['mean_tpot_ms']:>8.2f}  "
            f"{memfabric['mean_tpot_ms']:>7.2f}  {delta['mean_tpot_ms']:>10.2f}%  "
            f"{delta['output_throughput']:>17.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--prefill-devices", required=True)
    preflight.add_argument("--decode-devices", required=True)
    preflight.add_argument("--tp-size", type=int, required=True)
    preflight.add_argument("--input-lengths", required=True)
    preflight.add_argument("--output-len", type=int, required=True)
    preflight.add_argument("--max-model-len", type=int, required=True)
    preflight.add_argument("--max-num-seqs", type=int, required=True)
    preflight.add_argument("--max-concurrency", type=int, required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate-result")
    validate.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "preflight":
        prefill, decode, inputs = validate_profile_config(
            prefill_devices=args.prefill_devices,
            decode_devices=args.decode_devices,
            tp_size=args.tp_size,
            input_lengths=args.input_lengths,
            output_len=args.output_len,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_concurrency=args.max_concurrency,
        )
        print(f"PD profiling config: PASS; Prefill={prefill}, Decode={decode}, inputs={inputs}")
        return

    if args.command == "validate-result":
        result = load_result(args.result)
        print(
            "Benchmark result: PASS; "
            f"source={args.result}, completed={result['completed']}, failed={result['failed']}"
        )
        return

    summary = summarize_results(load_results(args.result_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"Summary saved to: {args.output}")


if __name__ == "__main__":
    main()

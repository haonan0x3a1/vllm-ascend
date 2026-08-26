# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest

PROFILE_TOOLS_PATH = Path(__file__).parents[2] / "benchmarks" / "dsv32_pd_transfer" / "profile_tools.py"
PROFILE_RUN_SCRIPT_PATH = Path(__file__).parents[2] / "benchmarks" / "dsv32_pd_transfer" / "run.sh"
RUNTIME_RUN_SCRIPT_PATH = (
    Path(__file__).parents[2] / "examples" / "disaggregated_prefill_v1" / "dsv32_sparse_offload_poc" / "run.sh"
)
SPARSE_OFFLOAD_PATH = Path(__file__).parents[2] / "vllm_ascend" / "attention" / "sparse_kv_offload.py"
CONNECTOR_PATH = (
    Path(__file__).parents[2]
    / "vllm_ascend"
    / "distributed"
    / "kv_transfer"
    / "kv_p2p"
    / "mooncake_layerwise_connector.py"
)
SPEC = importlib.util.spec_from_file_location("dsv32_pd_transfer_profile_tools", PROFILE_TOOLS_PATH)
assert SPEC is not None and SPEC.loader is not None
PROFILE_TOOLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE_TOOLS)


def test_targeted_profiler_enables_custom_scopes_without_changing_default() -> None:
    run_script = RUNTIME_RUN_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "ENABLE_TORCH_PROFILER=${ENABLE_TORCH_PROFILER:-false}" in run_script
    assert "export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1" in run_script
    assert 'profiler_args=(--profiler-config "$profiler_config")' in run_script


def test_benchmark_readiness_checks_both_roles_and_proxy_health() -> None:
    run_script = PROFILE_RUN_SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"http://127.0.0.1:$PREFILL_API_PORT/v1/models"' in run_script
    assert '"http://127.0.0.1:$DECODE_API_PORT/v1/models"' in run_script
    assert '"http://$HOST_IP:$PROXY_PORT/healthcheck"' in run_script
    assert '"http://$HOST_IP:$PROXY_PORT/v1/models"' not in run_script


def test_benchmark_validates_each_result_before_continuing() -> None:
    run_script = PROFILE_RUN_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'profile_tools.py" validate-result' in run_script
    assert '--result "$result_dir/$result_file"' in run_script
    assert 'ready >/dev/null' in run_script


def test_profile_scopes_cover_every_target_data_path_stage() -> None:
    source = SPARSE_OFFLOAD_PATH.read_text(encoding="utf-8") + CONNECTOR_PATH.read_text(encoding="utf-8")

    for scope in (
        "copy_full_kv_npu_to_host",
        "wait_prefill_full_kv_visible",
        "memfabric_full_kv_host_to_host",
        "mooncake_indexer_npu_to_npu",
        "layer_ack_round_trip",
        "decode_full_kv_visibility_fence",
        "gather_selected_kv_host_to_npu",
    ):
        assert f"dsv32_pd_transfer:{scope}" in source


def _result(mode: str, run: int, *, revision: str = "abc123") -> dict:
    result = {
        "transfer_mode": mode,
        "completed": 5,
        "failed": 0,
        "campaign": "same-host-v1",
        "revision": revision,
        "vllm_version": "0.23.0",
        "vllm_ascend_version": "0.19.1",
        "torch_npu_version": "2.10.0",
        "model_path": "/models/DeepSeek-V3.2-W4A8C8",
        "model_config_sha256": "deadbeef",
        "hf_overrides": '{"num_hidden_layers":61}',
        "prefill_devices": "0,1,2,3,4,5,6,7",
        "decode_devices": "8,9,10,11,12,13,14,15",
        "input_len": 3018,
        "output_len": 32,
        "num_prompts": 5,
        "num_warmups": 1,
        "max_model_len": 4096,
        "max_num_seqs": 1,
        "tp_size": 8,
        "max_num_batched_tokens": 256,
        "block_size": 128,
        "gpu_memory_utilization": 0.9,
        "engine_seed": 1024,
        "bench_seed": 0,
        "request_rate": "inf",
        "max_concurrency": 1,
        "enable_prefill_mc2": "true",
        "enable_mlapo": "false",
        "enable_flashcomm1": "false",
        "sparse_kv_offload_mode": "host",
        "memfabric_bm_protocol": "host_tcp",
        "_source": f"{mode}-{run}.json",
    }
    base = 100.0 if mode == "npu_staging" else 110.0
    for metric in PROFILE_TOOLS.SUMMARY_METRICS:
        result[metric] = base + run
    return result


def test_profile_config_requires_disjoint_tp8_devices() -> None:
    prefill, decode, inputs = PROFILE_TOOLS.validate_profile_config(
        prefill_devices="0,1,2,3,4,5,6,7",
        decode_devices="8,9,10,11,12,13,14,15",
        tp_size=8,
        input_lengths="1818,3018,3618",
        output_len=32,
        max_model_len=4096,
        max_num_seqs=1,
        max_concurrency=1,
    )

    assert prefill == list(range(8))
    assert decode == list(range(8, 16))
    assert inputs == [1818, 3018, 3618]


def test_profile_config_rejects_overlapping_device_sets() -> None:
    with pytest.raises(ValueError, match="overlap"):
        PROFILE_TOOLS.validate_profile_config(
            prefill_devices="0,1",
            decode_devices="1,2",
            tp_size=2,
            input_lengths="3018",
            output_len=32,
            max_model_len=4096,
            max_num_seqs=1,
            max_concurrency=1,
        )


def test_summary_pairs_equal_repetitions_and_reports_delta() -> None:
    results = [
        _result("npu_staging", 1),
        _result("npu_staging", 2),
        _result("memfabric_bm", 1),
        _result("memfabric_bm", 2),
    ]

    summary = PROFILE_TOOLS.summarize_results(results)

    comparison = summary["comparisons"][0]
    assert comparison["repetitions"] == 2
    assert comparison["npu_staging_mean"]["mean_ttft_ms"] == 101.5
    assert comparison["memfabric_bm_mean"]["mean_ttft_ms"] == 111.5
    assert comparison["memfabric_vs_npu_staging_pct"]["mean_ttft_ms"] == pytest.approx(9.8522167488)


def test_result_validation_rejects_failed_requests(tmp_path: Path) -> None:
    result = _result("memfabric_bm", 1)
    result["completed"] = 0
    result["failed"] = 5
    source = tmp_path / "failed.json"
    source.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match=r"completed=0, failed=5, expected=5"):
        PROFILE_TOOLS.load_result(source)


def test_summary_refuses_to_compare_different_revisions() -> None:
    with pytest.raises(ValueError, match="Missing modes"):
        PROFILE_TOOLS.summarize_results(
            [
                _result("npu_staging", 1, revision="baseline-revision"),
                _result("memfabric_bm", 1, revision="different-revision"),
            ]
        )


def test_trace_summary_reports_per_role_trace_means(tmp_path: Path) -> None:
    traces = tmp_path / "traces" / "memfabric_bm"
    prefill_0 = traces / "prefill" / "rank0" / "trace_view.json"
    prefill_1 = traces / "prefill" / "rank1" / "trace_view.json"
    decode_0 = traces / "decode" / "rank0" / "trace_view.json"
    for source in (prefill_0, prefill_1, decode_0):
        source.parent.mkdir(parents=True, exist_ok=True)

    prefill_0.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "dsv32_pd_transfer:memfabric_full_kv_host_to_host",
                        "ph": "X",
                        "dur": 2000,
                    },
                    {"name": "unrelated", "ph": "X", "dur": 9999},
                ]
            }
        ),
        encoding="utf-8",
    )
    prefill_1.write_text(
        json.dumps(
            [
                {
                    "name": "dsv32_pd_transfer:memfabric_full_kv_host_to_host",
                    "ph": "B",
                    "ts": 100,
                    "pid": 1,
                    "tid": 2,
                },
                {"ph": "E", "ts": 4100, "pid": 1, "tid": 2},
            ]
        ),
        encoding="utf-8",
    )
    decode_0.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "dsv32_pd_transfer:decode_full_kv_visibility_fence",
                        "ph": "X",
                        "dur": 500,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = PROFILE_TOOLS.summarize_trace_scopes(traces)

    by_stage = {(item["role"], item["stage"]): item for item in summary["stages"]}
    host_transfer = by_stage[
        ("prefill", "dsv32_pd_transfer:memfabric_full_kv_host_to_host")
    ]
    assert host_transfer["trace_files"] == 2
    assert host_transfer["events"] == 2
    assert host_transfer["mean_per_trace_total_ms"] == 3.0
    assert host_transfer["mean_event_ms"] == 3.0
    assert host_transfer["max_event_ms"] == 4.0

    fence = by_stage[
        ("decode", "dsv32_pd_transfer:decode_full_kv_visibility_fence")
    ]
    assert fence["mean_per_trace_total_ms"] == 0.5


def test_analyze_ascend_traces_converts_only_missing_outputs(tmp_path: Path) -> None:
    traces = tmp_path / "traces" / "memfabric_bm"
    pending = traces / "prefill" / "rank0_ascend_pt"
    existing = traces / "decode" / "rank0_ascend_pt"
    pending.mkdir(parents=True)
    existing_output = existing / "ASCEND_PROFILER_OUTPUT"
    existing_output.mkdir(parents=True)
    (existing_output / "trace_view.json").write_text("{}", encoding="utf-8")
    calls = []

    def fake_analyse(source: str) -> None:
        calls.append(source)
        output = Path(source) / "ASCEND_PROFILER_OUTPUT"
        output.mkdir()
        (output / "trace_view.json").write_text("{}", encoding="utf-8")
        print("verbose analyser output")

    analysis_log = tmp_path / "analysis.log"
    result = PROFILE_TOOLS.analyze_ascend_trace_dirs(
        traces,
        analysis_log,
        fake_analyse,
    )

    assert calls == [str(pending)]
    assert result["raw_trace_dirs"] == 2
    assert result["analyzed"] == 1
    assert result["skipped_existing"] == 1
    assert "verbose analyser output" in analysis_log.read_text(encoding="utf-8")


def test_trace_summary_rejects_missing_custom_scopes(tmp_path: Path) -> None:
    source = tmp_path / "trace_view.json"
    source.write_text(
        json.dumps({"traceEvents": [{"name": "unrelated", "ph": "X", "dur": 1}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No dsv32_pd_transfer"):
        PROFILE_TOOLS.summarize_trace_scopes(tmp_path)

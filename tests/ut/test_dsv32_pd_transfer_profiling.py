# SPDX-License-Identifier: Apache-2.0
import importlib.util
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


def test_benchmark_readiness_uses_the_proxy_healthcheck_route() -> None:
    run_script = PROFILE_RUN_SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"http://$HOST_IP:$PROXY_PORT/healthcheck"' in run_script
    assert '"http://$HOST_IP:$PROXY_PORT/v1/models"' not in run_script


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


def test_summary_refuses_to_compare_different_revisions() -> None:
    with pytest.raises(ValueError, match="Missing modes"):
        PROFILE_TOOLS.summarize_results(
            [
                _result("npu_staging", 1, revision="baseline-revision"),
                _result("memfabric_bm", 1, revision="different-revision"),
            ]
        )

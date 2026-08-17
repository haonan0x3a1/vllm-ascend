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

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "dsv32_sparse_offload" / "profile_tools.py"
SPEC = importlib.util.spec_from_file_location("dsv32_sparse_offload_profile_tools", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROFILE_TOOLS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILE_TOOLS
SPEC.loader.exec_module(PROFILE_TOOLS)


def _result(mode: str, run_id: int, ttft: float, tpot: float, throughput: float):
    result = {
        "campaign": "smoke-v1",
        "offload_mode": mode,
        "revision": "abc123",
        "vllm_version": "0.23.0",
        "vllm_ascend_version": "0.19.1",
        "torch_npu_version": "2.10.0",
        "model_path": "/models/dsv32",
        "model_config_sha256": "deadbeef",
        "devices": "0,1,2,3,4,5,6,7",
        "input_len": "3072",
        "output_len": "32",
        "num_prompts": 5,
        "max_model_len": "4096",
        "tp_size": "8",
        "max_num_batched_tokens": "256",
        "block_size": "128",
        "gpu_memory_utilization": "0.90",
        "engine_seed": "1024",
        "bench_seed": "0",
        "request_rate": "inf",
        "max_concurrency": 1,
        "enable_prefill_mc2": "true",
        "enable_mlapo": "false",
        "enable_flashcomm1": "false",
        "run_id": str(run_id),
        "completed": 5,
        "failed": 0,
    }
    for metric in PROFILE_TOOLS.SUMMARY_METRICS:
        result[metric] = 1.0
    result["mean_ttft_ms"] = ttft
    result["mean_tpot_ms"] = tpot
    result["output_throughput"] = throughput
    result["_source"] = f"{mode}-{run_id}.json"
    return result


def test_validate_profile_config_requires_fair_single_request_settings():
    devices, lengths = PROFILE_TOOLS.validate_profile_config(
        "0,1,2,3,4,5,6,7",
        8,
        "2048,3072",
        32,
        4096,
        1,
        1,
    )

    assert devices == list(range(8))
    assert lengths == [2048, 3072]

    with pytest.raises(ValueError, match="MAX_NUM_SEQS=1"):
        PROFILE_TOOLS.validate_profile_config("0", 1, "128", 32, 4096, 2, 1)
    with pytest.raises(ValueError, match="BENCH_MAX_CONCURRENCY=1"):
        PROFILE_TOOLS.validate_profile_config("0", 1, "128", 32, 4096, 1, 2)


def test_validate_profile_config_rejects_length_overflow():
    with pytest.raises(ValueError, match="exceeds MAX_MODEL_LEN"):
        PROFILE_TOOLS.validate_profile_config("0,1", 2, "4090", 32, 4096, 1, 1)


def test_summarize_pairs_repetitions_and_computes_host_delta():
    results = [
        _result("baseline", 1, 100.0, 10.0, 4.0),
        _result("baseline", 2, 120.0, 12.0, 4.0),
        _result("host", 1, 132.0, 13.2, 3.0),
        _result("host", 2, 132.0, 13.2, 3.0),
    ]

    summary = PROFILE_TOOLS.summarize_results(results)

    comparison = summary["comparisons"][0]
    assert comparison["repetitions"] == 2
    assert comparison["baseline_mean"]["mean_ttft_ms"] == 110.0
    assert comparison["host_mean"]["mean_ttft_ms"] == 132.0
    assert comparison["host_vs_baseline_pct"]["mean_ttft_ms"] == pytest.approx(20.0)
    assert comparison["host_vs_baseline_pct"]["output_throughput"] == pytest.approx(-25.0)


def test_summarize_rejects_missing_or_unbalanced_pairs():
    with pytest.raises(ValueError, match="Missing modes"):
        PROFILE_TOOLS.summarize_results([_result("baseline", 1, 100, 10, 4)])

    with pytest.raises(ValueError, match="repetition counts differ"):
        PROFILE_TOOLS.summarize_results(
            [
                _result("baseline", 1, 100, 10, 4),
                _result("baseline", 2, 100, 10, 4),
                _result("host", 1, 100, 10, 4),
            ]
        )

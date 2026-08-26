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

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "disaggregated_prefill_v1"
    / "dsv32_sparse_offload_poc"
    / "poc_tools.py"
)
RUN_SCRIPT_PATH = MODULE_PATH.with_name("run.sh")
SPEC = importlib.util.spec_from_file_location("dsv32_sparse_offload_poc_tools", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
POC_TOOLS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POC_TOOLS
SPEC.loader.exec_module(POC_TOOLS)


def test_service_log_capture_ignores_sigint_until_cleanup_finishes():
    run_script = RUN_SCRIPT_PATH.read_text(encoding="utf-8")

    assert '2>&1 | tee -i "$log_file"' in run_script


def test_runtime_config_exposes_opt_in_stage_metrics():
    run_script = RUN_SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"sparse_kv_stage_metrics": stage_metrics == "true"' in run_script
    assert "stage-summary)" in run_script
    assert '--stage-metrics-output "$STAGE_METRICS_OUTPUT"' in run_script
    assert '2>&1 | tee -i "$PROXY_LOG"' in run_script


def test_runtime_modules_load_torch_before_custom_extensions():
    imported_names = []

    def fake_importer(name: str) -> ModuleType:
        imported_names.append(name)
        return ModuleType(name)

    modules = POC_TOOLS.import_runtime_modules(fake_importer)

    assert imported_names == list(POC_TOOLS.RUNTIME_IMPORT_ORDER)
    assert list(modules) == list(POC_TOOLS.RUNTIME_IMPORT_ORDER)


def test_preflight_clears_inherited_role_device_visibility():
    environment = {
        "ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
        "KEEP_ME": "value",
    }

    removed = POC_TOOLS.clear_inherited_device_visibility(environment)

    assert removed == "8,9,10,11,12,13,14,15"
    assert environment == {"KEEP_ME": "value"}


def test_port_probe_matches_runtime_socket_reuse(monkeypatch):
    calls = []

    class FakeSocket:
        def setsockopt(self, level, option, value):
            calls.append(("setsockopt", level, option, value))

        def bind(self, address):
            calls.append(("bind", address))

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(POC_TOOLS.socket, "socket", FakeSocket)

    assert POC_TOOLS.can_bind(43172) == (True, None)
    assert calls == [
        (
            "setsockopt",
            POC_TOOLS.socket.SOL_SOCKET,
            POC_TOOLS.socket.SO_REUSEADDR,
            1,
        ),
        ("bind", ("0.0.0.0", 43172)),
        ("close",),
    ]


def test_unreserved_ephemeral_ports_are_rejected():
    assert POC_TOOLS.parse_port_range("32768 60999\n") == (32768, 60999)
    assert POC_TOOLS.parse_reserved_port_ranges("36000-36007, 37400") == (
        (36000, 36007),
        (37400, 37400),
    )
    assert POC_TOOLS.find_unreserved_ephemeral_ports(
        [18000, 36000, 36007, 36401, 37400, 61000],
        (32768, 60999),
        ((36000, 36007), (37400, 37400)),
    ) == [36401]


def test_memfabric_same_host_ports_are_distinct_per_tp_role():
    ports = POC_TOOLS.memfabric_bm_required_ports(
        tp_size=2,
        store_port_base=22200,
        hcom_port_base=22300,
    )

    assert ports == (
        22200,
        22201,
        22300,
        22301,
        22304,
        22305,
        22302,
        22303,
        22306,
        22307,
    )
    assert len(ports) == len(set(ports))


def test_memfabric_bm_api_is_resolved_from_package_export():
    package = ModuleType("memfabric_hybrid")
    bm_api = ModuleType("bm")
    package.bm = bm_api

    assert POC_TOOLS.resolve_memfabric_bm_api(package) is bm_api


def test_count_lifecycle_records_uses_specific_completion_events():
    request_id = "chatcmpl-test"
    prefill_log = "\n".join(
        [
            f"unrelated {request_id}",
            *(f"rank {rank} done_sending_msg {request_id}" for rank in range(8)),
        ]
    )
    decode_log = "\n".join(
        [
            f"scheduler {request_id}",
            *(f"rank {rank} Number of completed KV cache recv requests: 1 {request_id}" for rank in range(8)),
        ]
    )

    assert POC_TOOLS.count_lifecycle_records(request_id, prefill_log, decode_log) == (8, 8)


def test_memfabric_runtime_markers_require_every_tp_worker():
    tp_size = 2
    prefill_log = "\n".join(
        (
            *(POC_TOOLS.MEMFABRIC_BM_ALLOCATOR_INIT_MARKER for _ in range(tp_size)),
            *(POC_TOOLS.MEMFABRIC_BM_DATA_PLANE_MARKER for _ in range(tp_size)),
        )
    )
    decode_log = "\n".join(
        (
            *(POC_TOOLS.MEMFABRIC_BM_ALLOCATOR_INIT_MARKER for _ in range(tp_size)),
            *(POC_TOOLS.MEMFABRIC_BM_VISIBILITY_FENCE_MARKER for _ in range(tp_size)),
        )
    )

    assert POC_TOOLS.require_memfabric_bm_runtime_markers(prefill_log, decode_log, tp_size) == {
        "prefill_allocators": 2,
        "decode_allocators": 2,
        "prefill_data_plane": 2,
        "decode_visibility_fence": 2,
    }


def test_memfabric_runtime_markers_reject_a_silent_data_plane_fallback():
    tp_size = 2
    prefill_log = "\n".join(POC_TOOLS.MEMFABRIC_BM_ALLOCATOR_INIT_MARKER for _ in range(tp_size))
    decode_log = "\n".join(
        (
            *(POC_TOOLS.MEMFABRIC_BM_ALLOCATOR_INIT_MARKER for _ in range(tp_size)),
            *(POC_TOOLS.MEMFABRIC_BM_VISIBILITY_FENCE_MARKER for _ in range(tp_size)),
        )
    )

    try:
        POC_TOOLS.require_memfabric_bm_runtime_markers(prefill_log, decode_log, tp_size)
    except RuntimeError as exc:
        assert "prefill_data_plane" in str(exc)
    else:
        raise AssertionError("missing BM data-plane markers must fail validation")


def test_memfabric_shutdown_markers_require_every_tp_worker():
    tp_size = 2
    prefill_log = "\n".join(POC_TOOLS.MEMFABRIC_BM_ALLOCATOR_RELEASE_MARKER for _ in range(tp_size))
    decode_log = "\n".join(POC_TOOLS.MEMFABRIC_BM_ALLOCATOR_RELEASE_MARKER for _ in range(tp_size))

    assert POC_TOOLS.require_memfabric_bm_shutdown_markers(prefill_log, decode_log, tp_size) == {
        "prefill_releases": 2,
        "decode_releases": 2,
    }


def test_memfabric_shutdown_markers_reject_a_missing_worker_release():
    tp_size = 2
    prefill_log = POC_TOOLS.MEMFABRIC_BM_ALLOCATOR_RELEASE_MARKER
    decode_log = "\n".join(POC_TOOLS.MEMFABRIC_BM_ALLOCATOR_RELEASE_MARKER for _ in range(tp_size))

    try:
        POC_TOOLS.require_memfabric_bm_shutdown_markers(prefill_log, decode_log, tp_size)
    except RuntimeError as exc:
        assert "prefill_releases" in str(exc)
    else:
        raise AssertionError("missing allocator release markers must fail validation")


def test_pd_transfer_stage_summary_requires_all_workers_and_stages():
    prefill_stages = {
        stage: {"count": 2, "total_ms": 6.0, "mean_ms": 3.0, "max_ms": 4.0}
        for stage in POC_TOOLS.EXPECTED_PD_TRANSFER_STAGES["prefill"]
    }
    decode_stages = {
        stage: {"count": 2, "total_ms": 2.0, "mean_ms": 1.0, "max_ms": 1.5}
        for stage in POC_TOOLS.EXPECTED_PD_TRANSFER_STAGES["decode"]
    }

    def record(role: str, rank: int, stages: dict) -> str:
        payload = {
            "role": role,
            "tp_rank": rank,
            "request_ids": ["request-1"],
            "status": "completed",
            "stages": stages,
        }
        return f"prefix {POC_TOOLS.PD_TRANSFER_STAGE_METRICS_MARKER} {json.dumps(payload)}"

    summary = POC_TOOLS.summarize_pd_transfer_stage_metrics(
        "\n".join(record("prefill", rank, prefill_stages) for rank in range(2)),
        "\n".join(record("decode", rank, decode_stages) for rank in range(2)),
        2,
    )

    rows = {(row["role"], row["stage"]): row for row in summary["stages"]}
    host_to_host = rows[("prefill", "memfabric_full_kv_host_to_host")]
    assert host_to_host["workers"] == 2
    assert host_to_host["events"] == 4
    assert host_to_host["mean_per_worker_total_ms"] == 6.0
    assert host_to_host["mean_event_ms"] == 3.0


def test_pd_transfer_stage_summary_rejects_missing_worker():
    payload = {
        "role": "prefill",
        "tp_rank": 0,
        "status": "completed",
        "stages": {},
    }
    prefill_log = f"{POC_TOOLS.PD_TRANSFER_STAGE_METRICS_MARKER} {json.dumps(payload)}"

    try:
        POC_TOOLS.summarize_pd_transfer_stage_metrics(prefill_log, "", 2)
    except RuntimeError as exc:
        assert "expected 2 prefill worker records" in str(exc)
    else:
        raise AssertionError("missing stage-metrics worker record must fail")


def test_pd_transfer_stage_summary_rejects_duplicate_rank():
    def record(role: str, stages: dict) -> str:
        payload = {
            "role": role,
            "tp_rank": 0,
            "status": "completed",
            "stages": stages,
        }
        return f"{POC_TOOLS.PD_TRANSFER_STAGE_METRICS_MARKER} {json.dumps(payload)}"

    prefill_stages = {
        stage: {"count": 1, "total_ms": 1.0, "mean_ms": 1.0, "max_ms": 1.0}
        for stage in POC_TOOLS.EXPECTED_PD_TRANSFER_STAGES["prefill"]
    }
    decode_stages = {
        stage: {"count": 1, "total_ms": 1.0, "mean_ms": 1.0, "max_ms": 1.0}
        for stage in POC_TOOLS.EXPECTED_PD_TRANSFER_STAGES["decode"]
    }

    try:
        POC_TOOLS.summarize_pd_transfer_stage_metrics(
            "\n".join(record("prefill", prefill_stages) for _ in range(2)),
            "\n".join(record("decode", decode_stages) for _ in range(2)),
            2,
        )
    except RuntimeError as exc:
        assert "expected prefill TP ranks" in str(exc)
    else:
        raise AssertionError("duplicate stage-metrics TP rank must fail")


def test_find_fatal_lines_ignores_unrelated_warnings():
    matches = POC_TOOLS.find_fatal_lines(
        (
            ("prefill", "WARNING harmless\nRuntimeError: transfer failed"),
            ("decode", "Application startup complete"),
        )
    )

    assert matches == ["prefill: RuntimeError: transfer failed"]


def test_shutdown_filter_removes_engine_dead_after_manager_stopped():
    text = "\n".join(
        (
            "INFO [shutdown] MPClient: engine manager stopped",
            "ERROR AsyncLLM output_handler failed.",
            "ERROR Traceback (most recent call last):",
            "ERROR vllm.v1.engine.exceptions.EngineDeadError: stopped",
            "INFO [shutdown] API server: engine client stopped",
            "INFO Application shutdown complete.",
        )
    )

    filtered = POC_TOOLS.remove_expected_async_llm_shutdown_cascade(text)

    assert "Traceback" not in filtered
    assert "EngineDeadError" not in filtered
    assert "API server: engine client stopped" in filtered


def test_shutdown_filter_keeps_engine_dead_without_orderly_manager_stop():
    text = "\n".join(
        (
            "ERROR AsyncLLM output_handler failed.",
            "ERROR Traceback (most recent call last):",
            "ERROR vllm.v1.engine.exceptions.EngineDeadError: crashed",
            "INFO [shutdown] API server: engine client stopped",
        )
    )

    assert POC_TOOLS.remove_expected_async_llm_shutdown_cascade(text) == text


def test_parse_devices_rejects_duplicate_physical_cards():
    try:
        POC_TOOLS.parse_devices("0,1,1")
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate devices must be rejected")


def test_validation_cases_cover_both_topk_sides_and_request_reset():
    cases = POC_TOOLS.build_validation_cases()

    assert any(not case.expect_above_topk for case in cases)
    assert sum(case.expect_above_topk for case in cases) == 4
    assert cases[0].expected == cases[-1].expected == "alpha"

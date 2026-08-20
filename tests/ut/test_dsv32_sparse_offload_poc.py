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
SPEC = importlib.util.spec_from_file_location("dsv32_sparse_offload_poc_tools", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
POC_TOOLS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POC_TOOLS
SPEC.loader.exec_module(POC_TOOLS)


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


def test_memfabric_same_host_ports_are_distinct_per_tp_role():
    ports = POC_TOOLS.memfabric_bm_required_ports(
        tp_size=2,
        store_port_base=37200,
        hcom_port_base=37400,
    )

    assert ports == (37200, 37201, 37400, 37401, 37404, 37405)
    assert len(ports) == len(set(ports))


def test_memfabric_bm_api_is_resolved_from_package_export():
    package = ModuleType("memfabric_hybrid")
    bm_api = ModuleType("bm")
    setattr(package, "bm", bm_api)

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
            *(
                f"rank {rank} Number of completed KV cache recv requests: 1 {request_id}"
                for rank in range(8)
            ),
        ]
    )

    assert POC_TOOLS.count_lifecycle_records(request_id, prefill_log, decode_log) == (8, 8)


def test_find_fatal_lines_ignores_unrelated_warnings():
    matches = POC_TOOLS.find_fatal_lines(
        (
            ("prefill", "WARNING harmless\nRuntimeError: transfer failed"),
            ("decode", "Application startup complete"),
        )
    )

    assert matches == ["prefill: RuntimeError: transfer failed"]


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

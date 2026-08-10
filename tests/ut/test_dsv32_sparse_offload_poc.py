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

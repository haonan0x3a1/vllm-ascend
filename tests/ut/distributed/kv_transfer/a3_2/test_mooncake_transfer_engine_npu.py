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
# This file is a part of the vllm-ascend project.
#
"""Real Mooncake TransferEngine smoke test on two Ascend NPUs.

The shared UT ``conftest.py`` installs a fake ``mooncake.engine`` module so
CPU-only tests can import connector code.  Consequently, the native Mooncake
extension must be exercised in fresh subprocesses rather than in the pytest
process.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import IO, Any

import pytest
from packaging.version import Version

ALIGNMENT_BYTES = 2 * 1024 * 1024
BUFFER_BYTES = 2 * 1024 * 1024
BUFFER_PATTERN = 0x5A
HOST_DESTINATION_SENTINEL = -1
HOST_TRANSFER_OFFSET_BYTES = 4096
HOST_TRANSFER_BYTES = 64 * 1024
HOST_TRANSFER_REPEAT_COUNT = 20
HOST_TRANSFER_END_BYTES = HOST_TRANSFER_OFFSET_BYTES + HOST_TRANSFER_REPEAT_COUNT * HOST_TRANSFER_BYTES
HOST_TCP_FORCE_ENV = "MC_FORCE_TCP"
HOST_TCP_FORCE_VALUE = "1"
HOST_TCP_RUNTIME_MARKER = "MC_FORCE_TCP is set, using TCP transport only"
ASCEND_RUNTIME_MARKER = "install AscendDirectTransport"
HYBRID_INDEXER_SENTINEL = -1
HYBRID_INDEXER_TRANSFER_BYTES = 256 * 1024
HYBRID_INDEXER_FIRST_OFFSET_BYTES = 4096
HYBRID_INDEXER_GUARD_BYTES = 4096
HYBRID_INDEXER_SECOND_OFFSET_BYTES = (
    HYBRID_INDEXER_FIRST_OFFSET_BYTES + HYBRID_INDEXER_TRANSFER_BYTES + HYBRID_INDEXER_GUARD_BYTES
)
HYBRID_INDEXER_END_BYTES = HYBRID_INDEXER_SECOND_OFFSET_BYTES + HYBRID_INDEXER_TRANSFER_BYTES
HYBRID_INDEXER_FIRST_PATTERN = 0x2A
HYBRID_INDEXER_SECOND_PATTERN = 0x4B
MIXED_TRANSFER_OFFSET_BYTES = 4096
MIXED_TRANSFER_BYTES = 256 * 1024
MIXED_DESTINATION_SENTINEL = -1
MIXED_BUFFER_SPECS = (
    ("full_nope", "swapped", 0x11),
    ("full_rope", "swapped", 0x22),
    ("indexer", "npu", 0x33),
)
STAGED_BUFFER_SPECS = (
    ("full_nope", "npu", 0x11),
    ("full_rope", "npu", 0x22),
    ("indexer", "npu", 0x33),
)
FULL_KV_BUFFER_NAMES = ("full_nope", "full_rope")
INDEXER_BUFFER_NAME = "indexer"
CONNECTOR_BLOCK_VALUES = (
    (-1, 0x11, -1, 0x33),
    (-1, 0x22, -1, 0x44),
)
MIN_MOONCAKE_VERSION = Version("0.3.12.post1")
PROCESS_TIMEOUT_SECONDS = 60
HYBRID_PROCESS_TIMEOUT_SECONDS = 120
READY_MARKER = "MOONCAKE_NPU_SMOKE_READY="
RESULT_MARKER = "MOONCAKE_NPU_SMOKE_RESULT="


class _ProcessOutput:
    """Continuously drain a child pipe while exposing JSON marker waits."""

    def __init__(self, stream: IO[str]) -> None:
        self._stream = stream
        self._lines: list[str] = []
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._read,
            name="mooncake-smoke-output-reader",
            daemon=True,
        )
        self._thread.start()

    def _read(self) -> None:
        try:
            for line in self._stream:
                self._lines.append(line)
                self._queue.put(line)
        finally:
            self._queue.put(None)

    def wait_for_marker(
        self,
        marker: str,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {marker!r}.\nChild output:\n{self.output}")
            try:
                line = self._queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(f"Timed out waiting for {marker!r}.\nChild output:\n{self.output}") from exc
            if line is None:
                raise RuntimeError(f"Child exited before emitting {marker!r}.\nChild output:\n{self.output}")
            if line.startswith(marker):
                return json.loads(line.removeprefix(marker))

    def join(self, timeout: float = 5) -> None:
        self._thread.join(timeout=timeout)

    @property
    def output(self) -> str:
        return "".join(self._lines)


def _stop_process(
    process: subprocess.Popen[str],
    output: _ProcessOutput,
) -> None:
    """Stop a child process without leaving a native Mooncake worker behind."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    output.join()


def _allocate_aligned_buffer(
    torch_module,
    *,
    device_index: int,
    fill_value: int,
):
    raw = torch_module.empty(
        BUFFER_BYTES + ALIGNMENT_BYTES,
        dtype=torch_module.int8,
        device=f"npu:{device_index}",
    )
    aligned_address = (raw.data_ptr() + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES * ALIGNMENT_BYTES
    offset = aligned_address - raw.data_ptr()
    buffer = raw[offset : offset + BUFFER_BYTES]
    if buffer.data_ptr() % ALIGNMENT_BYTES != 0:
        raise RuntimeError(f"Buffer address {buffer.data_ptr():#x} is not 2 MiB aligned.")
    buffer.fill_(fill_value)
    return raw, buffer


def _allocate_aligned_host_buffer(
    torch_module,
    *,
    fill_value: int,
    pinned: bool,
):
    """Allocate a stable 2 MiB-aligned Host registration region."""
    raw = torch_module.empty(
        BUFFER_BYTES + ALIGNMENT_BYTES,
        dtype=torch_module.int8,
        device="cpu",
        pin_memory=pinned,
    )
    aligned_address = (raw.data_ptr() + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES * ALIGNMENT_BYTES
    offset = aligned_address - raw.data_ptr()
    buffer = raw[offset : offset + BUFFER_BYTES]
    if buffer.data_ptr() % ALIGNMENT_BYTES != 0:
        raise RuntimeError(f"Host buffer address {buffer.data_ptr():#x} is not 2 MiB aligned.")
    buffer.fill_(fill_value)
    return raw, buffer


def _allocate_connector_staging(
    torch_module,
    *,
    device_index: int,
    fill_value: int,
):
    raw, combined = _allocate_aligned_buffer(
        torch_module,
        device_index=device_index,
        fill_value=fill_value,
    )
    staging = (
        combined[: BUFFER_BYTES // 2].view(4, -1),
        combined[BUFFER_BYTES // 2 :].view(4, -1),
    )
    return raw, combined, staging


def _fill_connector_payload(staging) -> None:
    for tensor, expected_blocks in zip(staging, CONNECTOR_BLOCK_VALUES):
        for block_id, value in enumerate(expected_blocks):
            tensor[block_id].fill_(value)


def _verify_connector_staging_and_final(torch_module, staging, final) -> dict[str, Any]:
    """Read swapped Full KV through NPU and verify every physical block."""
    final_npu_copy = tuple(torch_module.empty_like(tensor) for tensor in staging)
    for npu_copy, swapped_tensor in zip(final_npu_copy, final):
        # Reading a framework swapped tensor directly with .cpu() can enter
        # devmm_h2h_copy and segfault on the validated CANN runtime. Mirror the
        # production Gather direction by restoring it to NPU first.
        npu_copy.copy_(swapped_tensor, non_blocking=False)
    torch_module.npu.synchronize()

    verification: dict[str, bool] = {}
    for tensor_name, staging_tensor, final_tensor, expected_blocks in zip(
        FULL_KV_BUFFER_NAMES,
        staging,
        final_npu_copy,
        CONNECTOR_BLOCK_VALUES,
    ):
        staging_cpu = staging_tensor.cpu()
        final_cpu = final_tensor.cpu()
        for block_id, expected_value in enumerate(expected_blocks):
            verification[f"{tensor_name}_staging_block_{block_id}"] = bool(
                torch_module.all(staging_cpu[block_id] == expected_value)
            )
            verification[f"{tensor_name}_final_block_{block_id}"] = bool(
                torch_module.all(final_cpu[block_id] == expected_value)
            )
    return verification


def _allocate_connector_final(torch_module, torch_npu_module, staging, *, device_index: int):
    final = tuple(
        torch_npu_module.empty_with_swapped_memory(
            tensor.shape,
            dtype=tensor.dtype,
            device=torch_module.device(f"npu:{device_index}"),
        )
        for tensor in staging
    )
    for tensor in final:
        tensor.fill_(-1)
    return final


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _allocate_mixed_buffer(
    torch_module,
    torch_npu_module,
    *,
    memory_kind: str,
    device_index: int,
    fill_value: int,
):
    """Match the production allocator for one sparse-offload cache tensor."""
    if memory_kind == "swapped":
        buffer = torch_npu_module.empty_with_swapped_memory(
            (BUFFER_BYTES,),
            dtype=torch_module.int8,
            device=torch_module.device(f"npu:{device_index}"),
        )
        owner = buffer
    elif memory_kind == "npu":
        owner, buffer = _allocate_aligned_buffer(
            torch_module,
            device_index=device_index,
            fill_value=fill_value,
        )
    else:
        raise ValueError(f"Unknown memory kind: {memory_kind!r}")
    buffer.fill_(fill_value)
    return owner, buffer


def _register_buffer(
    engine,
    buffer,
    *,
    memory_kind: str,
    device_index: int,
) -> None:
    if memory_kind == "npu":
        location = f"npu:{device_index}"
    elif memory_kind == "swapped":
        location = "cpu"
    else:
        raise ValueError(f"Unknown memory kind: {memory_kind!r}")

    result = engine.register_memory(
        buffer.data_ptr(),
        BUFFER_BYTES,
        location,
    )
    if result != 0:
        raise RuntimeError(
            f"Mooncake memory registration failed: memory_kind={memory_kind}, location={location!r}, result={result}"
        )


def _initialize_engine(device_index: int):
    import torch
    import torch_npu  # noqa: F401
    from mooncake.engine import TransferEngine
    from vllm.utils.network_utils import get_ip

    if torch.npu.device_count() < 2:
        raise RuntimeError("Mooncake NPU transfer smoke test requires at least two visible NPUs.")
    torch.npu.set_device(device_index)
    engine = TransferEngine()
    host = get_ip()
    result = engine.initialize(host, "P2PHANDSHAKE", "ascend", "")
    if result != 0:
        raise RuntimeError(f"TransferEngine initialization failed on NPU {device_index}: result={result}")
    return torch, engine, host


def _initialize_host_engine():
    """Initialize one engine from the Ascend wheel in TCP-only mode."""
    import torch
    import torch_npu  # noqa: F401
    from mooncake.engine import TransferEngine
    from vllm.utils.network_utils import get_ip

    if os.environ.get(HOST_TCP_FORCE_ENV) != HOST_TCP_FORCE_VALUE:
        raise RuntimeError(
            "The Ascend Mooncake wheel requires MC_FORCE_TCP=1 before "
            "engine initialization for a TCP-only Host transport probe."
        )
    engine = TransferEngine()
    host = get_ip()
    result = engine.initialize(host, "P2PHANDSHAKE", "tcp", "")
    if result != 0:
        raise RuntimeError(f"Host TransferEngine TCP initialization failed: result={result}")
    return torch, engine, host


def _initialize_coexisting_engines(device_index: int):
    """Create an Ascend TE, then a TCP-only TE in the same process."""
    if HOST_TCP_FORCE_ENV in os.environ:
        raise RuntimeError(
            "Hybrid probe must start without MC_FORCE_TCP so the first "
            "TransferEngine remains on the Ascend transport."
        )
    torch, ascend_engine, host = _initialize_engine(device_index)
    try:
        os.environ[HOST_TCP_FORCE_ENV] = HOST_TCP_FORCE_VALUE
        _, host_engine, host_engine_host = _initialize_host_engine()
    finally:
        os.environ.pop(HOST_TCP_FORCE_ENV, None)
    if host_engine_host != host:
        raise RuntimeError(
            "Ascend and Host TransferEngines resolved different local hosts: "
            f"ascend={host!r}, host={host_engine_host!r}."
        )
    ascend_port = int(ascend_engine.get_rpc_port())
    host_port = int(host_engine.get_rpc_port())
    if ascend_port == host_port:
        raise RuntimeError(f"Ascend and Host TransferEngines reused RPC port {ascend_port}.")
    return torch, ascend_engine, host_engine, host


def _register_host_buffer(engine, buffer) -> None:
    result = engine.register_memory(
        buffer.data_ptr(),
        BUFFER_BYTES,
        "cpu",
    )
    if result != 0:
        raise RuntimeError(f"Mooncake Host memory registration failed: result={result}")


def _host_payload_value(transfer_index: int) -> int:
    return transfer_index + 1


def _validate_host_transfer_layout() -> None:
    if HOST_TRANSFER_END_BYTES > BUFFER_BYTES:
        raise RuntimeError("Host transfer windows exceed the registered buffer.")


def _verify_host_transfer_buffer(torch_module, buffer) -> dict[str, Any]:
    _validate_host_transfer_layout()
    prefix_matches = bool(torch_module.all(buffer[:HOST_TRANSFER_OFFSET_BYTES] == HOST_DESTINATION_SENTINEL).item())
    payload_matches = []
    for transfer_index in range(HOST_TRANSFER_REPEAT_COUNT):
        start = HOST_TRANSFER_OFFSET_BYTES + transfer_index * HOST_TRANSFER_BYTES
        end = start + HOST_TRANSFER_BYTES
        payload_matches.append(
            bool(torch_module.all(buffer[start:end] == _host_payload_value(transfer_index)).item())
        )
    suffix_matches = bool(torch_module.all(buffer[HOST_TRANSFER_END_BYTES:] == HOST_DESTINATION_SENTINEL).item())
    return {
        "prefix_matches": prefix_matches,
        "payload_matches": payload_matches,
        "suffix_matches": suffix_matches,
        "matches": prefix_matches and all(payload_matches) and suffix_matches,
        "first_payload_byte": int(buffer[HOST_TRANSFER_OFFSET_BYTES].item()),
        "last_payload_byte": int(buffer[HOST_TRANSFER_END_BYTES - 1].item()),
    }


def _verify_hybrid_indexer_buffer(torch_module, buffer) -> dict[str, Any]:
    if HYBRID_INDEXER_END_BYTES > BUFFER_BYTES:
        raise RuntimeError("Hybrid Indexer transfer windows exceed the registered buffer.")
    received = buffer.cpu()
    first_end = HYBRID_INDEXER_FIRST_OFFSET_BYTES + HYBRID_INDEXER_TRANSFER_BYTES
    second_end = HYBRID_INDEXER_SECOND_OFFSET_BYTES + HYBRID_INDEXER_TRANSFER_BYTES
    prefix_matches = bool(
        torch_module.all(received[:HYBRID_INDEXER_FIRST_OFFSET_BYTES] == HYBRID_INDEXER_SENTINEL).item()
    )
    first_matches = bool(
        torch_module.all(
            received[HYBRID_INDEXER_FIRST_OFFSET_BYTES:first_end] == HYBRID_INDEXER_FIRST_PATTERN
        ).item()
    )
    middle_guard_matches = bool(
        torch_module.all(received[first_end:HYBRID_INDEXER_SECOND_OFFSET_BYTES] == HYBRID_INDEXER_SENTINEL).item()
    )
    second_matches = bool(
        torch_module.all(
            received[HYBRID_INDEXER_SECOND_OFFSET_BYTES:second_end] == HYBRID_INDEXER_SECOND_PATTERN
        ).item()
    )
    suffix_matches = bool(torch_module.all(received[second_end:] == HYBRID_INDEXER_SENTINEL).item())
    return {
        "prefix_matches": prefix_matches,
        "first_matches": first_matches,
        "middle_guard_matches": middle_guard_matches,
        "second_matches": second_matches,
        "suffix_matches": suffix_matches,
        "matches": (
            prefix_matches
            and first_matches
            and middle_guard_matches
            and second_matches
            and suffix_matches
        ),
    }


def _extract_json_marker(output: str, marker: str) -> dict[str, Any]:
    matches = [
        json.loads(line.removeprefix(marker))
        for line in output.splitlines()
        if line.startswith(marker)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {marker!r} record, found {len(matches)}.\n"
            f"Process output:\n{output}"
        )
    return matches[0]


def _unregister_buffers(
    engine,
    registered_pointers: list[int],
) -> list[tuple[int, int]]:
    failures: list[tuple[int, int]] = []
    for pointer in reversed(registered_pointers):
        result = engine.unregister_memory(pointer)
        if result != 0:
            failures.append((pointer, result))
    return failures


def _verify_mixed_buffer(
    torch_module,
    buffer,
    *,
    memory_kind: str,
    device_index: int,
    expected_value: int,
) -> dict[str, Any]:
    """Verify the transfer window and untouched guards for one cache tensor."""
    if memory_kind == "swapped":
        npu_copy = torch_module.empty(
            buffer.shape,
            dtype=buffer.dtype,
            device=f"npu:{device_index}",
        )
        npu_copy.copy_(buffer)
        torch_module.npu.synchronize()
        received = npu_copy.cpu()
    else:
        received = buffer.cpu()

    window_end = MIXED_TRANSFER_OFFSET_BYTES + MIXED_TRANSFER_BYTES
    prefix_matches = bool(torch_module.all(received[:MIXED_TRANSFER_OFFSET_BYTES] == MIXED_DESTINATION_SENTINEL).item())
    payload_matches = bool(torch_module.all(received[MIXED_TRANSFER_OFFSET_BYTES:window_end] == expected_value).item())
    suffix_matches = bool(torch_module.all(received[window_end:] == MIXED_DESTINATION_SENTINEL).item())
    return {
        "prefix_matches": prefix_matches,
        "payload_matches": payload_matches,
        "suffix_matches": suffix_matches,
        "first_payload_byte": int(received[MIXED_TRANSFER_OFFSET_BYTES].item()),
        "last_payload_byte": int(received[window_end - 1].item()),
    }


def _run_receiver() -> int:
    torch, engine, host = _initialize_engine(device_index=1)
    raw = buffer = None
    registered = False
    unregister_result = 0
    try:
        raw, buffer = _allocate_aligned_buffer(
            torch,
            device_index=1,
            fill_value=-1,
        )
        torch.npu.synchronize()
        result = engine.register_memory(buffer.data_ptr(), BUFFER_BYTES)
        if result != 0:
            raise RuntimeError(f"Receiver memory registration failed: result={result}")
        registered = True
        print(
            READY_MARKER
            + json.dumps(
                {
                    "host": host,
                    "port": engine.get_rpc_port(),
                    "address": buffer.data_ptr(),
                    "size": BUFFER_BYTES,
                }
            ),
            flush=True,
        )

        command = sys.stdin.readline().strip()
        if command != "VERIFY":
            raise RuntimeError(f"Receiver expected VERIFY command, received {command!r}.")

        torch.npu.synchronize()
        received = buffer.cpu()
        matches = bool(torch.all(received == BUFFER_PATTERN).item())
        print(
            RESULT_MARKER
            + json.dumps(
                {
                    "matches": matches,
                    "first_byte": int(received[0].item()),
                    "last_byte": int(received[-1].item()),
                }
            ),
            flush=True,
        )
        if not matches:
            raise RuntimeError("Receiver buffer does not match the sent pattern.")
        return 0
    finally:
        torch.npu.synchronize()
        if registered and buffer is not None:
            unregister_result = engine.unregister_memory(buffer.data_ptr())
        del engine
        del buffer
        del raw
        gc.collect()
        if unregister_result != 0:
            raise RuntimeError(f"Receiver memory unregistration failed: result={unregister_result}")


def _run_host_receiver(*, pinned: bool) -> int:
    _validate_host_transfer_layout()
    torch, engine, host = _initialize_host_engine()
    raw = buffer = None
    registered = False
    unregister_result = 0
    try:
        raw, buffer = _allocate_aligned_host_buffer(
            torch,
            fill_value=HOST_DESTINATION_SENTINEL,
            pinned=pinned,
        )
        _register_host_buffer(engine, buffer)
        registered = True
        print(
            READY_MARKER
            + json.dumps(
                {
                    "host": host,
                    "port": engine.get_rpc_port(),
                    "address": buffer.data_ptr(),
                    "size": BUFFER_BYTES,
                    "pinned": pinned,
                    "force_tcp": os.environ.get(HOST_TCP_FORCE_ENV),
                }
            ),
            flush=True,
        )

        command = sys.stdin.readline().strip()
        if command != "VERIFY":
            raise RuntimeError(f"Host receiver expected VERIFY command, received {command!r}.")

        verification = _verify_host_transfer_buffer(torch, buffer)
        print(
            RESULT_MARKER
            + json.dumps(
                {
                    **verification,
                    "pinned": pinned,
                }
            ),
            flush=True,
        )
        if not verification["matches"]:
            raise RuntimeError("Host receiver buffer does not match the sent payloads and guards.")
        return 0
    finally:
        if registered and buffer is not None:
            unregister_result = engine.unregister_memory(buffer.data_ptr())
        del engine
        del buffer
        del raw
        gc.collect()
        if unregister_result != 0:
            raise RuntimeError(f"Host receiver memory unregistration failed: result={unregister_result}")


def _run_hybrid_receiver() -> int:
    """Keep Ascend and TCP engines alive while receiving both memory paths."""
    _validate_host_transfer_layout()
    torch, ascend_engine, host_engine, host = _initialize_coexisting_engines(device_index=1)
    indexer_raw = indexer_buffer = host_raw = host_buffer = None
    indexer_registered = False
    host_registered = False
    indexer_unregister_result = 0
    host_unregister_result = 0
    try:
        indexer_raw, indexer_buffer = _allocate_aligned_buffer(
            torch,
            device_index=1,
            fill_value=HYBRID_INDEXER_SENTINEL,
        )
        host_raw, host_buffer = _allocate_aligned_host_buffer(
            torch,
            fill_value=HOST_DESTINATION_SENTINEL,
            pinned=True,
        )
        torch.npu.synchronize()
        _register_buffer(
            ascend_engine,
            indexer_buffer,
            memory_kind="npu",
            device_index=1,
        )
        indexer_registered = True
        _register_host_buffer(host_engine, host_buffer)
        host_registered = True

        print(
            READY_MARKER
            + json.dumps(
                {
                    "host": host,
                    "ascend_port": ascend_engine.get_rpc_port(),
                    "host_port": host_engine.get_rpc_port(),
                    "indexer_address": indexer_buffer.data_ptr(),
                    "host_address": host_buffer.data_ptr(),
                    "registration_size": BUFFER_BYTES,
                    "force_tcp_after_init": os.environ.get(HOST_TCP_FORCE_ENV),
                }
            ),
            flush=True,
        )

        command = sys.stdin.readline().strip()
        if command != "VERIFY":
            raise RuntimeError(f"Hybrid receiver expected VERIFY command, received {command!r}.")

        torch.npu.synchronize()
        indexer_verification = _verify_hybrid_indexer_buffer(torch, indexer_buffer)
        host_verification = _verify_host_transfer_buffer(torch, host_buffer)
        matches = indexer_verification["matches"] and host_verification["matches"]
        print(
            RESULT_MARKER
            + json.dumps(
                {
                    "matches": matches,
                    "indexer_verification": indexer_verification,
                    "host_verification": host_verification,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not matches:
            raise RuntimeError("Hybrid receiver observed corrupted NPU or Host data.")
        return 0
    finally:
        torch.npu.synchronize()
        if host_registered and host_buffer is not None:
            host_unregister_result = host_engine.unregister_memory(host_buffer.data_ptr())
        if indexer_registered and indexer_buffer is not None:
            indexer_unregister_result = ascend_engine.unregister_memory(indexer_buffer.data_ptr())
        del host_engine
        del ascend_engine
        del host_buffer
        del host_raw
        del indexer_buffer
        del indexer_raw
        gc.collect()
        if host_unregister_result != 0 or indexer_unregister_result != 0:
            raise RuntimeError(
                "Hybrid receiver memory unregistration failed: "
                f"host={host_unregister_result}, indexer={indexer_unregister_result}"
            )


def _run_mixed_receiver() -> int:
    import torch_npu

    torch, engine, host = _initialize_engine(device_index=1)
    owners = []
    buffers: dict[str, Any] = {}
    registered_pointers: list[int] = []
    unregister_failures: list[tuple[int, int]] = []
    try:
        metadata = []
        for name, memory_kind, _ in MIXED_BUFFER_SPECS:
            owner, buffer = _allocate_mixed_buffer(
                torch,
                torch_npu,
                memory_kind=memory_kind,
                device_index=1,
                fill_value=MIXED_DESTINATION_SENTINEL,
            )
            owners.append(owner)
            buffers[name] = buffer
            result = engine.register_memory(buffer.data_ptr(), BUFFER_BYTES)
            if result != 0:
                raise RuntimeError(f"Receiver {name} registration failed: result={result}")
            registered_pointers.append(buffer.data_ptr())
            metadata.append(
                {
                    "name": name,
                    "memory_kind": memory_kind,
                    "base_address": buffer.data_ptr(),
                    "target_address": (buffer.data_ptr() + MIXED_TRANSFER_OFFSET_BYTES),
                    "registration_size": BUFFER_BYTES,
                    "transfer_size": MIXED_TRANSFER_BYTES,
                    "base_alignment_mod": (buffer.data_ptr() % ALIGNMENT_BYTES),
                }
            )
        torch.npu.synchronize()
        print(
            READY_MARKER
            + json.dumps(
                {
                    "host": host,
                    "port": engine.get_rpc_port(),
                    "buffers": metadata,
                }
            ),
            flush=True,
        )

        command = sys.stdin.readline().strip()
        if command != "VERIFY":
            raise RuntimeError(f"Mixed receiver expected VERIFY command, got {command!r}.")

        torch.npu.synchronize()
        verification = {}
        for name, memory_kind, expected_value in MIXED_BUFFER_SPECS:
            verification[name] = _verify_mixed_buffer(
                torch,
                buffers[name],
                memory_kind=memory_kind,
                device_index=1,
                expected_value=expected_value,
            )
        matches = all(
            all(
                result[key]
                for key in (
                    "prefix_matches",
                    "payload_matches",
                    "suffix_matches",
                )
            )
            for result in verification.values()
        )
        print(
            RESULT_MARKER
            + json.dumps(
                {
                    "matches": matches,
                    "buffers": verification,
                }
            ),
            flush=True,
        )
        if not matches:
            raise RuntimeError("Mixed sparse-offload destination buffers failed verification.")
        return 0
    finally:
        torch.npu.synchronize()
        unregister_failures = _unregister_buffers(
            engine,
            registered_pointers,
        )
        del engine
        buffers.clear()
        owners.clear()
        gc.collect()
        if unregister_failures:
            raise RuntimeError(f"Mixed receiver memory unregistration failed: {unregister_failures}")


def _run_staged_receiver() -> int:
    """Receive into NPU staging, then copy Full KV into swapped memory."""
    import torch_npu

    torch, engine, host = _initialize_engine(device_index=1)
    owners = []
    staging_buffers: dict[str, Any] = {}
    final_buffers: dict[str, Any] = {}
    registered_pointers: list[int] = []
    unregister_failures: list[tuple[int, int]] = []
    try:
        metadata = []
        for name, memory_kind, _ in STAGED_BUFFER_SPECS:
            if memory_kind == "npu":
                owner, buffer = _allocate_aligned_buffer(
                    torch,
                    device_index=1,
                    fill_value=MIXED_DESTINATION_SENTINEL,
                )
            else:
                raise ValueError(f"Unsupported staged receiver memory kind: {memory_kind!r}")

            owners.append(owner)
            staging_buffers[name] = buffer
            _register_buffer(
                engine,
                buffer,
                memory_kind=memory_kind,
                device_index=1,
            )
            registered_pointers.append(buffer.data_ptr())
            metadata.append(
                {
                    "name": name,
                    "memory_kind": memory_kind,
                    "base_address": buffer.data_ptr(),
                    "target_address": (buffer.data_ptr() + MIXED_TRANSFER_OFFSET_BYTES),
                    "registration_size": BUFFER_BYTES,
                    "transfer_size": MIXED_TRANSFER_BYTES,
                    "base_alignment_mod": (buffer.data_ptr() % ALIGNMENT_BYTES),
                }
            )

        for name in FULL_KV_BUFFER_NAMES:
            owner, final_buffer = _allocate_mixed_buffer(
                torch,
                torch_npu,
                memory_kind="swapped",
                device_index=1,
                fill_value=MIXED_DESTINATION_SENTINEL,
            )
            owners.append(owner)
            final_buffers[name] = final_buffer

        torch.npu.synchronize()
        print(
            READY_MARKER
            + json.dumps(
                {
                    "host": host,
                    "port": engine.get_rpc_port(),
                    "buffers": metadata,
                }
            ),
            flush=True,
        )

        command = sys.stdin.readline().strip()
        if command != "VERIFY":
            raise RuntimeError(f"Staged receiver expected VERIFY command, got {command!r}.")

        torch.npu.synchronize()
        staging_verification = {}
        for name, memory_kind, expected_value in STAGED_BUFFER_SPECS:
            staging_verification[name] = _verify_mixed_buffer(
                torch,
                staging_buffers[name],
                memory_kind=memory_kind,
                device_index=1,
                expected_value=expected_value,
            )

        window_end = MIXED_TRANSFER_OFFSET_BYTES + MIXED_TRANSFER_BYTES
        for name in FULL_KV_BUFFER_NAMES:
            final_buffers[name][MIXED_TRANSFER_OFFSET_BYTES:window_end].copy_(
                staging_buffers[name][MIXED_TRANSFER_OFFSET_BYTES:window_end],
            )
        torch.npu.synchronize()

        final_verification = {
            name: _verify_mixed_buffer(
                torch,
                final_buffers[name],
                memory_kind="swapped",
                device_index=1,
                expected_value=next(
                    expected_value for spec_name, _, expected_value in STAGED_BUFFER_SPECS if spec_name == name
                ),
            )
            for name in FULL_KV_BUFFER_NAMES
        }
        final_verification[INDEXER_BUFFER_NAME] = staging_verification[INDEXER_BUFFER_NAME]

        verification_keys = (
            "prefix_matches",
            "payload_matches",
            "suffix_matches",
        )
        staging_matches = all(all(result[key] for key in verification_keys) for result in staging_verification.values())
        final_matches = all(all(result[key] for key in verification_keys) for result in final_verification.values())
        matches = staging_matches and final_matches
        print(
            RESULT_MARKER
            + json.dumps(
                {
                    "matches": matches,
                    "staging_buffers": staging_verification,
                    "final_buffers": final_verification,
                }
            ),
            flush=True,
        )
        if not matches:
            raise RuntimeError("Staged sparse-offload destination buffers failed verification.")
        return 0
    finally:
        torch.npu.synchronize()
        unregister_failures = _unregister_buffers(
            engine,
            registered_pointers,
        )
        del engine
        staging_buffers.clear()
        final_buffers.clear()
        owners.clear()
        gc.collect()
        if unregister_failures:
            raise RuntimeError(f"Staged receiver memory unregistration failed: {unregister_failures}")


def _run_sender(
    *,
    target_host: str,
    target_port: int,
    target_address: int,
) -> int:
    torch, engine, _ = _initialize_engine(device_index=0)
    raw = buffer = None
    registered = False
    unregister_result = 0
    try:
        raw, buffer = _allocate_aligned_buffer(
            torch,
            device_index=0,
            fill_value=BUFFER_PATTERN,
        )
        torch.npu.synchronize()
        result = engine.register_memory(buffer.data_ptr(), BUFFER_BYTES)
        if result != 0:
            raise RuntimeError(f"Sender memory registration failed: result={result}")
        registered = True

        session = f"{target_host}:{target_port}"
        result = engine.batch_transfer_sync_write(
            session,
            [buffer.data_ptr()],
            [target_address],
            [BUFFER_BYTES],
        )
        if result < 0:
            raise RuntimeError(f"Mooncake NPU transfer failed: session={session}, result={result}")
        print(
            RESULT_MARKER
            + json.dumps(
                {
                    "transfer_result": result,
                    "source_address": buffer.data_ptr(),
                    "target_address": target_address,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        torch.npu.synchronize()
        if registered and buffer is not None:
            unregister_result = engine.unregister_memory(buffer.data_ptr())
        del engine
        del buffer
        del raw
        gc.collect()
        if unregister_result != 0:
            raise RuntimeError(f"Sender memory unregistration failed: result={unregister_result}")


def _run_host_sender(
    *,
    target_host: str,
    target_port: int,
    target_address: int,
    pinned: bool,
) -> int:
    _validate_host_transfer_layout()
    torch, engine, _ = _initialize_host_engine()
    raw = buffer = None
    registered = False
    unregister_result = 0
    transfer_results = []
    try:
        raw, buffer = _allocate_aligned_host_buffer(
            torch,
            fill_value=0,
            pinned=pinned,
        )
        for transfer_index in range(HOST_TRANSFER_REPEAT_COUNT):
            start = HOST_TRANSFER_OFFSET_BYTES + transfer_index * HOST_TRANSFER_BYTES
            end = start + HOST_TRANSFER_BYTES
            buffer[start:end].fill_(_host_payload_value(transfer_index))
        _register_host_buffer(engine, buffer)
        registered = True

        session = f"{target_host}:{target_port}"
        for transfer_index in range(HOST_TRANSFER_REPEAT_COUNT):
            offset = HOST_TRANSFER_OFFSET_BYTES + transfer_index * HOST_TRANSFER_BYTES
            result = engine.batch_transfer_sync_write(
                session,
                [buffer.data_ptr() + offset],
                [target_address + offset],
                [HOST_TRANSFER_BYTES],
            )
            if result < 0:
                raise RuntimeError(
                    "Mooncake Host-to-Host TCP transfer failed: "
                    f"session={session}, transfer_index={transfer_index}, result={result}"
                )
            transfer_results.append(result)
        print(
            RESULT_MARKER
            + json.dumps(
                {
                    "transfer_results": transfer_results,
                    "source_address": buffer.data_ptr(),
                    "target_address": target_address,
                    "pinned": pinned,
                    "force_tcp": os.environ.get(HOST_TCP_FORCE_ENV),
                }
            ),
            flush=True,
        )
        return 0
    finally:
        if registered and buffer is not None:
            unregister_result = engine.unregister_memory(buffer.data_ptr())
        del engine
        del buffer
        del raw
        gc.collect()
        if unregister_result != 0:
            raise RuntimeError(f"Host sender memory unregistration failed: result={unregister_result}")


def _run_hybrid_sender(
    *,
    target_host: str,
    target_ascend_port: int,
    target_host_port: int,
    target_indexer_address: int,
    target_host_address: int,
) -> int:
    """Transfer Indexer A, Host Full KV, then Indexer B with coexisting TEs."""
    _validate_host_transfer_layout()
    torch, ascend_engine, host_engine, _ = _initialize_coexisting_engines(device_index=0)
    indexer_raw = indexer_buffer = host_raw = host_buffer = None
    indexer_registered = False
    host_registered = False
    indexer_unregister_result = 0
    host_unregister_result = 0
    host_transfer_results = []
    try:
        indexer_raw, indexer_buffer = _allocate_aligned_buffer(
            torch,
            device_index=0,
            fill_value=HYBRID_INDEXER_SENTINEL,
        )
        first_end = HYBRID_INDEXER_FIRST_OFFSET_BYTES + HYBRID_INDEXER_TRANSFER_BYTES
        second_end = HYBRID_INDEXER_SECOND_OFFSET_BYTES + HYBRID_INDEXER_TRANSFER_BYTES
        indexer_buffer[HYBRID_INDEXER_FIRST_OFFSET_BYTES:first_end].fill_(
            HYBRID_INDEXER_FIRST_PATTERN
        )
        indexer_buffer[HYBRID_INDEXER_SECOND_OFFSET_BYTES:second_end].fill_(
            HYBRID_INDEXER_SECOND_PATTERN
        )
        host_raw, host_buffer = _allocate_aligned_host_buffer(
            torch,
            fill_value=0,
            pinned=True,
        )
        for transfer_index in range(HOST_TRANSFER_REPEAT_COUNT):
            start = HOST_TRANSFER_OFFSET_BYTES + transfer_index * HOST_TRANSFER_BYTES
            end = start + HOST_TRANSFER_BYTES
            host_buffer[start:end].fill_(_host_payload_value(transfer_index))

        torch.npu.synchronize()
        _register_buffer(
            ascend_engine,
            indexer_buffer,
            memory_kind="npu",
            device_index=0,
        )
        indexer_registered = True
        _register_host_buffer(host_engine, host_buffer)
        host_registered = True

        ascend_session = f"{target_host}:{target_ascend_port}"
        host_session = f"{target_host}:{target_host_port}"
        first_indexer_result = ascend_engine.batch_transfer_sync_write(
            ascend_session,
            [indexer_buffer.data_ptr() + HYBRID_INDEXER_FIRST_OFFSET_BYTES],
            [target_indexer_address + HYBRID_INDEXER_FIRST_OFFSET_BYTES],
            [HYBRID_INDEXER_TRANSFER_BYTES],
        )
        if first_indexer_result < 0:
            raise RuntimeError(
                "First hybrid Ascend transfer failed: "
                f"session={ascend_session}, result={first_indexer_result}"
            )

        for transfer_index in range(HOST_TRANSFER_REPEAT_COUNT):
            offset = HOST_TRANSFER_OFFSET_BYTES + transfer_index * HOST_TRANSFER_BYTES
            result = host_engine.batch_transfer_sync_write(
                host_session,
                [host_buffer.data_ptr() + offset],
                [target_host_address + offset],
                [HOST_TRANSFER_BYTES],
            )
            if result < 0:
                raise RuntimeError(
                    "Hybrid Host transfer failed: "
                    f"session={host_session}, transfer_index={transfer_index}, result={result}"
                )
            host_transfer_results.append(result)

        second_indexer_result = ascend_engine.batch_transfer_sync_write(
            ascend_session,
            [indexer_buffer.data_ptr() + HYBRID_INDEXER_SECOND_OFFSET_BYTES],
            [target_indexer_address + HYBRID_INDEXER_SECOND_OFFSET_BYTES],
            [HYBRID_INDEXER_TRANSFER_BYTES],
        )
        if second_indexer_result < 0:
            raise RuntimeError(
                "Second hybrid Ascend transfer failed after TCP use: "
                f"session={ascend_session}, result={second_indexer_result}"
            )

        print(
            RESULT_MARKER
            + json.dumps(
                {
                    "first_indexer_result": first_indexer_result,
                    "host_transfer_results": host_transfer_results,
                    "second_indexer_result": second_indexer_result,
                    "ascend_port": ascend_engine.get_rpc_port(),
                    "host_port": host_engine.get_rpc_port(),
                    "force_tcp_after_init": os.environ.get(HOST_TCP_FORCE_ENV),
                }
            ),
            flush=True,
        )
        return 0
    finally:
        torch.npu.synchronize()
        if host_registered and host_buffer is not None:
            host_unregister_result = host_engine.unregister_memory(host_buffer.data_ptr())
        if indexer_registered and indexer_buffer is not None:
            indexer_unregister_result = ascend_engine.unregister_memory(indexer_buffer.data_ptr())
        del host_engine
        del ascend_engine
        del host_buffer
        del host_raw
        del indexer_buffer
        del indexer_raw
        gc.collect()
        if host_unregister_result != 0 or indexer_unregister_result != 0:
            raise RuntimeError(
                "Hybrid sender memory unregistration failed: "
                f"host={host_unregister_result}, indexer={indexer_unregister_result}"
            )


def _run_mixed_sender(
    *,
    target_host: str,
    target_port: int,
    target_metadata: list[dict[str, Any]],
) -> int:
    import torch_npu

    torch, engine, _ = _initialize_engine(device_index=0)
    owners = []
    buffers: dict[str, Any] = {}
    registered_pointers: list[int] = []
    unregister_failures: list[tuple[int, int]] = []
    try:
        for name, memory_kind, payload_value in MIXED_BUFFER_SPECS:
            owner, buffer = _allocate_mixed_buffer(
                torch,
                torch_npu,
                memory_kind=memory_kind,
                device_index=0,
                fill_value=0,
            )
            buffer[MIXED_TRANSFER_OFFSET_BYTES : (MIXED_TRANSFER_OFFSET_BYTES + MIXED_TRANSFER_BYTES)].fill_(
                payload_value
            )
            owners.append(owner)
            buffers[name] = buffer
            result = engine.register_memory(buffer.data_ptr(), BUFFER_BYTES)
            if result != 0:
                raise RuntimeError(f"Sender {name} registration failed: result={result}")
            registered_pointers.append(buffer.data_ptr())
        torch.npu.synchronize()

        target_by_name = {str(metadata["name"]): metadata for metadata in target_metadata}
        expected_names = {name for name, _, _ in MIXED_BUFFER_SPECS}
        if set(target_by_name) != expected_names:
            raise RuntimeError(
                "Mixed target metadata names do not match the source tuple: "
                f"expected={sorted(expected_names)}, "
                f"actual={sorted(target_by_name)}"
            )

        source_addresses = []
        target_addresses = []
        lengths = []
        for name, _, _ in MIXED_BUFFER_SPECS:
            target = target_by_name[name]
            if int(target["transfer_size"]) != MIXED_TRANSFER_BYTES:
                raise RuntimeError(f"Unexpected transfer size for {name}: {target['transfer_size']}")
            source_addresses.append(buffers[name].data_ptr() + MIXED_TRANSFER_OFFSET_BYTES)
            target_addresses.append(int(target["target_address"]))
            lengths.append(MIXED_TRANSFER_BYTES)

        session = f"{target_host}:{target_port}"
        result = engine.batch_transfer_sync_write(
            session,
            source_addresses,
            target_addresses,
            lengths,
        )
        if result < 0:
            raise RuntimeError(f"Mooncake mixed sparse-offload transfer failed: session={session}, result={result}")
        print(
            RESULT_MARKER
            + json.dumps(
                {
                    "transfer_result": result,
                    "source_addresses": source_addresses,
                    "target_addresses": target_addresses,
                    "lengths": lengths,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        torch.npu.synchronize()
        unregister_failures = _unregister_buffers(
            engine,
            registered_pointers,
        )
        del engine
        buffers.clear()
        owners.clear()
        gc.collect()
        if unregister_failures:
            raise RuntimeError(f"Mixed sender memory unregistration failed: {unregister_failures}")


def _run_staged_sender(
    *,
    target_host: str,
    target_port: int,
    target_metadata: list[dict[str, Any]],
) -> int:
    """Send Full KV and Indexer KV into Decode-side NPU staging."""
    torch, engine, _ = _initialize_engine(device_index=0)
    owners = []
    buffers: dict[str, Any] = {}
    registered_pointers: list[int] = []
    unregister_failures: list[tuple[int, int]] = []
    try:
        for name, _, payload_value in STAGED_BUFFER_SPECS:
            owner, buffer = _allocate_aligned_buffer(
                torch,
                device_index=0,
                fill_value=0,
            )
            buffer[MIXED_TRANSFER_OFFSET_BYTES : (MIXED_TRANSFER_OFFSET_BYTES + MIXED_TRANSFER_BYTES)].fill_(
                payload_value
            )
            owners.append(owner)
            buffers[name] = buffer
            _register_buffer(
                engine,
                buffer,
                memory_kind="npu",
                device_index=0,
            )
            registered_pointers.append(buffer.data_ptr())
        torch.npu.synchronize()

        target_by_name = {str(metadata["name"]): metadata for metadata in target_metadata}
        expected_names = {name for name, _, _ in STAGED_BUFFER_SPECS}
        if set(target_by_name) != expected_names:
            raise RuntimeError(
                "Staged target metadata names do not match the source tuple: "
                f"expected={sorted(expected_names)}, "
                f"actual={sorted(target_by_name)}"
            )
        for name in FULL_KV_BUFFER_NAMES:
            metadata = target_by_name[name]
            if metadata["memory_kind"] != "npu":
                raise RuntimeError(f"Staged Full-KV destination must be NPU memory: name={name!r}, metadata={metadata}")
            if int(metadata["transfer_size"]) != MIXED_TRANSFER_BYTES:
                raise RuntimeError(
                    f"Staged Full-KV transfer size does not match the test contract: name={name!r}, metadata={metadata}"
                )

        indexer_metadata = target_by_name[INDEXER_BUFFER_NAME]
        if indexer_metadata["memory_kind"] != "npu":
            raise RuntimeError(f"Staged Indexer destination must be NPU memory: metadata={indexer_metadata}")
        if int(indexer_metadata["transfer_size"]) != MIXED_TRANSFER_BYTES:
            raise RuntimeError(
                f"Staged Indexer transfer size does not match the test contract: metadata={indexer_metadata}"
            )

        session = f"{target_host}:{target_port}"
        # Keep Full KV and Indexer transfers separate because Full KV is copied
        # into swapped memory after arrival while Indexer KV remains on NPU.
        full_result = engine.batch_transfer_sync_write(
            session,
            [buffers[name].data_ptr() + MIXED_TRANSFER_OFFSET_BYTES for name in FULL_KV_BUFFER_NAMES],
            [int(target_by_name[name]["target_address"]) for name in FULL_KV_BUFFER_NAMES],
            [MIXED_TRANSFER_BYTES] * len(FULL_KV_BUFFER_NAMES),
        )
        if full_result < 0:
            raise RuntimeError(
                f"Mooncake staged Full-KV NPU-to-NPU transfer failed: session={session}, result={full_result}"
            )

        indexer_result = engine.batch_transfer_sync_write(
            session,
            [buffers[INDEXER_BUFFER_NAME].data_ptr() + MIXED_TRANSFER_OFFSET_BYTES],
            [int(target_by_name[INDEXER_BUFFER_NAME]["target_address"])],
            [MIXED_TRANSFER_BYTES],
        )
        if indexer_result < 0:
            raise RuntimeError(
                f"Mooncake staged Indexer NPU-to-NPU transfer failed: session={session}, result={indexer_result}"
            )

        print(
            RESULT_MARKER
            + json.dumps(
                {
                    "full_transfer_result": full_result,
                    "indexer_transfer_result": indexer_result,
                    "full_source_addresses": [
                        buffers[name].data_ptr() + MIXED_TRANSFER_OFFSET_BYTES for name in FULL_KV_BUFFER_NAMES
                    ],
                    "full_target_addresses": [
                        int(target_by_name[name]["target_address"]) for name in FULL_KV_BUFFER_NAMES
                    ],
                    "indexer_source_address": (buffers[INDEXER_BUFFER_NAME].data_ptr() + MIXED_TRANSFER_OFFSET_BYTES),
                    "indexer_target_address": int(target_by_name[INDEXER_BUFFER_NAME]["target_address"]),
                    "transfer_size": MIXED_TRANSFER_BYTES,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        torch.npu.synchronize()
        unregister_failures = _unregister_buffers(
            engine,
            registered_pointers,
        )
        del engine
        buffers.clear()
        owners.clear()
        gc.collect()
        if unregister_failures:
            raise RuntimeError(f"Staged sender memory unregistration failed: {unregister_failures}")


def _run_connector_persist() -> int:
    """Exercise the connector's NPU staging -> swapped Full-KV copy."""
    import torch
    import torch_npu

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
        KVCacheRecvingLayerThread,
        LayerMetadata,
        MooncakeAgentMetadata,
    )

    torch.npu.set_device(0)
    raw, combined_staging, staging = _allocate_connector_staging(
        torch,
        device_index=0,
        fill_value=0,
    )
    _fill_connector_payload(staging)
    final = _allocate_connector_final(
        torch,
        torch_npu,
        staging,
        device_index=0,
    )

    metadata = MooncakeAgentMetadata(
        te_rpc_port=0,
        layer_metadata={
            "layer0": LayerMetadata(
                tensor_group_idx=[0, 0],
                kv_caches_base_addr=[tensor.data_ptr() for tensor in staging],
                block_len=[tensor[0].nbytes for tensor in staging],
                block_size_scale=[1, 1],
            )
        },
    )
    receiver = KVCacheRecvingLayerThread(
        tp_rank=0,
        side_channel_port=0,
        tp_size=1,
        pd_head_ratio=1,
        local_engine_id="connector-persist-smoke",
        metadata=metadata,
        ready_event=threading.Event(),
        sparse_host_final_kv_caches={"layer0": final},
        sparse_host_staging_kv=staging,
    )
    receiver.persist_staged_layer("layer0", [1, 3])
    verification = _verify_connector_staging_and_final(torch, staging, final)
    result = {
        "matches": all(verification.values()),
        "staging_base_alignment_mod": staging[0].data_ptr() % ALIGNMENT_BYTES,
        "verification": verification,
    }
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
    del receiver, final, staging, combined_staging, raw
    gc.collect()
    torch.npu.synchronize()
    return 0 if result["matches"] else 1


def _run_connector_protocol_receiver() -> int:
    """Run the production layer receiver and persist its staged Full KV."""
    import gc
    from types import SimpleNamespace

    import torch_npu

    import vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector as connector_module
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
        KVCacheRecvingLayerThread,
        LayerMetadata,
        MooncakeAgentMetadata,
    )

    torch, engine, host = _initialize_engine(device_index=1)
    raw, combined_staging, staging = _allocate_connector_staging(
        torch,
        device_index=1,
        fill_value=-1,
    )
    final = _allocate_connector_final(
        torch,
        torch_npu,
        staging,
        device_index=1,
    )
    registered = False
    unregister_result = 0
    receiver = None
    try:
        _register_buffer(
            engine,
            combined_staging,
            memory_kind="npu",
            device_index=1,
        )
        registered = True
        layer_metadata = LayerMetadata(
            tensor_group_idx=[0, 0],
            kv_caches_base_addr=[tensor.data_ptr() for tensor in staging],
            block_len=[tensor[0].nbytes for tensor in staging],
            block_size_scale=[1, 1],
        )
        side_channel_port = _find_free_tcp_port()
        ready_event = threading.Event()
        connector_module.get_world_group = lambda: SimpleNamespace(local_rank=1)
        receiver = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=side_channel_port,
            tp_size=1,
            pd_head_ratio=1,
            local_engine_id="connector-protocol-receiver",
            metadata=MooncakeAgentMetadata(
                te_rpc_port=engine.get_rpc_port(),
                layer_metadata={"layer0": layer_metadata},
            ),
            ready_event=ready_event,
            sparse_host_final_kv_caches={"layer0": final},
            sparse_host_staging_kv=staging,
        )
        receiver.start()
        if not ready_event.wait(timeout=5):
            raise TimeoutError("Mooncake connector receiver side channel did not become ready.")
        print(
            READY_MARKER
            + json.dumps(
                {
                    "host": host,
                    "side_channel_port": side_channel_port,
                    "te_rpc_port": engine.get_rpc_port(),
                    "layer_metadata": {
                        "tensor_group_idx": layer_metadata.tensor_group_idx,
                        "kv_caches_base_addr": layer_metadata.kv_caches_base_addr,
                        "block_len": layer_metadata.block_len,
                        "block_size_scale": layer_metadata.block_size_scale,
                    },
                }
            ),
            flush=True,
        )

        command = sys.stdin.readline().strip()
        if command != "VERIFY":
            raise RuntimeError(f"Connector protocol receiver expected VERIFY, got {command!r}.")
        verification = _verify_connector_staging_and_final(torch, staging, final)
        result = {
            "matches": all(verification.values()),
            "verification": verification,
        }
        print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["matches"] else 1
    finally:
        torch.npu.synchronize()
        if registered:
            unregister_result = engine.unregister_memory(combined_staging.data_ptr())
        del receiver, engine, final, staging, combined_staging, raw
        gc.collect()
        if unregister_result != 0:
            raise RuntimeError(f"Connector receiver memory unregistration failed: result={unregister_result}")


def _run_connector_protocol_sender(
    *,
    target_host: str,
    target_side_channel_port: int,
    target_te_rpc_port: int,
    target_metadata: dict[str, Any],
) -> int:
    """Run the production layer sender through transfer, signal, and ACK."""
    import gc
    from types import SimpleNamespace

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
        KVCacheSendingLayerThread,
        LayerMetadata,
        MooncakeLayerwiseConnectorWorker,
        ReqMeta,
        SendTask,
    )

    torch, engine, _ = _initialize_engine(device_index=0)
    raw, combined_staging, staging = _allocate_connector_staging(
        torch,
        device_index=0,
        fill_value=-1,
    )
    _fill_connector_payload(staging)
    registered = False
    unregister_result = 0
    try:
        _register_buffer(
            engine,
            combined_staging,
            memory_kind="npu",
            device_index=0,
        )
        registered = True
        local_layer_metadata = LayerMetadata(
            tensor_group_idx=[0, 0],
            kv_caches_base_addr=[tensor.data_ptr() for tensor in staging],
            block_len=[tensor[0].nbytes for tensor in staging],
            block_size_scale=[1, 1],
        )
        remote_layer_metadata = LayerMetadata(**target_metadata)
        signal_worker = object.__new__(MooncakeLayerwiseConnectorWorker)
        signal_worker.layer_ack_timeout = 10.0
        sender = KVCacheSendingLayerThread(
            engine=engine,
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(mamba_cache_mode=None),
                speculative_config=None,
            ),
            kv_cache_config=SimpleNamespace(),
            kv_cache_specs=[object()],
            attn_resharding_group_idx=set(),
            total_layers=1,
            ready_event=threading.Event(),
            tp_size=1,
            tp_rank=0,
            pd_head_ratio=1,
            num_head_replica=1,
            layer_metadata={"layer0": local_layer_metadata},
            use_mla=True,
            use_attn_mamba_hybrid=False,
            k_buffer=None,
            v_buffer=None,
            enable_kv_quant=False,
            enable_c8_quant=False,
            resharding_stream=None,
            layer_callback_func=signal_worker.send_layer_staged_signal,
        )
        req_meta = ReqMeta(
            local_block_ids=[[1, 3]],
            token_ids=None,
            remote_block_ids=[[1, 3]],
            remote_block_size=[[4]],
            remote_engine_id="connector-protocol-receiver",
            remote_host=target_host,
            remote_port=target_side_channel_port,
            remote_te_rpc_port=target_te_rpc_port,
            remote_layer_metadata={"layer0": remote_layer_metadata},
            metaserver=None,
            remote_tp_size=1,
            remote_pcp_size=1,
            remote_dcp_size=1,
        )
        completion_event = threading.Event()
        send_task = SendTask(
            send_request={"connector-protocol123456789": req_meta},
            wait_event=SimpleNamespace(synchronize=torch.npu.synchronize),
            layer_idx=0,
            layer_name="layer0",
            group_rearrange_block_ids=[[]],
            completion_event=completion_event,
        )
        sender._handle_request(send_task)
        result = {
            "acknowledged": completion_event.is_set(),
            "error": None if send_task.error is None else str(send_task.error),
            "source_alignment_mod": staging[0].data_ptr() % ALIGNMENT_BYTES,
        }
        print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["acknowledged"] and result["error"] is None else 1
    finally:
        torch.npu.synchronize()
        if registered:
            unregister_result = engine.unregister_memory(combined_staging.data_ptr())
        del engine, staging, combined_staging, raw
        gc.collect()
        if unregister_result != 0:
            raise RuntimeError(f"Connector sender memory unregistration failed: result={unregister_result}")


def _require_native_mooncake() -> None:
    try:
        installed_version = Version(version("mooncake-transfer-engine-npu"))
    except PackageNotFoundError:
        pytest.skip("mooncake-transfer-engine-npu is not installed")
    if installed_version < MIN_MOONCAKE_VERSION:
        pytest.skip(
            "Mooncake NPU smoke test requires "
            f"mooncake-transfer-engine-npu>={MIN_MOONCAKE_VERSION}; "
            f"found {installed_version}"
        )
    for required_path in (Path("/dev/devmm_svm"), Path("/etc/hccn.conf")):
        if not required_path.exists():
            pytest.skip(f"Mooncake Ascend prerequisite is missing: {required_path}")


def test_mooncake_npu_to_npu_transfer():
    """Transfer a 2 MiB pattern from NPU 0 to NPU 1 and verify every byte."""
    _require_native_mooncake()
    command = [sys.executable, str(Path(__file__).resolve())]
    receiver = subprocess.Popen(
        [*command, "--role", "receiver"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if receiver.stdin is None or receiver.stdout is None:
        receiver.terminate()
        receiver.wait(timeout=5)
        raise RuntimeError("Failed to create receiver control pipes.")

    receiver_output = _ProcessOutput(receiver.stdout)
    try:
        ready = receiver_output.wait_for_marker(
            READY_MARKER,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        sender = subprocess.run(
            [
                *command,
                "--role",
                "sender",
                "--target-host",
                str(ready["host"]),
                "--target-port",
                str(ready["port"]),
                "--target-address",
                str(ready["address"]),
            ],
            capture_output=True,
            text=True,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        assert sender.returncode == 0, f"Mooncake sender failed.\nstdout:\n{sender.stdout}\nstderr:\n{sender.stderr}"

        receiver.stdin.write("VERIFY\n")
        receiver.stdin.flush()
        result = receiver_output.wait_for_marker(
            RESULT_MARKER,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        receiver.wait(timeout=PROCESS_TIMEOUT_SECONDS)
        receiver_output.join()
        assert receiver.returncode == 0, f"Mooncake receiver failed.\noutput:\n{receiver_output.output}"
        assert result["matches"], (
            f"Mooncake receiver observed corrupted data.\nresult={result}\noutput:\n{receiver_output.output}"
        )
    finally:
        if receiver.poll() is None:
            _stop_process(receiver, receiver_output)
        else:
            receiver_output.join()


@pytest.mark.parametrize(
    ("memory_kind", "pinned"),
    (
        pytest.param("regular", False, id="regular-host"),
        pytest.param("pinned", True, id="pinned-host"),
    ),
)
def test_mooncake_host_to_host_tcp_transfer(memory_kind: str, pinned: bool):
    """Verify the Ascend wheel's TCP-only Host transport capability."""
    _require_native_mooncake()
    command = [sys.executable, str(Path(__file__).resolve())]
    child_environment = os.environ.copy()
    child_environment[HOST_TCP_FORCE_ENV] = HOST_TCP_FORCE_VALUE
    receiver = subprocess.Popen(
        [
            *command,
            "--role",
            "host_receiver",
            "--host-memory-kind",
            memory_kind,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_environment,
    )
    if receiver.stdin is None or receiver.stdout is None:
        receiver.terminate()
        receiver.wait(timeout=5)
        raise RuntimeError("Failed to create Host receiver control pipes.")

    receiver_output = _ProcessOutput(receiver.stdout)
    try:
        ready = receiver_output.wait_for_marker(
            READY_MARKER,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        assert ready["pinned"] is pinned
        assert ready["force_tcp"] == HOST_TCP_FORCE_VALUE
        sender = subprocess.run(
            [
                *command,
                "--role",
                "host_sender",
                "--host-memory-kind",
                memory_kind,
                "--target-host",
                str(ready["host"]),
                "--target-port",
                str(ready["port"]),
                "--target-address",
                str(ready["address"]),
            ],
            capture_output=True,
            text=True,
            timeout=PROCESS_TIMEOUT_SECONDS,
            env=child_environment,
        )
        assert sender.returncode == 0, (
            "Mooncake Host sender failed.\n"
            f"stdout:\n{sender.stdout}\n"
            f"stderr:\n{sender.stderr}"
        )

        receiver.stdin.write("VERIFY\n")
        receiver.stdin.flush()
        result = receiver_output.wait_for_marker(
            RESULT_MARKER,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        receiver.wait(timeout=PROCESS_TIMEOUT_SECONDS)
        receiver_output.join()
        assert receiver.returncode == 0, (
            "Mooncake Host receiver failed.\n"
            f"output:\n{receiver_output.output}"
        )
        assert result["matches"], (
            "Mooncake Host receiver observed corrupted data.\n"
            f"result={result}\noutput:\n{receiver_output.output}"
        )
        assert all(result["payload_matches"])
        sender_output = sender.stdout + sender.stderr
        assert HOST_TCP_RUNTIME_MARKER in receiver_output.output, (
            "Mooncake Host receiver did not emit the expected TCP-only "
            f"runtime marker.\noutput:\n{receiver_output.output}"
        )
        assert HOST_TCP_RUNTIME_MARKER in sender_output, (
            "Mooncake Host sender did not emit the expected TCP-only "
            f"runtime marker.\noutput:\n{sender_output}"
        )
        print(
            "Mooncake Host-to-Host TCP path verified: "
            + json.dumps(
                {
                    "memory_kind": memory_kind,
                    "repeat_count": HOST_TRANSFER_REPEAT_COUNT,
                    "transfer_bytes": HOST_TRANSFER_BYTES,
                    "verification": result,
                },
                sort_keys=True,
            )
        )
    finally:
        if receiver.poll() is None:
            _stop_process(receiver, receiver_output)
        else:
            receiver_output.join()


def test_mooncake_ascend_and_tcp_engines_coexist():
    """Verify one worker can retain independent Ascend and TCP engines."""
    _require_native_mooncake()
    command = [sys.executable, str(Path(__file__).resolve())]
    child_environment = os.environ.copy()
    child_environment.pop(HOST_TCP_FORCE_ENV, None)
    receiver = subprocess.Popen(
        [*command, "--role", "hybrid_receiver"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_environment,
    )
    if receiver.stdin is None or receiver.stdout is None:
        receiver.terminate()
        receiver.wait(timeout=5)
        raise RuntimeError("Failed to create hybrid receiver control pipes.")

    receiver_output = _ProcessOutput(receiver.stdout)
    try:
        ready = receiver_output.wait_for_marker(
            READY_MARKER,
            timeout=HYBRID_PROCESS_TIMEOUT_SECONDS,
        )
        assert ready["ascend_port"] != ready["host_port"]
        assert ready["force_tcp_after_init"] is None
        sender = subprocess.run(
            [
                *command,
                "--role",
                "hybrid_sender",
                "--target-host",
                str(ready["host"]),
                "--target-ascend-port",
                str(ready["ascend_port"]),
                "--target-host-port",
                str(ready["host_port"]),
                "--target-indexer-address",
                str(ready["indexer_address"]),
                "--target-host-address",
                str(ready["host_address"]),
            ],
            capture_output=True,
            text=True,
            timeout=HYBRID_PROCESS_TIMEOUT_SECONDS,
            env=child_environment,
        )
        sender_output = sender.stdout + sender.stderr
        assert sender.returncode == 0, (
            "Mooncake hybrid sender failed.\n"
            f"stdout:\n{sender.stdout}\n"
            f"stderr:\n{sender.stderr}"
        )
        sender_result = _extract_json_marker(sender.stdout, RESULT_MARKER)
        assert sender_result["ascend_port"] != sender_result["host_port"]
        assert sender_result["force_tcp_after_init"] is None
        assert len(sender_result["host_transfer_results"]) == HOST_TRANSFER_REPEAT_COUNT

        receiver.stdin.write("VERIFY\n")
        receiver.stdin.flush()
        result = receiver_output.wait_for_marker(
            RESULT_MARKER,
            timeout=HYBRID_PROCESS_TIMEOUT_SECONDS,
        )
        receiver.wait(timeout=HYBRID_PROCESS_TIMEOUT_SECONDS)
        receiver_output.join()
        assert receiver.returncode == 0, (
            "Mooncake hybrid receiver failed.\n"
            f"output:\n{receiver_output.output}"
        )
        assert result["matches"], (
            "Mooncake hybrid receiver observed corrupted data.\n"
            f"result={result}\noutput:\n{receiver_output.output}"
        )
        assert ASCEND_RUNTIME_MARKER in receiver_output.output, (
            "Hybrid receiver did not initialize an Ascend transport.\n"
            f"output:\n{receiver_output.output}"
        )
        assert HOST_TCP_RUNTIME_MARKER in receiver_output.output, (
            "Hybrid receiver did not initialize a TCP-only transport.\n"
            f"output:\n{receiver_output.output}"
        )
        assert ASCEND_RUNTIME_MARKER in sender_output, (
            "Hybrid sender did not initialize an Ascend transport.\n"
            f"output:\n{sender_output}"
        )
        assert HOST_TCP_RUNTIME_MARKER in sender_output, (
            "Hybrid sender did not initialize a TCP-only transport.\n"
            f"output:\n{sender_output}"
        )
        print(
            "Mooncake Ascend/TCP dual-engine coexistence verified: "
            + json.dumps(
                {
                    "sequence": [
                        "ascend-indexer-a",
                        "tcp-host-full-kv",
                        "ascend-indexer-b",
                    ],
                    "sender": sender_result,
                    "receiver": result,
                },
                sort_keys=True,
            )
        )
    finally:
        if receiver.poll() is None:
            _stop_process(receiver, receiver_output)
        else:
            receiver_output.join()


def test_mooncake_sparse_offload_staged_memory_transfer():
    """Verify NPU staging followed by a local write into swapped Full KV."""
    _require_native_mooncake()
    command = [sys.executable, str(Path(__file__).resolve())]
    receiver = subprocess.Popen(
        [*command, "--role", "staged_receiver"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if receiver.stdin is None or receiver.stdout is None:
        receiver.terminate()
        receiver.wait(timeout=5)
        raise RuntimeError("Failed to create staged receiver control pipes.")

    receiver_output = _ProcessOutput(receiver.stdout)
    try:
        ready = receiver_output.wait_for_marker(
            READY_MARKER,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        sender = subprocess.run(
            [
                *command,
                "--role",
                "staged_sender",
                "--target-host",
                str(ready["host"]),
                "--target-port",
                str(ready["port"]),
                "--target-metadata",
                json.dumps(ready["buffers"], separators=(",", ":")),
            ],
            capture_output=True,
            text=True,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        assert sender.returncode == 0, (
            f"Mooncake staged sender failed.\nstdout:\n{sender.stdout}\nstderr:\n{sender.stderr}"
        )

        receiver.stdin.write("VERIFY\n")
        receiver.stdin.flush()
        result = receiver_output.wait_for_marker(
            RESULT_MARKER,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        receiver.wait(timeout=PROCESS_TIMEOUT_SECONDS)
        receiver_output.join()
        assert receiver.returncode == 0, f"Mooncake staged receiver failed.\noutput:\n{receiver_output.output}"
        assert result["matches"], (
            f"Mooncake staged receiver observed corrupted data.\nresult={result}\noutput:\n{receiver_output.output}"
        )
        print(
            "Mooncake sparse-offload staged path verified: "
            + json.dumps(
                {
                    "receiver_layout": ready["buffers"],
                    "staging_verification": result["staging_buffers"],
                    "final_verification": result["final_buffers"],
                },
                sort_keys=True,
            )
        )
    finally:
        if receiver.poll() is None:
            _stop_process(receiver, receiver_output)
        else:
            receiver_output.join()


def test_mooncake_connector_persists_staging_into_swapped_full_kv():
    """Verify the production connector's staged block persistence on NPU."""
    _require_native_mooncake()
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--role",
            "connector_persist",
        ],
        capture_output=True,
        text=True,
        timeout=PROCESS_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"Mooncake connector staging persistence failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    marker_line = next(line for line in result.stdout.splitlines() if line.startswith(RESULT_MARKER))
    payload = json.loads(marker_line.removeprefix(RESULT_MARKER))
    assert payload["matches"]
    assert payload["staging_base_alignment_mod"] == 0


def test_mooncake_layerwise_connector_protocol_persists_before_ack():
    """Verify production send/receive threads transfer, persist, then ACK."""
    _require_native_mooncake()
    command = [sys.executable, str(Path(__file__).resolve())]
    receiver = subprocess.Popen(
        [*command, "--role", "connector_protocol_receiver"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if receiver.stdin is None or receiver.stdout is None:
        receiver.terminate()
        receiver.wait(timeout=5)
        raise RuntimeError("Failed to create connector protocol receiver control pipes.")

    receiver_output = _ProcessOutput(receiver.stdout)
    try:
        ready = receiver_output.wait_for_marker(
            READY_MARKER,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        sender = subprocess.run(
            [
                *command,
                "--role",
                "connector_protocol_sender",
                "--target-host",
                str(ready["host"]),
                "--target-side-channel-port",
                str(ready["side_channel_port"]),
                "--target-te-port",
                str(ready["te_rpc_port"]),
                "--target-metadata",
                json.dumps(ready["layer_metadata"], separators=(",", ":")),
            ],
            capture_output=True,
            text=True,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        assert sender.returncode == 0, (
            f"Mooncake connector protocol sender failed.\nstdout:\n{sender.stdout}\nstderr:\n{sender.stderr}"
        )
        sender_marker = next(line for line in sender.stdout.splitlines() if line.startswith(RESULT_MARKER))
        sender_result = json.loads(sender_marker.removeprefix(RESULT_MARKER))
        assert sender_result["acknowledged"]
        assert sender_result["error"] is None
        assert sender_result["source_alignment_mod"] == 0

        receiver.stdin.write("VERIFY\n")
        receiver.stdin.flush()
        receiver_result = receiver_output.wait_for_marker(
            RESULT_MARKER,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        receiver.wait(timeout=PROCESS_TIMEOUT_SECONDS)
        receiver_output.join()
        assert receiver.returncode == 0, (
            f"Mooncake connector protocol receiver failed.\noutput:\n{receiver_output.output}"
        )
        assert receiver_result["matches"], (
            "Mooncake connector ACK arrived without correct Full-KV "
            f"persistence.\nresult={receiver_result}\noutput:\n{receiver_output.output}"
        )
        print(
            "Mooncake layerwise connector protocol verified: "
            + json.dumps(
                {
                    "sender": sender_result,
                    "receiver": receiver_result,
                },
                sort_keys=True,
            )
        )
    finally:
        if receiver.poll() is None:
            _stop_process(receiver, receiver_output)
        else:
            receiver_output.join()


@pytest.mark.xfail(
    reason=(
        "Diagnostic only: swapped tensor.data_ptr() exposes an SVM alias, not "
        "the original aclrtMallocHost address required by Mooncake/HIXL "
        "registration. Run this exact test with --runxfail only when "
        "re-checking a new CANN/Mooncake runtime."
    ),
    run=False,
)
def test_mooncake_sparse_offload_mixed_memory_transfer():
    """Diagnose direct swapped-Host/NPU transfer on a candidate runtime."""
    _require_native_mooncake()
    command = [sys.executable, str(Path(__file__).resolve())]
    receiver = subprocess.Popen(
        [*command, "--role", "mixed_receiver"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if receiver.stdin is None or receiver.stdout is None:
        receiver.terminate()
        receiver.wait(timeout=5)
        raise RuntimeError("Failed to create mixed receiver control pipes.")

    receiver_output = _ProcessOutput(receiver.stdout)
    try:
        ready = receiver_output.wait_for_marker(
            READY_MARKER,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        sender = subprocess.run(
            [
                *command,
                "--role",
                "mixed_sender",
                "--target-host",
                str(ready["host"]),
                "--target-port",
                str(ready["port"]),
                "--target-metadata",
                json.dumps(ready["buffers"], separators=(",", ":")),
            ],
            capture_output=True,
            text=True,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        assert sender.returncode == 0, (
            f"Mooncake mixed sender failed.\nstdout:\n{sender.stdout}\nstderr:\n{sender.stderr}"
        )

        receiver.stdin.write("VERIFY\n")
        receiver.stdin.flush()
        result = receiver_output.wait_for_marker(
            RESULT_MARKER,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        receiver.wait(timeout=PROCESS_TIMEOUT_SECONDS)
        receiver_output.join()
        assert receiver.returncode == 0, f"Mooncake mixed receiver failed.\noutput:\n{receiver_output.output}"
        assert result["matches"], (
            f"Mooncake mixed receiver observed corrupted data.\nresult={result}\noutput:\n{receiver_output.output}"
        )
        print(
            "Mooncake sparse-offload mixed batch verified: "
            + json.dumps(
                {
                    "receiver_layout": ready["buffers"],
                    "verification": result["buffers"],
                },
                sort_keys=True,
            )
        )
    finally:
        if receiver.poll() is None:
            _stop_process(receiver, receiver_output)
        else:
            receiver_output.join()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        choices=(
            "receiver",
            "sender",
            "host_receiver",
            "host_sender",
            "hybrid_receiver",
            "hybrid_sender",
            "mixed_receiver",
            "mixed_sender",
            "staged_receiver",
            "staged_sender",
            "connector_persist",
            "connector_protocol_receiver",
            "connector_protocol_sender",
        ),
        required=True,
    )
    parser.add_argument("--target-host")
    parser.add_argument("--target-port", type=int)
    parser.add_argument("--target-address", type=int)
    parser.add_argument("--target-metadata")
    parser.add_argument("--target-side-channel-port", type=int)
    parser.add_argument("--target-te-port", type=int)
    parser.add_argument("--target-ascend-port", type=int)
    parser.add_argument("--target-host-port", type=int)
    parser.add_argument("--target-indexer-address", type=int)
    parser.add_argument("--target-host-address", type=int)
    parser.add_argument(
        "--host-memory-kind",
        choices=("regular", "pinned"),
        default="regular",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.role == "receiver":
        raise SystemExit(_run_receiver())
    if args.role == "host_receiver":
        raise SystemExit(_run_host_receiver(pinned=args.host_memory_kind == "pinned"))
    if args.role == "host_sender":
        if args.target_host is None or args.target_port is None or args.target_address is None:
            raise SystemExit("Host sender requires --target-host, --target-port, and --target-address")
        raise SystemExit(
            _run_host_sender(
                target_host=args.target_host,
                target_port=args.target_port,
                target_address=args.target_address,
                pinned=args.host_memory_kind == "pinned",
            )
        )
    if args.role == "hybrid_receiver":
        raise SystemExit(_run_hybrid_receiver())
    if args.role == "hybrid_sender":
        if (
            args.target_host is None
            or args.target_ascend_port is None
            or args.target_host_port is None
            or args.target_indexer_address is None
            or args.target_host_address is None
        ):
            raise SystemExit(
                "hybrid sender requires --target-host, --target-ascend-port, "
                "--target-host-port, --target-indexer-address, and --target-host-address"
            )
        raise SystemExit(
            _run_hybrid_sender(
                target_host=args.target_host,
                target_ascend_port=args.target_ascend_port,
                target_host_port=args.target_host_port,
                target_indexer_address=args.target_indexer_address,
                target_host_address=args.target_host_address,
            )
        )
    if args.role == "mixed_receiver":
        raise SystemExit(_run_mixed_receiver())
    if args.role == "staged_receiver":
        raise SystemExit(_run_staged_receiver())
    if args.role == "connector_persist":
        raise SystemExit(_run_connector_persist())
    if args.role == "connector_protocol_receiver":
        raise SystemExit(_run_connector_protocol_receiver())
    if args.role == "connector_protocol_sender":
        if (
            args.target_host is None
            or args.target_side_channel_port is None
            or args.target_te_port is None
            or args.target_metadata is None
        ):
            raise SystemExit(
                "connector protocol sender requires --target-host, "
                "--target-side-channel-port, --target-te-port, and --target-metadata"
            )
        raise SystemExit(
            _run_connector_protocol_sender(
                target_host=args.target_host,
                target_side_channel_port=args.target_side_channel_port,
                target_te_rpc_port=args.target_te_port,
                target_metadata=json.loads(args.target_metadata),
            )
        )
    if args.role == "mixed_sender":
        if args.target_host is None or args.target_port is None or args.target_metadata is None:
            raise SystemExit("mixed sender requires --target-host, --target-port, and --target-metadata")
        raise SystemExit(
            _run_mixed_sender(
                target_host=args.target_host,
                target_port=args.target_port,
                target_metadata=json.loads(args.target_metadata),
            )
        )
    if args.role == "staged_sender":
        if args.target_host is None or args.target_port is None or args.target_metadata is None:
            raise SystemExit("staged sender requires --target-host, --target-port, and --target-metadata")
        raise SystemExit(
            _run_staged_sender(
                target_host=args.target_host,
                target_port=args.target_port,
                target_metadata=json.loads(args.target_metadata),
            )
        )
    if args.target_host is None or args.target_port is None or args.target_address is None:
        raise SystemExit("sender requires --target-host, --target-port, and --target-address")
    raise SystemExit(
        _run_sender(
            target_host=args.target_host,
            target_port=args.target_port,
            target_address=args.target_address,
        )
    )

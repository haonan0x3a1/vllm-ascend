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
import queue
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
MIN_MOONCAKE_VERSION = Version("0.3.12.post1")
PROCESS_TIMEOUT_SECONDS = 60
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
    raw, combined_staging = _allocate_aligned_buffer(
        torch,
        device_index=0,
        fill_value=0,
    )
    staging = (
        combined_staging[: BUFFER_BYTES // 2].view(4, -1),
        combined_staging[BUFFER_BYTES // 2 :].view(4, -1),
    )
    staging[0][1].fill_(0x11)
    staging[1][1].fill_(0x22)
    staging[0][3].fill_(0x33)
    staging[1][3].fill_(0x44)
    final = (
        torch_npu.empty_with_swapped_memory(
            staging[0].shape,
            dtype=torch.int8,
            device=torch.device("npu:0"),
        ),
        torch_npu.empty_with_swapped_memory(
            staging[1].shape,
            dtype=torch.int8,
            device=torch.device("npu:0"),
        ),
    )
    for tensor in final:
        tensor.fill_(-1)

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
    final_npu_copy = tuple(torch.empty_like(tensor) for tensor in staging)
    for npu_copy, swapped_tensor in zip(final_npu_copy, final):
        # Reading a framework swapped tensor directly with .cpu() can enter
        # devmm_h2h_copy and segfault on the validated CANN runtime. Mirror the
        # production Gather direction by restoring it to NPU first.
        npu_copy.copy_(swapped_tensor, non_blocking=False)
    torch.npu.synchronize()
    verification = {
        "full_nope_block_1": torch.equal(final_npu_copy[0][1].cpu(), staging[0][1].cpu()),
        "full_nope_block_3": torch.equal(final_npu_copy[0][3].cpu(), staging[0][3].cpu()),
        "full_rope_block_1": torch.equal(final_npu_copy[1][1].cpu(), staging[1][1].cpu()),
        "full_rope_block_3": torch.equal(final_npu_copy[1][3].cpu(), staging[1][3].cpu()),
        "full_nope_block_0_untouched": bool(torch.all(final_npu_copy[0][0].cpu() == -1)),
        "full_nope_block_2_untouched": bool(torch.all(final_npu_copy[0][2].cpu() == -1)),
        "full_rope_block_0_untouched": bool(torch.all(final_npu_copy[1][0].cpu() == -1)),
        "full_rope_block_2_untouched": bool(torch.all(final_npu_copy[1][2].cpu() == -1)),
    }
    result = {
        "matches": all(verification.values()),
        "staging_base_alignment_mod": staging[0].data_ptr() % ALIGNMENT_BYTES,
        "verification": verification,
    }
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
    del receiver, final_npu_copy, final, staging, combined_staging, raw
    gc.collect()
    torch.npu.synchronize()
    return 0 if result["matches"] else 1


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
            "mixed_receiver",
            "mixed_sender",
            "staged_receiver",
            "staged_sender",
            "connector_persist",
        ),
        required=True,
    )
    parser.add_argument("--target-host")
    parser.add_argument("--target-port", type=int)
    parser.add_argument("--target-address", type=int)
    parser.add_argument("--target-metadata")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.role == "receiver":
        raise SystemExit(_run_receiver())
    if args.role == "mixed_receiver":
        raise SystemExit(_run_mixed_receiver())
    if args.role == "staged_receiver":
        raise SystemExit(_run_staged_receiver())
    if args.role == "connector_persist":
        raise SystemExit(_run_connector_persist())
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

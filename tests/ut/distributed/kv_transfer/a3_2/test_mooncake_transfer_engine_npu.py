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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        choices=("receiver", "sender"),
        required=True,
    )
    parser.add_argument("--target-host")
    parser.add_argument("--target-port", type=int)
    parser.add_argument("--target-address", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.role == "receiver":
        raise SystemExit(_run_receiver())
    if args.target_host is None or args.target_port is None or args.target_address is None:
        raise SystemExit("sender requires --target-host, --target-port, and --target-address")
    raise SystemExit(
        _run_sender(
            target_host=args.target_host,
            target_port=args.target_port,
            target_address=args.target_address,
        )
    )

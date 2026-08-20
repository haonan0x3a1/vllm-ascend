# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.utils import (
    memfabric_bm_allocator as allocator_module,
)
from vllm_ascend.distributed.kv_transfer.utils.memfabric_bm_allocator import (
    MEMFABRIC_BM_ADDRESS_POLL_INTERVAL_SECONDS,
    MEMFABRIC_BM_DRAM_ALIGNMENT_BYTES,
    MEMFABRIC_BM_TENSOR_ALIGNMENT_BYTES,
    SMEM_BM_FLAG_DRAM_MAP_HOST_VA,
    MemFabricBMAllocationPlan,
    MemFabricBMFullKVAllocator,
    MemFabricBMRuntimeConfig,
)


class _KVTransferConfig:
    def __init__(self, role: str, raw: dict) -> None:
        self.kv_role = role
        self._raw = raw

    def get_from_extra_config(self, key: str, default=None):
        if key == "memfabric_bm":
            return self._raw
        return default


def _runtime_config(role: str, *, tp_rank: int = 3, **overrides):
    raw = {
        "protocol": "host_tcp",
        "store_host": "172.16.0.146",
        "store_port_base": 37200,
        "nic_ip": "172.16.0.146",
        "hcom_port_base": 37400,
        "pool_bytes": MEMFABRIC_BM_DRAM_ALIGNMENT_BYTES,
        "bm_id": 74,
        "start_store_role": "kv_consumer",
    }
    raw.update(overrides)
    return MemFabricBMRuntimeConfig.from_kv_transfer_config(
        _KVTransferConfig(role, raw),
        tp_rank=tp_rank,
        device_id=tp_rank,
    )


def test_allocation_plan_preserves_tensor_alignment() -> None:
    plan = MemFabricBMAllocationPlan(3 * MEMFABRIC_BM_TENSOR_ALIGNMENT_BYTES)

    assert plan.reserve(1) == 0
    assert plan.reserve(1) == MEMFABRIC_BM_TENSOR_ALIGNMENT_BYTES
    assert plan.cursor_bytes == MEMFABRIC_BM_TENSOR_ALIGNMENT_BYTES + 1


def test_allocation_plan_fails_before_exceeding_pool() -> None:
    plan = MemFabricBMAllocationPlan(MEMFABRIC_BM_TENSOR_ALIGNMENT_BYTES)
    plan.reserve(MEMFABRIC_BM_TENSOR_ALIGNMENT_BYTES)

    with pytest.raises(MemoryError, match="pool exhausted"):
        plan.reserve(1)


def test_same_host_tp_pair_uses_one_store_and_distinct_hcom_ports() -> None:
    producer = _runtime_config("kv_producer")
    consumer = _runtime_config("kv_consumer")

    assert producer.rank_id == 0
    assert consumer.rank_id == 1
    assert producer.store_url == consumer.store_url == "tcp://172.16.0.146:37203"
    assert producer.nic_url == "tcp://172.16.0.146:37412"
    assert consumer.nic_url == "tcp://172.16.0.146:37413"
    assert not producer.starts_store
    assert consumer.starts_store


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"protocol": "device_rdma"}, "protocol"),
        ({"pool_bytes": 1}, "positive multiple of 1 GiB"),
        ({"protocol": "host_rdma", "nic_ip": "127.0.0.1"}, "non-loopback"),
        ({"peer_join_timeout_seconds": 0}, "positive finite number"),
        ({"unexpected": True}, "unsupported keys"),
    ],
)
def test_runtime_config_rejects_unsafe_values(
    overrides: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _runtime_config("kv_producer", **overrides)


def test_runtime_config_requires_nested_memfabric_config() -> None:
    config = SimpleNamespace(
        kv_role="kv_producer",
        get_from_extra_config=lambda key, default=None: default,
    )

    with pytest.raises(ValueError, match="memfabric_bm dictionary"):
        MemFabricBMRuntimeConfig.from_kv_transfer_config(
            config,
            tp_rank=0,
            device_id=0,
        )


def test_allocator_initializes_and_releases_runtime_in_order(monkeypatch) -> None:
    calls = []

    class _Handle:
        def join(self):
            calls.append("join")
            return 0

        def peer_rank_ptr(self, *args):
            return 0x280040000000

        def gva_to_va(self, pointer, *args):
            return pointer

        def leave(self):
            calls.append("leave")
            return 0

        def destroy(self):
            calls.append("destroy")

    class _BmConfig:
        def set_nic(self, url):
            calls.append(("nic", url))

    class _BM:
        BmConfig = _BmConfig
        BmDataOpType = SimpleNamespace(HOST_TCP="tcp", HOST_RDMA="rdma")
        BmMemType = SimpleNamespace(
            HOST="host",
            LOCAL_HOST="local_host",
            LOCAL_DEVICE="local_device",
        )

        def initialize(self, *args):
            calls.append("bm_initialize")
            return 0

        def create2(self, **kwargs):
            calls.append(("create2", kwargs))
            return _Handle()

        def uninitialize(self, device_id):
            calls.append(("bm_uninitialize", device_id))

    class _MF:
        def set_log_level(self, level):
            calls.append(("log_level", level))

        def initialize(self):
            calls.append("mf_initialize")
            return 0

        def uninitialize(self):
            calls.append("mf_uninitialize")

    config = _runtime_config("kv_consumer")
    allocator = MemFabricBMFullKVAllocator(config)
    monkeypatch.setattr(allocator, "_load_runtime", lambda: (_MF(), _BM()))
    monkeypatch.setattr(allocator, "_assert_cpu_mapping", lambda: "rw-s")
    monkeypatch.setattr(
        allocator_module.torch,
        "npu",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

    allocator.initialize()
    allocator.close()

    create_call = next(value for value in calls if isinstance(value, tuple) and value[0] == "create2")
    assert create_call[1]["flags"] == SMEM_BM_FLAG_DRAM_MAP_HOST_VA
    assert calls.index("mf_initialize") < calls.index("bm_initialize") < calls.index("join")
    assert calls[-4:] == [
        "leave",
        "destroy",
        ("bm_uninitialize", 3),
        "mf_uninitialize",
    ]


def test_allocator_waits_for_peer_address_translations(monkeypatch) -> None:
    translation_round = 0
    sleeps = []

    class _Handle:
        def join(self):
            return 0

        def peer_rank_ptr(self, *args):
            return 0x280040000000

        def gva_to_va(self, pointer, memory_type):
            nonlocal translation_round
            if memory_type == "local_device":
                translation_round += 1
            if translation_round < 2:
                return 0
            return pointer

        def leave(self):
            return 0

        def destroy(self):
            pass

    class _BmConfig:
        def set_nic(self, url):
            pass

    class _BM:
        BmConfig = _BmConfig
        BmDataOpType = SimpleNamespace(HOST_TCP="tcp", HOST_RDMA="rdma")
        BmMemType = SimpleNamespace(
            HOST="host",
            LOCAL_HOST="local_host",
            LOCAL_DEVICE="local_device",
        )

        def initialize(self, *args):
            return 0

        def create2(self, **kwargs):
            return _Handle()

        def uninitialize(self, device_id):
            pass

    class _MF:
        def set_log_level(self, level):
            pass

        def initialize(self):
            return 0

        def uninitialize(self):
            pass

    allocator = MemFabricBMFullKVAllocator(
        _runtime_config(
            "kv_consumer",
            peer_join_timeout_seconds=10,
        )
    )
    monkeypatch.setattr(allocator, "_load_runtime", lambda: (_MF(), _BM()))
    monkeypatch.setattr(allocator, "_assert_cpu_mapping", lambda: "rw-s")
    monkeypatch.setattr(allocator_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        allocator_module.torch,
        "npu",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

    allocator.initialize()

    assert sleeps == 2 * [MEMFABRIC_BM_ADDRESS_POLL_INTERVAL_SECONDS]
    assert allocator.local_gva == 0x280040000000
    assert allocator.local_host_va == allocator.local_gva
    assert allocator.local_device_va == allocator.local_gva
    allocator.close()


def test_allocator_reports_peer_address_timeout(monkeypatch) -> None:
    class _Handle:
        def peer_rank_ptr(self, *args):
            return 0x280040000000

        def gva_to_va(self, pointer, memory_type):
            return 0

    allocator = MemFabricBMFullKVAllocator(
        _runtime_config(
            "kv_consumer",
            peer_join_timeout_seconds=1,
        )
    )
    allocator._handle = _Handle()
    allocator._bm = SimpleNamespace(
        BmMemType=SimpleNamespace(
            HOST="host",
            LOCAL_HOST="local_host",
            LOCAL_DEVICE="local_device",
        )
    )
    monotonic_values = iter((100.0, 101.0))
    monkeypatch.setattr(
        allocator_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Timed out waiting for the MemFabric BM peer.*"
            "gva=0x280040000000, host=0x0, device=0x0"
        ),
    ):
        allocator._wait_for_local_addresses()

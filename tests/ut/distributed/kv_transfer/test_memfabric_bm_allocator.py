# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.utils import (
    memfabric_bm_allocator as allocator_module,
)
from vllm_ascend.distributed.kv_transfer.utils.memfabric_bm_allocator import (
    MEMFABRIC_BM_ACK,
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
        "store_port_base": 22200,
        "nic_ip": "172.16.0.146",
        "hcom_port_base": 22300,
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


def test_same_host_tp_pair_uses_store_rank_and_one_hcom_base() -> None:
    producer = _runtime_config("kv_producer")
    consumer = _runtime_config("kv_consumer")

    assert producer.rank_id == 1
    assert consumer.rank_id == 0
    assert producer.store_url == consumer.store_url == "tcp://172.16.0.146:22203"
    assert producer.nic_url == "tcp://172.16.0.146:22312"
    assert consumer.nic_url == "tcp://172.16.0.146:22312"
    assert (
        producer.create_rendezvous_port
        == consumer.create_rendezvous_port
        == 22314
    )
    assert (
        producer.join_rendezvous_port
        == consumer.join_rendezvous_port
        == 22315
    )
    assert not producer.starts_store
    assert consumer.starts_store


def test_rank_mapping_tracks_a_producer_owned_store() -> None:
    producer = _runtime_config(
        "kv_producer",
        start_store_role="kv_producer",
    )
    consumer = _runtime_config(
        "kv_consumer",
        start_store_role="kv_producer",
    )

    assert producer.rank_id == 0
    assert consumer.rank_id == 1


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
            calls.append("peer_rank_ptr")
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
        allocator,
        "_rendezvous_with_peer",
        lambda **kwargs: calls.append(("rendezvous", kwargs["stage"])),
    )
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
    assert calls.index("mf_initialize") < calls.index("bm_initialize")
    assert calls.index(create_call) < calls.index(("rendezvous", "created"))
    assert calls.index(("rendezvous", "created")) < calls.index("join")
    assert calls.index("join") < calls.index(("rendezvous", "joined"))
    assert calls.index(("rendezvous", "joined")) < calls.index("peer_rank_ptr")
    assert calls[-4:] == [
        "leave",
        "destroy",
        ("bm_uninitialize", 3),
        "mf_uninitialize",
    ]


class _FakeConnection:
    def __init__(self, response: bytes) -> None:
        self.response = response
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def settimeout(self, timeout):
        pass

    def recv(self, nbytes):
        return self.response

    def sendall(self, message):
        self.sent.append(message)


def test_store_owner_waits_for_peer_stage_without_querying_bm(monkeypatch) -> None:
    allocator = MemFabricBMFullKVAllocator(_runtime_config("kv_consumer"))
    peer_message = allocator._ready_message("created", 1)
    connection = _FakeConnection(peer_message)
    listener_calls = []

    class _FakeListener:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def setsockopt(self, *args):
            pass

        def bind(self, address):
            listener_calls.append(("bind", address))

        def listen(self, backlog):
            listener_calls.append(("listen", backlog))

        def settimeout(self, timeout):
            listener_calls.append(("timeout", timeout))

        def accept(self):
            listener_calls.append(("accept",))
            return connection, ("172.16.0.146", 12345)

    monkeypatch.setattr(
        allocator_module.socket,
        "socket",
        lambda *args: _FakeListener(),
    )

    allocator._rendezvous_with_peer(
        stage="created",
        port=allocator.config.create_rendezvous_port,
    )

    assert ("bind", ("172.16.0.146", 37414)) in listener_calls
    assert connection.sent == [MEMFABRIC_BM_ACK]


def test_non_store_rank_announces_stage_and_waits_for_ack(monkeypatch) -> None:
    allocator = MemFabricBMFullKVAllocator(_runtime_config("kv_producer"))
    connection = _FakeConnection(MEMFABRIC_BM_ACK)
    connect_calls = []

    def _create_connection(address, timeout):
        connect_calls.append((address, timeout))
        return connection

    monkeypatch.setattr(
        allocator_module.socket,
        "create_connection",
        _create_connection,
    )

    allocator._rendezvous_with_peer(
        stage="created",
        port=allocator.config.create_rendezvous_port,
    )

    assert connect_calls[0][0] == ("172.16.0.146", 37414)
    assert connection.sent == [allocator._ready_message("created", 1)]

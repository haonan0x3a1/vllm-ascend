# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MemFabric BM-backed Full-KV allocator for sparse KV offload.

The allocator owns one MemFabric DRAM pool per tensor-parallel worker. Its
LOCAL_DEVICE view is wrapped as raw int8 NPU tensors so the model runner can
reuse the existing KV-cache reshape and Gather contracts. The matching
LOCAL_HOST/GVA view is retained for the later Host-to-Host data plane.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch_npu
from vllm.logger import logger

MEMFABRIC_BM_DRAM_ALIGNMENT_BYTES = 1 << 30
MEMFABRIC_BM_TENSOR_ALIGNMENT_BYTES = 2 << 20
SMEM_BM_FLAG_DRAM_MAP_HOST_VA = 1 << 9
MEMFABRIC_BM_WORLD_SIZE = 2
MEMFABRIC_BM_PREFILL_RANK = 0
MEMFABRIC_BM_DECODE_RANK = 1
MEMFABRIC_BM_HCOM_PORT_STRIDE = 4


def _align_up(value: int, alignment: int) -> int:
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"alignment must be a positive power of two, got {alignment}")
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class MemFabricBMAllocation:
    offset: int
    nbytes: int
    device_ptr: int
    gva: int


class MemFabricBMAllocationPlan:
    """Alignment-aware bump allocator over one BM DRAM pool."""

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError(f"capacity_bytes must be positive, got {capacity_bytes}")
        self.capacity_bytes = capacity_bytes
        self.cursor_bytes = 0

    def reserve(
        self,
        nbytes: int,
        *,
        alignment: int = MEMFABRIC_BM_TENSOR_ALIGNMENT_BYTES,
    ) -> int:
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        offset = _align_up(self.cursor_bytes, alignment)
        end = offset + nbytes
        if end > self.capacity_bytes:
            raise MemoryError(
                "MemFabric BM Full-KV pool exhausted: "
                f"requested={nbytes}, aligned_offset={offset}, "
                f"capacity={self.capacity_bytes}"
            )
        self.cursor_bytes = end
        return offset


@dataclass(frozen=True)
class MemFabricBMRuntimeConfig:
    protocol: str
    store_host: str
    store_port_base: int
    nic_ip: str
    hcom_port_base: int
    pool_bytes: int
    bm_id: int
    start_store_role: str
    role: str
    tp_rank: int
    device_id: int
    log_level: int = 1

    @classmethod
    def from_kv_transfer_config(
        cls,
        kv_transfer_config: Any,
        *,
        tp_rank: int,
        device_id: int,
    ) -> "MemFabricBMRuntimeConfig":
        raw = kv_transfer_config.get_from_extra_config("memfabric_bm", None)
        if not isinstance(raw, dict):
            raise ValueError(
                "sparse_kv_transfer_mode='memfabric_bm' requires a "
                "kv_connector_extra_config.memfabric_bm dictionary."
            )
        supported_keys = {
            "protocol",
            "store_host",
            "store_port_base",
            "nic_ip",
            "hcom_port_base",
            "pool_bytes",
            "bm_id",
            "start_store_role",
            "log_level",
        }
        unknown_keys = sorted(set(raw) - supported_keys)
        if unknown_keys:
            raise ValueError(f"memfabric_bm contains unsupported keys: {unknown_keys}")

        role = kv_transfer_config.kv_role
        config = cls(
            protocol=str(raw.get("protocol", "host_tcp")),
            store_host=str(raw.get("store_host", "")),
            store_port_base=int(raw.get("store_port_base", 37200)),
            nic_ip=str(raw.get("nic_ip", "")),
            hcom_port_base=int(raw.get("hcom_port_base", 37400)),
            pool_bytes=int(raw.get("pool_bytes", MEMFABRIC_BM_DRAM_ALIGNMENT_BYTES)),
            bm_id=int(raw.get("bm_id", 74)),
            start_store_role=str(raw.get("start_store_role", "kv_consumer")),
            role=str(role),
            tp_rank=tp_rank,
            device_id=device_id,
            log_level=int(raw.get("log_level", 1)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.protocol not in {"host_tcp", "host_rdma"}:
            raise ValueError(
                "memfabric_bm.protocol must be 'host_tcp' or 'host_rdma', "
                f"got {self.protocol!r}"
            )
        if self.role not in {"kv_producer", "kv_consumer"}:
            raise ValueError(
                "memfabric_bm requires kv_role='kv_producer' or "
                f"'kv_consumer', got {self.role!r}"
            )
        if self.start_store_role not in {"kv_producer", "kv_consumer"}:
            raise ValueError(
                "memfabric_bm.start_store_role must be 'kv_producer' or "
                f"'kv_consumer', got {self.start_store_role!r}"
            )
        for name, value in (
            ("store_host", self.store_host),
            ("nic_ip", self.nic_ip),
        ):
            try:
                parsed = ipaddress.ip_address(value)
            except ValueError as exc:
                raise ValueError(f"memfabric_bm.{name} must be an IP address, got {value!r}") from exc
            if parsed.version != 4:
                raise ValueError(f"memfabric_bm.{name} currently requires IPv4, got {value!r}")
        if self.protocol == "host_rdma" and ipaddress.ip_address(self.nic_ip).is_loopback:
            raise ValueError("memfabric_bm HOST_RDMA requires a non-loopback RDMA NIC IPv4 address")
        if self.pool_bytes <= 0 or self.pool_bytes % MEMFABRIC_BM_DRAM_ALIGNMENT_BYTES:
            raise ValueError(
                "memfabric_bm.pool_bytes must be a positive multiple of 1 GiB, "
                f"got {self.pool_bytes}"
            )
        if self.tp_rank < 0 or self.device_id < 0:
            raise ValueError(
                "memfabric_bm tp_rank and device_id must be non-negative, "
                f"got tp_rank={self.tp_rank}, device_id={self.device_id}"
            )
        if self.bm_id < 0:
            raise ValueError(f"memfabric_bm.bm_id must be non-negative, got {self.bm_id}")
        for name, port in (
            ("store", self.store_port),
            ("hcom", self.hcom_port),
        ):
            if not 1 <= port <= 65535:
                raise ValueError(f"memfabric_bm {name} port is outside [1, 65535]: {port}")

    @property
    def rank_id(self) -> int:
        if self.role == "kv_producer":
            return MEMFABRIC_BM_PREFILL_RANK
        return MEMFABRIC_BM_DECODE_RANK

    @property
    def store_port(self) -> int:
        return self.store_port_base + self.tp_rank

    @property
    def hcom_port(self) -> int:
        # Prefill and Decode may run as separate processes on one physical
        # host during the same-node gate. Give both ranks distinct listener
        # ports while keeping all TP-pair ranges disjoint.
        return (
            self.hcom_port_base
            + self.tp_rank * MEMFABRIC_BM_HCOM_PORT_STRIDE
            + self.rank_id
        )

    @property
    def store_url(self) -> str:
        return f"tcp://{self.store_host}:{self.store_port}"

    @property
    def nic_url(self) -> str:
        return f"tcp://{self.nic_ip}:{self.hcom_port}"

    @property
    def starts_store(self) -> bool:
        return self.role == self.start_store_role


class MemFabricBMFullKVAllocator:
    """Own a BM pool and return raw int8 NPU aliases into its DRAM pages."""

    def __init__(self, config: MemFabricBMRuntimeConfig) -> None:
        self.config = config
        self.plan = MemFabricBMAllocationPlan(config.pool_bytes)
        self.allocations: list[MemFabricBMAllocation] = []
        self._mf: Any | None = None
        self._bm: Any | None = None
        self._handle: Any | None = None
        self._mf_initialized = False
        self._bm_initialized = False
        self._joined = False
        self._closed = False
        self.local_gva = 0
        self.local_host_va = 0
        self.local_device_va = 0

    def _load_runtime(self) -> tuple[Any, Any]:
        try:
            import memfabric_hybrid as mf
            from memfabric_hybrid import bm
        except ImportError as exc:
            raise RuntimeError(
                "sparse_kv_transfer_mode='memfabric_bm' requires the "
                "memfabric_hybrid Python package and its runtime libraries."
            ) from exc
        return mf, bm

    def _data_op_type(self) -> Any:
        assert self._bm is not None
        if self.config.protocol == "host_tcp":
            return self._bm.BmDataOpType.HOST_TCP
        return self._bm.BmDataOpType.HOST_RDMA

    def _assert_cpu_mapping(self) -> str:
        end_address = self.local_host_va + self.config.pool_bytes
        maps_path = Path("/proc/self/maps")
        if not maps_path.exists():
            raise RuntimeError("MemFabric BM Host mapping validation requires /proc/self/maps")
        for line in maps_path.read_text(encoding="utf-8").splitlines():
            address_range, permissions, *_ = line.split(maxsplit=2)
            start_text, end_text = address_range.split("-", maxsplit=1)
            mapping_start = int(start_text, 16)
            mapping_end = int(end_text, 16)
            if mapping_start <= self.local_host_va and end_address <= mapping_end:
                if "r" not in permissions or "w" not in permissions:
                    raise RuntimeError(
                        "MemFabric BM LOCAL_HOST mapping is not CPU read/write: "
                        f"permissions={permissions}"
                    )
                return permissions
        raise RuntimeError(
            "MemFabric BM LOCAL_HOST is not present in /proc/self/maps; "
            "SMEM_BM_FLAG_DRAM_MAP_HOST_VA may be ineffective"
        )

    def initialize(self) -> None:
        if self._closed:
            raise RuntimeError("MemFabric BM allocator is already closed")
        if self._handle is not None:
            return

        try:
            self._initialize_runtime()
        except BaseException:
            try:
                self.close()
            except BaseException:
                logger.exception(
                    "Failed to release a partially initialized MemFabric BM "
                    "Full-KV allocator."
                )
            raise

    def _initialize_runtime(self) -> None:
        self._mf, self._bm = self._load_runtime()
        self._mf.set_log_level(self.config.log_level)
        mf_result = self._mf.initialize()
        if mf_result != 0:
            raise RuntimeError(f"memfabric_hybrid.initialize failed: result={mf_result}")
        self._mf_initialized = True

        bm_config = self._bm.BmConfig()
        bm_config.rank_id = self.config.rank_id
        bm_config.start_store = self.config.starts_store
        bm_config.unified_address_space = True
        bm_config.set_nic(self.config.nic_url)
        bm_result = self._bm.initialize(
            self.config.store_url,
            MEMFABRIC_BM_WORLD_SIZE,
            self.config.device_id,
            bm_config,
        )
        if bm_result != 0:
            raise RuntimeError(f"MemFabric BM initialize failed: result={bm_result}")
        self._bm_initialized = True

        self._handle = self._bm.create2(
            id=self.config.bm_id,
            local_dram_size=self.config.pool_bytes,
            max_dram_size=self.config.pool_bytes,
            data_op_type=self._data_op_type(),
            flags=SMEM_BM_FLAG_DRAM_MAP_HOST_VA,
        )
        if self._handle is None:
            raise RuntimeError("MemFabric BM create2 returned None")
        join_result = self._handle.join()
        if join_result != 0:
            raise RuntimeError(f"MemFabric BM join failed: result={join_result}")
        self._joined = True

        self.local_gva = self._handle.peer_rank_ptr(
            self.config.rank_id,
            self._bm.BmMemType.HOST,
        )
        self.local_host_va = self._handle.gva_to_va(
            self.local_gva,
            self._bm.BmMemType.LOCAL_HOST,
        )
        self.local_device_va = self._handle.gva_to_va(
            self.local_gva,
            self._bm.BmMemType.LOCAL_DEVICE,
        )
        if not self.local_gva or not self.local_host_va or not self.local_device_va:
            raise RuntimeError(
                "MemFabric BM did not expose the required local addresses: "
                f"gva=0x{self.local_gva:x}, host=0x{self.local_host_va:x}, "
                f"device=0x{self.local_device_va:x}"
            )
        if not (
            self.local_gva == self.local_host_va == self.local_device_va
        ):
            raise RuntimeError(
                "MemFabric BM Full-KV integration requires one unified local "
                "GVA/Host/NPU address, got "
                f"gva=0x{self.local_gva:x}, host=0x{self.local_host_va:x}, "
                f"device=0x{self.local_device_va:x}"
            )
        if self.local_device_va % MEMFABRIC_BM_TENSOR_ALIGNMENT_BYTES:
            raise RuntimeError(
                "MemFabric BM LOCAL_DEVICE base must be 2 MiB aligned, got "
                f"0x{self.local_device_va:x}"
            )
        permissions = self._assert_cpu_mapping()
        logger.info(
            "Initialized MemFabric BM Full-KV allocator: role=%s, tp_rank=%d, "
            "rank_id=%d, protocol=%s, pool_bytes=%d, gva=0x%x, permissions=%s.",
            self.config.role,
            self.config.tp_rank,
            self.config.rank_id,
            self.config.protocol,
            self.config.pool_bytes,
            self.local_gva,
            permissions,
        )

    def allocate_int8(
        self,
        numel: int,
        *,
        device: torch.device,
        alignment: int = MEMFABRIC_BM_TENSOR_ALIGNMENT_BYTES,
    ) -> torch.Tensor:
        self.initialize()
        nbytes = int(numel)
        offset = self.plan.reserve(nbytes, alignment=alignment)
        data_ptr = self.local_device_va + offset
        construct_storage = getattr(
            torch_npu._C,
            "_construct_storage_from_data_pointer",
            None,
        )
        construct_tensor = getattr(
            torch_npu._C,
            "_construct_NPU_Tensor_From_Storage_And_Metadata",
            None,
        )
        if not callable(construct_storage) or not callable(construct_tensor):
            raise RuntimeError(
                "MemFabric BM Full-KV allocation requires torch_npu external "
                "storage and Tensor construction APIs."
            )
        storage = construct_storage(data_ptr, device, nbytes)
        metadata = {
            "data_ptr": data_ptr,
            "device": device,
            "nbytes": nbytes,
            "dtype": torch.int8,
            "size": (numel,),
            "stride": (1,),
            "storage_offset": 0,
        }
        tensor = construct_tensor(metadata, storage)
        allocation = MemFabricBMAllocation(
            offset=offset,
            nbytes=nbytes,
            device_ptr=data_ptr,
            gva=self.local_gva + offset,
        )
        self.allocations.append(allocation)
        return tensor

    def owns_device_range(self, data_ptr: int, nbytes: int) -> bool:
        if nbytes <= 0:
            return False
        start = self.local_device_va
        return start <= data_ptr and data_ptr + nbytes <= start + self.config.pool_bytes

    def close(self) -> None:
        """Release BM after every external Tensor alias has been destroyed."""
        if self._closed:
            return
        if torch.npu.is_available():
            torch.npu.synchronize()
        if self._handle is not None and self._joined:
            leave_result = self._handle.leave()
            if leave_result != 0:
                raise RuntimeError(f"MemFabric BM leave failed: result={leave_result}")
            self._joined = False
        if self._handle is not None:
            self._handle.destroy()
            self._handle = None
        if self._bm_initialized and self._bm is not None:
            self._bm.uninitialize(self.config.device_id)
            self._bm_initialized = False
        if self._mf_initialized and self._mf is not None:
            self._mf.uninitialize()
            self._mf_initialized = False
        self._closed = True
        logger.info(
            "Released MemFabric BM Full-KV allocator: role=%s, tp_rank=%d, "
            "allocations=%d, used_bytes=%d.",
            self.config.role,
            self.config.tp_rank,
            len(self.allocations),
            self.plan.cursor_bytes,
        )

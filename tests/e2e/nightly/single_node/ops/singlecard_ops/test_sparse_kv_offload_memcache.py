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

import math
import os
import uuid

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.config import ParallelConfig

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    MmcDirect,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVTransferThread
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

RUN_MEMCACHE_INTEGRATION_TEST = os.getenv("VLLM_ASCEND_RUN_MEMCACHE_INTEGRATION_TEST") == "1"

pytestmark = [
    pytest.mark.skipif(
        not RUN_MEMCACHE_INTEGRATION_TEST,
        reason=("Set VLLM_ASCEND_RUN_MEMCACHE_INTEGRATION_TEST=1 to run the external MemCache integration test."),
    ),
    pytest.mark.skipif(
        not torch.npu.is_available(),
        reason="The sparse KV offload MemCache integration test requires an Ascend NPU.",
    ),
    pytest.mark.skipif(
        get_ascend_device_type() != AscendDeviceType.A3,
        reason="Sparse KV offload host mode currently supports only Ascend A3.",
    ),
]

BLOCK_SIZE = 128


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _make_npu_indexer_cache() -> tuple[torch.Tensor, ...]:
    return (
        torch.empty(
            (1, BLOCK_SIZE, 4),
            dtype=torch.bfloat16,
            device="npu",
        ),
    )


def _make_swapped_cache(shape: tuple[int, ...]) -> torch.Tensor:
    dtype = torch.bfloat16
    dtype_size = torch.empty((), dtype=dtype).element_size()
    raw_storage = torch_npu.empty_with_swapped_memory(
        (math.prod(shape) * dtype_size,),
        dtype=torch.int8,
        device="npu",
    )
    return raw_storage.view(dtype).view(shape)


def _make_sparse_offload_cache() -> tuple[torch.Tensor, ...]:
    return (
        _make_swapped_cache((1, BLOCK_SIZE, 1, 2)),
        _make_swapped_cache((1, BLOCK_SIZE, 1, 1)),
        torch.empty(
            (1, BLOCK_SIZE, 4),
            dtype=torch.bfloat16,
            device="npu",
        ),
    )


def _make_sparse_offload_staging(cache: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty_like(cache[0], device="npu"),
        torch.empty_like(cache[1], device="npu"),
        cache[2],
    )


def _make_transfer_arrays(
    base_gva: int,
    tensors: tuple[torch.Tensor, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = np.asarray(
        [_tensor_nbytes(tensor) for tensor in tensors],
        dtype=np.int64,
    )
    offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(sizes[:-1], dtype=np.int64),
        )
    )
    gvas = base_gva + offsets
    local_addrs = np.asarray(
        [tensor.data_ptr() for tensor in tensors],
        dtype=np.int64,
    )
    return gvas, local_addrs, sizes


def _copy_npu_cache(
    backend: MemcacheBackend,
    gvas: np.ndarray,
    addrs: np.ndarray,
    sizes: np.ndarray,
    *,
    is_save: bool,
) -> int:
    transfer = KVTransferThread.__new__(KVTransferThread)
    transfer.m_store = backend
    transfer.num_addrs_per_block = len(gvas)
    direction = MmcDirect.COPY_L2G.value if is_save else MmcDirect.COPY_G2L.value
    return transfer._batch_copy_with_limits(
        gvas,
        addrs,
        sizes,
        direction,
        max_transfer_blocks=0,
        max_transfer_bytes=0,
    )


def test_memcache_round_trip_npu_resident_indexer_cache():
    """Verify the supported MemCache path used by the Indexer cache."""
    MemcacheBackend.validate_gva_layerwise_api()
    torch.npu.set_device(0)
    source = _make_npu_indexer_cache()
    destination = _make_npu_indexer_cache()

    source[0].copy_(
        torch.arange(
            source[0].numel(),
            dtype=torch.float32,
            device="npu",
        ).reshape(source[0].shape)
    )
    for tensor in destination:
        tensor.fill_(-1)
    torch.npu.synchronize()

    backend = MemcacheBackend(
        ParallelConfig(),
        local_rank=0,
        init_bm=True,
    )
    all_tensors = (*source, *destination)
    backend.register_buffer(
        [tensor.data_ptr() for tensor in all_tensors],
        [_tensor_nbytes(tensor) for tensor in all_tensors],
    )

    key = f"vllm-ascend-sparse-offload-indexer-smoke-{uuid.uuid4().hex}"
    total_bytes = sum(_tensor_nbytes(tensor) for tensor in source)
    allocated_gvas = backend.batch_alloc([key], [total_bytes])
    assert len(allocated_gvas) == 1
    assert allocated_gvas[0] > 0
    assert backend.store is not None
    lease_acquired = False
    try:
        source_gvas, source_addrs, source_sizes = _make_transfer_arrays(
            allocated_gvas[0],
            source,
        )
        save_result = _copy_npu_cache(
            backend,
            source_gvas,
            source_addrs,
            source_sizes,
            is_save=True,
        )
        assert save_result == 0

        key_info = backend.batch_get_key_info([key])
        assert len(key_info) == 1
        assert key_info[0].size() > 0
        lease_result = backend.batch_add_lease([key])
        assert all(result == 0 for result in lease_result)
        lease_acquired = True

        load_gva_list = key_info[0].gva_list()
        assert load_gva_list
        destination_gvas, destination_addrs, destination_sizes = _make_transfer_arrays(
            load_gva_list[0],
            destination,
        )
        load_result = _copy_npu_cache(
            backend,
            destination_gvas,
            destination_addrs,
            destination_sizes,
            is_save=False,
        )
        assert load_result == 0
        torch.npu.synchronize()

        for expected, actual in zip(source, destination):
            torch.testing.assert_close(actual.cpu(), expected.cpu())
    finally:
        if lease_acquired:
            backend.batch_remove_lease([key])
        backend.store.remove(key)


def test_memcache_round_trip_sparse_offload_through_npu_staging():
    """Verify swapped Full KV through NPU staging using public L2G/G2L."""
    MemcacheBackend.validate_gva_layerwise_api()
    torch.npu.set_device(0)

    source = _make_sparse_offload_cache()
    destination = _make_sparse_offload_cache()
    source_staging = _make_sparse_offload_staging(source)
    destination_staging = _make_sparse_offload_staging(destination)

    for offset, tensor in zip((0, 4096, 8192), source):
        tensor.copy_(
            torch.arange(
                tensor.numel(),
                dtype=torch.float32,
                device="npu",
            ).reshape(tensor.shape)
            + offset
        )
    for tensor in destination:
        tensor.fill_(-1)

    # Producer-side staging: Full KV originates in swapped Host memory while
    # the Indexer cache is already NPU-resident.
    source_staging[0].copy_(source[0])
    source_staging[1].copy_(source[1])
    torch.npu.synchronize()

    backend = MemcacheBackend(
        ParallelConfig(),
        local_rank=0,
        init_bm=True,
    )
    staged_tensors = (*source_staging, *destination_staging)
    backend.register_buffer(
        [tensor.data_ptr() for tensor in staged_tensors],
        [_tensor_nbytes(tensor) for tensor in staged_tensors],
    )

    key = f"vllm-ascend-sparse-offload-staging-smoke-{uuid.uuid4().hex}"
    total_bytes = sum(_tensor_nbytes(tensor) for tensor in source_staging)
    allocated_gvas = backend.batch_alloc([key], [total_bytes])
    assert len(allocated_gvas) == 1
    assert allocated_gvas[0] > 0
    assert backend.store is not None
    lease_acquired = False
    try:
        source_gvas, source_addrs, source_sizes = _make_transfer_arrays(
            allocated_gvas[0],
            source_staging,
        )
        save_result = _copy_npu_cache(
            backend,
            source_gvas,
            source_addrs,
            source_sizes,
            is_save=True,
        )
        assert save_result == 0

        key_info = backend.batch_get_key_info([key])
        assert len(key_info) == 1
        assert key_info[0].size() > 0
        lease_result = backend.batch_add_lease([key])
        assert all(result == 0 for result in lease_result)
        lease_acquired = True

        load_gva_list = key_info[0].gva_list()
        assert load_gva_list
        destination_gvas, destination_addrs, destination_sizes = _make_transfer_arrays(
            load_gva_list[0],
            destination_staging,
        )
        load_result = _copy_npu_cache(
            backend,
            destination_gvas,
            destination_addrs,
            destination_sizes,
            is_save=False,
        )
        assert load_result == 0
        torch.npu.synchronize()

        # Consumer-side staging: Full KV lands in NPU memory first, then moves
        # into swapped Host memory. The Indexer destination is already final.
        destination[0].copy_(destination_staging[0])
        destination[1].copy_(destination_staging[1])
        torch.npu.synchronize()

        for expected, actual in zip(source, destination):
            torch.testing.assert_close(actual.cpu(), expected.cpu())
    finally:
        if lease_acquired:
            backend.batch_remove_lease([key])
        backend.store.remove(key)

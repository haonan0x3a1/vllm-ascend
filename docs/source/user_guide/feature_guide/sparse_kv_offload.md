# DeepSeek-V3.2 Sparse KV Offload

## Overview

This experimental path implements the DeepSeek-V3.2 DSA sparse KV data flow on
Atlas A3:

1. The full MLA latent/RoPE KV cache resides in Host-backed swapped memory.
2. The Lightning Indexer cache remains on the NPU.
3. Prefill and decode writes use one NPU staging workspace shared by all
   layers. A later
   chunked-prefill chunk first restores that layer's prior physical blocks
   from Full Host KV, then persists only the token rows touched by the current
   forward back into Host KV. Decode also persists its newly generated token
   row before Top-K Gather.
4. Lightning Indexer produces Top-K token indices for a decode step.
5. `npu_gather_selection_kv_cache` reads Full Host KV and materializes selected
   KV on the NPU.
6. The existing Sparse FlashAttention operator consumes selected KV using
   local indices.

The default `mirror` mode remains available as a lower-risk integration check.
It keeps Full KV on the NPU and mirrors updated physical blocks into swapped
memory. It validates Gather and SFA correctness but does not save HBM.

`host` mode is the memory-saving implementation. It automatically reserves the
number of logical blocks required by `max_model_len`, including vLLM's null
block, while allocating only the Indexer cache and runtime workspaces in HBM.
Before allocation, the worker validates the real persistent NPU footprint
(Indexer cache, transfer alignment, shared Prefill workspace, per-layer
Selected KV, and selection metadata) against the memory left by model
profiling. This keeps the logical-block override from hiding an insufficient
HBM budget.

## Prerequisites

- Atlas A3 hardware.
- A CANN and torch-npu version compatible with the vLLM Ascend environment.
- The `custom_ops` wheel built from `cann-recipes-infer`, including
  `torch_npu.npu_gather_selection_kv_cache`.
- Online P/D staging requires the Ascend Mooncake TransferEngine package
  `mooncake-transfer-engine-npu>=0.3.12.post1`.
- A DeepSeek-V3.2 DSA model with `index_topk=2048`.
- MLAPO remains an optional optimization for checkpoints supported by the
  existing A3 SFA MLAPO path. Other quantization schemes, including Dynamic
  W8A8 Attention weights, use the shared NPU staging path.

Before serving a model, run the focused configuration, workspace, and
model-runner tests:

```bash
pytest -sv \
  tests/ut/test_ascend_config.py::TestSparseKVOffloadConfig \
  tests/ut/attention/test_sparse_kv_offload.py \
  tests/ut/worker/a2/test_model_runner_v1.py::TestNPUModelRunnerKVCache \
  tests/ut/distributed/ascend_store/test_pool_scheduler.py \
  tests/ut/distributed/ascend_store/test_pool_worker.py
```

Then run the NPU tests that explicitly create Full KV in swapped memory:

```bash
pytest -sv \
  tests/ut/attention/a3_2/test_sparse_kv_offload_npu.py
```

Importing `custom_ops` successfully is not sufficient evidence: the test must
exercise the Host/Swapped source path.

The NPU tests validate both modes, exercise prefill row persistence in host
mode, and compare full-KV SFA with Host-Gather-selected-KV SFA for BF16 and
FP16.

## Usage

### Host Full-KV mode

```bash
vllm serve /path/to/DeepSeek-V3.2-Exp \
  --enforce-eager \
  --max-num-seqs 1 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --additional-config \
  '{"enable_mlapo":false,"sparse_kv_offload":{"enabled":true,"mode":"host"}}'
```

### Mirror validation mode

Replace `"mode":"host"` with `"mode":"mirror"`. Mirror mode cannot be combined
with a KV Connector.

### Online P/D disaggregation

Host Full-KV mode supports an experimental Online P/D path through
`MooncakeLayerwiseConnector`. Mooncake does not register the swapped/SVM Full-KV
addresses directly. Instead, both workers register the existing aligned,
cross-layer NPU Prefill workspace as a one-layer transfer staging cache:

1. Prefill computes one layer into the shared NPU staging cache.
2. Mooncake transfers that layer into the Decode worker's NPU staging cache.
3. Decode synchronously persists the received physical blocks into that
   layer's swapped Full KV and acknowledges the layer.
4. Only after the ACK may Prefill and Decode reuse their staging cache for the
   next layer. The NPU-resident Lightning Indexer cache is transferred directly.

This first correctness path intentionally serializes layer reuse. It does not
claim transfer/compute overlap. It also requires the same MLA tensor-parallel
layout on Prefill and Decode workers.

Both workers must enable Host mode and configure the layerwise connector. Use
`kv_producer` on Prefill and `kv_consumer` on Decode; the metaserver/proxy and
per-role topology fields follow the standard Mooncake layerwise P/D guide:

```text
--additional-config \
  '{"enable_mlapo":false,"sparse_kv_offload":{"enabled":true,"mode":"host"}}' \
--kv-transfer-config \
  '{"kv_connector":"MooncakeLayerwiseConnector",
    "kv_role":"kv_producer",
    "kv_port":"30000",
    "kv_connector_extra_config":{
      "prefill":{"dp_size":1,"tp_size":1},
      "decode":{"dp_size":1,"tp_size":1}}}'
```

Change only `kv_role` to `kv_consumer` for the Decode worker and use the actual
matching topology. The current sparse Host validation still requires
`--max-num-seqs 1`, `--block-size 128`, eager mode, and prefix caching disabled.

The feature is disabled by default. When disabled, allocation and SFA execution
remain unchanged.

For a new environment, validate startup first with `--load-format dummy`, then
repeat the same request with real model weights and compare the generated output
against the feature-disabled eager baseline. A dummy-weight run verifies wiring
and memory allocation only; it is not an accuracy result.

## Current Limitations

- One request at a time.
- Eager execution only.
- BF16 or FP16 KV cache only; sparse C8 is unsupported.
- No speculative decoding or MTP.
- No prefix caching.
- No DSA context parallelism, PCP, or DCP.
- Online P/D Host mode currently requires `MooncakeLayerwiseConnector` and NPU
  staging; direct swapped/SVM registration is unsupported.
- The staged Online P/D path is correctness-first: one request, equal P/D MLA
  tensor-parallel layouts, and synchronous per-layer ACK before staging reuse.
- The current NPU integration test covers a same-node staged transfer. A
  cross-node model-serving result is still required before claiming end-to-end
  Online P/D support in a deployment.
- Selected KV state is reused across decode steps and reset at Prefill/request
  boundaries. The current device buffer holds one Top-K working set rather than
  a larger configurable LRU pool.
- Chunked-prefill context restoration and Prefill row persistence are
  synchronous for correctness.
- Gather/SFA overlap, asynchronous prefill D2H, and performance claims require
  separate profiling.

# DeepSeek-V3.2 Sparse KV Offload

## Overview

This experimental path implements the DeepSeek-V3.2 DSA sparse KV data flow on
Atlas A3:

1. The full MLA latent/RoPE KV cache resides in Host-backed swapped memory.
2. The Lightning Indexer cache remains on the NPU.
3. Prefill uses one NPU workspace shared by all layers and persists only the
   token rows touched by the current forward into Full Host KV.
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

## Prerequisites

- Atlas A3 hardware.
- A CANN and torch-npu version compatible with the vLLM Ascend environment.
- The `custom_ops` wheel built from `cann-recipes-infer`, including
  `torch_npu.npu_gather_selection_kv_cache`.
- A DeepSeek-V3.2 DSA model with `index_topk=2048`.
- Host mode requires the A3 `npu_mla_prolog_v3` path and a supported W8A8
  checkpoint.

Before serving a model, run the focused configuration, workspace, and
model-runner tests:

```bash
pytest -sv \
  tests/ut/test_ascend_config.py::TestSparseKVOffloadConfig \
  tests/ut/attention/test_sparse_kv_offload.py \
  tests/ut/worker/a2/test_model_runner_v1.py::TestNPUModelRunnerKVCache
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
  '{"enable_mlapo":true,"sparse_kv_offload":{"enabled":true,"mode":"host"}}'
```

### Mirror validation mode

Replace `"mode":"host"` with `"mode":"mirror"`. Mirror mode cannot be combined
with a KV Connector.

### Online P/D disaggregation

Host mode supports the existing layerwise `AscendStoreConnector` with the
memcache backend. Add the following connector configuration to both commands,
using `kv_producer` for Prefill and `kv_consumer` for Decode:

```bash
--kv-transfer-config '{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "backend": "memcache",
    "mooncake_rpc_port": "0",
    "use_layerwise": true
  }
}'
```

Use the dedicated layerwise proxy and memcache setup described in
[Layerwise KV Pool](layerwise_kv_pool.md). The mixed cache tuple is transferred
layer by layer: Full MLA KV targets swapped memory while the Indexer cache
targets NPU memory.

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
- P/D currently supports only layerwise `AscendStoreConnector` with memcache.
- Selected KV state is reused across decode steps and reset at Prefill/request
  boundaries. The current device buffer holds one Top-K working set rather than
  a larger configurable LRU pool.
- Prefill row persistence is synchronous for correctness.
- Gather/SFA overlap, asynchronous prefill D2H, and performance claims require
  separate profiling.

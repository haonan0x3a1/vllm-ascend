# Sparse KV Offload Mirror PoC

## Overview

This experimental path validates the DeepSeek-V3.2 DSA sparse KV data flow on
Ascend:

1. The framework writes the full MLA KV cache to NPU memory as usual.
2. Physical KV blocks updated by the current forward are copied to
   Host-backed swapped memory.
3. Lightning Indexer produces the Top-K token indices for a decode step.
4. `npu_gather_selection_kv_cache` reads the full KV from swapped memory and
   materializes a selected KV cache on the NPU.
5. The existing Sparse FlashAttention operator consumes the selected KV cache
   using local indices.

`mirror` mode intentionally keeps the original full KV cache on the NPU. It is
a correctness and integration PoC, not an NPU-memory-saving implementation.

## Prerequisites

- Atlas A3 hardware.
- A CANN and torch-npu version compatible with the vLLM Ascend environment.
- The `custom_ops` wheel built from `cann-recipes-infer`, including
  `torch_npu.npu_gather_selection_kv_cache`.
- A DeepSeek-V3.2 DSA model with `index_topk=2048`.

Before serving a model, run the focused configuration and CPU workspace tests:

```bash
pytest -sv \
  tests/ut/test_ascend_config.py::TestSparseKVOffloadConfig \
  tests/ut/attention/test_sparse_kv_offload.py
```

Then run the NPU test that explicitly creates full KV in swapped memory:

```bash
pytest -sv \
  tests/ut/attention/a3_2/test_sparse_kv_offload_npu.py
```

Importing `custom_ops` successfully is not sufficient evidence: the test must
exercise the Host/Swapped source path.

The NPU test also compares the existing full-KV SFA output with the
Host/Swapped Gather-to-selected-KV SFA output for BF16 and FP16.

## Usage

```bash
vllm serve /path/to/DeepSeek-V3.2-Exp \
  --enforce-eager \
  --max-num-seqs 1 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --additional-config \
  '{"sparse_kv_offload":{"enabled":true,"mode":"mirror"}}'
```

The feature is disabled by default. When disabled, the existing SFA execution
path is unchanged.

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
- No KV Connector or P/D disaggregation.
- Selected KV state is reset every decode step; Top-K reuse is disabled.
- Updated physical blocks are copied synchronously for correctness.
- Full KV remains allocated on the NPU, so this mode does not reduce HBM use.

The next implementation stage replaces the NPU full-KV allocation with
Host-resident full KV plus a layerwise prefill workspace. P/D disaggregation
can then reuse the existing Mooncake connector while sending full MLA KV to
Host storage and keeping the Indexer cache on the NPU.

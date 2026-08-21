# DeepSeek-V3.2 same-host PD transfer profiling

This runner compares the two real 8-NPU Prefill + 8-NPU Decode paths at one
fixed revision:

- `npu_staging`: the verified Mooncake Ascend Full-KV baseline.
- `memfabric_bm`: Full KV uses BM Host-to-Host while Mooncake carries only the
  Indexer KV.

It answers two different questions in separate runs:

1. Headline A/B: TTFT, TPOT, E2E, and throughput with profiling disabled.
2. Stage breakdown: one explicitly profiled MemFabric request using the trace
   scopes below.

## One-time setup

```bash
cd /workspace/w50062541/code/vllm-ascend/benchmarks/dsv32_pd_transfer
cp --update=none config.example.env config.env
bash run.sh preflight
```

`config.env` points to the already working runtime `config.env`; machine paths
and service settings therefore still have one source of truth.

## Headline A/B

First set `SPARSE_KV_TRANSFER_MODE=npu_staging` and
`ENABLE_TORCH_PROFILER=false` in the runtime config. Start Decode, Prefill, and
proxy as usual, then run:

```bash
bash run.sh bench-suite npu_staging
```

Stop all three services and confirm all 16 NPUs are free. Change only
`SPARSE_KV_TRANSFER_MODE=memfabric_bm`, restart the three services, and run:

```bash
bash run.sh bench-suite memfabric_bm
bash run.sh summarize
```

The summary refuses to pair results if the revision, model hash, device map,
request shape, or important serving settings differ. Positive latency deltas
mean MemFabric is slower; positive throughput deltas mean it is faster.

## One profiled MemFabric request

Profiler traces are diagnostic evidence, not headline performance. Set these
runtime values before starting the MemFabric services:

```bash
ENABLE_TORCH_PROFILER=true
TORCH_PROFILER_DIR=/workspace/w50062541/output/dsv32-pd-transfer-profile/traces
```

After all services are ready:

```bash
bash run.sh profile-one 3018
```

The script starts both engine profilers directly, sends one request through the
proxy, then stops both profilers. Search the traces for:

- `dsv32_pd_transfer:copy_full_kv_npu_to_host`
- `dsv32_pd_transfer:wait_prefill_full_kv_visible`
- `dsv32_pd_transfer:memfabric_full_kv_host_to_host`
- `dsv32_pd_transfer:mooncake_indexer_npu_to_npu`
- `dsv32_pd_transfer:layer_ack_round_trip`
- `dsv32_pd_transfer:decode_full_kv_visibility_fence`
- `dsv32_pd_transfer:gather_selected_kv_host_to_npu`

The Gather scope must be interpreted from its NPU kernels in the trace. Its CPU
scope duration is dispatch overhead, not the Host-to-NPU Gather execution time.

## Current evidence boundary

This runner measures same-physical-host `host_tcp`. It does not prove
cross-host `HOST_RDMA`, production concurrency, or that same-host TCP predicts
cross-host bandwidth. Those remain separate gates.

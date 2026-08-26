# DeepSeek-V3.2 same-host PD transfer profiling

This runner compares the two real 8-NPU Prefill + 8-NPU Decode paths at one
fixed revision:

- `npu_staging`: the verified Mooncake Ascend Full-KV baseline.
- `memfabric_bm`: Full KV uses BM Host-to-Host while Mooncake carries only the
  Indexer KV.

It answers two different questions in separate runs:

1. Headline A/B: TTFT, TPOT, E2E, and throughput with profiling disabled.
2. Background Host stages: one MemFabric request with low-overhead aggregate
   timers in the Connector transfer threads.
3. NPU trace scopes: one explicitly profiled MemFabric request using the trace
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
Each repetition must complete every request and leave Prefill, Decode, and the
proxy healthy; the suite stops immediately when either check fails.

## One background-stage MemFabric request

Torch profiler scopes are thread-local in this runtime. The model-execution
thread therefore captured the NPU-to-Host Full-KV copy and Host-to-NPU Gather,
but did not capture the Connector sender/receiver threads. Use the opt-in Host
timers for those missing stages.

Set these values in the runtime config before starting the services:

```bash
ENABLE_TORCH_PROFILER=false
SPARSE_KV_STAGE_METRICS=true
```

After all services are ready, run exactly one request:

```bash
cd /workspace/w50062541/code/vllm-ascend/benchmarks/dsv32_pd_transfer
bash run.sh stage-one 3018

cd /workspace/w50062541/code/vllm-ascend/examples/disaggregated_prefill_v1/dsv32_sparse_offload_poc
bash run.sh stage-summary
```

`stage-summary` requires one completed record from every Prefill and Decode TP
rank. It reports Host-observed synchronous durations for:

- Prefill visibility-event wait;
- MemFabric Full-KV Host-to-Host copy;
- Mooncake Indexer NPU-to-NPU transfer;
- layer ACK round trip;
- Decode Full-KV visibility fence.

The aggregate state is shared by work handled on one background thread, so this
probe deliberately sends one request with no concurrency. These durations are
not NPU kernel timings and are not added together as end-to-end latency.

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
bash run.sh analyze-profile
bash run.sh summarize-profile
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

Only `copy_full_kv_npu_to_host` and
`gather_selected_kv_host_to_npu` are expected from the current raw Ascend
trace. The five background-thread stages are covered by the Host-observed probe
above; their absence from the raw trace is not a fallback to the old data path.

The Gather scope must be interpreted from its NPU kernels in the trace. Its CPU
scope duration is dispatch overhead, not the Host-to-NPU Gather execution time.
Ascend first writes raw `*_ascend_pt` directories. `analyze-profile` performs
the documented offline `torch_npu.profiler.profiler.analyse()` conversion for
each worker and writes its verbose output to
`profiler/$BENCH_CAMPAIGN/trace-analysis.log`. It does not require a running
model, but needs additional disk space for `ASCEND_PROFILER_OUTPUT`.
`summarize-profile` scans the generated Chrome-trace JSON files and prints each
custom scope's event count, mean per-trace-file total, mean event duration,
and maximum event duration. It keeps the source-file list in
`profiler/$BENCH_CAMPAIGN/stage-summary.json`; totals across TP ranks must not be
interpreted as wall-clock latency.

## Current evidence boundary

This runner measures same-physical-host `host_tcp`. It does not prove
cross-host `HOST_RDMA`, production concurrency, or that same-host TCP predicts
cross-host bandwidth. Those remain separate gates.

# DeepSeek-V3.2 Sparse KV Offload profiling

This runner compares two single-instance serving configurations with the same
real model, NPU set, and benchmark requests:

- `baseline`: the framework-owned Full KV cache stays in NPU HBM.
- `host`: the Full KV cache uses Host-backed swapped memory and Decode gathers
  the selected Top-K rows into an NPU workspace.

The first profiling gate deliberately excludes P/D disaggregation and
Mooncake. It measures the local Host Sparse KV Offload cost before transport is
added to the critical path.

## Scope of the first gate

The runner uses `vllm bench serve` because it reports user-visible TTFT, TPOT,
ITL, E2EL, and token throughput. It is a single-instance serving benchmark,
not an offline engine benchmark and not a maximum-concurrency throughput test.
The Host implementation currently requires `max_num_seqs=1`, so both modes
are run at concurrency one for a fair comparison.

Do not enable a profiler while collecting headline A/B numbers. Profiler runs
perturb timing and belong to the next operator-breakdown gate.

## One-time server configuration

```bash
cd benchmarks/dsv32_sparse_offload
cp --update=none config.example.env config.env
```

Adjust the machine-specific paths and NPU IDs in `config.env`. The file is
ignored by Git.

The checked-in defaults are a short first pass: 3072 input tokens, 32 output
tokens, five measured requests, one warmup, and three repetitions. After the
short pass is stable, increase `BENCH_OUTPUT_LEN` to 64 or 128 and
`BENCH_NUM_PROMPTS` to at least 10 for a more stable TPOT distribution.
Set a new `BENCH_CAMPAIGN` name for every independent experiment; results from
different campaigns are kept in separate directories instead of overwritten.

## Run the A/B experiment

Run preflight first:

```bash
bash run.sh preflight
```

Start the baseline service in terminal 1:

```bash
bash run.sh serve baseline
```

In terminal 2, wait for readiness and run the complete baseline suite:

```bash
bash run.sh ready baseline
bash run.sh bench-suite baseline
```

Stop terminal 1 with `Ctrl-C`. Confirm the configured NPUs are free, then start
the Host service in terminal 1:

```bash
bash run.sh serve host
```

Run the matching Host suite in terminal 2:

```bash
bash run.sh ready host
bash run.sh bench-suite host
bash run.sh summarize
```

The raw vLLM result JSON files are stored below
`$OUTPUT_DIR/dsv32-sparse-offload-profile/raw/$BENCH_CAMPAIGN`. The campaign
summary is written below `$OUTPUT_DIR/dsv32-sparse-offload-profile` and printed
as a table.
Each suite also records raw `npu-smi info` and `free -b` snapshots before and
after the measurements. These snapshots establish the loaded-service memory
state; they are not a high-frequency peak-memory trace.

For one targeted run instead of the configured suite:

```bash
bash run.sh bench baseline 3072 1
bash run.sh bench host 3072 1
```

## Fairness requirements

- Run the modes sequentially on the same physical NPUs. Do not run them side
  by side.
- Confirm with `npu-smi info` that no foreign process uses those NPUs.
- Keep model revision, vLLM/vLLM-Ascend revision, CANN/custom OPP, TP/EP,
  eager mode, BF16 KV, seed, input/output lengths, and benchmark load equal.
- Keep prefix caching disabled and use random prompts with a fixed seed.
- Treat the first run after service startup as warmup; compare all configured
  repetitions rather than a single result.
- Record OOM or the largest successful context separately. Latency and capacity
  answer different questions.

The summary reports `host_vs_baseline_pct` as
`(host - baseline) / baseline * 100`. Positive values mean an increase. That is
a regression for latency metrics but an improvement for throughput metrics.
All request-level samples from the configured repetitions are pooled before
means, medians, and p99 values are calculated; the runner does not average
per-repetition percentiles.

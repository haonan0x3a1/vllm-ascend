#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_PATH=${DSV32_PD_PROFILE_CONFIG:-"$SCRIPT_DIR/config.env"}
ACTION=${1:-help}

show_help() {
    cat <<'EOF'
Usage: bash run.sh <command> [arguments]

Commands:
  preflight                     Validate the paired A/B configuration.
  ready                         Check Prefill, Decode, and the P/D proxy.
  bench <mode> <input> <run>    Run one headline measurement.
  bench-suite <mode>            Run every configured input and repetition.
  profile-one <input>           Capture one MemFabric BM profiler request.
  summarize                     Compare npu_staging with memfabric_bm.
  help                          Show this help.

The runtime services are still started by the disaggregated Prefill example.
The mode passed here must equal SPARSE_KV_TRANSFER_MODE in its config.env.
EOF
}

if [[ "$ACTION" == "help" || "$ACTION" == "-h" || "$ACTION" == "--help" ]]; then
    show_help
    exit 0
fi
if [[ ! -r "$CONFIG_PATH" ]]; then
    echo "Missing profiling config: $CONFIG_PATH" >&2
    echo "Create it with: cp $SCRIPT_DIR/config.example.env $SCRIPT_DIR/config.env" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"
if [[ ! -r "${RUNTIME_CONFIG:-}" ]]; then
    echo "RUNTIME_CONFIG is missing or unreadable: ${RUNTIME_CONFIG:-<unset>}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$RUNTIME_CONFIG"
ENABLE_TORCH_PROFILER=${ENABLE_TORCH_PROFILER:-false}
if [[ "$ENABLE_TORCH_PROFILER" != "true" && "$ENABLE_TORCH_PROFILER" != "false" ]]; then
    echo "ENABLE_TORCH_PROFILER must be true or false, got: $ENABLE_TORCH_PROFILER" >&2
    exit 1
fi

required_variables=(
    REPO_DIR MODEL_PATH OUTPUT_DIR CANN_ENV CUSTOM_OPP_ENV MOONCAKE_PYTHON
    HOST_IP PROXY_PORT PREFILL_API_PORT DECODE_API_PORT PREFILL_DEVICES
    DECODE_DEVICES TP_SIZE SERVED_MODEL_NAME HF_OVERRIDES MAX_MODEL_LEN MAX_NUM_SEQS
    MAX_NUM_BATCHED_TOKENS BLOCK_SIZE GPU_MEMORY_UTILIZATION SEED
    ENABLE_PREFILL_MC2 ENABLE_MLAPO ENABLE_FLASHCOMM1 SPARSE_KV_OFFLOAD_MODE
    SPARSE_KV_TRANSFER_MODE MEMFABRIC_BM_PROTOCOL BENCH_CAMPAIGN BENCH_INPUT_LENGTHS BENCH_OUTPUT_LEN
    BENCH_NUM_PROMPTS BENCH_NUM_WARMUPS BENCH_REPETITIONS BENCH_SEED
    BENCH_REQUEST_RATE BENCH_MAX_CONCURRENCY
)
for variable in "${required_variables[@]}"; do
    if [[ -z "${!variable:-}" ]]; then
        echo "Required config variable is missing: $variable" >&2
        exit 1
    fi
done

RAW_RESULT_DIR="$OUTPUT_DIR/dsv32-pd-transfer-profile/raw/$BENCH_CAMPAIGN"
PROFILE_RESULT_DIR="$OUTPUT_DIR/dsv32-pd-transfer-profile/profiler/$BENCH_CAMPAIGN"
SUMMARY_PATH="$OUTPUT_DIR/dsv32-pd-transfer-profile/summary-$BENCH_CAMPAIGN.json"

validate_mode() {
    local mode=$1
    if [[ "$mode" != "npu_staging" && "$mode" != "memfabric_bm" ]]; then
        echo "Mode must be npu_staging or memfabric_bm, got: $mode" >&2
        exit 1
    fi
    if [[ "$mode" != "$SPARSE_KV_TRANSFER_MODE" ]]; then
        echo "Requested mode $mode does not match runtime SPARSE_KV_TRANSFER_MODE=$SPARSE_KV_TRANSFER_MODE." >&2
        echo "Stop all three services, change the runtime config, then restart Decode, Prefill, and proxy." >&2
        exit 1
    fi
}

require_headline_mode() {
    if [[ "$ENABLE_TORCH_PROFILER" != "false" ]]; then
        echo "Headline benchmarks require ENABLE_TORCH_PROFILER=false." >&2
        exit 1
    fi
}

prepare_environment() {
    # shellcheck disable=SC1090
    source "$CANN_ENV"
    # shellcheck disable=SC1090
    source "$CUSTOM_OPP_ENV"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    export NO_PROXY="localhost,127.0.0.1,::1,$HOST_IP"
    export no_proxy="$NO_PROXY"
    mkdir -p "$RAW_RESULT_DIR" "$PROFILE_RESULT_DIR"
}

ready() {
    curl --noproxy '*' --max-time 30 --fail-with-body --show-error \
        "http://127.0.0.1:$PREFILL_API_PORT/v1/models" >/dev/null
    curl --noproxy '*' --max-time 30 --fail-with-body --show-error \
        "http://127.0.0.1:$DECODE_API_PORT/v1/models" >/dev/null
    curl --noproxy '*' --max-time 30 --fail-with-body --show-error \
        "http://$HOST_IP:$PROXY_PORT/healthcheck"
    echo
}

preflight() {
    if [[ "$MEMFABRIC_BM_PROTOCOL" != "host_tcp" ]]; then
        echo "This same-host profiling gate requires MEMFABRIC_BM_PROTOCOL=host_tcp." >&2
        exit 1
    fi
    "$MOONCAKE_PYTHON" "$SCRIPT_DIR/profile_tools.py" preflight \
        --prefill-devices "$PREFILL_DEVICES" \
        --decode-devices "$DECODE_DEVICES" \
        --tp-size "$TP_SIZE" \
        --input-lengths "$BENCH_INPUT_LENGTHS" \
        --output-len "$BENCH_OUTPUT_LEN" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-concurrency "$BENCH_MAX_CONCURRENCY"
    echo "Repository revision: $(git -C "$REPO_DIR" rev-parse HEAD)"
    echo "Runtime transfer mode: $SPARSE_KV_TRANSFER_MODE"
    echo "Confirm npu-smi has no foreign processes before starting either suite."
}

run_bench() {
    local mode=$1
    local input_len=$2
    local run_id=$3
    local result_dir=$4
    local num_prompts=$5
    local num_warmups=$6
    validate_mode "$mode"
    ready >/dev/null

    "$MOONCAKE_PYTHON" -c '
import sys
input_len, output_len, max_len = map(int, sys.argv[1:])
if input_len <= 0 or input_len + output_len > max_len:
    raise SystemExit(f"invalid lengths: input={input_len}, output={output_len}, max={max_len}")
' "$input_len" "$BENCH_OUTPUT_LEN" "$MAX_MODEL_LEN"

    local revision model_config_sha256 result_file bench_log
    local vllm_version vllm_ascend_version torch_npu_version
    revision=$(git -C "$REPO_DIR" rev-parse HEAD)
    model_config_sha256=$("$MOONCAKE_PYTHON" -c '
import hashlib
import pathlib
import sys
print(hashlib.sha256((pathlib.Path(sys.argv[1]) / "config.json").read_bytes()).hexdigest())
' "$MODEL_PATH")
    read -r vllm_version vllm_ascend_version torch_npu_version < <(
        "$MOONCAKE_PYTHON" -c '
from importlib.metadata import version
print(version("vllm"), version("vllm-ascend"), version("torch-npu"))
'
    )

    mkdir -p "$result_dir"
    result_file="$mode-isl${input_len}-osl${BENCH_OUTPUT_LEN}-run${run_id}.json"
    bench_log="$result_dir/${result_file%.json}.log"
    cd "$REPO_DIR"
    "$MOONCAKE_PYTHON" -m vllm.entrypoints.cli.main bench serve \
        --backend openai \
        --base-url "http://$HOST_IP:$PROXY_PORT" \
        --endpoint /v1/completions \
        --model "$SERVED_MODEL_NAME" \
        --tokenizer "$MODEL_PATH" \
        --tokenizer-mode deepseek_v32 \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$BENCH_OUTPUT_LEN" \
        --random-range-ratio 0 \
        --num-prompts "$num_prompts" \
        --num-warmups "$num_warmups" \
        --request-rate "$BENCH_REQUEST_RATE" \
        --max-concurrency "$BENCH_MAX_CONCURRENCY" \
        --ignore-eos \
        --temperature 0 \
        --seed "$BENCH_SEED" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --save-result \
        --save-detailed \
        --result-dir "$result_dir" \
        --result-filename "$result_file" \
        --metadata \
            "campaign=$BENCH_CAMPAIGN" \
            "transfer_mode=$mode" \
            "revision=$revision" \
            "vllm_version=$vllm_version" \
            "vllm_ascend_version=$vllm_ascend_version" \
            "torch_npu_version=$torch_npu_version" \
            "model_path=$MODEL_PATH" \
            "model_config_sha256=$model_config_sha256" \
            "hf_overrides=$HF_OVERRIDES" \
            "prefill_devices=$PREFILL_DEVICES" \
            "decode_devices=$DECODE_DEVICES" \
            "input_len=$input_len" \
            "output_len=$BENCH_OUTPUT_LEN" \
            "num_prompts=$num_prompts" \
            "num_warmups=$num_warmups" \
            "max_model_len=$MAX_MODEL_LEN" \
            "max_num_seqs=$MAX_NUM_SEQS" \
            "tp_size=$TP_SIZE" \
            "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" \
            "block_size=$BLOCK_SIZE" \
            "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
            "engine_seed=$SEED" \
            "bench_seed=$BENCH_SEED" \
            "request_rate=$BENCH_REQUEST_RATE" \
            "max_concurrency=$BENCH_MAX_CONCURRENCY" \
            "enable_prefill_mc2=$ENABLE_PREFILL_MC2" \
            "enable_mlapo=$ENABLE_MLAPO" \
            "enable_flashcomm1=$ENABLE_FLASHCOMM1" \
            "sparse_kv_offload_mode=$SPARSE_KV_OFFLOAD_MODE" \
            "memfabric_bm_protocol=$MEMFABRIC_BM_PROTOCOL" \
        2>&1 | tee "$bench_log"
    "$MOONCAKE_PYTHON" "$SCRIPT_DIR/profile_tools.py" validate-result \
        --result "$result_dir/$result_file"
    ready >/dev/null
}

capture_snapshot() {
    local mode=$1
    local stage=$2
    local snapshot_dir="$OUTPUT_DIR/dsv32-pd-transfer-profile/snapshots/$BENCH_CAMPAIGN"
    mkdir -p "$snapshot_dir"
    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "mode=$mode"
        echo "stage=$stage"
        echo "revision=$(git -C "$REPO_DIR" rev-parse HEAD)"
        echo "prefill_devices=$PREFILL_DEVICES"
        echo "decode_devices=$DECODE_DEVICES"
        echo "===== npu-smi info ====="
        npu-smi info
        echo "===== free -b ====="
        free -b
    } > "$snapshot_dir/$mode-$stage.txt"
}

bench_suite() {
    local mode=$1
    validate_mode "$mode"
    require_headline_mode
    capture_snapshot "$mode" before
    local input_len run_id
    IFS=',' read -r -a input_lengths <<< "$BENCH_INPUT_LENGTHS"
    for input_len in "${input_lengths[@]}"; do
        input_len=${input_len//[[:space:]]/}
        for ((run_id = 1; run_id <= BENCH_REPETITIONS; run_id++)); do
            echo "===== mode=$mode input_len=$input_len run=$run_id/$BENCH_REPETITIONS ====="
            run_bench "$mode" "$input_len" "$run_id" "$RAW_RESULT_DIR" \
                "$BENCH_NUM_PROMPTS" "$BENCH_NUM_WARMUPS"
        done
    done
    capture_snapshot "$mode" after
}

profile_control() {
    local action=$1
    local port
    for port in "$PREFILL_API_PORT" "$DECODE_API_PORT"; do
        curl --noproxy '*' --max-time 60 --fail-with-body --show-error \
            -X POST "http://127.0.0.1:$port/${action}_profile"
    done
}

profile_one() {
    local input_len=$1
    validate_mode memfabric_bm
    if [[ "$ENABLE_TORCH_PROFILER" != "true" ]]; then
        echo "profile-one requires ENABLE_TORCH_PROFILER=true before service startup." >&2
        exit 1
    fi
    trap 'profile_control stop >/dev/null 2>&1 || true' EXIT
    profile_control start
    local status=0
    run_bench memfabric_bm "$input_len" profile "$PROFILE_RESULT_DIR" 1 0 || status=$?
    profile_control stop || status=$?
    trap - EXIT
    return "$status"
}

prepare_environment
case "$ACTION" in
    preflight) preflight ;;
    ready) ready ;;
    bench)
        require_headline_mode
        run_bench "${2:-}" "${3:-}" "${4:-}" "$RAW_RESULT_DIR" "$BENCH_NUM_PROMPTS" "$BENCH_NUM_WARMUPS"
        ;;
    bench-suite) bench_suite "${2:-}" ;;
    profile-one) profile_one "${2:-}" ;;
    summarize)
        "$MOONCAKE_PYTHON" "$SCRIPT_DIR/profile_tools.py" summarize \
            --result-dir "$RAW_RESULT_DIR" \
            --output "$SUMMARY_PATH"
        ;;
    *)
        echo "Unknown command: $ACTION" >&2
        show_help >&2
        exit 1
        ;;
esac

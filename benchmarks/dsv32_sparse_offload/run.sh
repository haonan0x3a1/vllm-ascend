#!/usr/bin/env bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_PATH=${DSV32_PROFILE_CONFIG:-"$SCRIPT_DIR/config.env"}
ACTION=${1:-help}

show_help() {
    cat <<'EOF'
Usage: bash run.sh <command> [arguments]

Commands:
  preflight                         Validate paths, imports, configuration, and port.
  serve <baseline|host>             Start one real-weight service in the foreground.
  ready <baseline|host>             Check service readiness and model identity.
  bench <mode> <input_len> <run_id> Run one vllm bench serve measurement.
  bench-suite <baseline|host>       Run all configured lengths and repetitions.
  summarize                         Pair baseline/Host JSON files and report deltas.
  help                              Show this help.

Set DSV32_PROFILE_CONFIG to use a config outside this directory. Otherwise the
script loads ./config.env, which is intentionally ignored by Git.
EOF
}

if [[ "$ACTION" == "help" || "$ACTION" == "-h" || "$ACTION" == "--help" ]]; then
    show_help
    exit 0
fi

if [[ ! -r "$CONFIG_PATH" ]]; then
    echo "Missing config: $CONFIG_PATH" >&2
    echo "Create it with: cp $SCRIPT_DIR/config.example.env $SCRIPT_DIR/config.env" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

required_variables=(
    WORKSPACE_DIR REPO_DIR MODEL_PATH LOG_DIR OUTPUT_DIR CANN_ENV CUSTOM_OPP_ENV
    RUNTIME_PYTHON API_HOST API_PORT DEVICES TP_SIZE SERVED_MODEL_PREFIX
    HF_OVERRIDES MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS BLOCK_SIZE
    GPU_MEMORY_UTILIZATION ENGINE_SEED HCCL_OP_EXPANSION_MODE HCCL_BUFFSIZE
    OMP_NUM_THREADS ENABLE_PREFILL_MC2 ENABLE_MLAPO ENABLE_FLASHCOMM1
    BENCH_INPUT_LENGTHS BENCH_OUTPUT_LEN BENCH_NUM_PROMPTS BENCH_NUM_WARMUPS
    BENCH_REPETITIONS BENCH_CAMPAIGN BENCH_SEED BENCH_REQUEST_RATE
    BENCH_MAX_CONCURRENCY
)
for variable in "${required_variables[@]}"; do
    if [[ -z "${!variable:-}" ]]; then
        echo "Required config variable is missing: $variable" >&2
        exit 1
    fi
done

if [[ ! "$BENCH_CAMPAIGN" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "BENCH_CAMPAIGN may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 1
fi

RAW_RESULT_DIR="$OUTPUT_DIR/dsv32-sparse-offload-profile/raw/$BENCH_CAMPAIGN"
SUMMARY_PATH="$OUTPUT_DIR/dsv32-sparse-offload-profile/summary-$BENCH_CAMPAIGN.json"

prepare_environment() {
    if [[ ! -r "$CANN_ENV" ]]; then
        echo "CANN environment script is missing: $CANN_ENV" >&2
        exit 1
    fi
    if [[ ! -r "$CUSTOM_OPP_ENV" ]]; then
        echo "Custom OPP environment script is missing: $CUSTOM_OPP_ENV" >&2
        exit 1
    fi
    if [[ ! -x "$RUNTIME_PYTHON" ]]; then
        echo "Runtime Python is missing or not executable: $RUNTIME_PYTHON" >&2
        exit 1
    fi

    # The custom OPP must be loaded after the base CANN environment.
    # shellcheck disable=SC1090
    source "$CANN_ENV"
    # shellcheck disable=SC1090
    source "$CUSTOM_OPP_ENV"

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    unset VLLM_USE_V1
    unset VLLM_ASCEND_ENABLE_MLAPO
    unset VLLM_ASCEND_ENABLE_FLASHCOMM1

    export NO_PROXY="localhost,127.0.0.1,::1,$API_HOST"
    export no_proxy="$NO_PROXY"
    export HCCL_OP_EXPANSION_MODE
    export HCCL_BUFFSIZE
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    mkdir -p "$LOG_DIR" "$RAW_RESULT_DIR"
}

validate_mode() {
    case "${1:-}" in
        baseline|host) ;;
        *) echo "Mode must be one of: baseline, host" >&2; exit 1 ;;
    esac
}

served_model_name() {
    echo "$SERVED_MODEL_PREFIX-$1"
}

build_additional_config() {
    local mode=$1
    "$RUNTIME_PYTHON" -c '
import json
import sys

prefill_mc2, mlapo, flashcomm1, mode = sys.argv[1:]
as_bool = {"true": True, "false": False}
print(json.dumps({
    "enable_prefill_mc2": as_bool[prefill_mc2],
    "enable_mlapo": as_bool[mlapo],
    "enable_flashcomm1": as_bool[flashcomm1],
    "sparse_kv_offload": {
        "enabled": mode == "host",
        "mode": "host" if mode == "host" else "mirror",
    },
}, separators=(",", ":")))
' "$ENABLE_PREFILL_MC2" "$ENABLE_MLAPO" "$ENABLE_FLASHCOMM1" "$mode"
}

preflight() {
    local inherited_visibility=${ASCEND_RT_VISIBLE_DEVICES:-}
    unset ASCEND_RT_VISIBLE_DEVICES

    "$RUNTIME_PYTHON" - "$DEVICES" "$REPO_DIR" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

import torch
import torch_npu
import custom_ops
import vllm_ascend
import vllm_ascend.vllm_ascend_C

devices = [int(item) for item in sys.argv[1].split(",")]
repo_dir = Path(sys.argv[2]).resolve()
source = Path(vllm_ascend.__file__).resolve()
device_count = torch.npu.device_count()

print("Runtime package versions:")
for package in ("vllm", "vllm-ascend", "torch", "torch-npu"):
    print(f"  {package}: {version(package)}")
print("vllm-ascend source:", source)
print("NPU available:", torch.npu.is_available())
print("Physical NPU count:", device_count)
print(
    "Gather operator registered:",
    hasattr(torch_npu, "npu_gather_selection_kv_cache"),
)
if repo_dir not in source.parents:
    raise SystemExit(f"vllm-ascend is not loaded from {repo_dir}: {source}")
if max(devices) >= device_count:
    raise SystemExit(
        f"Configured physical device {max(devices)} is outside device count {device_count}"
    )
PY

    if [[ -n "$inherited_visibility" ]]; then
        export ASCEND_RT_VISIBLE_DEVICES=$inherited_visibility
        echo "Ignored inherited ASCEND_RT_VISIBLE_DEVICES='$inherited_visibility' for physical topology probe."
    fi

    [[ -d "$MODEL_PATH" ]] || { echo "Model path is missing: $MODEL_PATH" >&2; exit 1; }
    [[ -d "$REPO_DIR/.git" ]] || { echo "Repository path is invalid: $REPO_DIR" >&2; exit 1; }

    "$RUNTIME_PYTHON" "$SCRIPT_DIR/profile_tools.py" preflight \
        --devices "$DEVICES" \
        --tp-size "$TP_SIZE" \
        --input-lengths "$BENCH_INPUT_LENGTHS" \
        --output-len "$BENCH_OUTPUT_LEN" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-concurrency "$BENCH_MAX_CONCURRENCY" \
        --host "$API_HOST" \
        --port "$API_PORT"

    echo "Repository revision: $(git -C "$REPO_DIR" rev-parse HEAD)"
    echo "Confirm separately that npu-smi shows no foreign processes on: $DEVICES"
    echo "Preflight PASSED"
}

serve() {
    local mode=$1
    validate_mode "$mode"
    export ASCEND_RT_VISIBLE_DEVICES=$DEVICES
    local model_name additional_config log_file
    model_name=$(served_model_name "$mode")
    additional_config=$(build_additional_config "$mode")
    log_file="$LOG_DIR/dsv32-sparse-offload-profile-$BENCH_CAMPAIGN-$mode.log"

    cd "$WORKSPACE_DIR"
    "$RUNTIME_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --served-model-name "$model_name" \
        --tensor-parallel-size "$TP_SIZE" \
        --enable-expert-parallel \
        --hf-overrides "$HF_OVERRIDES" \
        --seed "$ENGINE_SEED" \
        --dtype bfloat16 \
        --kv-cache-dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --block-size "$BLOCK_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --no-enable-prefix-caching \
        --enforce-eager \
        --additional-config "$additional_config" \
        2>&1 | tee "$log_file"
}

ready() {
    local mode=$1
    validate_mode "$mode"
    local expected response
    expected=$(served_model_name "$mode")
    response=$(curl --noproxy '*' --max-time 30 --fail-with-body --show-error \
        "http://$API_HOST:$API_PORT/v1/models")
    "$RUNTIME_PYTHON" -c '
import json
import sys

expected = sys.argv[1]
payload = json.loads(sys.argv[2])
models = [item["id"] for item in payload.get("data", [])]
if expected not in models:
    raise SystemExit(f"Expected model {expected!r}, got {models!r}")
print(f"Service ready: {expected}")
' "$expected" "$response"
}

bench() {
    local mode=$1
    local input_len=$2
    local run_id=$3
    validate_mode "$mode"
    ready "$mode"

    "$RUNTIME_PYTHON" -c '
import sys
input_len, output_len, max_len = map(int, sys.argv[1:])
if input_len <= 0 or input_len + output_len > max_len:
    raise SystemExit(
        f"invalid lengths: input={input_len}, output={output_len}, max={max_len}"
    )
' "$input_len" "$BENCH_OUTPUT_LEN" "$MAX_MODEL_LEN"

    local revision model_name result_file bench_log model_config_sha256
    local vllm_version vllm_ascend_version torch_npu_version
    revision=$(git -C "$REPO_DIR" rev-parse HEAD)
    model_config_sha256=$(sha256sum "$MODEL_PATH/config.json" | awk '{print $1}')
    read -r vllm_version vllm_ascend_version torch_npu_version < <(
        "$RUNTIME_PYTHON" -c '
from importlib.metadata import version
print(version("vllm"), version("vllm-ascend"), version("torch-npu"))
'
    )
    model_name=$(served_model_name "$mode")
    result_file="$mode-isl${input_len}-osl${BENCH_OUTPUT_LEN}-run${run_id}.json"
    bench_log="$LOG_DIR/dsv32-sparse-offload-bench-$BENCH_CAMPAIGN-$mode-isl${input_len}-run${run_id}.log"

    cd "$REPO_DIR"
    "$RUNTIME_PYTHON" -m vllm.entrypoints.cli.main bench serve \
        --backend openai \
        --base-url "http://$API_HOST:$API_PORT" \
        --endpoint /v1/completions \
        --model "$model_name" \
        --tokenizer "$MODEL_PATH" \
        --tokenizer-mode deepseek_v32 \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$BENCH_OUTPUT_LEN" \
        --random-range-ratio 0 \
        --num-prompts "$BENCH_NUM_PROMPTS" \
        --num-warmups "$BENCH_NUM_WARMUPS" \
        --request-rate "$BENCH_REQUEST_RATE" \
        --max-concurrency "$BENCH_MAX_CONCURRENCY" \
        --ignore-eos \
        --temperature 0 \
        --seed "$BENCH_SEED" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --save-result \
        --save-detailed \
        --result-dir "$RAW_RESULT_DIR" \
        --result-filename "$result_file" \
        --metadata \
            "campaign=$BENCH_CAMPAIGN" \
            "offload_mode=$mode" \
            "revision=$revision" \
            "vllm_version=$vllm_version" \
            "vllm_ascend_version=$vllm_ascend_version" \
            "torch_npu_version=$torch_npu_version" \
            "model_path=$MODEL_PATH" \
            "model_config_sha256=$model_config_sha256" \
            "devices=$DEVICES" \
            "input_len=$input_len" \
            "output_len=$BENCH_OUTPUT_LEN" \
            "max_model_len=$MAX_MODEL_LEN" \
            "tp_size=$TP_SIZE" \
            "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" \
            "block_size=$BLOCK_SIZE" \
            "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
            "engine_seed=$ENGINE_SEED" \
            "bench_seed=$BENCH_SEED" \
            "run_id=$run_id" \
            "enable_prefill_mc2=$ENABLE_PREFILL_MC2" \
            "enable_mlapo=$ENABLE_MLAPO" \
            "enable_flashcomm1=$ENABLE_FLASHCOMM1" \
        2>&1 | tee "$bench_log"
}

capture_snapshot() {
    local mode=$1
    local stage=$2
    local snapshot_dir="$OUTPUT_DIR/dsv32-sparse-offload-profile/snapshots/$BENCH_CAMPAIGN"
    mkdir -p "$snapshot_dir"
    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "mode=$mode"
        echo "stage=$stage"
        echo "revision=$(git -C "$REPO_DIR" rev-parse HEAD)"
        echo "devices=$DEVICES"
        echo
        echo "===== npu-smi info ====="
        npu-smi info
        echo
        echo "===== free -b ====="
        free -b
    } > "$snapshot_dir/$mode-$stage.txt"
}

bench_suite() {
    local mode=$1
    validate_mode "$mode"
    local input_len run_id
    capture_snapshot "$mode" before
    IFS=',' read -r -a input_lengths <<< "$BENCH_INPUT_LENGTHS"
    for input_len in "${input_lengths[@]}"; do
        input_len=${input_len//[[:space:]]/}
        for ((run_id = 1; run_id <= BENCH_REPETITIONS; run_id++)); do
            echo "===== mode=$mode input_len=$input_len run=$run_id/$BENCH_REPETITIONS ====="
            bench "$mode" "$input_len" "$run_id"
        done
    done
    capture_snapshot "$mode" after
}

summarize() {
    "$RUNTIME_PYTHON" "$SCRIPT_DIR/profile_tools.py" summarize \
        --result-dir "$RAW_RESULT_DIR" \
        --output "$SUMMARY_PATH"
}

prepare_environment

case "$ACTION" in
    preflight) preflight ;;
    serve) serve "${2:-}" ;;
    ready) ready "${2:-}" ;;
    bench) bench "${2:-}" "${3:-}" "${4:-}" ;;
    bench-suite) bench_suite "${2:-}" ;;
    summarize) summarize ;;
    *)
        echo "Unknown command: $ACTION" >&2
        show_help >&2
        exit 1
        ;;
esac

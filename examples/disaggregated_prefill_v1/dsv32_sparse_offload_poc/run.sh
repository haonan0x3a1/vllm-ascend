#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_PATH=${DSV32_POC_CONFIG:-"$SCRIPT_DIR/config.env"}
ACTION=${1:-help}

show_help() {
    cat <<'EOF'
Usage: bash run.sh <command> [role]

Commands:
  preflight             Validate paths, runtime APIs, device mapping, and ports.
  probe-host-transfer   Validate Mooncake TCP Host-to-Host transport.
  probe-hybrid-transfer Validate coexisting Ascend and TCP Mooncake engines.
  decode                Start the Decode service in the foreground.
  prefill               Start the Prefill service in the foreground.
  proxy                 Start the P/D proxy in the foreground.
  ready <role>          Check decode, prefill, or proxy readiness.
  validate              Run the final below/above-Top-K request suite.
  collect               Archive logs, results, revisions, and checksums.
  help                  Show this help.

Set DSV32_POC_CONFIG to use a config outside this directory. Otherwise the
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
    MOONCAKE_PYTHON HOST_IP PROXY_PORT PREFILL_API_PORT DECODE_API_PORT
    PREFILL_KV_PORT_BASE DECODE_KV_PORT_BASE PREFILL_DEVICES DECODE_DEVICES
    TP_SIZE SERVED_MODEL_NAME HF_OVERRIDES MAX_MODEL_LEN INDEX_TOPK MAX_NUM_SEQS
    MAX_NUM_BATCHED_TOKENS BLOCK_SIZE GPU_MEMORY_UTILIZATION SEED
    HCCL_OP_EXPANSION_MODE HCCL_BUFFSIZE OMP_NUM_THREADS ASCEND_TRANSFER_TIMEOUT
    VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT ENABLE_PREFILL_MC2 ENABLE_MLAPO
    ENABLE_FLASHCOMM1 SPARSE_KV_OFFLOAD_MODE
)
for variable in "${required_variables[@]}"; do
    if [[ -z "${!variable:-}" ]]; then
        echo "Required config variable is missing: $variable" >&2
        exit 1
    fi
done

PREFILL_LOG="$LOG_DIR/dsv32-pd-real61-4k-prefill-tp8.log"
DECODE_LOG="$LOG_DIR/dsv32-pd-real61-4k-decode-tp8.log"
PROXY_LOG="$LOG_DIR/dsv32-pd-real61-4k-proxy.log"
HOST_TRANSFER_LOG="$LOG_DIR/dsv32-mooncake-host-transfer-probe.log"
HYBRID_TRANSFER_LOG="$LOG_DIR/dsv32-mooncake-hybrid-transfer-probe.log"
VALIDATION_OUTPUT="$OUTPUT_DIR/dsv32-pd-real61-4k-final-suite.json"
PROXY_SCRIPT="$REPO_DIR/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py"
HOST_TRANSFER_TEST="tests/ut/distributed/kv_transfer/a3_2/"
HOST_TRANSFER_TEST+="test_mooncake_transfer_engine_npu.py::test_mooncake_host_to_host_tcp_transfer"
HYBRID_TRANSFER_TEST="tests/ut/distributed/kv_transfer/a3_2/"
HYBRID_TRANSFER_TEST+="test_mooncake_transfer_engine_npu.py::test_mooncake_ascend_and_tcp_engines_coexist"

prepare_environment() {
    if [[ ! -r "$CANN_ENV" ]]; then
        echo "CANN environment script is missing: $CANN_ENV" >&2
        exit 1
    fi
    if [[ ! -r "$CUSTOM_OPP_ENV" ]]; then
        echo "Custom OPP environment script is missing: $CUSTOM_OPP_ENV" >&2
        exit 1
    fi
    if [[ ! -x "$MOONCAKE_PYTHON" ]]; then
        echo "Mooncake Python is missing or not executable: $MOONCAKE_PYTHON" >&2
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

    export NO_PROXY="localhost,127.0.0.1,::1,$HOST_IP"
    export no_proxy="$NO_PROXY"
    export VLLM_HOST_IP="$HOST_IP"
    export HCCL_OP_EXPANSION_MODE
    export HCCL_BUFFSIZE
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ASCEND_TRANSFER_TIMEOUT
    export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
    export ASCEND_TRANSPORT_PRINT=1

    mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
}

build_kv_transfer_config() {
    local role=$1
    local port_base=$2
    local engine_id=$3
    "$MOONCAKE_PYTHON" -c '
import json
import sys

role, port_base, engine_id, tp_size = sys.argv[1:]
print(json.dumps({
    "kv_connector": "MooncakeLayerwiseConnector",
    "kv_role": role,
    "kv_port": port_base,
    "engine_id": engine_id,
    "kv_connector_extra_config": {
        "prefill": {"dp_size": 1, "tp_size": int(tp_size)},
        "decode": {"dp_size": 1, "tp_size": int(tp_size)},
    },
}, separators=(",", ":")))
' "$role" "$port_base" "$engine_id" "$TP_SIZE"
}

build_additional_config() {
    "$MOONCAKE_PYTHON" -c '
import json
import sys

prefill_mc2, mlapo, flashcomm1, mode = sys.argv[1:]
as_bool = {"true": True, "false": False}
print(json.dumps({
    "enable_prefill_mc2": as_bool[prefill_mc2],
    "enable_mlapo": as_bool[mlapo],
    "enable_flashcomm1": as_bool[flashcomm1],
    "sparse_kv_offload": {"enabled": True, "mode": mode},
}, separators=(",", ":")))
' "$ENABLE_PREFILL_MC2" "$ENABLE_MLAPO" "$ENABLE_FLASHCOMM1" "$SPARSE_KV_OFFLOAD_MODE"
}

serve_role() {
    local role=$1
    local devices api_port kv_port engine_id log_file kv_role
    if [[ "$role" == "decode" ]]; then
        devices=$DECODE_DEVICES
        api_port=$DECODE_API_PORT
        kv_port=$DECODE_KV_PORT_BASE
        engine_id=dsv32-d-real61-4k
        log_file=$DECODE_LOG
        kv_role=kv_consumer
    else
        devices=$PREFILL_DEVICES
        api_port=$PREFILL_API_PORT
        kv_port=$PREFILL_KV_PORT_BASE
        engine_id=dsv32-p-real61-4k
        log_file=$PREFILL_LOG
        kv_role=kv_producer
    fi

    export ASCEND_RT_VISIBLE_DEVICES=$devices
    local additional_config
    additional_config=$(build_additional_config)
    local kv_transfer_config
    kv_transfer_config=$(build_kv_transfer_config "$kv_role" "$kv_port" "$engine_id")

    cd "$WORKSPACE_DIR"
    "$MOONCAKE_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$api_port" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --tensor-parallel-size "$TP_SIZE" \
        --enable-expert-parallel \
        --hf-overrides "$HF_OVERRIDES" \
        --seed "$SEED" \
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
        --kv-transfer-config "$kv_transfer_config" \
        2>&1 | tee "$log_file"
}

ready() {
    local role=${1:-}
    local url
    case "$role" in
        decode) url="http://127.0.0.1:$DECODE_API_PORT/v1/models" ;;
        prefill) url="http://127.0.0.1:$PREFILL_API_PORT/v1/models" ;;
        proxy) url="http://$HOST_IP:$PROXY_PORT/healthcheck" ;;
        *) echo "ready requires one of: decode, prefill, proxy" >&2; exit 1 ;;
    esac
    curl --noproxy '*' --max-time 30 --fail-with-body --show-error "$url"
    echo
}

prepare_environment

case "$ACTION" in
    preflight)
        "$MOONCAKE_PYTHON" "$SCRIPT_DIR/poc_tools.py" preflight \
            --model-path "$MODEL_PATH" \
            --repo-dir "$REPO_DIR" \
            --proxy-script "$PROXY_SCRIPT" \
            --host-ip "$HOST_IP" \
            --prefill-devices "$PREFILL_DEVICES" \
            --decode-devices "$DECODE_DEVICES" \
            --tp-size "$TP_SIZE" \
            --max-model-len "$MAX_MODEL_LEN" \
            --index-topk "$INDEX_TOPK" \
            --proxy-port "$PROXY_PORT" \
            --prefill-api-port "$PREFILL_API_PORT" \
            --decode-api-port "$DECODE_API_PORT" \
            --prefill-kv-port-base "$PREFILL_KV_PORT_BASE" \
            --decode-kv-port-base "$DECODE_KV_PORT_BASE"
        ;;
    probe-host-transfer)
        (
            unset ASCEND_RT_VISIBLE_DEVICES
            cd "$REPO_DIR"
            "$MOONCAKE_PYTHON" -m pytest -sv "$HOST_TRANSFER_TEST" \
                2>&1 | tee "$HOST_TRANSFER_LOG"
            if ! grep -Eq '(^|[^0-9])2 passed([^0-9]|$)' "$HOST_TRANSFER_LOG"; then
                echo "Host transfer probe did not complete both cases: $HOST_TRANSFER_LOG" >&2
                exit 1
            fi
            echo "Host transfer probe evidence: $HOST_TRANSFER_LOG"
        )
        ;;
    probe-hybrid-transfer)
        (
            if [[ ! "${ASCEND_RT_VISIBLE_DEVICES:-}" =~ ^[0-9]+,[0-9]+$ ]]; then
                echo "Hybrid transfer probe requires exactly two explicitly selected free NPUs." >&2
                echo "Example: ASCEND_RT_VISIBLE_DEVICES=8,9 bash run.sh probe-hybrid-transfer" >&2
                exit 1
            fi
            echo "Hybrid transfer probe physical NPUs: $ASCEND_RT_VISIBLE_DEVICES"
            unset MC_FORCE_TCP
            cd "$REPO_DIR"
            "$MOONCAKE_PYTHON" -m pytest -sv "$HYBRID_TRANSFER_TEST" \
                2>&1 | tee "$HYBRID_TRANSFER_LOG"
            if ! grep -Eq '(^|[^0-9])1 passed([^0-9]|$)' "$HYBRID_TRANSFER_LOG"; then
                echo "Hybrid transfer probe did not pass: $HYBRID_TRANSFER_LOG" >&2
                exit 1
            fi
            echo "Hybrid transfer probe evidence: $HYBRID_TRANSFER_LOG"
        )
        ;;
    decode|prefill)
        serve_role "$ACTION"
        ;;
    proxy)
        cd "$REPO_DIR"
        "$MOONCAKE_PYTHON" "$PROXY_SCRIPT" \
            --host "$HOST_IP" \
            --port "$PROXY_PORT" \
            --prefiller-hosts "$HOST_IP" \
            --prefiller-ports "$PREFILL_API_PORT" \
            --decoder-hosts "$HOST_IP" \
            --decoder-ports "$DECODE_API_PORT" \
            2>&1 | tee "$PROXY_LOG"
        ;;
    ready)
        ready "${2:-}"
        ;;
    validate)
        "$MOONCAKE_PYTHON" "$SCRIPT_DIR/poc_tools.py" validate \
            --url "http://$HOST_IP:$PROXY_PORT/v1/chat/completions" \
            --model "$SERVED_MODEL_NAME" \
            --index-topk "$INDEX_TOPK" \
            --tp-size "$TP_SIZE" \
            --output "$VALIDATION_OUTPUT" \
            --prefill-log "$PREFILL_LOG" \
            --decode-log "$DECODE_LOG" \
            --proxy-log "$PROXY_LOG"
        ;;
    collect)
        "$MOONCAKE_PYTHON" "$SCRIPT_DIR/poc_tools.py" collect \
            --output-dir "$OUTPUT_DIR" \
            --repo-dir "$REPO_DIR" \
            --model-path "$MODEL_PATH" \
            --model "$SERVED_MODEL_NAME" \
            --max-model-len "$MAX_MODEL_LEN" \
            --index-topk "$INDEX_TOPK" \
            --tp-size "$TP_SIZE" \
            --prefill-devices "$PREFILL_DEVICES" \
            --decode-devices "$DECODE_DEVICES" \
            --prefill-kv-port-base "$PREFILL_KV_PORT_BASE" \
            --decode-kv-port-base "$DECODE_KV_PORT_BASE" \
            --validation-output "$VALIDATION_OUTPUT" \
            --prefill-log "$PREFILL_LOG" \
            --decode-log "$DECODE_LOG" \
            --proxy-log "$PROXY_LOG"
        ;;
    *)
        echo "Unknown command: $ACTION" >&2
        show_help >&2
        exit 1
        ;;
esac

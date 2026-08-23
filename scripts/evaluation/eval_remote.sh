#!/usr/bin/env bash

set -euo pipefail

GN0_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$GN0_ROOT"

# Evaluate GN-Bench navigation through generic remote WebSocket policies.
# The remote policy may return either discrete VLN-CE actions or nav_delta chunks.

MODEL_NAME="remote_nav"
CONFIG_PATH="configs/gn_bench/interiorgs/test_seen.yaml"

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
RESULT_DIR="tmp/${MODEL_NAME}_${TIMESTAMP}"

STALL_RECOVERY_FLAG=""
MAX_ACTIONS_PER_QUERY=8
ACTION_FORMAT="auto"
REMOTE_SEND_STATE_FLAG=""
WAM_TRACE_MODE="none"
WAM_ROLLOUT_BASE_SEED="0"
DAGGER_FLAG=""
DAGGER_OUTPUT_ROOT=""
DAGGER_FUTURE_ACTIONS="32"
DAGGER_HISTORY_SAMPLING="uniform"
DAGGER_HISTORY_END="current"
DAGGER_STOP_RADIUS="1.5"
DAGGER_RESCUE_HORIZON="8"
DAGGER_STOP_OVERRIDE_FLAG=""
DAGGER_STUCK_RESCUE_FLAG=""

START_IDX="0"
END_IDX=""
NUM_GPUS=8
PROCS_PER_GPU=1
STALL_PATIENCE=16

SERVER_HOST="127.0.0.1"
SERVER_PORT="8000"
SERVER_PORTS=""
REMOTE_REQUEST_TIMEOUT="600"
REMOTE_CONNECT_TIMEOUT="600"
REMOTE_METADATA_TIMEOUT="2"
REMOTE_STOP_EPS="1e-3"
REMOTE_MIN_CHUNK_TRANSLATION="0"
REMOTE_TRANSLATION_FRAME="chunk_start"
REMOTE_MAX_TRANSLATION="0.0"
REMOTE_MAX_YAW="0.0"
SAVE_MEDIA="none"
REMOTE_VIDEO_FPS="5"
REMOTE_VIDEO_CODEC="h264"

print_help() {
    cat <<'EOF'
Usage: bash scripts/evaluation/eval_remote.sh [OPTIONS]

Evaluate GN-Bench episodes against one remote policy server per worker.

Evaluation:
  --exp-config PATH                    Experiment YAML (default: configs/gn_bench/interiorgs/test_seen.yaml)
  --result-dir PATH                    Evaluation results directory (default: timestamped under tmp/)
  --start-idx N                        Inclusive episode start index (default: 0)
  --end-idx N                          Exclusive episode end index (default: dataset end)
  --max-actions-per-query N            Returned actions executed before re-querying (default: 8)
  --action-format MODE                 auto, discrete, or nav_delta (default: auto)
  --stall-patience N                   Consecutive unchanged-distance steps before recovery/stop (default: 16)
  --enable-stall-recovery              Backtrack on the first two stall triggers

Worker Scheduling:
  --num-gpus N                         Number of GN-Bench GPU slots (default: 8)
  --procs-per-gpu N                    Worker processes per GPU (default: 1)

Remote Server:
  --server-host HOST                   Policy server host (default: 127.0.0.1)
  --server-port PORT                   Base policy server port (default: 8000)
  --server-ports PORTS                 Comma-separated explicit worker ports
  --remote-request-timeout SEC         Inference request timeout (default: 600)
  --remote-connect-timeout SEC         WebSocket connection timeout (default: 600)
  --remote-metadata-timeout SEC        Server metadata timeout (default: 2)
  --remote-send-state                  Include nav_pose state in requests
  --remote-stop-eps X                  Near-zero action STOP threshold (default: 1e-3)
  --remote-min-chunk-translation M     Stop below total chunk translation; 0 disables (default: 0)
  --remote-translation-frame MODE      chunk_start or current (default: chunk_start)
  --remote-max-translation M           Per-action translation cap; 0 disables (default: 0)
  --remote-max-yaw RAD                 Per-action yaw cap in radians; 0 disables (default: 0)

Media Output:
  --save-media MODE                    images, video, or none (default: none)
  --remote-video-fps FPS               Episode video frame rate (default: 5)
  --remote-video-codec CODEC           Episode video codec (default: h264)

DAgger:
  --dagger                             Enable DAgger collection
  --dagger-output-root PATH            DAgger dataset root (required with --dagger)
  --dagger-future-actions N            Future teacher actions to save (default: 32)
  --dagger-history-sampling MODE       uniform or recent (default: uniform)
  --dagger-history-end MODE            current or previous (default: current)
  --dagger-stop-radius M               Teacher STOP radius (default: 1.5)
  --dagger-rescue-horizon N            Teacher rescue horizon (default: 8)
  --dagger-stop-override               Enable model-STOP override (default)
  --no-dagger-stop-override            Disable model-STOP override
  --dagger-stuck-rescue                Enable stuck-policy teacher rescue (default)
  --no-dagger-stuck-rescue             Disable stuck-policy teacher rescue

WAM Trace:
  --wam-trace-mode MODE                none or sidecar (default: none)
  --wam-rollout-base-seed N            Non-negative rollout base seed (default: 0)

  -h, --help                           Show this help message
EOF
}

require_option_value() {
    local option="$1"
    local value="${2-}"
    if [[ -z "$value" || "$value" == --* ]]; then
        echo "Error: $option requires a value." >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) print_help; exit 0 ;;
        --enable-stall-recovery) STALL_RECOVERY_FLAG="--enable-stall-recovery"; shift ;;
        --exp-config) require_option_value "$1" "${2-}"; CONFIG_PATH="$2"; shift 2 ;;
        --max-actions-per-query) require_option_value "$1" "${2-}"; MAX_ACTIONS_PER_QUERY="$2"; shift 2 ;;
        --action-format) require_option_value "$1" "${2-}"; ACTION_FORMAT="$2"; shift 2 ;;
        --remote-send-state) REMOTE_SEND_STATE_FLAG="--remote-send-state"; shift ;;
        --wam-trace-mode) require_option_value "$1" "${2-}"; WAM_TRACE_MODE="$2"; shift 2 ;;
        --wam-rollout-base-seed) require_option_value "$1" "${2-}"; WAM_ROLLOUT_BASE_SEED="$2"; shift 2 ;;
        --dagger) DAGGER_FLAG="--dagger"; shift ;;
        --dagger-output-root) require_option_value "$1" "${2-}"; DAGGER_OUTPUT_ROOT="$2"; shift 2 ;;
        --dagger-future-actions) require_option_value "$1" "${2-}"; DAGGER_FUTURE_ACTIONS="$2"; shift 2 ;;
        --dagger-history-sampling) require_option_value "$1" "${2-}"; DAGGER_HISTORY_SAMPLING="$2"; shift 2 ;;
        --dagger-history-end) require_option_value "$1" "${2-}"; DAGGER_HISTORY_END="$2"; shift 2 ;;
        --dagger-stop-radius) require_option_value "$1" "${2-}"; DAGGER_STOP_RADIUS="$2"; shift 2 ;;
        --dagger-rescue-horizon) require_option_value "$1" "${2-}"; DAGGER_RESCUE_HORIZON="$2"; shift 2 ;;
        --dagger-stop-override) DAGGER_STOP_OVERRIDE_FLAG=""; shift ;;
        --no-dagger-stop-override) DAGGER_STOP_OVERRIDE_FLAG="--no-dagger-stop-override"; shift ;;
        --dagger-stuck-rescue) DAGGER_STUCK_RESCUE_FLAG=""; shift ;;
        --no-dagger-stuck-rescue) DAGGER_STUCK_RESCUE_FLAG="--no-dagger-stuck-rescue"; shift ;;
        --start-idx) require_option_value "$1" "${2-}"; START_IDX="$2"; shift 2 ;;
        --end-idx) require_option_value "$1" "${2-}"; END_IDX="$2"; shift 2 ;;
        --num-gpus) require_option_value "$1" "${2-}"; NUM_GPUS="$2"; shift 2 ;;
        --procs-per-gpu) require_option_value "$1" "${2-}"; PROCS_PER_GPU="$2"; shift 2 ;;
        --stall-patience) require_option_value "$1" "${2-}"; STALL_PATIENCE="$2"; shift 2 ;;
        --result-dir) require_option_value "$1" "${2-}"; RESULT_DIR="$2"; shift 2 ;;
        --server-host) require_option_value "$1" "${2-}"; SERVER_HOST="$2"; shift 2 ;;
        --server-port) require_option_value "$1" "${2-}"; SERVER_PORT="$2"; shift 2 ;;
        --server-ports) require_option_value "$1" "${2-}"; SERVER_PORTS="$2"; shift 2 ;;
        --remote-request-timeout) require_option_value "$1" "${2-}"; REMOTE_REQUEST_TIMEOUT="$2"; shift 2 ;;
        --remote-connect-timeout) require_option_value "$1" "${2-}"; REMOTE_CONNECT_TIMEOUT="$2"; shift 2 ;;
        --remote-metadata-timeout) require_option_value "$1" "${2-}"; REMOTE_METADATA_TIMEOUT="$2"; shift 2 ;;
        --remote-stop-eps) require_option_value "$1" "${2-}"; REMOTE_STOP_EPS="$2"; shift 2 ;;
        --remote-min-chunk-translation) require_option_value "$1" "${2-}"; REMOTE_MIN_CHUNK_TRANSLATION="$2"; shift 2 ;;
        --remote-translation-frame) require_option_value "$1" "${2-}"; REMOTE_TRANSLATION_FRAME="$2"; shift 2 ;;
        --remote-max-translation) require_option_value "$1" "${2-}"; REMOTE_MAX_TRANSLATION="$2"; shift 2 ;;
        --remote-max-yaw) require_option_value "$1" "${2-}"; REMOTE_MAX_YAW="$2"; shift 2 ;;
        --save-media) require_option_value "$1" "${2-}"; SAVE_MEDIA="$2"; shift 2 ;;
        --remote-video-fps) require_option_value "$1" "${2-}"; REMOTE_VIDEO_FPS="$2"; shift 2 ;;
        --remote-video-codec) require_option_value "$1" "${2-}"; REMOTE_VIDEO_CODEC="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_number() {
    [[ "$1" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]
}

is_positive_number() {
    is_number "$1" && awk -v value="$1" 'BEGIN { exit !(value > 0) }'
}

is_nonnegative_number() {
    is_number "$1" && awk -v value="$1" 'BEGIN { exit !(value >= 0) }'
}

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: experiment config not found: $CONFIG_PATH" >&2
    exit 1
fi
if [[ -n "$DAGGER_FLAG" && -z "${DAGGER_OUTPUT_ROOT// }" ]]; then
    echo "Error: --dagger-output-root is required when --dagger is set." >&2
    exit 1
fi
if ! is_positive_integer "$NUM_GPUS" || ! is_positive_integer "$PROCS_PER_GPU"; then
    echo "Error: --num-gpus and --procs-per-gpu must be positive integers." >&2
    exit 1
fi
if ! is_positive_integer "$MAX_ACTIONS_PER_QUERY"; then
    echo "Error: --max-actions-per-query must be a positive integer." >&2
    exit 1
fi
if ! is_nonnegative_integer "$START_IDX"; then
    echo "Error: --start-idx must be a non-negative integer." >&2
    exit 1
fi
if [[ -n "$END_IDX" && "$END_IDX" != "-1" ]]; then
    if ! is_nonnegative_integer "$END_IDX" || (( END_IDX < START_IDX )); then
        echo "Error: --end-idx must be -1 or an integer greater than or equal to --start-idx." >&2
        exit 1
    fi
fi
if ! is_nonnegative_integer "$STALL_PATIENCE"; then
    echo "Error: --stall-patience must be a non-negative integer." >&2
    exit 1
fi
if ! is_positive_integer "$SERVER_PORT" || (( SERVER_PORT > 65535 )); then
    echo "Error: --server-port must be an integer in [1, 65535]." >&2
    exit 1
fi
if ! is_positive_number "$REMOTE_REQUEST_TIMEOUT" || ! is_positive_number "$REMOTE_CONNECT_TIMEOUT" || ! is_positive_number "$REMOTE_METADATA_TIMEOUT"; then
    echo "Error: remote timeouts must be positive numbers." >&2
    exit 1
fi
if ! is_nonnegative_number "$REMOTE_STOP_EPS" || ! is_nonnegative_number "$REMOTE_MIN_CHUNK_TRANSLATION" || ! is_nonnegative_number "$REMOTE_MAX_TRANSLATION" || ! is_nonnegative_number "$REMOTE_MAX_YAW"; then
    echo "Error: remote action thresholds must be non-negative numbers." >&2
    exit 1
fi
if ! is_positive_number "$REMOTE_VIDEO_FPS"; then
    echo "Error: --remote-video-fps must be a positive number." >&2
    exit 1
fi
if ! is_positive_integer "$DAGGER_FUTURE_ACTIONS" || ! is_positive_integer "$DAGGER_RESCUE_HORIZON" || ! is_nonnegative_number "$DAGGER_STOP_RADIUS"; then
    echo "Error: invalid DAgger action count, rescue horizon, or stop radius." >&2
    exit 1
fi
if [[ "$ACTION_FORMAT" != "auto" && "$ACTION_FORMAT" != "discrete" && "$ACTION_FORMAT" != "nav_delta" ]]; then
    echo "Error: --action-format must be auto, discrete, or nav_delta." >&2
    exit 1
fi
if [[ "$REMOTE_TRANSLATION_FRAME" != "chunk_start" && "$REMOTE_TRANSLATION_FRAME" != "current" ]]; then
    echo "Error: --remote-translation-frame must be chunk_start or current." >&2
    exit 1
fi
if [[ "$SAVE_MEDIA" != "images" && "$SAVE_MEDIA" != "video" && "$SAVE_MEDIA" != "none" ]]; then
    echo "Error: --save-media must be images, video, or none." >&2
    exit 1
fi
if [[ "$DAGGER_HISTORY_SAMPLING" != "uniform" && "$DAGGER_HISTORY_SAMPLING" != "recent" ]]; then
    echo "Error: --dagger-history-sampling must be uniform or recent." >&2
    exit 1
fi
if [[ "$DAGGER_HISTORY_END" != "current" && "$DAGGER_HISTORY_END" != "previous" ]]; then
    echo "Error: --dagger-history-end must be current or previous." >&2
    exit 1
fi
if [[ "$WAM_TRACE_MODE" != "none" && "$WAM_TRACE_MODE" != "sidecar" ]]; then
    echo "Error: --wam-trace-mode must be none or sidecar." >&2
    exit 1
fi
if ! is_nonnegative_integer "$WAM_ROLLOUT_BASE_SEED"; then
    echo "Error: --wam-rollout-base-seed must be a non-negative integer." >&2
    exit 1
fi

TOTAL_WORKERS=$(( NUM_GPUS * PROCS_PER_GPU ))
if [[ -z "$SERVER_PORTS" ]] && (( SERVER_PORT + TOTAL_WORKERS - 1 > 65535 )); then
    echo "Error: the generated server-port range exceeds 65535." >&2
    exit 1
fi

mkdir -p "$RESULT_DIR"

echo "Starting evaluation: $MODEL_NAME"
echo "Experiment config: $CONFIG_PATH"
echo "Results will be saved to: $RESULT_DIR"
echo "Action format: $ACTION_FORMAT"
echo "Max actions per query: $MAX_ACTIONS_PER_QUERY"
echo "Stall patience: $STALL_PATIENCE"
echo "Stall recovery: $([[ -n "$STALL_RECOVERY_FLAG" ]] && echo true || echo false)"
echo "Remote stop epsilon: $REMOTE_STOP_EPS"
echo "Remote minimum chunk translation: $REMOTE_MIN_CHUNK_TRANSLATION m"
echo "WAM trace mode: $WAM_TRACE_MODE"
if [[ "$WAM_TRACE_MODE" == "sidecar" ]]; then
    echo "WAM rollout base seed: $WAM_ROLLOUT_BASE_SEED"
fi
if [[ -n "$DAGGER_FLAG" ]]; then
    echo "DAgger: true"
    echo "DAgger output root: $DAGGER_OUTPUT_ROOT"
    echo "DAgger future actions: $DAGGER_FUTURE_ACTIONS"
    echo "DAgger history sampling: $DAGGER_HISTORY_SAMPLING"
    echo "DAgger history end: $DAGGER_HISTORY_END"
    echo "DAgger stop radius: $DAGGER_STOP_RADIUS"
    echo "DAgger rescue horizon: $DAGGER_RESCUE_HORIZON"
else
    echo "DAgger: false"
fi
echo "Media output: $SAVE_MEDIA"
if [[ "$SAVE_MEDIA" == "video" ]]; then
    echo "Video fps: $REMOTE_VIDEO_FPS"
    echo "Video codec: $REMOTE_VIDEO_CODEC"
fi

END_IDX_DESC="${END_IDX:-end}"
echo "Evaluating episodes from $START_IDX to $END_IDX_DESC"
echo "GN-Bench GPU slots: $NUM_GPUS"
echo "Processes per GPU: $PROCS_PER_GPU"
echo "Total GN-Bench workers: $TOTAL_WORKERS"

END_IDX_ARGS=()
if [[ -n "$END_IDX" ]]; then
    END_IDX_ARGS=(--end-idx "$END_IDX")
fi

SERVER_PORT_ARRAY=()
if [[ -n "$SERVER_PORTS" ]]; then
    IFS=',' read -r -a SERVER_PORT_ARRAY <<< "$SERVER_PORTS"
    if [[ "${#SERVER_PORT_ARRAY[@]}" -ne "$TOTAL_WORKERS" ]]; then
        echo "Error: --server-ports must contain exactly $TOTAL_WORKERS comma-separated ports." >&2
        exit 1
    fi
    for explicit_port in "${SERVER_PORT_ARRAY[@]}"; do
        if ! is_positive_integer "$explicit_port" || (( explicit_port > 65535 )); then
            echo "Error: every --server-ports entry must be an integer in [1, 65535]." >&2
            exit 1
        fi
    done
else
    for ((idx=0; idx<TOTAL_WORKERS; idx++)); do
        SERVER_PORT_ARRAY+=( "$(( SERVER_PORT + idx ))" )
    done
fi

echo "Remote server host: $SERVER_HOST"
echo "Remote server ports: ${SERVER_PORT_ARRAY[*]}"
echo "Note: this script starts GN-Bench workers only. Start one remote policy server per port first."

cleanup() {
    trap - SIGINT SIGTERM
    echo "Terminating..."
    if [[ ${#WORKER_PIDS[@]} -gt 0 ]]; then
        kill "${WORKER_PIDS[@]}" 2>/dev/null || true
        wait "${WORKER_PIDS[@]}" 2>/dev/null || true
    fi
    exit 130
}

trap cleanup SIGINT SIGTERM
WORKER_PIDS=()

for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
    for ((slot=0; slot<PROCS_PER_GPU; slot++)); do
        split_id=$(( gpu_id * PROCS_PER_GPU + slot ))
        worker_server_port="${SERVER_PORT_ARRAY[$split_id]}"

        echo "[Worker $split_id/$TOTAL_WORKERS] Launching on GN-Bench GPU $gpu_id (slot $slot), server port $worker_server_port..."

        CUDA_VISIBLE_DEVICES=$gpu_id python -m gn0.evaluation.remote \
            --exp-config "$CONFIG_PATH" \
            --split-num "$TOTAL_WORKERS" \
            --split-id "$split_id" \
            --result-dir "$RESULT_DIR" \
            --max-actions-per-query "$MAX_ACTIONS_PER_QUERY" \
            --action-format "$ACTION_FORMAT" \
            --start-idx "$START_IDX" \
            --stall-patience "$STALL_PATIENCE" \
            --server-host "$SERVER_HOST" \
            --server-port "$worker_server_port" \
            --remote-request-timeout "$REMOTE_REQUEST_TIMEOUT" \
            --remote-connect-timeout "$REMOTE_CONNECT_TIMEOUT" \
            --remote-metadata-timeout "$REMOTE_METADATA_TIMEOUT" \
            --remote-stop-eps "$REMOTE_STOP_EPS" \
            --remote-min-chunk-translation "$REMOTE_MIN_CHUNK_TRANSLATION" \
            --remote-translation-frame "$REMOTE_TRANSLATION_FRAME" \
            --remote-max-translation "$REMOTE_MAX_TRANSLATION" \
            --remote-max-yaw "$REMOTE_MAX_YAW" \
            --save-media "$SAVE_MEDIA" \
            "${END_IDX_ARGS[@]}" \
            --remote-video-fps "$REMOTE_VIDEO_FPS" \
            --remote-video-codec "$REMOTE_VIDEO_CODEC" \
            --wam-trace-mode "$WAM_TRACE_MODE" \
            --wam-rollout-base-seed "$WAM_ROLLOUT_BASE_SEED" \
            --dagger-output-root "$DAGGER_OUTPUT_ROOT" \
            --dagger-future-actions "$DAGGER_FUTURE_ACTIONS" \
            --dagger-history-sampling "$DAGGER_HISTORY_SAMPLING" \
            --dagger-history-end "$DAGGER_HISTORY_END" \
            --dagger-stop-radius "$DAGGER_STOP_RADIUS" \
            --dagger-rescue-horizon "$DAGGER_RESCUE_HORIZON" \
            $DAGGER_FLAG \
            $DAGGER_STOP_OVERRIDE_FLAG \
            $DAGGER_STUCK_RESCUE_FLAG \
            $REMOTE_SEND_STATE_FLAG \
            $STALL_RECOVERY_FLAG &

        WORKER_PIDS+=("$!")
    done
done

WORKER_STATUS=0
for worker_pid in "${WORKER_PIDS[@]}"; do
    if ! wait "$worker_pid"; then
        WORKER_STATUS=1
    fi
done
trap - SIGINT SIGTERM

if (( WORKER_STATUS != 0 )); then
    echo "Remote evaluation failed; inspect the worker logs above." >&2
    exit "$WORKER_STATUS"
fi

echo "All workers finished."

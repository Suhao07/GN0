#!/usr/bin/env bash

set -euo pipefail

GN0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GN0_ROOT"

MODEL_NAME="bae"
MODEL_PATH="model_zoo/GN-BAE"
CONFIG_PATH="configs/gn_bench/interiorgs/test_seen.yaml"
PROMPT_TYPE="V1"
MAX_ACTIONS_PER_QUERY=1

TIMESTAMP="$(date "+%Y%m%d_%H%M%S")"
RESULT_DIR="tmp/${MODEL_NAME}_${TIMESTAMP}"

START_IDX=0
END_IDX=""
NUM_GPUS=1
PROCS_PER_GPU=1
STALL_PATIENCE=16

DAGGER="false"
DAGGER_STOP_RADIUS="1.5"
SAVE_MEDIA="none"
VIDEO_FPS="5"
VIDEO_CODEC="h264"

print_help() {
    cat <<'EOF'
Usage: bash scripts/evaluation/eval_bae.sh [OPTIONS]

Evaluate the local BAE policy on a GN-Bench dataset split.

Model:
  --model-path PATH                   Model checkpoint or identifier (default: model_zoo/GN-BAE)
  --prompt-type MODE                  V1, V2, or V3 (default: V1)
  --max-actions-per-query N           Predicted actions executed before re-querying (default: 1)

Evaluation:
  --exp-config PATH                   Experiment YAML (default: configs/gn_bench/interiorgs/test_seen.yaml)
  --result-dir PATH                   Evaluation results directory (default: timestamped under tmp/)
  --start-idx N                       Inclusive episode start index (default: 0)
  --end-idx N                         Exclusive episode end index (default: dataset end)
  --stall-patience N                  Consecutive unchanged-distance steps before STOP (default: 16)
  --dagger                            Use the BAE DAgger agent
  --dagger-stop-radius M              DAgger oracle STOP radius in meters (default: 1.5)

Worker Scheduling:
  --num-gpus N                        Number of GN-Bench GPU slots (default: 1)
  --procs-per-gpu N                   Worker processes per GPU (default: 1)

Media Output:
  --save-media MODE                   images, video, or none (default: none)
  --video-fps FPS                     Episode video frame rate (default: 5)
  --video-codec CODEC                 Episode video codec (default: h264)

  -h, --help                          Show this help message
EOF
}

require_option_value() {
    local option="$1"
    local value="${2-}"
    if [[ -z "$value" || "$value" == --* ]]; then
        echo "Error: $option requires a value." >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            exit 0
            ;;
        --dagger)
            DAGGER="true"
            shift
            ;;
        --dagger-stop-radius)
            require_option_value "$1" "${2-}"
            DAGGER_STOP_RADIUS="$2"
            shift 2
            ;;
        --exp-config)
            require_option_value "$1" "${2-}"
            CONFIG_PATH="$2"
            shift 2
            ;;
        --model-path)
            require_option_value "$1" "${2-}"
            MODEL_PATH="$2"
            shift 2
            ;;
        --prompt-type)
            require_option_value "$1" "${2-}"
            PROMPT_TYPE="$2"
            shift 2
            ;;
        --max-actions-per-query)
            require_option_value "$1" "${2-}"
            MAX_ACTIONS_PER_QUERY="$2"
            shift 2
            ;;
        --start-idx)
            require_option_value "$1" "${2-}"
            START_IDX="$2"
            shift 2
            ;;
        --end-idx)
            require_option_value "$1" "${2-}"
            END_IDX="$2"
            shift 2
            ;;
        --num-gpus)
            require_option_value "$1" "${2-}"
            NUM_GPUS="$2"
            shift 2
            ;;
        --procs-per-gpu)
            require_option_value "$1" "${2-}"
            PROCS_PER_GPU="$2"
            shift 2
            ;;
        --stall-patience)
            require_option_value "$1" "${2-}"
            STALL_PATIENCE="$2"
            shift 2
            ;;
        --result-dir)
            require_option_value "$1" "${2-}"
            RESULT_DIR="$2"
            shift 2
            ;;
        --save-media)
            require_option_value "$1" "${2-}"
            SAVE_MEDIA="$2"
            shift 2
            ;;
        --video-fps)
            require_option_value "$1" "${2-}"
            VIDEO_FPS="$2"
            shift 2
            ;;
        --video-codec)
            require_option_value "$1" "${2-}"
            VIDEO_CODEC="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            exit 2
            ;;
    esac
done

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_number() {
    [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] &&
        awk -v value="$1" 'BEGIN { exit !(value > 0) }'
}

is_nonnegative_number() {
    [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]
}

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: experiment config not found: $CONFIG_PATH" >&2
    exit 2
fi
if [[ -z "${MODEL_PATH// }" ]]; then
    echo "Error: --model-path cannot be empty." >&2
    exit 2
fi
if [[ "$PROMPT_TYPE" != "V1" && "$PROMPT_TYPE" != "V2" && "$PROMPT_TYPE" != "V3" ]]; then
    echo "Error: --prompt-type must be V1, V2, or V3." >&2
    exit 2
fi
if ! is_positive_integer "$MAX_ACTIONS_PER_QUERY"; then
    echo "Error: --max-actions-per-query must be a positive integer." >&2
    exit 2
fi
if ! is_nonnegative_integer "$START_IDX"; then
    echo "Error: --start-idx must be a non-negative integer." >&2
    exit 2
fi
if [[ -n "$END_IDX" && "$END_IDX" != "-1" ]]; then
    if ! is_nonnegative_integer "$END_IDX" || (( END_IDX < START_IDX )); then
        echo "Error: --end-idx must be -1 or greater than or equal to --start-idx." >&2
        exit 2
    fi
fi
if ! is_positive_integer "$NUM_GPUS" || ! is_positive_integer "$PROCS_PER_GPU"; then
    echo "Error: --num-gpus and --procs-per-gpu must be positive integers." >&2
    exit 2
fi
if ! is_nonnegative_integer "$STALL_PATIENCE"; then
    echo "Error: --stall-patience must be a non-negative integer." >&2
    exit 2
fi
if ! is_nonnegative_number "$DAGGER_STOP_RADIUS"; then
    echo "Error: --dagger-stop-radius must be a non-negative number." >&2
    exit 2
fi
if [[ "$SAVE_MEDIA" != "images" && "$SAVE_MEDIA" != "video" && "$SAVE_MEDIA" != "none" ]]; then
    echo "Error: --save-media must be images, video, or none." >&2
    exit 2
fi
if ! is_positive_number "$VIDEO_FPS"; then
    echo "Error: --video-fps must be a positive number." >&2
    exit 2
fi
if [[ "$DAGGER" == "true" && "$SAVE_MEDIA" != "images" ]]; then
    echo "Error: --dagger requires --save-media images." >&2
    exit 2
fi

TOTAL_WORKERS=$((NUM_GPUS * PROCS_PER_GPU))

mkdir -p "$RESULT_DIR"

echo "Starting evaluation: $MODEL_NAME with prompt $PROMPT_TYPE"
echo "Model: $MODEL_PATH"
echo "Experiment config: $CONFIG_PATH"
echo "Results will be saved to: $RESULT_DIR"
echo "Max actions per query: $MAX_ACTIONS_PER_QUERY"
echo "Stall patience: $STALL_PATIENCE"
echo "Episode range: [$START_IDX, ${END_IDX:-end})"
echo "GN-Bench GPU slots: $NUM_GPUS"
echo "Processes per GPU: $PROCS_PER_GPU"
echo "Total workers: $TOTAL_WORKERS"
echo "DAgger: $DAGGER"
if [[ "$DAGGER" == "true" ]]; then
    echo "DAgger stop radius: ${DAGGER_STOP_RADIUS}m"
fi
echo "Media output: $SAVE_MEDIA"

END_IDX_ARGS=()
if [[ -n "$END_IDX" ]]; then
    END_IDX_ARGS=(--end-idx "$END_IDX")
fi

DAGGER_ARGS=()
if [[ "$DAGGER" == "true" ]]; then
    DAGGER_ARGS=(--dagger --dagger-stop-radius "$DAGGER_STOP_RADIUS")
fi

WORKER_PIDS=()
cleanup() {
    trap - SIGINT SIGTERM
    echo "Terminating BAE evaluation workers..."
    if [[ ${#WORKER_PIDS[@]} -gt 0 ]]; then
        kill "${WORKER_PIDS[@]}" 2>/dev/null || true
        wait "${WORKER_PIDS[@]}" 2>/dev/null || true
    fi
    exit 130
}
trap cleanup SIGINT SIGTERM

for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
    for ((slot=0; slot<PROCS_PER_GPU; slot++)); do
        split_id=$((gpu_id * PROCS_PER_GPU + slot))
        echo "[Worker $split_id/$TOTAL_WORKERS] Launching on GN-Bench GPU $gpu_id (slot $slot)..."
        CUDA_VISIBLE_DEVICES="$gpu_id" python -m gn0.evaluation.local \
            --exp-config "$CONFIG_PATH" \
            --split-num "$TOTAL_WORKERS" \
            --split-id "$split_id" \
            --model-path "$MODEL_PATH" \
            --result-dir "$RESULT_DIR" \
            --model-name "$MODEL_NAME" \
            --prompt-type "$PROMPT_TYPE" \
            --max-actions-per-query "$MAX_ACTIONS_PER_QUERY" \
            --start-idx "$START_IDX" \
            --stall-patience "$STALL_PATIENCE" \
            --save-media "$SAVE_MEDIA" \
            --video-fps "$VIDEO_FPS" \
            --video-codec "$VIDEO_CODEC" \
            "${END_IDX_ARGS[@]}" \
            "${DAGGER_ARGS[@]}" &
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
    echo "BAE evaluation failed; inspect the worker logs above." >&2
    exit "$WORKER_STATUS"
fi

echo "All BAE evaluation workers finished."

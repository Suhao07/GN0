#!/usr/bin/env bash

# Render GN-Matrix trajectory splits without launching a navigation policy.

set -euo pipefail

GN0_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG_PATH="configs/gn_bench/interiorgs/train_seen.yaml"
RESULT_DIR="tmp/rendered_interiorgs_train_seen"
START_IDX=0
END_IDX=""
NUM_GPUS=1
PROCS_PER_GPU=1

usage() {
    cat <<'EOF'
Usage: bash scripts/evaluation/render_dataset.sh [options]

Render a GN-Matrix dataset split to <result-dir>/<scene>/<trajectory_id>.mp4.

Options:
  --exp-config PATH       Experiment configuration YAML.
  --result-dir PATH       Output directory for rendered videos.
  --start-idx N           First episode index, inclusive (default: 0).
  --end-idx N             Last episode index, exclusive (default: dataset end).
  --num-gpus N            Number of GPU slots (default: 1).
  --procs-per-gpu N       Worker processes per GPU (default: 1).
  -h, --help              Show this help message.
EOF
}

require_value() {
    local option=$1
    local value=${2-}
    if [[ -z "$value" ]]; then
        echo "Error: $option requires a value." >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-config)
            require_value "$1" "${2-}"
            CONFIG_PATH=$2
            shift 2
            ;;
        --result-dir)
            require_value "$1" "${2-}"
            RESULT_DIR=$2
            shift 2
            ;;
        --start-idx)
            require_value "$1" "${2-}"
            START_IDX=$2
            shift 2
            ;;
        --end-idx)
            require_value "$1" "${2-}"
            END_IDX=$2
            shift 2
            ;;
        --num-gpus)
            require_value "$1" "${2-}"
            NUM_GPUS=$2
            shift 2
            ;;
        --procs-per-gpu)
            require_value "$1" "${2-}"
            PROCS_PER_GPU=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$START_IDX" =~ ^[0-9]+$ ]]; then
    echo "Error: --start-idx must be a non-negative integer." >&2
    exit 2
fi
if [[ -n "$END_IDX" ]] && ! [[ "$END_IDX" =~ ^[0-9]+$ ]]; then
    echo "Error: --end-idx must be a non-negative integer." >&2
    exit 2
fi
if [[ -n "$END_IDX" ]] && (( END_IDX < START_IDX )); then
    echo "Error: --end-idx must be greater than or equal to --start-idx." >&2
    exit 2
fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num-gpus must be a positive integer." >&2
    exit 2
fi
if ! [[ "$PROCS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --procs-per-gpu must be a positive integer." >&2
    exit 2
fi

cd "$GN0_ROOT"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: experiment config not found: $CONFIG_PATH" >&2
    exit 2
fi
mkdir -p "$RESULT_DIR"

TOTAL_WORKERS=$((NUM_GPUS * PROCS_PER_GPU))
END_IDX_DESC=${END_IDX:-end}
END_IDX_ARGS=()
if [[ -n "$END_IDX" ]]; then
    END_IDX_ARGS=(--end-idx "$END_IDX")
fi

echo "GN-Bench Dataset Rendering"
echo "Experiment config : $CONFIG_PATH"
echo "Output directory  : $RESULT_DIR"
echo "Episode range     : [$START_IDX, $END_IDX_DESC)"
echo "GPU slots         : $NUM_GPUS"
echo "Processes per GPU : $PROCS_PER_GPU"
echo "Total workers     : $TOTAL_WORKERS"

worker_pids=()
cleanup() {
    if [[ ${#worker_pids[@]} -gt 0 ]]; then
        kill "${worker_pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
    for ((slot=0; slot<PROCS_PER_GPU; slot++)); do
        split_id=$((gpu_id * PROCS_PER_GPU + slot))
        echo "[Renderer $split_id/$TOTAL_WORKERS] Launching on GPU $gpu_id"
        CUDA_VISIBLE_DEVICES=$gpu_id python -m gn0.evaluation.local \
            --render-dataset \
            --exp-config "$CONFIG_PATH" \
            --result-dir "$RESULT_DIR" \
            --split-num "$TOTAL_WORKERS" \
            --split-id "$split_id" \
            --start-idx "$START_IDX" \
            "${END_IDX_ARGS[@]}" &
        worker_pids+=("$!")
    done
done

status=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "$worker_pid"; then
        status=1
    fi
done
trap - INT TERM

if (( status != 0 )); then
    echo "Dataset rendering failed; inspect the worker logs above." >&2
    exit "$status"
fi

echo "Dataset rendering completed: $RESULT_DIR"

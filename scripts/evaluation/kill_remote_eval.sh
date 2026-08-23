#!/usr/bin/env bash

set -euo pipefail

DRY_RUN="false"
GRACE_PERIOD=5

print_help() {
    cat <<'EOF'
Usage: bash scripts/evaluation/kill_remote_eval.sh [OPTIONS]

Stop remote GN-Bench evaluation processes owned by the current user and
launched from this GN0 checkout. Policy servers are intentionally unaffected.

Options:
  --dry-run             List matching processes without sending signals
  --grace-period SEC    Seconds before SIGKILL (default: 5)
  -h, --help            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="true"; shift ;;
        --grace-period)
            [[ -n "${2-}" ]] || { echo "Error: --grace-period requires a value." >&2; exit 2; }
            GRACE_PERIOD="$2"
            shift 2
            ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Error: unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ "$GRACE_PERIOD" =~ ^[0-9]+$ ]] || { echo "Error: --grace-period must be non-negative." >&2; exit 2; }

GN0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CURRENT_UID="$(id -u)"

process_cwd() {
    if [[ -e "/proc/$1/cwd" ]]; then
        readlink "/proc/$1/cwd" 2>/dev/null || true
    elif command -v lsof >/dev/null 2>&1; then
        lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1
    fi
}

TARGET_PIDS=()
mapfile -t CANDIDATES < <(pgrep -u "$CURRENT_UID" -f 'gn0[.]evaluation[.]remote' || true)
for pid in "${CANDIDATES[@]}"; do
    [[ "$pid" != "$$" && "$(process_cwd "$pid")" == "$GN0_ROOT" ]] || continue
    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    [[ "$command" =~ ^([^[:space:]]*/)?python(3([.][0-9]+)?)?[[:space:]]+-m[[:space:]]+gn0[.]evaluation[.]remote([[:space:]]|$) ]] || continue
    TARGET_PIDS+=("$pid")
done

if [[ ${#TARGET_PIDS[@]} -eq 0 ]]; then
    echo "No scoped remote evaluation processes found."
    exit 0
fi

echo "Scoped remote evaluation PIDs: ${TARGET_PIDS[*]}"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run: no signals sent."
    exit 0
fi

kill -TERM "${TARGET_PIDS[@]}" 2>/dev/null || true
for ((elapsed=0; elapsed<GRACE_PERIOD; elapsed++)); do
    RUNNING=()
    for pid in "${TARGET_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && RUNNING+=("$pid")
    done
    [[ ${#RUNNING[@]} -eq 0 ]] && break
    sleep 1
done

RUNNING=()
for pid in "${TARGET_PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && RUNNING+=("$pid")
done
if [[ ${#RUNNING[@]} -gt 0 ]]; then
    echo "Sending SIGKILL to remaining PIDs: ${RUNNING[*]}"
    kill -KILL "${RUNNING[@]}" 2>/dev/null || true
fi

echo "Remote evaluation processes stopped."

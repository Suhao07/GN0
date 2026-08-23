import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def sanitize_metric(value):
    """Converts the value to float and turns NaN/Inf into 0.0."""
    try:
        val = float(value)
        return val if math.isfinite(val) else 0.0
    except (ValueError, TypeError):
        return 0.0


METRIC_KEYS = (
    "success",
    "spl",
    "distance_to_goal",
    "oracle_success",
    "path_length",
)


@dataclass(frozen=True)
class MetricSummary:
    episodes: int
    trajectory_length: float
    navigation_error: float
    oracle_success_count: int
    oracle_success: float
    success_count: int
    success_rate: float
    spl_total: float
    spl: float


def load_metric_records(log_dir: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for file_path in sorted(log_dir.glob("*.json")):
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON value is not an object")
            records.append(
                {key: sanitize_metric(payload.get(key, 0)) for key in METRIC_KEYS}
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            print(f"Warning: skipped {file_path.name}: {exc}", file=sys.stderr)
    return records


def summarize_metric_records(records: list[dict[str, float]]) -> MetricSummary:
    total = len(records)
    if total == 0:
        raise ValueError("cannot summarize an empty record set")

    totals = {key: sum(record[key] for record in records) for key in METRIC_KEYS}
    return MetricSummary(
        episodes=total,
        trajectory_length=totals["path_length"] / total,
        navigation_error=totals["distance_to_goal"] / total,
        oracle_success_count=int(totals["oracle_success"]),
        oracle_success=totals["oracle_success"] / total,
        success_count=int(totals["success"]),
        success_rate=totals["success"] / total,
        spl_total=totals["spl"],
        spl=totals["spl"] / total,
    )


def format_metric_summary(summary: MetricSummary, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(asdict(summary), indent=2, sort_keys=True)

    if output_format != "table":
        raise ValueError(f"unsupported output format: {output_format}")

    episode_total = f"{summary.episodes:,}"
    rows = (
        ("TL", "Trajectory length", f"{summary.trajectory_length:.3f}"),
        ("NE ↓", "Navigation error", f"{summary.navigation_error:.3f}"),
        (
            "OS ↑",
            "Oracle success",
            f"{summary.oracle_success:.1%}  "
            f"({summary.oracle_success_count:,}/{episode_total})",
        ),
        (
            "SR ↑",
            "Success rate",
            f"{summary.success_rate:.1%}  ({summary.success_count:,}/{episode_total})",
        ),
        (
            "SPL ↑",
            "Success weighted by path length",
            f"{summary.spl:.1%}  (mean {summary.spl:.3f})",
        ),
    )
    key_width = max(len(key) for key, _, _ in rows) + 2
    value_width = max(len(value) for _, _, value in rows)
    lines = [
        f"{key:<{key_width}}{value:<{value_width + 2}}{label}"
        for key, label, value in rows
    ]
    title = f"GN0 Evaluation Summary · {episode_total} episodes"
    rule = "─" * max(len(title), *(len(line) for line in lines))
    return "\n".join((title, rule, *lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate metrics from JSON logs in a directory."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Root directory containing the 'log' folder.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("table", "json"),
        default="table",
        help="Output format (default: table).",
    )
    args = parser.parse_args()

    log_dir = Path(args.path) / "log"
    if not log_dir.is_dir():
        print(f"Error: log directory '{log_dir}' not found.", file=sys.stderr)
        return 1

    records = load_metric_records(log_dir)
    total = len(records)
    if total == 0:
        print("Error: no valid evaluation logs found.", file=sys.stderr)
        return 1

    summary = summarize_metric_records(records)
    print(format_metric_summary(summary, args.output_format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

TRACKED_STOP_REASONS = (
    "action_near_zero",
    "chunk_low_translation",
    "distance_to_goal_unchanged",
)


def _is_success(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def summarize_stop_reasons(
    records: Iterable[dict[str, Any]],
) -> dict[str, Counter]:
    summary = {"success": Counter(), "failure": Counter()}
    for record in records:
        group = "success" if _is_success(record.get("success", 0.0)) else "failure"
        reason = record.get("stop")
        if reason is None or not str(reason).strip():
            reason = "missing"
        summary[group][str(reason)] += 1
    return summary


def _load_records(log_dir: Path) -> tuple[list[dict[str, Any]], int]:
    records = []
    invalid = 0
    for path in sorted(log_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON value is not an object")
            records.append(payload)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid += 1
            print(f"Warning: skipping {path.name}: {exc}", file=sys.stderr)
    return records, invalid


def _group_breakdown(counts: Counter) -> dict[str, Any]:
    total = sum(counts.values())
    other_counts = {
        reason: count
        for reason, count in sorted(counts.items())
        if reason not in TRACKED_STOP_REASONS and reason != "missing"
    }
    reasons = {reason: counts[reason] for reason in TRACKED_STOP_REASONS}
    reasons["other"] = sum(other_counts.values())
    reasons["missing"] = counts["missing"]
    return {
        "episodes": total,
        "reasons": reasons,
        "other_details": other_counts,
    }


def build_stop_reason_report(
    summary: dict[str, Counter], invalid_logs: int = 0
) -> dict[str, Any]:
    success = _group_breakdown(summary["success"])
    failure = _group_breakdown(summary["failure"])
    return {
        "valid_episodes": success["episodes"] + failure["episodes"],
        "invalid_logs": invalid_logs,
        "success": success,
        "failure": failure,
    }


def _format_count(count: int, total: int) -> str:
    ratio = count / total if total else 0.0
    return f"{count:,} ({ratio:.1%})"


def format_stop_reason_report(report: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report, indent=2, sort_keys=True)
    if output_format != "table":
        raise ValueError(f"unsupported output format: {output_format}")

    success = report["success"]
    failure = report["failure"]
    success_total = success["episodes"]
    failure_total = failure["episodes"]
    reason_order = (*TRACKED_STOP_REASONS, "other", "missing")

    success_header = f"Success (n={success_total:,})"
    failure_header = f"Failure (n={failure_total:,})"
    rows = [
        (
            reason,
            _format_count(success["reasons"][reason], success_total),
            _format_count(failure["reasons"][reason], failure_total),
        )
        for reason in reason_order
    ]

    other_reasons = sorted(
        set(success["other_details"]) | set(failure["other_details"])
    )
    detail_rows = [
        (
            reason,
            _format_count(success["other_details"].get(reason, 0), success_total),
            _format_count(failure["other_details"].get(reason, 0), failure_total),
        )
        for reason in other_reasons
    ]

    first_column = ("Stop reason", *(row[0] for row in rows), *other_reasons)
    second_column = (
        success_header,
        *(row[1] for row in rows),
        *(row[1] for row in detail_rows),
    )
    reason_width = max(len(value) for value in first_column) + 2
    success_width = max(len(value) for value in second_column) + 2

    def format_row(row: tuple[str, str, str]) -> str:
        reason, success_value, failure_value = row
        return (
            f"{reason:<{reason_width}}{success_value:<{success_width}}{failure_value}"
        )

    header = format_row(("Stop reason", success_header, failure_header))
    body = [format_row(row) for row in rows]
    title = f"GN0 Stop-Reason Summary · {report['valid_episodes']:,} valid episodes"
    invalid_line = None
    if report["invalid_logs"]:
        invalid_line = f"Invalid logs skipped · {report['invalid_logs']:,}"

    all_lines = [title, header, *body]
    if invalid_line:
        all_lines.insert(1, invalid_line)
    if detail_rows:
        all_lines.extend(("", "Other stop reasons", *map(format_row, detail_rows)))

    rule_width = max(len(line) for line in all_lines)
    all_lines.insert(1 if not invalid_line else 2, "─" * rule_width)
    return "\n".join(all_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare stop-reason proportions for successful and failed episodes."
    )
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Evaluation root containing log/*.json, or the log directory itself.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("table", "json"),
        default="table",
        help="Output format (default: table).",
    )
    args = parser.parse_args()

    log_dir = args.path if args.path.name == "log" else args.path / "log"
    if not log_dir.is_dir():
        parser.error(f"log directory not found: {log_dir}")

    records, invalid = _load_records(log_dir)
    if not records:
        parser.error(f"no valid JSON logs found in: {log_dir}")

    summary = summarize_stop_reasons(records)
    report = build_stop_reason_report(summary, invalid)
    print(format_stop_reason_report(report, args.output_format))


if __name__ == "__main__":
    main()

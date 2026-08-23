"""Shared, crash-safe helpers for GN0 evaluation entry points."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER = logging.getLogger(__name__)


def json_default(value: Any) -> Any:
    """Convert NumPy values to JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not serializable")


def get_episode_id(ref_json: str | Path) -> str:
    ref_path = Path(ref_json)
    return f"{ref_path.parent.name}_{ref_path.stem}"


def validate_split_args(split_num: int, split_id: int) -> None:
    if split_num <= 0:
        raise ValueError(f"split_num must be > 0, got {split_num}")
    if split_id < 0 or split_id >= split_num:
        raise ValueError(
            "split_id must be in [0, split_num), got "
            f"split_id={split_id}, split_num={split_num}"
        )


def shard_episodes(episodes: list, split_num: int, split_id: int) -> list:
    """Return a deterministic, disjoint strided shard."""
    validate_split_args(split_num, split_id)
    return episodes[split_id::split_num]


def make_log_dir(result_dir: str | Path) -> Path:
    log_dir = Path(result_dir) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def resolve_end_idx(episodes: list, end_idx: int | None) -> int:
    if end_idx is None or end_idx == -1:
        return len(episodes)
    if end_idx < -1:
        raise ValueError(f"end_idx must be >= 0, -1, or omitted; got {end_idx}")
    return int(end_idx)


def load_finished_episode_ids(log_dir: Path) -> set[str]:
    """Load only complete, internally consistent episode logs for resume."""
    finished_ids: set[str] = set()
    for path in sorted(log_dir.glob("*.json")):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring unreadable evaluation log %s: %s", path, exc)
            continue

        if not isinstance(payload, dict):
            LOGGER.warning("Ignoring non-object evaluation log %s", path)
            continue
        episode_id = payload.get("id")
        if episode_id != path.stem:
            LOGGER.warning(
                "Ignoring evaluation log %s because id=%r does not match filename",
                path,
                episode_id,
            )
            continue
        if (
            payload.get("status") != "complete"
            or payload.get("episode_complete") is not True
        ):
            LOGGER.warning("Ignoring incomplete evaluation log %s", path)
            continue
        finished_ids.add(episode_id)
    return finished_ids


def filter_pending_episodes(episodes: Iterable, finished_ids: set[str]) -> list:
    return [
        episode
        for episode in episodes
        if get_episode_id(episode.ref_json) not in finished_ids
    ]


def select_eval_episodes(
    dataset_split,
    log_dir: Path,
    start_idx: int,
    end_idx: int | None,
) -> tuple[list, int, int, int]:
    resolved_end_idx = resolve_end_idx(dataset_split.episodes, end_idx)
    finished_ids = load_finished_episode_ids(log_dir)
    selected = dataset_split.episodes[start_idx:resolved_end_idx]
    pending = filter_pending_episodes(selected, finished_ids)
    return pending, resolved_end_idx, len(selected) - len(pending), len(finished_ids)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a durable JSON object on the destination filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, default=json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

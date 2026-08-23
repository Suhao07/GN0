"""DAgger collection helpers for remote DreamZero navigation rollouts.

The collector writes the simple format discussed for DreamZero: every query
state becomes one synthetic InteriorGS LeRobot episode with 33 policy-history
frames followed by 33 oracle future frames.  A step_filter keeps only the
boundary sample trainable, so the existing InteriorGS loader can read it.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from GN_Bench.tasks.nav.oracle_trajectory import (
    DEFAULT_DAGGER_HEADING_LOOKAHEAD_M,
    DEFAULT_DAGGER_RESAMPLE_STEP_M,
    DEFAULT_DAGGER_STOP_RADIUS_M,
    DEFAULT_DAGGER_YAW_SMOOTH_WINDOW,
    TrajectoryPlanningError,
    build_smoothed_astar_trajectory,
)
from scipy.spatial.transform import Rotation as R

PAST_FRAMES = 33
FUTURE_FRAMES = 33
EPISODE_LENGTH = PAST_FRAMES + FUTURE_FRAMES
CHUNK_SIZE = 1000
VIDEO_FPS = 3.3333333333333335
VIDEO_CODEC = "h264"
VIDEO_ENCODER = "libx264"
VIDEO_PIX_FMT = "yuv420p"
VIDEO_KEY = "observation.images.rgb"
EXTRINSIC_KEY = "observation.camera_extrinsic"
LANGUAGE_KEY = "annotation.language.language_instruction"
ASTAR_RESAMPLE_STEP_M = DEFAULT_DAGGER_RESAMPLE_STEP_M
DAGGER_STOP_RADIUS_M = DEFAULT_DAGGER_STOP_RADIUS_M
DAGGER_HEADING_LOOKAHEAD_M = DEFAULT_DAGGER_HEADING_LOOKAHEAD_M
DAGGER_YAW_SMOOTH_WINDOW = DEFAULT_DAGGER_YAW_SMOOTH_WINDOW
DAGGER_HISTORY_SAMPLING = "uniform"
DAGGER_HISTORY_END = "current"


class InteriorGSDaggerCollector:
    def __init__(
        self,
        output_root: str | Path,
        *,
        worker_id: int = 0,
        past_frames: int = PAST_FRAMES,
        future_actions: int = FUTURE_FRAMES - 1,
        fps: float = VIDEO_FPS,
        stop_radius_m: float = DAGGER_STOP_RADIUS_M,
        stop_aware: bool = True,
        heading_lookahead_m: float = DAGGER_HEADING_LOOKAHEAD_M,
        yaw_smooth_window: int = DAGGER_YAW_SMOOTH_WINDOW,
        history_sampling: str = DAGGER_HISTORY_SAMPLING,
        history_end: str = DAGGER_HISTORY_END,
    ) -> None:
        self.output_root = Path(output_root)
        self.worker_id = int(worker_id)
        self.past_frames = int(past_frames)
        self.future_actions = int(future_actions)
        self.future_frames = self.future_actions + 1
        self.episode_length = self.past_frames + self.future_frames
        self.fps = float(fps)
        self.astar_resample_step_m = ASTAR_RESAMPLE_STEP_M
        self.stop_radius_m = max(0.0, float(stop_radius_m))
        self.stop_aware = bool(stop_aware)
        self.heading_lookahead_m = max(0.0, float(heading_lookahead_m))
        self.yaw_smooth_window = int(yaw_smooth_window)
        self.history_sampling = str(history_sampling or DAGGER_HISTORY_SAMPLING).lower()
        if self.history_sampling not in {"recent", "uniform"}:
            raise ValueError(
                "DAgger history_sampling must be 'recent' or 'uniform', "
                f"got {history_sampling!r}"
            )
        self.history_end = str(history_end or DAGGER_HISTORY_END).lower()
        if self.history_end not in {"current", "previous"}:
            raise ValueError(
                "DAgger history_end must be 'current' or 'previous', "
                f"got {history_end!r}"
            )

        if self.past_frames != PAST_FRAMES or self.future_frames != FUTURE_FRAMES:
            raise ValueError(
                "The simple DreamZero DAgger format expects 33 past frames and "
                "33 future frames."
            )

        self.meta_dir = self.output_root / "meta"
        self.lock_path = self._write_lock_path(self.output_root)
        self.next_episode_path = self.meta_dir / "next_episode_index.txt"
        self._base_metadata_ready = False
        self.timing_enabled = os.environ.get("DAGGER_TIMING", "0").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    @staticmethod
    def _write_lock_path(output_root: Path) -> Path:
        lock_dir = Path(
            os.environ.get("DAGGER_WRITE_LOCK_DIR", "/tmp/gn0_dagger_locks")
        )
        digest = hashlib.sha1(str(output_root).encode("utf-8")).hexdigest()[:16]
        return lock_dir / f"dagger_write_{digest}.lock"

    def collect(
        self,
        *,
        sim,
        instruction: str,
        episode_id: str,
        query_step: int,
        history_frames: list[np.ndarray],
        history_positions: list[np.ndarray],
        history_rotations: list[np.ndarray],
        goal_position,
        distance_to_goal: float | None = None,
        policy_motion_m: float | None = None,
    ) -> Path | None:
        # Write one 66-frame LeRobot DAgger episode for the current query.

        total_start = time.perf_counter()
        timings: dict[str, float] = {}

        if goal_position is None:
            print("[DAgger] skip: missing goal_position")
            return None
        if not history_frames:
            print("[DAgger] skip: missing policy history frames")
            return None

        current_state = sim.get_agent_state()
        start_position = np.asarray(current_state.position, dtype=np.float32).copy()
        start_rotation = np.asarray(current_state.rotation, dtype=np.float32).copy()

        stage_start = time.perf_counter()
        future_positions, future_rotations, reason, astar_meta = (
            self._build_astar_future(
                sim=sim,
                start_position=start_position,
                start_rotation=start_rotation,
                goal_position=goal_position,
                horizon=self.future_actions,
            )
        )
        timings["astar"] = time.perf_counter() - stage_start
        if future_positions is None or future_rotations is None:
            self._print_timing(
                episode_id=episode_id,
                query_step=query_step,
                timings=timings,
                total_start=total_start,
                status=f"skip:{reason}",
            )
            print(f"[DAgger] skip: oracle failed ({reason})")
            return None

        stage_start = time.perf_counter()
        rendered_future_frames = self._render_poses(
            sim=sim,
            positions=future_positions[1:],
            rotations=future_rotations[1:],
        )
        timings["render_32"] = time.perf_counter() - stage_start
        if len(rendered_future_frames) != self.future_actions:
            self._print_timing(
                episode_id=episode_id,
                query_step=query_step,
                timings=timings,
                total_start=total_start,
                status=f"skip:rendered_{len(rendered_future_frames)}",
            )
            print(
                f"[DAgger] skip: rendered {len(rendered_future_frames)} future frames, "
                f"expected {self.future_actions}"
            )
            return None
        future_frames = [self._ensure_rgb(history_frames[-1])] + rendered_future_frames

        stage_start = time.perf_counter()
        past_frames, past_positions, past_rotations, selected_history_indices = (
            self._select_past(
                history_frames=history_frames,
                history_positions=history_positions,
                history_rotations=history_rotations,
                current_position=start_position,
                current_rotation=start_rotation,
            )
        )
        timings["select_past"] = time.perf_counter() - stage_start

        frames = past_frames + future_frames
        positions = past_positions + future_positions
        rotations = past_rotations + future_rotations
        if len(frames) != self.episode_length:
            raise RuntimeError(f"Bad DAgger episode length: {len(frames)}")

        stage_start = time.perf_counter()
        episode_index = self._allocate_episode_index()
        timings["allocate"] = time.perf_counter() - stage_start
        chunk_index = episode_index // CHUNK_SIZE

        video_path = (
            self.output_root
            / "videos"
            / f"chunk-{chunk_index:03d}"
            / VIDEO_KEY
            / f"episode_{episode_index:06d}.mp4"
        )
        parquet_path = (
            self.output_root
            / "data"
            / f"chunk-{chunk_index:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )

        video_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        stage_start = time.perf_counter()
        self._write_video(video_path, frames)
        timings["write_video"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        self._write_parquet(
            parquet_path=parquet_path,
            episode_index=episode_index,
            task_index=episode_index,
            positions=positions,
            rotations=rotations,
        )
        timings["write_parquet"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        self._append_metadata(
            episode_index=episode_index,
            task_index=episode_index,
            instruction=instruction,
            episode_id=episode_id,
            query_step=query_step,
            oracle_reason=reason,
            astar_meta=astar_meta,
            frame_shape=frames[0].shape,
            goal_position=goal_position,
            history_indices=selected_history_indices,
            history_frame_count=len(history_frames),
            distance_to_goal=distance_to_goal,
            policy_motion_m=policy_motion_m,
        )
        timings["write_meta"] = time.perf_counter() - stage_start
        self._print_timing(
            episode_id=episode_id,
            query_step=query_step,
            timings=timings,
            total_start=total_start,
            episode_index=episode_index,
            status="ok",
        )
        print(
            f"[DAgger] wrote episode={episode_index} source={episode_id} "
            f"query_step={query_step} oracle_reason={reason}"
        )
        return parquet_path

    def build_teacher_nav_deltas(
        self,
        *,
        sim,
        goal_position,
        horizon: int,
    ) -> tuple[list[np.ndarray], str, dict[str, Any]]:
        if goal_position is None:
            return [], "missing_goal_position", {}
        current_state = sim.get_agent_state()
        start_position = np.asarray(current_state.position, dtype=np.float32).copy()
        start_rotation = np.asarray(current_state.rotation, dtype=np.float32).copy()
        positions, rotations, reason, meta = self._build_astar_future(
            sim=sim,
            start_position=start_position,
            start_rotation=start_rotation,
            goal_position=goal_position,
            horizon=max(1, int(horizon)),
        )
        if positions is None or rotations is None:
            return [], reason, meta
        return self._nav_deltas_from_poses(positions, rotations), reason, meta

    @staticmethod
    def _nav_deltas_from_poses(positions, rotations) -> list[np.ndarray]:
        deltas = []
        points = [np.asarray(position, dtype=np.float32) for position in positions]
        rots = [np.asarray(rotation, dtype=np.float32) for rotation in rotations]
        for index in range(max(0, min(len(points), len(rots)) - 1)):
            curr_rot = R.from_quat(rots[index])
            next_rot = R.from_quat(rots[index + 1])
            world_delta = points[index + 1] - points[index]
            local_delta = curr_rot.inv().apply(
                np.array([world_delta[0], world_delta[1], 0.0], dtype=np.float32)
            )
            yaw0 = curr_rot.as_euler("zyx", degrees=False)[0]
            yaw1 = next_rot.as_euler("zyx", degrees=False)[0]
            dyaw = (float(yaw1 - yaw0) + math.pi) % (2.0 * math.pi) - math.pi
            deltas.append(
                np.asarray([local_delta[0], local_delta[1], dyaw], dtype=np.float32)
            )
        return deltas

    def _print_timing(
        self,
        *,
        episode_id: str,
        query_step: int,
        timings: dict[str, float],
        total_start: float,
        episode_index: int | None = None,
        status: str,
    ) -> None:
        if not self.timing_enabled:
            return
        total = time.perf_counter() - total_start
        episode_part = f"episode={episode_index} " if episode_index is not None else ""
        ordered_keys = [
            "astar",
            "render_32",
            "select_past",
            "allocate",
            "write_video",
            "write_parquet",
            "write_meta",
        ]
        parts = " ".join(
            f"{key}={float(timings.get(key, 0.0)):.3f}s" for key in ordered_keys
        )
        print(
            f"[DAggerTiming] {episode_part}source={episode_id} "
            f"query_step={query_step} status={status} {parts} total={total:.3f}s"
        )

    def _build_astar_future(
        self,
        *,
        sim,
        start_position,
        start_rotation,
        goal_position,
        horizon: int,
    ):
        if goal_position is None or len(goal_position) < 2:
            return None, None, "missing_goal_position", {}

        try:
            trajectory = build_smoothed_astar_trajectory(
                sim=sim,
                start_position=start_position,
                start_rotation=start_rotation,
                goal_position=goal_position,
                horizon=horizon,
                step_m=self.astar_resample_step_m,
                stop_radius_m=self.stop_radius_m,
                stop_aware=self.stop_aware,
                heading_lookahead_m=self.heading_lookahead_m,
                yaw_smooth_window=self.yaw_smooth_window,
            )
        except TrajectoryPlanningError as exc:
            return None, None, exc.reason, {}

        return (
            trajectory.positions,
            trajectory.rotations,
            trajectory.reason,
            trajectory.metadata,
        )

    def _render_poses(self, *, sim, positions, rotations) -> list[np.ndarray]:
        frames = []
        saved_state = sim.get_agent_state()
        saved_position = np.asarray(saved_state.position, dtype=np.float32).copy()
        saved_rotation = np.asarray(saved_state.rotation, dtype=np.float32).copy()
        saved_actor_cursor = getattr(sim, "actor_render_cursor", None)
        try:
            for position, rotation in zip(positions, rotations):
                obs = sim.get_observations_at(
                    position=np.asarray(position, dtype=np.float32).tolist(),
                    rotation=np.asarray(rotation, dtype=np.float32).tolist(),
                    keep_agent_at_new_pose=False,
                )
                if obs is None or "rgb" not in obs:
                    break
                frames.append(self._ensure_rgb(obs["rgb"]))
        finally:
            sim.set_agent_state(saved_position, saved_rotation, reset_sensors=False)
            if saved_actor_cursor is not None:
                sim.actor_render_cursor = saved_actor_cursor
        return frames

    def _select_past(
        self,
        *,
        history_frames,
        history_positions,
        history_rotations,
        current_position,
        current_rotation,
    ):
        indexed_triples = list(
            enumerate(zip(history_frames, history_positions, history_rotations))
        )
        history_pool = indexed_triples
        if self.history_end == "previous" and len(indexed_triples) > 1:
            history_pool = indexed_triples[:-1]
        if self.history_sampling == "uniform":
            past = self._select_uniform_history(history_pool)
        else:
            past = history_pool
        if not past:
            current_frame = history_frames[-1]
            past = [
                (
                    max(0, len(history_frames) - 1),
                    (current_frame, current_position, current_rotation),
                )
            ]

        selected = past[-self.past_frames :]
        while len(selected) < self.past_frames:
            selected.insert(0, selected[0])

        indices = [int(item[0]) for item in selected]
        frames = [self._ensure_rgb(item[1][0]) for item in selected]
        positions = [
            np.asarray(item[1][1], dtype=np.float32).copy() for item in selected
        ]
        rotations = [
            np.asarray(item[1][2], dtype=np.float32).copy() for item in selected
        ]
        return frames, positions, rotations, indices

    def _select_uniform_history(self, indexed_triples):
        if not indexed_triples:
            return []
        if len(indexed_triples) == 1:
            return [indexed_triples[0]] * self.past_frames
        indices = np.linspace(
            0,
            len(indexed_triples) - 1,
            num=self.past_frames,
            dtype=np.int64,
        ).tolist()
        return [indexed_triples[int(index)] for index in indices]

    @staticmethod
    def _json_float_list(value) -> list[float] | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return []
        return [float(item) for item in arr.tolist()]

    @staticmethod
    def _ensure_rgb(frame) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame HWC, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _pose_to_extrinsic(position, rotation) -> list[list[float]]:
        yaw = R.from_quat(rotation).as_euler("zyx", degrees=False)[0]
        rot_world_agent = R.from_euler("z", yaw, degrees=False).as_matrix()
        cam_to_nav = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = (rot_world_agent @ cam_to_nav).astype(np.float32)
        extrinsic[:3, 3] = np.asarray(position, dtype=np.float32)[:3]
        return extrinsic.tolist()

    def _write_video(self, path: Path, frames: list[np.ndarray]) -> None:
        if not frames:
            raise ValueError(f"No frames to write for {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.stem}.tmp.{os.getpid()}.mp4")

        first = self._ensure_rgb(frames[0])
        height, width = first.shape[:2]
        self._write_h264_video(tmp_path, frames, width=width, height=height)
        os.replace(tmp_path, path)

    def _write_h264_video(
        self,
        path: Path,
        frames: list[np.ndarray],
        *,
        width: int,
        height: int,
    ) -> None:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            str(path),
            fps=self.fps,
            codec=VIDEO_ENCODER,
            quality=8,
            macro_block_size=1,
        )
        try:
            for frame in frames:
                rgb = self._ensure_rgb(frame)
                if rgb.shape[:2] != (height, width):
                    rgb = cv2.resize(
                        rgb,
                        (width, height),
                        interpolation=cv2.INTER_AREA,
                    )
                writer.append_data(rgb)
        finally:
            writer.close()

    def _write_parquet(
        self,
        *,
        parquet_path: Path,
        episode_index: int,
        task_index: int,
        positions,
        rotations,
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        extrinsics = [
            self._pose_to_extrinsic(position, rotation)
            for position, rotation in zip(positions, rotations)
        ]
        num_rows = len(extrinsics)
        table = pa.Table.from_arrays(
            [
                pa.array(
                    extrinsics,
                    type=pa.list_(pa.list_(pa.float32(), list_size=4), list_size=4),
                ),
                pa.array([int(episode_index)] * num_rows, type=pa.int64()),
                pa.array(list(range(num_rows)), type=pa.int64()),
                pa.array([int(task_index)] * num_rows, type=pa.int64()),
                pa.array([int(task_index)] * num_rows, type=pa.int64()),
            ],
            names=[
                EXTRINSIC_KEY,
                "episode_index",
                "frame_index",
                "task_index",
                LANGUAGE_KEY,
            ],
        )
        pq.write_table(table, parquet_path)

    def _allocate_episode_index(self) -> int:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        with _locked_file(self.lock_path):
            next_index = self._read_stored_next_episode_index()
            if next_index is None:
                next_index = self._discover_next_episode_index()

            while self._episode_artifacts_exist(next_index):
                next_index += 1

            self.next_episode_path.write_text(f"{next_index + 1}\n", encoding="utf-8")
        return next_index

    def _read_stored_next_episode_index(self) -> int | None:
        if not self.next_episode_path.exists():
            return None
        text = self.next_episode_path.read_text(encoding="utf-8").strip()
        try:
            next_index = int(text or "0")
        except ValueError:
            return None
        if next_index < 0:
            return None
        return next_index

    def _discover_next_episode_index(self) -> int:
        max_index = -1
        episodes_path = self.meta_dir / "episodes.jsonl"
        if episodes_path.exists():
            with episodes_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        max_index = max(
                            max_index, int(json.loads(line)["episode_index"])
                        )
                    except Exception:
                        continue

        return max_index + 1

    def _episode_artifacts_exist(self, episode_index: int) -> bool:
        chunk_index = int(episode_index) // CHUNK_SIZE
        parquet_path = (
            self.output_root
            / "data"
            / f"chunk-{chunk_index:03d}"
            / f"episode_{int(episode_index):06d}.parquet"
        )
        video_path = (
            self.output_root
            / "videos"
            / f"chunk-{chunk_index:03d}"
            / VIDEO_KEY
            / f"episode_{int(episode_index):06d}.mp4"
        )
        return parquet_path.exists() or video_path.exists()

    def _append_metadata(
        self,
        *,
        episode_index: int,
        task_index: int,
        instruction: str,
        episode_id: str,
        query_step: int,
        oracle_reason: str,
        astar_meta: dict[str, Any],
        frame_shape,
        goal_position=None,
        history_indices: list[int] | None = None,
        history_frame_count: int | None = None,
        distance_to_goal: float | None = None,
        policy_motion_m: float | None = None,
    ) -> None:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        goal_position_world = self._json_float_list(goal_position)
        used_history_indices = [int(index) for index in (history_indices or [])]
        inferred_history_source = f"client:{self.history_sampling}"
        history_frame_count = (
            None if history_frame_count is None else int(history_frame_count)
        )
        history_includes_current = bool(
            used_history_indices
            and history_frame_count is not None
            and max(used_history_indices) >= max(0, history_frame_count - 1)
        )
        with _locked_file(self.lock_path):
            self._ensure_base_metadata(height=height, width=width)
            _append_jsonl(
                self.meta_dir / "tasks.jsonl",
                {"task_index": int(task_index), "task": str(instruction or "")},
            )
            _append_jsonl(
                self.meta_dir / "episodes.jsonl",
                {
                    "episode_index": int(episode_index),
                    "tasks": [str(instruction or "")],
                    "length": int(self.episode_length),
                    "scene": str(episode_id).split("_")[0],
                    "trajectory": str(episode_id),
                    "source": "remote_dagger",
                    "source_episode_id": str(episode_id),
                    "query_step": int(query_step),
                    "worker_id": int(self.worker_id),
                    "goal_position_world": goal_position_world,
                    "history_sampling": str(self.history_sampling),
                    "history_end": str(self.history_end),
                    "history_index_source": inferred_history_source,
                    "history_frame_indices": used_history_indices,
                    "history_frame_count": history_frame_count,
                    "history_includes_current": history_includes_current,
                    "oracle_reason": str(oracle_reason),
                    "distance_to_goal": (
                        None if distance_to_goal is None else float(distance_to_goal)
                    ),
                    "policy_motion_m": (
                        None if policy_motion_m is None else float(policy_motion_m)
                    ),
                    **astar_meta,
                    "video_fps": float(self.fps),
                    "video_height": int(height),
                    "video_width": int(width),
                    "video_codec": VIDEO_CODEC,
                    "video_encoder": VIDEO_ENCODER,
                },
            )
            blocked = [i for i in range(self.episode_length) if i != self.past_frames]
            _append_jsonl(
                self.meta_dir / "step_filter.jsonl",
                {"episode_index": int(episode_index), "step_indices": blocked},
            )
            self._update_info_counts(episode_index=episode_index)

    def _ensure_base_metadata(self, *, height: int, width: int) -> None:
        if self._base_metadata_ready:
            return

        modality_path = self.meta_dir / "modality.json"
        if not modality_path.exists():
            modality_path.write_text(
                json.dumps(
                    {
                        "video": {"rgb": {"original_key": VIDEO_KEY}},
                        "camera": {"extrinsic": {"original_key": EXTRINSIC_KEY}},
                        "annotation": {"language.language_instruction": {}},
                    },
                    indent=4,
                )
                + "\n",
                encoding="utf-8",
            )

        info_path = self.meta_dir / "info.json"
        if not info_path.exists():
            info = {
                "codebase_version": "v2.0",
                "robot_type": "interiorgs",
                "total_episodes": 0,
                "total_frames": 0,
                "total_tasks": 0,
                "total_videos": 1,
                "total_chunks": 0,
                "chunks_size": CHUNK_SIZE,
                "fps": float(self.fps),
                "splits": {"train": "0:100"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    VIDEO_KEY: {
                        "dtype": "video",
                        "shape": [int(height), int(width), 3],
                        "names": ["height", "width", "channel"],
                        "video_info": {
                            "video.fps": float(self.fps),
                            "video.codec": VIDEO_CODEC,
                            "video.pix_fmt": VIDEO_PIX_FMT,
                            "video.is_depth_map": False,
                            "has_audio": False,
                        },
                    },
                    EXTRINSIC_KEY: {
                        "dtype": "float32",
                        "shape": [4, 4],
                        "names": ["row", "col"],
                    },
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "frame_index": {"dtype": "int64", "shape": [1]},
                    "task_index": {"dtype": "int64", "shape": [1]},
                    LANGUAGE_KEY: {"dtype": "int64", "shape": [1]},
                },
                "conversion": {
                    "source": "GN0 remote DreamZero DAgger",
                    "format": "66_frame_simple_window",
                    "past_frames": PAST_FRAMES,
                    "future_frames": FUTURE_FRAMES,
                    "oracle": "astar_world_resampled",
                    "astar_resample_step_m": ASTAR_RESAMPLE_STEP_M,
                    "stop_aware": bool(self.stop_aware),
                    "stop_radius_m": float(self.stop_radius_m),
                    "heading_lookahead_m": float(self.heading_lookahead_m),
                    "yaw_smooth_window": int(self.yaw_smooth_window),
                    "history_sampling": str(self.history_sampling),
                    "history_end": str(self.history_end),
                    "history_includes_current": self.history_end == "current",
                    "history_indices": "per_episode_episodes_jsonl",
                    "history_index_source": "per_episode",
                    "goal_position_world": "per_episode_episodes_jsonl",
                    "video_encoder": VIDEO_ENCODER,
                },
            }
            info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
        else:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            features = info.setdefault("features", {})
            video_feature = features.setdefault(VIDEO_KEY, {})
            video_info = video_feature.setdefault("video_info", {})
            conversion = info.setdefault("conversion", {})
            changed = False
            if video_info.get("video.codec") != VIDEO_CODEC:
                video_info["video.codec"] = VIDEO_CODEC
                changed = True
            if video_info.get("video.pix_fmt") != VIDEO_PIX_FMT:
                video_info["video.pix_fmt"] = VIDEO_PIX_FMT
                changed = True
            if video_info.get("video.fps") != float(self.fps):
                video_info["video.fps"] = float(self.fps)
                changed = True
            if conversion.get("oracle") != "astar_world_resampled":
                conversion["oracle"] = "astar_world_resampled"
                changed = True
            if conversion.get("astar_resample_step_m") != ASTAR_RESAMPLE_STEP_M:
                conversion["astar_resample_step_m"] = ASTAR_RESAMPLE_STEP_M
                changed = True
            if conversion.get("stop_aware") != bool(self.stop_aware):
                conversion["stop_aware"] = bool(self.stop_aware)
                changed = True
            if conversion.get("stop_radius_m") != float(self.stop_radius_m):
                conversion["stop_radius_m"] = float(self.stop_radius_m)
                changed = True
            if conversion.get("heading_lookahead_m") != float(self.heading_lookahead_m):
                conversion["heading_lookahead_m"] = float(self.heading_lookahead_m)
                changed = True
            if conversion.get("yaw_smooth_window") != int(self.yaw_smooth_window):
                conversion["yaw_smooth_window"] = int(self.yaw_smooth_window)
                changed = True
            if conversion.get("history_sampling") != str(self.history_sampling):
                conversion["history_sampling"] = str(self.history_sampling)
                changed = True
            if conversion.get("history_end") != str(self.history_end):
                conversion["history_end"] = str(self.history_end)
                changed = True
            if conversion.get("history_indices") != "per_episode_episodes_jsonl":
                conversion["history_indices"] = "per_episode_episodes_jsonl"
                changed = True
            if conversion.get("history_index_source") != "per_episode":
                conversion["history_index_source"] = "per_episode"
                changed = True
            if conversion.get("goal_position_world") != "per_episode_episodes_jsonl":
                conversion["goal_position_world"] = "per_episode_episodes_jsonl"
                changed = True
            expected_includes_current = self.history_end == "current"
            if conversion.get("history_includes_current") != expected_includes_current:
                conversion["history_includes_current"] = expected_includes_current
                changed = True
            if conversion.get("video_encoder") != VIDEO_ENCODER:
                conversion["video_encoder"] = VIDEO_ENCODER
                changed = True
            if "astar_source_frame_stride" in conversion:
                conversion.pop("astar_source_frame_stride", None)
                changed = True
            if changed:
                info_path.write_text(
                    json.dumps(info, indent=4) + "\n", encoding="utf-8"
                )

        self._base_metadata_ready = True

    def _update_info_counts(self, *, episode_index: int) -> None:
        info_path = self.meta_dir / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))

        total_episodes = self._safe_int(info.get("total_episodes"), 0) + 1
        total_frames = self._safe_int(info.get("total_frames"), 0) + int(
            self.episode_length
        )
        total_chunks = max(
            self._safe_int(info.get("total_chunks"), 0),
            int(episode_index) // CHUNK_SIZE + 1,
            int(math.ceil(max(total_episodes, 1) / CHUNK_SIZE)),
        )

        info["total_episodes"] = int(total_episodes)
        info["total_tasks"] = int(total_episodes)
        info["total_frames"] = int(total_frames)
        info["total_chunks"] = int(total_chunks)
        info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class _locked_file:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self.handle

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
        return False

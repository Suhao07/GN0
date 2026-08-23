"""GN0 agent wrapper for generic remote navigation policies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from GN_Bench.core.agent import Agent
from GN_Bench.tasks.nav.oracle_trajectory import DEFAULT_DAGGER_STOP_RADIUS_M
from scipy.spatial.transform import Rotation as R

from gn0.media import EpisodeMediaRecorder
from gn0.remote import RemoteNavClient

ACTION_STOP = 0
STOP_REASON_ACTION_NEAR_ZERO = "action_near_zero"
STOP_REASON_CHUNK_LOW_TRANSLATION = "chunk_low_translation"
ACTION_ALIASES = {
    "stop": 0,
    "<stop>": 0,
    "move_forward": 1,
    "forward": 1,
    "fwd": 1,
    "<fwd>": 1,
    "turn_left": 2,
    "left": 2,
    "<left>": 2,
    "turn_right": 3,
    "right": 3,
    "<right>": 3,
}
DAGGER_STUCK_WINDOW = 8
DAGGER_STUCK_MOTION_EPS = 0.03
DAGGER_SAME_STATE_POS_EPS = 0.02
DAGGER_SAME_STATE_YAW_EPS = math.radians(3.0)
DAGGER_STOP_RADIUS_M = DEFAULT_DAGGER_STOP_RADIUS_M
DAGGER_TEACHER_RESCUE_HORIZON = 8
DAGGER_STUCK_OVERRIDE_STEPS = 5
DAGGER_STUCK_TAKEOVER_STEPS = 15
GN0_WAM_ENV_SCHEMA_VERSION = "gn0-wam-env/v1"
MAX_ROLLOUT_SEED = 2**63 - 1


def stable_wam_rollout_seed(
    base_seed: int,
    policy_version: int,
    episode_id: str,
    query_id: int,
) -> int:
    """Derive a process-independent request seed for offline behavior collection."""

    payload = (
        f"{int(base_seed)}\0{int(policy_version)}\0{str(episode_id)}\0{int(query_id)}"
    ).encode("utf-8")
    return (
        int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        & MAX_ROLLOUT_SEED
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> str:
    data = (
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return hashlib.sha256(data).hexdigest()


class RemoteNavAgent(Agent):
    """Calls a remote WebSocket policy and adapts actions to GN0.

    Supported response schemas:
      - {"action_type": "discrete", "actions": [0, 1, 2, 3]}
      - {"action_type": "nav_delta", "actions": [[dx, dy, dyaw], ...]}
      - Legacy nested nav_delta: {"action.nav_delta": [[dx, dy, dyaw], ...]}

    The agent executes at most ``action_num`` returned actions. If the server
    returns fewer actions, the next request sends exactly the RGB frames observed
    while executing those actions.
    """

    def __init__(
        self,
        model_path: str | None = None,
        result_path: str = "tmp/remote_nav",
        prompt_type: str = "V1",
        action_num: int = 1,
        server_host: str = "127.0.0.1",
        server_port: int = 8000,
        request_timeout: float = 120.0,
        connect_timeout: float = 600.0,
        metadata_timeout: float = 2.0,
        stop_distance: float | None = None,
        stop_eps: float = 1e-3,
        chunk_stop_translation: float = 0.15,
        translation_frame: str = "chunk_start",
        max_translation: float = 0.0,
        max_yaw: float = 0.0,
        save_images: bool = False,
        save_video: bool = False,
        video_fps: float = 5.0,
        video_codec: str = "h264",
        action_format: str = "auto",
        send_state: bool = False,
        wam_trace_mode: str = "none",
        wam_rollout_base_seed: int = 0,
        dagger: bool = False,
        dagger_output_root: str | None = None,
        dagger_worker_id: int = 0,
        dagger_future_actions: int = 32,
        dagger_stop_radius: float = DAGGER_STOP_RADIUS_M,
        dagger_rescue_horizon: int = DAGGER_TEACHER_RESCUE_HORIZON,
        dagger_history_sampling: str = "uniform",
        dagger_history_end: str = "current",
        dagger_stop_override: bool = True,
        dagger_stuck_rescue: bool = True,
        **_: Any,
    ) -> None:
        del model_path, prompt_type, stop_distance
        self.result_path = result_path
        self.action_num = max(1, int(action_num))
        self.stop_eps = max(0.0, float(stop_eps))
        self.chunk_stop_translation = max(0.0, float(chunk_stop_translation))
        self.translation_frame = str(translation_frame)
        if self.translation_frame not in {"chunk_start", "current"}:
            raise ValueError(
                "translation_frame must be 'chunk_start' or 'current', "
                f"got {self.translation_frame!r}"
            )
        self.max_translation = float(max_translation)
        self.max_yaw = float(max_yaw)
        self.save_video = bool(save_video)
        self.save_images = bool(save_images) and not self.save_video
        media_mode = (
            "video" if self.save_video else "images" if self.save_images else "none"
        )
        self.media = EpisodeMediaRecorder(
            result_path,
            mode=media_mode,
            fps=video_fps,
            codec=video_codec,
        )
        self.action_format = self._normalize_action_type(action_format)
        if self.action_format not in {"auto", "discrete", "nav_delta"}:
            raise ValueError(
                "action_format must be auto, discrete, or nav_delta, "
                f"got {action_format!r}"
            )
        self.send_state = bool(send_state)
        self.wam_trace_mode = str(wam_trace_mode or "none").strip().lower()
        if self.wam_trace_mode not in {"none", "sidecar"}:
            raise ValueError(
                f"wam_trace_mode must be 'none' or 'sidecar', got {wam_trace_mode!r}"
            )
        self.wam_rollout_base_seed = int(wam_rollout_base_seed)
        if not 0 <= self.wam_rollout_base_seed <= MAX_ROLLOUT_SEED:
            raise ValueError(
                f"wam_rollout_base_seed must be in [0, {MAX_ROLLOUT_SEED}], "
                f"got {self.wam_rollout_base_seed}"
            )
        self.dagger_enabled = bool(dagger)
        self.dagger_stop_radius = max(0.0, float(dagger_stop_radius))
        self.dagger_rescue_horizon = max(1, int(dagger_rescue_horizon))
        self.dagger_stop_override = bool(dagger_stop_override)
        self.dagger_stuck_rescue = bool(dagger_stuck_rescue)
        self.dagger_collector = None
        if self.dagger_enabled:
            if not str(dagger_output_root or "").strip():
                raise ValueError("dagger_output_root is required when dagger=True")
            from gn0.collection import InteriorGSDaggerCollector

            self.dagger_collector = InteriorGSDaggerCollector(
                dagger_output_root,
                worker_id=int(dagger_worker_id),
                future_actions=int(dagger_future_actions),
                stop_radius_m=self.dagger_stop_radius,
                stop_aware=True,
                history_sampling=dagger_history_sampling,
                history_end=dagger_history_end,
            )

        print(
            "Initialize remote nav client: "
            f"ws://{server_host}:{server_port}, action_num={self.action_num}, "
            f"action_format={self.action_format}, "
            f"chunk_stop_translation={self.chunk_stop_translation}, "
            f"translation_frame={self.translation_frame}, "
            f"save_images={self.save_images}, save_video={self.save_video}, "
            f"wam_trace_mode={self.wam_trace_mode}, "
            f"dagger={self.dagger_enabled}, "
            f"dagger_stop_radius={self.dagger_stop_radius}, "
            f"dagger_history_sampling={dagger_history_sampling}, "
            f"dagger_history_end={dagger_history_end}"
        )
        self.client = RemoteNavClient(
            host=server_host,
            port=server_port,
            request_timeout=request_timeout,
            connect_timeout=connect_timeout,
            metadata_timeout=metadata_timeout,
        )
        print("Remote nav server metadata:", self.client.metadata)
        self.wam_trace_metadata: dict[str, Any] = {}
        self.wam_policy_version: int | None = None
        if self.wam_trace_mode == "sidecar":
            trace_metadata = self.client.metadata.get("wam_trace")
            if not isinstance(trace_metadata, dict):
                raise RuntimeError(
                    "wam_trace_mode='sidecar' requires server WAM trace capability metadata"
                )
            supported_modes = trace_metadata.get("modes", [])
            if not trace_metadata.get("enabled") or "sidecar" not in supported_modes:
                raise RuntimeError(
                    "Remote server does not advertise enabled sidecar WAM trace capture"
                )
            if trace_metadata.get("policy_version") is None:
                raise RuntimeError(
                    "Remote WAM trace metadata is missing policy_version"
                )
            self.wam_policy_version = int(trace_metadata["policy_version"])
            if self.wam_policy_version < 0:
                raise RuntimeError("Remote WAM policy_version must be non-negative")
            self.wam_trace_metadata = dict(trace_metadata)

        self.episode_id: str | None = None
        self.pending_actions: list[Any] = []
        self.pending_action_metadata: list[dict[str, Any]] = []
        self.pending_action_type = "auto"
        self._pending_chunk_stop_distance: float | None = None
        self.episode_stop_reason: str | None = None
        self.chunk_base_rotation: R | None = None
        self.frames_since_query: list[np.ndarray] = []
        self.query_count = 0
        self.step_idx = 0
        self.history_actions: list[Any] = []
        self.history_frames: list[np.ndarray] = []
        self.history_positions: list[np.ndarray] = []
        self.history_rotations: list[np.ndarray] = []
        self.last_dagger_collect_pose: np.ndarray | None = None
        self.teacher_actions: list[np.ndarray] = []
        self.teacher_action_reason: str | None = None
        self.dagger_stuck_steps = 0
        self.dagger_entered_stop_radius = False
        self.dagger_truncated_after_goal_exit = False
        self.wam_query_records: list[dict[str, Any]] = []
        self.wam_primitive_records: list[dict[str, Any]] = []
        self._pending_primitive_record: dict[str, Any] | None = None
        self._episode_manifest_ref: dict[str, Any] | None = None
        self._episode_manifest_finalized = False
        self._episode_attempt_id: str | None = None

    def reset(self, episode_ref, sim=None) -> None:
        del sim
        self.close_episode(complete=False)
        episode_ref = Path(episode_ref)
        self.episode_id = f"{episode_ref.parent.name}_{episode_ref.stem}"
        self.pending_actions = []
        self.pending_action_metadata = []
        self.pending_action_type = "auto"
        self._pending_chunk_stop_distance = None
        self.episode_stop_reason = None
        self.chunk_base_rotation = None
        self.frames_since_query = []
        self.query_count = 0
        self.step_idx = 0
        self.history_actions = []
        self.history_frames = []
        self.history_positions = []
        self.history_rotations = []
        self.last_dagger_collect_pose = None
        self.teacher_actions = []
        self.teacher_action_reason = None
        self.dagger_stuck_steps = 0
        self.dagger_entered_stop_radius = False
        self.dagger_truncated_after_goal_exit = False
        self.wam_query_records = []
        self.wam_primitive_records = []
        self._pending_primitive_record = None
        self._episode_manifest_ref = None
        self._episode_manifest_finalized = False
        self._episode_attempt_id = uuid.uuid4().hex
        self.media.start_episode(self.episode_id)
        try:
            self.client.reset()
        except Exception as exc:
            print(
                "[RemoteNav] reset request failed, continuing with session_id reset: "
                f"{exc}"
            )
        print("RemoteNav Reset Complete for Episode:", self.episode_id)

    def act(self, observations, info):
        sim = observations.get("sim")
        if sim is None:
            raise ValueError(
                "RemoteNavAgent requires observations['sim'] from the GN0 remote evaluator"
            )

        rgb = observations.get("rgb")
        if rgb is not None:
            self._record_rgb(rgb)
            if self.dagger_enabled:
                self._record_policy_history(rgb, sim)
            if self.save_images or self.save_video:
                self._save_observation_images(sim, rgb)

        _ = info

        if self._maybe_truncate_after_dagger_goal_exit(info):
            return self._stop_for_dagger_goal_exit()

        if self.teacher_actions:
            self._collect_dagger_after_action(observations, info, sim)
            return self._pop_teacher_action(sim)

        if not self.pending_actions:
            if self._maybe_arm_stuck_teacher_rescue(observations, info, sim):
                self._collect_dagger_after_action(observations, info, sim)
                return self._pop_teacher_action(sim)
            self._query_new_actions(observations, sim)
            self._collect_dagger_after_action(observations, info, sim)
        else:
            self._collect_dagger_after_action(observations, info, sim)
            if self._maybe_arm_stuck_teacher_rescue(observations, info, sim):
                return self._pop_teacher_action(sim)

        if self._pending_chunk_stop_distance is not None:
            chunk_distance = self._pending_chunk_stop_distance
            self._pending_chunk_stop_distance = None
            self.pending_actions = []
            self.pending_action_metadata = []
            stop_action = np.zeros(3, dtype=np.float32)
            if self._maybe_arm_stop_teacher_rescue(observations, info, sim):
                return self._pop_teacher_action(sim)
            self.history_actions.append(self._history_value(stop_action, "nav_delta"))
            self.step_idx += 1
            self._set_pending_primitive(
                commanded_action=stop_action,
                action_type="nav_delta",
                action_source="policy_chunk_low_translation",
                action_metadata=None,
            )
            self.episode_stop_reason = STOP_REASON_CHUNK_LOW_TRANSLATION
            print(
                "[RemoteNav] STOP: chunk translation "
                f"{chunk_distance:.6f}m < {self.chunk_stop_translation:.6f}m"
            )
            return {"action": ACTION_STOP}

        if not self.pending_actions:
            self.episode_stop_reason = "empty_action_chunk"
            self._set_pending_primitive(
                commanded_action=ACTION_STOP,
                action_type="discrete",
                action_source="policy_empty_chunk",
                action_metadata=None,
            )
            return {"action": ACTION_STOP}

        action = self.pending_actions.pop(0)
        action_metadata = (
            self.pending_action_metadata.pop(0)
            if self.pending_action_metadata
            else None
        )
        action_type = self.pending_action_type
        self.history_actions.append(self._history_value(action, action_type))
        self.step_idx += 1

        if action_type == "discrete":
            action_id = int(action)
            if action_id == ACTION_STOP:
                self.episode_stop_reason = "policy_discrete_stop"
                self.pending_actions = []
                self.pending_action_metadata = []
            self._set_pending_primitive(
                commanded_action=action_id,
                action_type="discrete",
                action_source="policy",
                action_metadata=action_metadata,
            )
            return {"action": action_id}

        nav_delta = np.asarray(action, dtype=np.float32)
        if self._is_stop_nav_delta(nav_delta):
            if self._maybe_arm_stop_teacher_rescue(observations, info, sim):
                if self.history_actions:
                    self.history_actions.pop()
                self.step_idx = max(0, self.step_idx - 1)
                return self._pop_teacher_action(sim)
            self.pending_actions = []
            self.pending_action_metadata = []
            self._set_pending_primitive(
                commanded_action=nav_delta,
                action_type="nav_delta",
                action_source="policy",
                action_metadata=action_metadata,
            )
            self.episode_stop_reason = STOP_REASON_ACTION_NEAR_ZERO
            return {"action": ACTION_STOP}
        self.dagger_stuck_steps = 0
        self._set_pending_primitive(
            commanded_action=nav_delta,
            action_type="nav_delta",
            action_source="policy",
            action_metadata=action_metadata,
        )
        return self._nav_delta_to_env_action(sim, nav_delta)

    def make_stall_recovery_action(
        self,
        observations,
        sim,
        distance: float,
    ) -> dict[str, Any]:
        rgb = observations.get("rgb")
        if rgb is not None:
            self._record_rgb(rgb)
            if self.dagger_enabled:
                self._record_policy_history(rgb, sim)
            if self.save_images or self.save_video:
                self._save_observation_images(sim, rgb)

        self.pending_actions = []
        self.pending_action_metadata = []
        self._pending_chunk_stop_distance = None
        self.chunk_base_rotation = None
        self.teacher_actions = []
        self.teacher_action_reason = None
        self.dagger_stuck_steps = 0
        self.episode_stop_reason = None

        nav_delta = np.asarray([-abs(float(distance)), 0.0, 0.0], dtype=np.float32)
        self.history_actions.append(self._history_value(nav_delta, "nav_delta"))
        self.step_idx += 1
        self._set_pending_primitive(
            commanded_action=nav_delta,
            action_type="nav_delta",
            action_source="distance_to_goal_unchanged_backtrack",
            action_metadata=None,
        )
        return self._nav_delta_to_env_action(
            sim,
            nav_delta,
            translation_frame="current",
        )

    def _query_new_actions(self, observations, sim) -> None:
        rgb = observations.get("rgb")
        instruction_payload = observations.get("instruction", {})
        if isinstance(instruction_payload, dict):
            instruction = instruction_payload.get("text", "")
        else:
            instruction = str(instruction_payload or "")

        if rgb is None:
            raise ValueError("RemoteNavAgent requires observations['rgb']")
        if self.episode_id is None:
            raise RuntimeError("RemoteNavAgent.act called before reset")

        agent_state = sim.get_agent_state()
        self.chunk_base_rotation = R.from_quat(agent_state.rotation)

        current_rgb, image_chunk = self._build_request_images(rgb)
        state = self._nav_pose(sim) if self.send_state else None
        query_id = self.query_count
        rollout_seed = None
        if self.wam_trace_mode == "sidecar":
            assert self.wam_policy_version is not None
            rollout_seed = stable_wam_rollout_seed(
                self.wam_rollout_base_seed,
                self.wam_policy_version,
                self.episode_id,
                query_id,
            )
        response = self.client.infer(
            rgb=current_rgb,
            images=image_chunk,
            instruction=instruction,
            session_id=self.episode_id,
            state=state,
            rollout_seed=rollout_seed,
            query_id=query_id if rollout_seed is not None else None,
            wam_trace_mode=self.wam_trace_mode,
        )
        trace_ref = self._validate_wam_trace_response(
            response,
            expected_query_id=query_id,
            expected_seed=rollout_seed,
        )
        self.query_count += 1
        self.frames_since_query = []

        action_type, actions = self._extract_actions(response)
        chunk_distance = None
        chunk_stop = False
        if action_type == "nav_delta":
            chunk_stop, chunk_distance = self._nav_delta_chunk_stop_decision(actions)

        keep = 0 if chunk_stop else min(self.action_num, len(actions))
        self.pending_actions = [] if chunk_stop else list(actions[:keep])
        self.pending_action_type = action_type
        self._pending_chunk_stop_distance = chunk_distance if chunk_stop else None
        self.pending_action_metadata = []
        if trace_ref is not None:
            query_record_index = len(self.wam_query_records)
            self.wam_query_records.append(
                {
                    "query_id": query_id,
                    "query_step": self.step_idx,
                    "rollout_seed": int(rollout_seed),
                    "trace_ref": trace_ref,
                    "returned_action_count": len(actions),
                    "kept_action_count": keep,
                    "primitive_execution_mask": [0] * len(actions),
                }
            )
            self.pending_action_metadata = [
                {
                    "query_record_index": query_record_index,
                    "query_id": query_id,
                    "trace_id": trace_ref["trace_id"],
                    "chunk_index": chunk_index,
                }
                for chunk_index in range(keep)
            ]

        request_frames = 1 if image_chunk is None else int(image_chunk.shape[0])
        chunk_summary = (
            ""
            if chunk_distance is None
            else f" chunk_translation={chunk_distance:.6f}m chunk_stop={chunk_stop}"
        )
        print(
            f"[RemoteNav] step={self.step_idx} request_frames={request_frames} "
            f"action_type={action_type} returned={len(actions)} keep={keep}"
            f"{chunk_summary}"
        )

    def _validate_wam_trace_response(
        self,
        response: dict[str, Any],
        *,
        expected_query_id: int,
        expected_seed: int | None,
    ) -> dict[str, Any] | None:
        if self.wam_trace_mode == "none":
            return None
        trace_ref = response.get("wam_trace_ref")
        if not isinstance(trace_ref, dict):
            raise RuntimeError("Strict WAM sidecar response is missing wam_trace_ref")
        required = (
            "schema_version",
            "transport",
            "root_id",
            "trace_id",
            "manifest_relpath",
            "manifest_sha256",
            "tensor_sha256",
            "policy_version",
            "query_id",
            "rollout_seed",
        )
        missing = [key for key in required if key not in trace_ref]
        if missing:
            raise RuntimeError(f"WAM trace reference is missing fields: {missing}")

        normalized = _json_safe(trace_ref)
        encoded = json.dumps(normalized, sort_keys=True).encode("utf-8")
        if len(encoded) > 64 * 1024:
            raise RuntimeError(
                f"WAM trace reference must remain small, got {len(encoded)} bytes"
            )
        expected_schema = self.wam_trace_metadata.get("schema_version")
        expected_root_id = self.wam_trace_metadata.get("root_id")
        if normalized["schema_version"] != expected_schema:
            raise RuntimeError(
                "WAM trace schema mismatch: "
                f"server={expected_schema!r}, response={normalized['schema_version']!r}"
            )
        if normalized["transport"] != "server_sidecar":
            raise RuntimeError(
                f"WAM trace transport must be server_sidecar, got {normalized['transport']!r}"
            )
        if normalized["root_id"] != expected_root_id:
            raise RuntimeError(
                f"WAM trace root_id mismatch: {normalized['root_id']!r} != {expected_root_id!r}"
            )
        if int(normalized["query_id"]) != int(expected_query_id):
            raise RuntimeError(
                f"WAM response query_id mismatch: {normalized['query_id']} != {expected_query_id}"
            )
        if expected_seed is None or int(normalized["rollout_seed"]) != int(
            expected_seed
        ):
            raise RuntimeError(
                "WAM response rollout_seed does not echo the deterministic request seed"
            )
        if int(normalized["policy_version"]) != int(self.wam_policy_version):
            raise RuntimeError(
                "WAM response policy_version differs from connection metadata"
            )
        for key, expected in (
            ("query_id", expected_query_id),
            ("rollout_seed", expected_seed),
            ("policy_version", self.wam_policy_version),
        ):
            if key not in response or int(
                np.asarray(response[key]).reshape(()).item()
            ) != int(expected):
                raise RuntimeError(f"WAM top-level response does not echo {key}")

        manifest_path = Path(str(normalized["manifest_relpath"]))
        if manifest_path.is_absolute() or ".." in manifest_path.parts:
            raise RuntimeError(
                f"WAM manifest_relpath must be a safe relative path, got {manifest_path}"
            )
        for checksum_key in ("manifest_sha256", "tensor_sha256"):
            checksum = str(normalized[checksum_key])
            if len(checksum) != 64 or any(
                char not in "0123456789abcdef" for char in checksum
            ):
                raise RuntimeError(f"Invalid {checksum_key}: {checksum!r}")
        if not str(normalized["trace_id"]).strip():
            raise RuntimeError("WAM trace_id cannot be empty")
        return normalized

    def _set_pending_primitive(
        self,
        *,
        commanded_action: Any,
        action_type: str,
        action_source: str,
        action_metadata: dict[str, Any] | None,
    ) -> None:
        metadata = dict(action_metadata or {})
        self._pending_primitive_record = {
            **metadata,
            "commanded_action": self._history_value(commanded_action, action_type),
            "action_type": str(action_type),
            "action_source": str(action_source),
        }

    @staticmethod
    def _trace_metrics(metrics: Any) -> dict[str, Any]:
        if not isinstance(metrics, dict):
            return {}
        keys = (
            "distance_to_goal",
            "success",
            "spl",
            "path_length",
            "oracle_success",
            "collisions",
            "collision",
        )
        return {key: _json_safe(metrics[key]) for key in keys if key in metrics}

    def record_step_result(
        self,
        *,
        env_action: dict[str, Any],
        before_metrics: dict[str, Any],
        after_metrics: dict[str, Any],
        episode_over: bool,
        action_source: str | None = None,
    ) -> None:
        """Join one executed primitive to its immutable behavior-policy trace."""

        if self.wam_trace_mode == "none":
            self._pending_primitive_record = None
            return
        record = dict(self._pending_primitive_record or {})
        self._pending_primitive_record = None
        if action_source is not None:
            record["action_source"] = str(action_source)
        source = str(record.get("action_source", "episode_loop"))
        query_record_index = record.get("query_record_index")
        chunk_index = record.get("chunk_index")
        policy_gradient_mask = 0
        if (
            source == "policy"
            and query_record_index is not None
            and chunk_index is not None
        ):
            query_index = int(query_record_index)
            action_index = int(chunk_index)
            if not 0 <= query_index < len(self.wam_query_records):
                raise RuntimeError(f"Invalid WAM query record index: {query_index}")
            execution_mask = self.wam_query_records[query_index][
                "primitive_execution_mask"
            ]
            if not 0 <= action_index < len(execution_mask):
                raise RuntimeError(f"Invalid WAM chunk action index: {action_index}")
            execution_mask[action_index] = 1
            policy_gradient_mask = 1
        record.update(
            {
                "primitive_index": len(self.wam_primitive_records),
                "action_source": source,
                "policy_gradient_mask": policy_gradient_mask,
                "env_action": _json_safe(env_action),
                "metrics_before": self._trace_metrics(before_metrics),
                "metrics_after": self._trace_metrics(after_metrics),
                "episode_over": bool(episode_over),
            }
        )
        self.wam_primitive_records.append(record)

    def _collect_dagger_after_action(self, observations, info, sim) -> None:
        if self.dagger_collector is None:
            return
        if self.step_idx <= 0:
            return
        if self.episode_id is None:
            return

        instruction_payload = observations.get("instruction", {})
        if isinstance(instruction_payload, dict):
            instruction = instruction_payload.get("text", "")
        else:
            instruction = str(instruction_payload or "")

        current_pose = self._nav_pose(sim)
        skip_reason = self._dagger_duplicate_stuck_reason(current_pose)
        if skip_reason is not None:
            print(f"[DAgger] skip: {skip_reason}")
            return

        written_path = self.dagger_collector.collect(
            sim=sim,
            instruction=instruction,
            episode_id=self.episode_id,
            query_step=self.step_idx,
            history_frames=self.history_frames,
            history_positions=self.history_positions,
            history_rotations=self.history_rotations,
            goal_position=observations.get("goal_position"),
            distance_to_goal=self._distance_to_goal(info),
            policy_motion_m=self._finite_motion_or_none(self._last_policy_motion()),
        )
        if written_path is not None:
            self.last_dagger_collect_pose = current_pose.copy()

    def _maybe_truncate_after_dagger_goal_exit(self, info) -> bool:
        if not self.dagger_enabled or self.dagger_stop_radius <= 0.0:
            return False
        if self.dagger_truncated_after_goal_exit:
            return True

        dtg = self._distance_to_goal(info)
        if dtg is None:
            return False
        if dtg <= self.dagger_stop_radius:
            self.dagger_entered_stop_radius = True
            return False
        if not self.dagger_entered_stop_radius:
            return False

        self.dagger_truncated_after_goal_exit = True
        self.pending_actions = []
        self.pending_action_metadata = []
        self.teacher_actions = []
        self.teacher_action_reason = None
        print(
            "[DAgger] truncate: left_stop_radius_after_entry "
            f"distance_to_goal={dtg:.3f}m stop_radius={self.dagger_stop_radius:.3f}m"
        )
        return True

    def _stop_for_dagger_goal_exit(self) -> dict[str, int]:
        self.episode_stop_reason = "dagger_goal_exit"
        self._set_pending_primitive(
            commanded_action=ACTION_STOP,
            action_type="discrete",
            action_source="dagger_goal_exit_truncate",
            action_metadata=None,
        )
        return {"action": ACTION_STOP}

    def _maybe_arm_stop_teacher_rescue(self, observations, info, sim) -> bool:
        if not (self.dagger_enabled and self.dagger_stop_override):
            return False
        if not self._outside_dagger_stop_radius(info):
            return False
        return self._arm_teacher_actions(
            observations=observations,
            sim=sim,
            reason="stop_outside_goal",
            horizon=1,
        )

    def _maybe_arm_stuck_teacher_rescue(self, observations, info, sim) -> bool:
        if not (self.dagger_enabled and self.dagger_stuck_rescue):
            return False
        if not self._outside_dagger_stop_radius(info):
            self.dagger_stuck_steps = 0
            return False
        immediate_motion = self._last_policy_motion()
        recent_motion = self._recent_policy_motion(DAGGER_STUCK_WINDOW)
        is_stuck = (
            immediate_motion <= DAGGER_STUCK_MOTION_EPS
            or recent_motion <= DAGGER_STUCK_MOTION_EPS
        )
        if not is_stuck:
            self.dagger_stuck_steps = 0
            return False

        self.dagger_stuck_steps += 1
        if self.dagger_stuck_steps < DAGGER_STUCK_OVERRIDE_STEPS:
            return False

        horizon = (
            self.dagger_rescue_horizon
            if self.dagger_stuck_steps >= DAGGER_STUCK_TAKEOVER_STEPS
            else 1
        )
        reason = (
            "stuck_teacher_rescue "
            f"stuck_steps={self.dagger_stuck_steps} "
            f"immediate_motion={immediate_motion:.3f}m "
            f"recent_motion={recent_motion:.3f}m"
        )
        if self._arm_teacher_actions(
            observations=observations, sim=sim, reason=reason, horizon=horizon
        ):
            self.dagger_stuck_steps = 0
            return True
        return False

    def _arm_teacher_actions(
        self, *, observations, sim, reason: str, horizon: int
    ) -> bool:
        if self.dagger_collector is None:
            return False
        goal_position = observations.get("goal_position")
        try:
            deltas, oracle_reason, _ = self.dagger_collector.build_teacher_nav_deltas(
                sim=sim,
                goal_position=goal_position,
                horizon=max(1, int(horizon)),
            )
        except Exception as exc:
            print(f"[DAgger] teacher rescue failed ({reason}): {exc}")
            return False

        valid_actions = []
        for delta in deltas:
            arr = np.asarray(delta, dtype=np.float32).reshape(-1)[:3]
            if arr.shape[0] == 3 and np.isfinite(arr).all():
                valid_actions.append(arr.astype(np.float32))
        if not valid_actions:
            print(f"[DAgger] teacher rescue unavailable ({reason}): {oracle_reason}")
            return False
        if self._is_stop_nav_delta(valid_actions[0]):
            print(f"[DAgger] teacher rescue produced STOP ({reason}): {oracle_reason}")
            return False

        keep = max(1, min(int(horizon), len(valid_actions)))
        self.teacher_actions = valid_actions[:keep]
        self.teacher_action_reason = f"{reason}; oracle={oracle_reason}"
        self.pending_actions = []
        self.pending_action_metadata = []
        self.pending_action_type = "nav_delta"
        print(
            f"[DAgger] teacher rescue armed: reason={self.teacher_action_reason} "
            f"actions={len(self.teacher_actions)}"
        )
        return True

    def _pop_teacher_action(self, sim) -> dict[str, Any]:
        if not self.teacher_actions:
            self.episode_stop_reason = "dagger_teacher_empty"
            self._set_pending_primitive(
                commanded_action=ACTION_STOP,
                action_type="discrete",
                action_source="teacher",
                action_metadata=None,
            )
            return {"action": ACTION_STOP}
        nav_delta = np.asarray(self.teacher_actions.pop(0), dtype=np.float32)
        self.history_actions.append(self._history_value(nav_delta, "nav_delta"))
        self.step_idx += 1
        self._set_pending_primitive(
            commanded_action=nav_delta,
            action_type="nav_delta",
            action_source="teacher",
            action_metadata=None,
        )
        if self._is_stop_nav_delta(nav_delta):
            self.teacher_actions = []
            self.episode_stop_reason = "dagger_teacher_stop"
            return {"action": ACTION_STOP}
        print(
            f"[DAgger] executing teacher action step={self.step_idx} "
            f"remaining={len(self.teacher_actions)} reason={self.teacher_action_reason}"
        )
        return self._nav_delta_to_env_action(
            sim, nav_delta, translation_frame="current"
        )

    def _dagger_duplicate_stuck_reason(self, current_pose: np.ndarray) -> str | None:
        if self.last_dagger_collect_pose is None:
            return None

        last_pose = np.asarray(self.last_dagger_collect_pose, dtype=np.float32)
        current_pose = np.asarray(current_pose, dtype=np.float32)
        pos_delta = float(np.linalg.norm(current_pose[:2] - last_pose[:2]))
        yaw_delta = self._angle_abs_diff(current_pose[2], last_pose[2])
        if (
            pos_delta > DAGGER_SAME_STATE_POS_EPS
            or yaw_delta > DAGGER_SAME_STATE_YAW_EPS
        ):
            return None

        immediate_motion = self._last_policy_motion()
        if immediate_motion <= DAGGER_STUCK_MOTION_EPS:
            return (
                "repeated_stuck_state "
                f"pos_delta={pos_delta:.3f}m "
                f"yaw_delta={math.degrees(yaw_delta):.1f}deg "
                f"immediate_motion={immediate_motion:.3f}m"
            )

        recent_motion = self._recent_policy_motion(DAGGER_STUCK_WINDOW)
        if recent_motion > DAGGER_STUCK_MOTION_EPS:
            return None

        return (
            "repeated_stuck_state "
            f"pos_delta={pos_delta:.3f}m "
            f"yaw_delta={math.degrees(yaw_delta):.1f}deg "
            f"recent_motion={recent_motion:.3f}m"
        )

    def _distance_to_goal(self, info) -> float | None:
        if not isinstance(info, dict):
            return None
        value = info.get("distance_to_goal")
        try:
            out = float(value)
        except Exception:
            return None
        return out if math.isfinite(out) else None

    def _outside_dagger_stop_radius(self, info) -> bool:
        dtg = self._distance_to_goal(info)
        if dtg is None:
            return False
        return dtg > self.dagger_stop_radius

    @staticmethod
    def _finite_motion_or_none(value: float) -> float | None:
        try:
            value = float(value)
        except Exception:
            return None
        return value if math.isfinite(value) else None

    def _last_policy_motion(self) -> float:
        if len(self.history_positions) < 2:
            return float("inf")
        recent = np.asarray(self.history_positions[-2:], dtype=np.float32)
        return float(np.linalg.norm(recent[-1, :2] - recent[0, :2]))

    def _recent_policy_motion(self, window: int) -> float:
        if len(self.history_positions) < 2:
            return float("inf")
        count = min(len(self.history_positions), max(2, int(window) + 1))
        recent = np.asarray(self.history_positions[-count:], dtype=np.float32)
        return float(np.linalg.norm(recent[-1, :2] - recent[0, :2]))

    @staticmethod
    def _angle_abs_diff(a: float, b: float) -> float:
        diff = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
        return abs(diff)

    def _record_rgb(self, rgb: np.ndarray) -> None:
        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            return
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self.frames_since_query.append(np.ascontiguousarray(frame.copy()))
        max_history = max(64, self.action_num * 4)
        if len(self.frames_since_query) > max_history:
            self.frames_since_query = self.frames_since_query[-max_history:]

    def _record_policy_history(self, rgb: np.ndarray, sim) -> None:
        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            return
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        state = sim.get_agent_state()
        self.history_frames.append(np.ascontiguousarray(frame.copy()))
        self.history_positions.append(
            np.asarray(state.position, dtype=np.float32).copy()
        )
        self.history_rotations.append(
            np.asarray(state.rotation, dtype=np.float32).copy()
        )
        max_history = 4096
        if len(self.history_frames) > max_history:
            self.history_frames = self.history_frames[-max_history:]
            self.history_positions = self.history_positions[-max_history:]
            self.history_rotations = self.history_rotations[-max_history:]

    def _build_request_images(
        self, current_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        current = np.asarray(current_rgb)
        if current.dtype != np.uint8:
            current = np.clip(current, 0, 255).astype(np.uint8)
        if self.query_count == 0 or not self.frames_since_query:
            return current, None
        return current, np.stack(self.frames_since_query, axis=0)

    def _extract_actions(self, response: dict[str, Any]) -> tuple[str, list[Any]]:
        action_type, raw_actions = self._select_action_payload(response)
        action_type = self._resolve_action_type(action_type, raw_actions)
        if action_type == "discrete":
            return action_type, self._normalize_discrete_actions(raw_actions)
        return action_type, self._normalize_nav_delta_actions(raw_actions)

    def _select_action_payload(self, response: dict[str, Any]) -> tuple[str, Any]:
        explicit_type = (
            response.get("action_type")
            or response.get("action.type")
            or response.get("action_format")
        )
        action_type = (
            self._normalize_action_type(explicit_type) if explicit_type else "auto"
        )

        if "actions" in response:
            return action_type, response["actions"]
        if "action.nav_delta" in response:
            return "nav_delta", response["action.nav_delta"]
        if "action.discrete" in response:
            return "discrete", response["action.discrete"]
        if "action.vlnce" in response:
            return "discrete", response["action.vlnce"]
        if "action" in response:
            return action_type, response["action"]
        raise KeyError(f"Remote response has no action field: {response.keys()}")

    def _resolve_action_type(self, action_type: str, raw_actions: Any) -> str:
        if self.action_format != "auto":
            return self.action_format
        if action_type != "auto":
            return action_type

        if isinstance(raw_actions, (str, bytes)):
            return "discrete"
        arr = np.asarray(raw_actions)
        if arr.shape == ():
            return "discrete"
        if arr.dtype.kind in {"i", "u", "b"}:
            return "discrete"
        if arr.ndim >= 2 and arr.shape[-1] >= 3:
            return "nav_delta"
        if arr.ndim == 1 and arr.shape[0] == 3 and arr.dtype.kind == "f":
            return "nav_delta"
        return "discrete"

    @staticmethod
    def _normalize_action_type(value: Any) -> str:
        if value is None:
            return "auto"
        text = str(value).strip().lower().replace("-", "_")
        aliases = {
            "": "auto",
            "auto": "auto",
            "vlnce": "discrete",
            "vln_ce": "discrete",
            "discrete": "discrete",
            "int": "discrete",
            "nav": "nav_delta",
            "delta": "nav_delta",
            "nav_delta": "nav_delta",
            "continuous": "nav_delta",
            "relative_pose": "nav_delta",
            "relative": "nav_delta",
        }
        return aliases.get(text, text)

    @staticmethod
    def _normalize_discrete_actions(raw_actions: Any) -> list[int]:
        if isinstance(raw_actions, (str, bytes)):
            raw_actions = [raw_actions]
        arr = np.asarray(raw_actions, dtype=object)
        if arr.shape == ():
            values = [arr.item()]
        else:
            values = arr.reshape(-1).tolist()

        actions: list[int] = []
        for value in values:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if isinstance(value, str):
                key = value.strip().lower()
                if key not in ACTION_ALIASES:
                    raise ValueError(f"Unknown discrete action token: {value!r}")
                action = ACTION_ALIASES[key]
            else:
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError(f"Boolean is not a discrete action: {value!r}")
                action = int(value)
                if action != value:
                    raise ValueError(f"Discrete action must be an integer: {value!r}")
            if action not in {0, 1, 2, 3}:
                raise ValueError(f"Discrete action must be in [0, 3], got {action}")
            actions.append(action)
        return actions

    @staticmethod
    def _normalize_nav_delta_actions(raw_actions: Any) -> list[np.ndarray]:
        arr = np.asarray(raw_actions, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] < 3:
                raise ValueError(f"Expected nav_delta shape (N, >=3), got {arr.shape}")
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[-1] < 3:
            raise ValueError(f"Expected nav_delta shape (N, >=3), got {arr.shape}")
        arr = arr[:, :3]
        if not np.isfinite(arr).all():
            raise ValueError("nav_delta actions must contain only finite values")
        return [row.astype(np.float32) for row in arr]

    def _is_stop_nav_delta(self, nav_delta: np.ndarray) -> bool:
        action = np.asarray(nav_delta, dtype=np.float32)[:3]
        return bool(np.all(np.abs(action) <= self.stop_eps))

    def _nav_delta_chunk_stop_decision(self, actions: list[Any]) -> tuple[bool, float]:
        total_translation = sum(
            math.hypot(float(action[0]), float(action[1])) for action in actions
        )
        should_stop = (
            bool(actions)
            and self.chunk_stop_translation > 0.0
            and total_translation < self.chunk_stop_translation
        )
        return should_stop, total_translation

    def _nav_delta_to_env_action(
        self,
        sim,
        nav_delta: np.ndarray,
        *,
        translation_frame: str | None = None,
    ) -> dict[str, Any]:
        dx, dy, dyaw = [float(v) for v in nav_delta[:3]]
        if self.max_translation > 0:
            dist = math.hypot(dx, dy)
            if dist > self.max_translation:
                scale = self.max_translation / max(dist, 1e-8)
                dx *= scale
                dy *= scale
        if self.max_yaw > 0:
            dyaw = max(-self.max_yaw, min(self.max_yaw, dyaw))

        state = sim.get_agent_state()
        curr_pos = np.array(state.position, dtype=np.float32)
        curr_rot = R.from_quat(state.rotation)

        frame = translation_frame or self.translation_frame
        if frame == "chunk_start" and self.chunk_base_rotation is not None:
            trans_rot = self.chunk_base_rotation
        else:
            trans_rot = curr_rot

        world_delta = trans_rot.apply(np.array([dx, dy, 0.0], dtype=np.float32))
        new_pos = curr_pos + world_delta.astype(np.float32)
        new_pos[2] = curr_pos[2]

        new_rot = curr_rot * R.from_euler("z", dyaw, degrees=False)
        return {
            "action": {
                "position": new_pos.astype(float).tolist(),
                "rotation": new_rot.as_quat().astype(float).tolist(),
            }
        }

    @staticmethod
    def _nav_pose(sim) -> np.ndarray:
        state = sim.get_agent_state()
        pos = np.asarray(state.position, dtype=np.float32)
        yaw = R.from_quat(state.rotation).as_euler("zyx", degrees=False)[0]
        return np.asarray([pos[0], pos[1], yaw], dtype=np.float32)

    @staticmethod
    def _history_value(action: Any, action_type: str) -> Any:
        if action_type == "discrete":
            return int(action)
        return np.asarray(action, dtype=np.float32).astype(float).tolist()

    def close_episode(
        self,
        *,
        complete: bool = False,
        final_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        self.media.close()
        if self.wam_trace_mode == "none" or self.episode_id is None:
            return None
        if self._episode_manifest_finalized:
            return self._episode_manifest_ref

        status = "complete" if complete else "incomplete"
        attempt_id = self._episode_attempt_id or uuid.uuid4().hex
        if final_metrics is not None:
            published_final_metrics = self._trace_metrics(final_metrics)
        elif self.wam_primitive_records:
            published_final_metrics = dict(
                self.wam_primitive_records[-1].get("metrics_after", {})
            )
        else:
            published_final_metrics = {}
        if not complete:
            termination_reason = "collection_interrupted"
        elif not self.wam_primitive_records:
            termination_reason = "no_primitives"
        elif self.wam_primitive_records[-1].get("action_source") == "early_stop":
            termination_reason = "early_stop_rotation"
        elif (
            self.wam_primitive_records[-1].get("action_source")
            == "dagger_goal_exit_truncate"
        ):
            termination_reason = "dagger_left_stop_radius"
        elif self.wam_primitive_records[-1].get("episode_over"):
            termination_reason = "environment_done"
        else:
            termination_reason = "evaluation_step_limit"
        manifest = {
            "schema_version": GN0_WAM_ENV_SCHEMA_VERSION,
            "status": status,
            "episode_id": self.episode_id,
            "group_id": self.episode_id,
            "rollout_id": attempt_id,
            "attempt_id": attempt_id,
            "behavior_policy_version": self.wam_policy_version,
            "rollout_base_seed": self.wam_rollout_base_seed,
            "query_count": len(self.wam_query_records),
            "primitive_count": len(self.wam_primitive_records),
            "termination_reason": termination_reason,
            "queries": self.wam_query_records,
            "primitives": self.wam_primitive_records,
            "final_metrics": published_final_metrics,
        }
        relative_path = (
            Path(self.episode_id)
            / "wam_trace"
            / ("READY.json" if complete else f"INCOMPLETE.{attempt_id}.json")
        )
        manifest_path = Path(self.result_path) / relative_path
        manifest_sha256 = _atomic_write_json(manifest_path, manifest)
        manifest_ref = {
            "schema_version": GN0_WAM_ENV_SCHEMA_VERSION,
            "status": status,
            "manifest_relpath": str(relative_path),
            "manifest_sha256": manifest_sha256,
            "episode_id": self.episode_id,
            "behavior_policy_version": self.wam_policy_version,
        }
        self._episode_manifest_ref = manifest_ref
        self._episode_manifest_finalized = True
        return manifest_ref

    def close(self) -> None:
        self.close_episode(complete=False)
        self.client.close()

    def _save_observation_images(self, sim, rgb: np.ndarray) -> None:
        bev_traj = sim.get_bev_map_with_trajectory()
        self.media.record(rgb, "rgb", self.step_idx)
        self.media.record(bev_traj, "bev_traj", self.step_idx)

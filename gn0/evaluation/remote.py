#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from GN_Bench import Env
from GN_Bench.datasets import make_dataset
from GN_Bench.tasks.nav.oracle_trajectory import DEFAULT_DAGGER_STOP_RADIUS_M
from tqdm import trange

from gn0.benchmark import get_config
from gn0.evaluation.common import (
    atomic_write_json,
    get_episode_id,
    make_log_dir,
    select_eval_episodes,
    shard_episodes,
    validate_split_args,
)

LOGGER = logging.getLogger(__name__)

TARGET_METRICS = (
    "distance_to_goal",
    "success",
    "spl",
    "path_length",
    "oracle_success",
)
DATASET_SEED = 42
STOP_ACTION_ID = 0
STOP_REASON_DISTANCE_TO_GOAL_UNCHANGED = "distance_to_goal_unchanged"
STALL_RECOVERY_STEPS = 8
STALL_RECOVERY_BACKTRACK_METERS = 0.08
STALL_RECOVERY_MAX_ATTEMPTS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GN_Bench evaluation against a remote navigation server"
    )

    io_group = parser.add_argument_group("Paths & IO")
    io_group.add_argument(
        "--exp-config", type=str, required=True, help="Config YAML path"
    )
    io_group.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Evaluation results directory",
    )

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--max-actions-per-query",
        type=int,
        default=8,
        help="Maximum number of returned remote actions to execute before the next query",
    )
    eval_group.add_argument("--split-num", type=int, required=True)
    eval_group.add_argument("--split-id", type=int, required=True)
    eval_group.add_argument("--start-idx", type=int, default=0, help="StartIndex")
    eval_group.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Exclusive end index. Omit to run through the dataset end.",
    )
    eval_group.add_argument(
        "--enable-stall-recovery",
        action="store_true",
        help=(
            "Recover from navigation stalls by backing up 8 x 0.08m on the "
            "first two triggers; stop on the third trigger."
        ),
    )
    eval_group.add_argument(
        "--stall-patience",
        type=int,
        default=None,
        help="Consecutive unchanged-distance steps before recovery or STOP.",
    )
    remote_group = parser.add_argument_group("Remote Navigation Client")
    remote_group.add_argument("--server-host", type=str, default="127.0.0.1")
    remote_group.add_argument("--server-port", type=int, default=8000)
    remote_group.add_argument(
        "--remote-request-timeout",
        type=float,
        default=600.0,
    )
    remote_group.add_argument("--remote-connect-timeout", type=float, default=600.0)
    remote_group.add_argument(
        "--remote-stop-eps",
        type=float,
        default=1e-3,
        help="Treat a model nav_delta with all |dx,dy,dyaw| <= eps as STOP.",
    )
    remote_group.add_argument(
        "--remote-min-chunk-translation",
        type=float,
        default=0.0,
        help=(
            "Stop before executing a nav_delta chunk when "
            "sum(sqrt(dx^2 + dy^2)) is below this distance in meters. "
            "Set to 0 to disable."
        ),
    )
    remote_group.add_argument(
        "--remote-translation-frame",
        type=str,
        default="chunk_start",
        choices=["chunk_start", "current"],
    )
    remote_group.add_argument(
        "--remote-max-translation",
        type=float,
        default=0.0,
        help="Maximum translation per action in meters; 0 disables clipping.",
    )
    remote_group.add_argument(
        "--remote-max-yaw",
        type=float,
        default=0.0,
        help="Maximum absolute yaw per action in radians; 0 disables clipping.",
    )
    remote_group.add_argument(
        "--save-media",
        type=str,
        choices=("images", "video", "none"),
        default="none",
        help="Episode media output mode. Direct Python invocation defaults to none.",
    )
    remote_group.add_argument("--remote-video-fps", type=float, default=5.0)
    remote_group.add_argument("--remote-video-codec", type=str, default="h264")
    remote_group.add_argument(
        "--action-format",
        type=str,
        default="auto",
        choices=["auto", "discrete", "nav_delta"],
        help="How to interpret remote actions. auto supports action_type, action.nav_delta, and integer action lists.",
    )
    remote_group.add_argument(
        "--remote-metadata-timeout",
        type=float,
        default=2.0,
        help="Seconds to wait for optional server metadata after WebSocket connect.",
    )
    remote_group.add_argument("--remote-send-state", action="store_true")
    remote_group.add_argument(
        "--wam-trace-mode",
        type=str,
        default="none",
        choices=("none", "sidecar"),
        help="Opt in to canonical server-side WAM trace sidecars for later offline training.",
    )
    remote_group.add_argument(
        "--wam-rollout-base-seed",
        type=int,
        default=0,
        help="Stable base seed used with behavior policy version, episode, and query id.",
    )

    dagger_group = parser.add_argument_group("DAgger")
    dagger_group.add_argument("--dagger", action="store_true")
    dagger_group.add_argument(
        "--dagger-output-root",
        type=str,
        default=None,
        help="InteriorGS LeRobot DAgger dataset root. Required when --dagger is set.",
    )
    dagger_group.add_argument("--dagger-future-actions", type=int, default=32)
    dagger_group.add_argument(
        "--dagger-history-sampling",
        type=str,
        default="uniform",
        choices=("recent", "uniform"),
        help="History frames saved in DAgger episodes. Defaults to uniform to match server history.",
    )
    dagger_group.add_argument(
        "--dagger-history-end",
        type=str,
        default="current",
        choices=("current", "previous"),
        help=(
            "Newest frame eligible for DAgger history sampling. Use previous for "
            "2D strict-past history and current for VGGT."
        ),
    )
    dagger_group.add_argument(
        "--dagger-stop-radius",
        type=float,
        default=DEFAULT_DAGGER_STOP_RADIUS_M,
        help="Distance-to-goal radius where DAgger oracle should STOP/freeze.",
    )
    dagger_group.add_argument(
        "--dagger-rescue-horizon",
        type=int,
        default=8,
        help="Number of A* teacher nav_delta actions to queue for takeover rescue.",
    )
    dagger_group.add_argument(
        "--dagger-stop-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace model STOP outside the DAgger stop radius (default: enabled).",
    )
    dagger_group.add_argument(
        "--dagger-stuck-rescue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use A* teacher rescue when a DAgger rollout stalls (default: enabled).",
    )

    args = parser.parse_args()
    args.remote_save_images = args.save_media == "images"
    args.remote_save_video = args.save_media == "video"
    del args.save_media

    if args.dagger and not str(args.dagger_output_root or "").strip():
        parser.error("--dagger-output-root is required when --dagger is set")
    if args.max_actions_per_query <= 0:
        parser.error("--max-actions-per-query must be positive")
    if args.start_idx < 0:
        parser.error("--start-idx must be non-negative")
    if (
        args.end_idx is not None
        and args.end_idx != -1
        and args.end_idx < args.start_idx
    ):
        parser.error("--end-idx must be -1 or greater than or equal to --start-idx")
    if args.stall_patience is not None and args.stall_patience < 0:
        parser.error("--stall-patience must be non-negative")
    for option, value in (
        ("--remote-request-timeout", args.remote_request_timeout),
        ("--remote-connect-timeout", args.remote_connect_timeout),
        ("--remote-metadata-timeout", args.remote_metadata_timeout),
        ("--remote-video-fps", args.remote_video_fps),
    ):
        if value <= 0:
            parser.error(f"{option} must be positive")
    for option, value in (
        ("--remote-stop-eps", args.remote_stop_eps),
        ("--remote-min-chunk-translation", args.remote_min_chunk_translation),
        ("--remote-max-translation", args.remote_max_translation),
        ("--remote-max-yaw", args.remote_max_yaw),
        ("--dagger-stop-radius", args.dagger_stop_radius),
    ):
        if value < 0:
            parser.error(f"{option} must be non-negative")
    return args


def load_dataset_split(config):
    dataset = make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(DATASET_SEED)
    return dataset.get_splits(1)[0]


def build_agent(
    result_dir,
    max_actions_per_query,
    server_host,
    server_port,
    remote_request_timeout,
    remote_connect_timeout,
    remote_stop_eps,
    remote_min_chunk_translation,
    remote_translation_frame,
    remote_max_translation,
    remote_max_yaw,
    remote_save_images,
    remote_save_video,
    remote_video_fps,
    remote_video_codec,
    action_format,
    remote_metadata_timeout,
    remote_send_state,
    wam_trace_mode,
    wam_rollout_base_seed,
    dagger,
    dagger_output_root,
    dagger_worker_id,
    dagger_future_actions,
    dagger_history_sampling,
    dagger_history_end,
    dagger_stop_radius,
    dagger_rescue_horizon,
    dagger_stop_override,
    dagger_stuck_rescue,
):
    from gn0.agents.remote import RemoteNavAgent

    return RemoteNavAgent(
        model_path=None,
        result_path=result_dir,
        action_num=max_actions_per_query,
        server_host=server_host,
        server_port=server_port,
        request_timeout=remote_request_timeout,
        connect_timeout=remote_connect_timeout,
        metadata_timeout=remote_metadata_timeout,
        stop_eps=remote_stop_eps,
        chunk_stop_translation=remote_min_chunk_translation,
        translation_frame=remote_translation_frame,
        max_translation=remote_max_translation,
        max_yaw=remote_max_yaw,
        save_images=remote_save_images,
        save_video=remote_save_video,
        video_fps=remote_video_fps,
        video_codec=remote_video_codec,
        action_format=action_format,
        send_state=remote_send_state,
        wam_trace_mode=wam_trace_mode,
        wam_rollout_base_seed=wam_rollout_base_seed,
        dagger=dagger,
        dagger_output_root=dagger_output_root,
        dagger_worker_id=dagger_worker_id,
        dagger_future_actions=dagger_future_actions,
        dagger_history_sampling=dagger_history_sampling,
        dagger_history_end=dagger_history_end,
        dagger_stop_radius=dagger_stop_radius,
        dagger_rescue_horizon=dagger_rescue_horizon,
        dagger_stop_override=dagger_stop_override,
        dagger_stuck_rescue=dagger_stuck_rescue,
    )


def run_exp(
    exp_config,
    split_num,
    split_id,
    start_idx=0,
    end_idx=None,
    stall_patience=None,
    **kwargs,
):
    validate_split_args(split_num, split_id)

    config_opts = None
    if stall_patience is not None:
        config_opts = ["EVAL.STALL_PATIENCE", int(stall_patience)]
    config = get_config(exp_config, opts=config_opts)
    log_dir = make_log_dir(kwargs["result_dir"])
    dataset_split = load_dataset_split(config)

    global_episodes, resolved_end_idx, skipped, finished_count = select_eval_episodes(
        dataset_split, log_dir, start_idx, end_idx
    )

    if split_id == 0:
        print(
            f"Global Eval Range: [{start_idx}, {resolved_end_idx}] "
            f"(pending: {len(global_episodes)}, skipped in range: {skipped}, "
            f"existing complete logs: {finished_count})"
        )

    dataset_split.episodes = (
        shard_episodes(global_episodes, split_num, split_id) if global_episodes else []
    )

    print(
        f"[Split {split_id}/{split_num}] Assigned {len(dataset_split.episodes)} episodes."
    )

    if not dataset_split.episodes:
        print(f"[Split {split_id}/{split_num}] No episodes to process.")
        return

    evaluate_agent(config, split_id, dataset_split, log_dir, **kwargs)


def evaluate_agent(
    config,
    split_id,
    dataset,
    log_dir,
    result_dir,
    max_actions_per_query,
    server_host,
    server_port,
    remote_request_timeout,
    remote_connect_timeout,
    remote_stop_eps,
    remote_min_chunk_translation,
    remote_translation_frame,
    remote_max_translation,
    remote_max_yaw,
    remote_save_images,
    remote_save_video,
    remote_video_fps,
    remote_video_codec,
    action_format,
    remote_metadata_timeout,
    remote_send_state,
    wam_trace_mode,
    wam_rollout_base_seed,
    dagger,
    dagger_output_root,
    dagger_future_actions,
    dagger_history_sampling,
    dagger_history_end,
    dagger_stop_radius,
    dagger_rescue_horizon,
    dagger_stop_override,
    dagger_stuck_rescue,
    enable_stall_recovery,
    **kwargs,
) -> None:
    del kwargs
    env = Env(config.TASK_CONFIG, dataset)
    agent = build_agent(
        result_dir=result_dir,
        max_actions_per_query=max_actions_per_query,
        server_host=server_host,
        server_port=server_port,
        remote_request_timeout=remote_request_timeout,
        remote_connect_timeout=remote_connect_timeout,
        remote_stop_eps=remote_stop_eps,
        remote_min_chunk_translation=remote_min_chunk_translation,
        remote_translation_frame=remote_translation_frame,
        remote_max_translation=remote_max_translation,
        remote_max_yaw=remote_max_yaw,
        remote_save_images=remote_save_images,
        remote_save_video=remote_save_video,
        remote_video_fps=remote_video_fps,
        remote_video_codec=remote_video_codec,
        action_format=action_format,
        remote_metadata_timeout=remote_metadata_timeout,
        remote_send_state=remote_send_state,
        wam_trace_mode=wam_trace_mode,
        wam_rollout_base_seed=wam_rollout_base_seed,
        dagger=dagger,
        dagger_output_root=dagger_output_root,
        dagger_worker_id=split_id,
        dagger_future_actions=dagger_future_actions,
        dagger_history_sampling=dagger_history_sampling,
        dagger_history_end=dagger_history_end,
        dagger_stop_radius=dagger_stop_radius,
        dagger_rescue_horizon=dagger_rescue_horizon,
        dagger_stop_override=dagger_stop_override,
        dagger_stuck_rescue=dagger_stuck_rescue,
    )

    desc = f"{config.EVAL.IDENTIFICATION}-{split_id}"
    try:
        for _ in trange(len(env.episodes), desc=desc):
            episode_completed = False
            episode_manifest_ref = None
            final_metrics = None
            stop_reason = None
            try:
                stop_diagnostics = {}
                stop_reason = run_episode(
                    env,
                    agent,
                    config,
                    enable_stall_recovery=enable_stall_recovery,
                    stop_diagnostics=stop_diagnostics,
                )
                final_metrics = env.get_metrics()
                episode_completed = True
            finally:
                close_episode = getattr(agent, "close_episode", None)
                if callable(close_episode):
                    episode_manifest_ref = close_episode(
                        complete=episode_completed,
                        final_metrics=final_metrics,
                    )
            save_episode_metrics(
                env,
                log_dir,
                final_info=final_metrics,
                wam_episode_manifest=episode_manifest_ref,
                stop_reason=stop_reason,
                stop_diagnostics=stop_diagnostics,
                execution_stats={
                    "environment_steps": agent.executed_steps,
                    "motion_steps": agent.executed_motion_steps,
                    "stop_steps": agent.executed_steps - agent.executed_motion_steps,
                    "query_count": agent.query_count,
                },
            )
    finally:
        close_agent = getattr(agent, "close", None)
        if callable(close_agent):
            close_agent()


def run_episode(
    env: Env,
    agent,
    config,
    *,
    enable_stall_recovery: bool = False,
    stop_diagnostics: dict[str, Any] | None = None,
) -> str:
    obs = env.reset()
    agent.reset(Path(env.current_episode.ref_json), sim=env.sim)

    success_distance = config.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE
    reached_goal = env.get_metrics()["distance_to_goal"] < success_distance
    initial_false_stop = False
    goal_nonstop_steps = 0

    last_dtg = float("inf")
    stall_steps = 0
    stop_reason: str | None = None
    recovery_attempts = 0
    recovery_steps_remaining = 0

    for step in range(config.EVAL.EARLY_STOP_STEPS + 1):
        if env.episode_over:
            if stop_reason is None:
                stop_reason = "environment_done"
            break

        info = env.get_metrics()
        curr_dtg = info.get("distance_to_goal", 0)

        obs.update(
            {"sim": env.sim, "goal_position": env.current_episode.goals[0].position}
        )

        if recovery_steps_remaining > 0:
            action = agent.make_stall_recovery_action(
                obs,
                env.sim,
                STALL_RECOVERY_BACKTRACK_METERS,
            )
            action_source = "stall_recovery"
            recovery_steps_remaining -= 1
            stall_steps = 0
            last_dtg = float("inf") if recovery_steps_remaining == 0 else curr_dtg
        else:
            stall_steps = 0 if curr_dtg != last_dtg else stall_steps + 1
            last_dtg = curr_dtg

            if stall_steps > config.EVAL.STALL_PATIENCE:
                if (
                    enable_stall_recovery
                    and recovery_attempts < STALL_RECOVERY_MAX_ATTEMPTS
                ):
                    recovery_attempts += 1
                    recovery_steps_remaining = STALL_RECOVERY_STEPS - 1
                    stall_steps = 0
                    LOGGER.warning(
                        "[Stall Recovery] attempt=%d/%d step=%d; backing up %d x %.2fm",
                        recovery_attempts,
                        STALL_RECOVERY_MAX_ATTEMPTS,
                        step,
                        STALL_RECOVERY_STEPS,
                        STALL_RECOVERY_BACKTRACK_METERS,
                    )
                    action = agent.make_stall_recovery_action(
                        obs,
                        env.sim,
                        STALL_RECOVERY_BACKTRACK_METERS,
                    )
                    action_source = "stall_recovery"
                else:
                    LOGGER.warning("[Early Stop] Stall threshold at step %s", step)
                    action = {"action": STOP_ACTION_ID}
                    action_source = "early_stop"
                    stop_reason = STOP_REASON_DISTANCE_TO_GOAL_UNCHANGED
            else:
                action = agent.act(obs, info)
                action_source = None
                agent_stop_reason = getattr(agent, "episode_stop_reason", None)
                if agent_stop_reason is not None:
                    stop_reason = str(agent_stop_reason)

        # 使用官方成功半径；恢复/强制截断的 STOP 不计作模型起点误停。
        in_goal = info["distance_to_goal"] < success_distance
        is_stop = action["action"] == STOP_ACTION_ID
        if step == 0 and is_stop and action_source is None:
            initial_false_stop = not in_goal and not str(stop_reason).startswith("dagger")
        goal_nonstop_steps += int(in_goal and not is_stop)
        obs = env.step(action)
        after_info = env.get_metrics()
        reached_goal |= after_info["distance_to_goal"] < success_distance
        record_step_result = getattr(agent, "record_step_result", None)
        if callable(record_step_result):
            record_step_result(
                env_action=action,
                before_metrics=info,
                after_metrics=after_info,
                episode_over=env.episode_over,
                action_source=action_source,
            )

    if stop_diagnostics is not None:
        stop_diagnostics.update(
            initial_false_stop=initial_false_stop,
            reached_goal=bool(reached_goal),
            goal_nonstop_steps=goal_nonstop_steps,
            # 到达过成功半径但未成功结束，包括离开目标后才停止。
            missed_goal_stop=bool(reached_goal and not env.get_metrics()["success"]),
            action_residual=agent.stop_action_residual,
        )

    if stop_reason is not None:
        return stop_reason
    if env.episode_over:
        return "environment_done"
    return "evaluation_step_limit"


def save_episode_metrics(
    env: Env,
    log_dir: Path,
    *,
    final_info: dict[str, Any] | None = None,
    wam_episode_manifest: dict[str, Any] | None = None,
    stop_reason: str | None = None,
    execution_stats: dict[str, int] | None = None,
    stop_diagnostics: dict[str, Any] | None = None,
) -> None:
    if final_info is None:
        final_info = env.get_metrics()
    results = {k: final_info[k] for k in TARGET_METRICS if k in final_info}
    results["id"] = get_episode_id(env.current_episode.ref_json)
    results["status"] = "complete"
    results["episode_complete"] = True
    results["stop"] = str(stop_reason or "unknown")
    if stop_diagnostics is not None:
        results["stop_diagnostics"] = stop_diagnostics
    if execution_stats is not None:
        results["execution"] = execution_stats
    if wam_episode_manifest is not None:
        if wam_episode_manifest.get("status") != "complete":
            raise RuntimeError(
                "Final metrics may only reference a complete GN0 WAM episode manifest"
            )
        results["wam_episode_manifest"] = wam_episode_manifest

    save_path = log_dir / f"{results['id']}.json"
    atomic_write_json(save_path, results)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    run_exp(**vars(args))


if __name__ == "__main__":
    main()

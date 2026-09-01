#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

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
    resolve_end_idx,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GN_Bench Navigation Evaluation")

    io_group = parser.add_argument_group("Paths & IO")
    io_group.add_argument(
        "--exp-config", type=str, required=True, help="Config YAML path"
    )
    io_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model weights location (required for evaluation, unused for rendering)",
    )
    io_group.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Evaluation results directory",
    )

    model_group = parser.add_argument_group("Model & Task")
    model_group.add_argument("--model-name", type=str, default="bae")
    model_group.add_argument(
        "--prompt-type", type=str, default="V3", choices=["V1", "V2", "V3"]
    )
    model_group.add_argument(
        "--max-actions-per-query",
        type=int,
        default=1,
        help="Maximum number of predicted actions to execute before the next query",
    )
    model_group.add_argument("--split-num", type=int, required=True)
    model_group.add_argument("--split-id", type=int, required=True)

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--dagger", action="store_true")
    eval_group.add_argument(
        "--dagger-stop-radius",
        type=float,
        default=DEFAULT_DAGGER_STOP_RADIUS_M,
        help="Distance-to-goal radius where the BAE DAgger oracle should STOP.",
    )
    eval_group.add_argument("--start-idx", type=int, default=0, help="StartIndex")
    eval_group.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Exclusive end index. Omit to run through the dataset end.",
    )

    renderer_group = parser.add_argument_group("Trajectory Video Renderer")
    renderer_group.add_argument(
        "--render-dataset",
        action="store_true",
        help="Render each smoothed dataset trajectory to <result-dir>/<scene>/<trajectory>.mp4.",
    )
    eval_group.add_argument(
        "--stall-patience",
        type=int,
        default=None,
        help="Consecutive unchanged-distance steps before stopping.",
    )

    media_group = parser.add_argument_group("Media Output")
    media_group.add_argument(
        "--save-media",
        choices=("images", "video", "none"),
        default="none",
        help="Episode media output mode (default: none).",
    )
    media_group.add_argument("--video-fps", type=float, default=5.0)
    media_group.add_argument("--video-codec", type=str, default="h264")

    args = parser.parse_args()
    if not args.render_dataset and not args.model_path:
        parser.error("--model-path is required unless --render-dataset is set")
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
    if args.dagger_stop_radius < 0:
        parser.error("--dagger-stop-radius must be non-negative")
    if args.split_num <= 0:
        parser.error("--split-num must be positive")
    if args.split_id < 0 or args.split_id >= args.split_num:
        parser.error("--split-id must be smaller than --split-num")
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    if args.dagger and args.save_media != "images":
        parser.error("--dagger requires --save-media images")
    return args


def load_dataset_split(config):
    dataset = make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(DATASET_SEED)
    return dataset.get_splits(1)[0]


def select_render_episodes(dataset_split, start_idx: int, end_idx: int | None):
    end_idx = resolve_end_idx(dataset_split.episodes, end_idx)
    return dataset_split.episodes[start_idx:end_idx], end_idx


def resolve_renderer_settings(config):
    renderer_cfg = getattr(config.EVAL, "RENDERER", None)

    def pick(name: str, default):
        if renderer_cfg is not None and name in renderer_cfg:
            return renderer_cfg[name]
        return default

    return {
        "video_fps": float(pick("VIDEO_FPS", 10.0)),
        "video_codec": str(pick("VIDEO_CODEC", "libx264")),
        "smooth_window": int(pick("SMOOTH_WINDOW", 9)),
        "resample_step_m": float(pick("RESAMPLE_STEP_M", 0.05)),
        "heading_lookahead_m": float(pick("HEADING_LOOKAHEAD_M", 0.5)),
        "yaw_smooth_window": int(pick("YAW_SMOOTH_WINDOW", 9)),
    }


def build_agent(
    dagger: bool,
    model_path,
    result_dir,
    prompt_type,
    max_actions_per_query,
    save_media,
    video_fps,
    video_codec,
    dagger_stop_radius,
):
    if dagger:
        from gn0.agents.bae_dagger import BAEAgent
    else:
        from gn0.agents.bae import BAEAgent

    agent_kwargs = {}
    if dagger:
        agent_kwargs["dagger_stop_radius"] = dagger_stop_radius

    return BAEAgent(
        model_path,
        result_dir,
        prompt_type,
        max_actions_per_query,
        save_media=save_media,
        video_fps=video_fps,
        video_codec=video_codec,
        **agent_kwargs,
    )


def run_exp(
    exp_config,
    split_num,
    split_id,
    start_idx=0,
    end_idx=None,
    render_dataset=False,
    stall_patience=None,
    **kwargs,
):
    validate_split_args(split_num, split_id)

    config_opts = None
    if stall_patience is not None:
        config_opts = ["EVAL.STALL_PATIENCE", int(stall_patience)]
    config = get_config(exp_config, opts=config_opts)
    result_dir = kwargs["result_dir"]
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    log_dir = None if render_dataset else make_log_dir(result_dir)
    dataset_split = load_dataset_split(config)

    if render_dataset:
        global_episodes, resolved_end_idx = select_render_episodes(
            dataset_split, start_idx, end_idx
        )
    else:
        global_episodes, resolved_end_idx, skipped, finished_count = (
            select_eval_episodes(dataset_split, log_dir, start_idx, end_idx)
        )

    if split_id == 0:
        mode = "Render" if render_dataset else "Eval"
        if render_dataset:
            detail = f"total target: {len(global_episodes)}"
        else:
            detail = (
                f"pending: {len(global_episodes)}, skipped in range: {skipped}, "
                f"existing complete logs: {finished_count}"
            )
        print(f"Global {mode} Range: [{start_idx}, {resolved_end_idx}] ({detail})")

    dataset_split.episodes = (
        shard_episodes(global_episodes, split_num, split_id) if global_episodes else []
    )

    print(
        f"[Split {split_id}/{split_num}] GPU assigned {len(dataset_split.episodes)} episodes."
    )

    if not dataset_split.episodes:
        print(f"[Split {split_id}/{split_num}] No episodes to process.")
        return

    if render_dataset:
        renderer_settings = resolve_renderer_settings(config)
        render_trajectory_videos(
            config=config,
            split_id=split_id,
            dataset=dataset_split,
            result_dir=result_dir,
            **renderer_settings,
        )
        return

    evaluate_agent(config, split_id, dataset_split, log_dir, **kwargs)


def render_trajectory_videos(
    config,
    split_id,
    dataset,
    result_dir,
    video_fps,
    video_codec,
    smooth_window,
    resample_step_m,
    heading_lookahead_m,
    yaw_smooth_window,
) -> None:
    from gn0.collection import TrajectoryVideoCollector

    env = Env(config.TASK_CONFIG, dataset)
    renderer = TrajectoryVideoCollector(
        result_dir,
        fps=video_fps,
        codec=video_codec,
        smooth_window=smooth_window,
        resample_step_m=resample_step_m,
        heading_lookahead_m=heading_lookahead_m,
        yaw_smooth_window=yaw_smooth_window,
    )
    print(
        f"[Renderer] settings: fps={video_fps}, "
        f"codec={video_codec}, smooth_window={smooth_window}, "
        f"resample_step_m={resample_step_m}, "
        f"heading_lookahead_m={heading_lookahead_m}, "
        f"yaw_smooth_window={yaw_smooth_window}"
    )

    desc = f"{config.EVAL.IDENTIFICATION}-renderer-{split_id}"
    for _ in trange(len(env.episodes), desc=desc):
        env.reset()
        video_path, frame_count = renderer.collect(
            episode=env.current_episode,
            sim=env.sim,
        )
        print(f"[Renderer] {video_path} ({frame_count} frames)")


def evaluate_agent(
    config,
    split_id,
    dataset,
    log_dir,
    model_path,
    result_dir,
    prompt_type,
    max_actions_per_query,
    dagger,
    save_media,
    video_fps,
    video_codec,
    dagger_stop_radius,
    **kwargs,
) -> None:
    env = Env(config.TASK_CONFIG, dataset)
    agent = build_agent(
        dagger,
        model_path,
        result_dir,
        prompt_type,
        max_actions_per_query,
        save_media,
        video_fps,
        video_codec,
        dagger_stop_radius,
    )

    desc = f"{config.EVAL.IDENTIFICATION}-{split_id}"
    try:
        for _ in trange(len(env.episodes), desc=desc):
            run_episode(env, agent, config)
            close_episode = getattr(agent, "close_episode", None)
            if callable(close_episode):
                close_episode()
            save_episode_metrics(env, log_dir)
    finally:
        close_agent = getattr(agent, "close", None)
        if callable(close_agent):
            close_agent()


def run_episode(env: Env, agent, config) -> None:
    obs = env.reset()
    agent.reset(Path(env.current_episode.ref_json), sim=env.sim)

    last_dtg = float("inf")
    stall_steps = 0

    for step in range(config.EVAL.EARLY_STOP_STEPS + 1):
        if env.episode_over:
            break

        info = env.get_metrics()
        curr_dtg = info.get("distance_to_goal", 0)

        stall_steps = 0 if curr_dtg != last_dtg else stall_steps + 1
        last_dtg = curr_dtg

        obs.update(
            {"sim": env.sim, "goal_position": env.current_episode.goals[0].position}
        )

        if stall_steps > config.EVAL.STALL_PATIENCE:
            LOGGER.warning("[Early Stop] Stall patience exceeded at step %s", step)
            action = {"action": STOP_ACTION_ID}
        else:
            action = agent.act(obs, info)

        obs = env.step(action)


def save_episode_metrics(env: Env, log_dir: Path) -> None:
    final_info = env.get_metrics()
    results = {k: final_info[k] for k in TARGET_METRICS if k in final_info}
    results["id"] = get_episode_id(env.current_episode.ref_json)
    results["status"] = "complete"
    results["episode_complete"] = True

    save_path = log_dir / f"{results['id']}.json"
    atomic_write_json(save_path, results)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    run_exp(**vars(args))


if __name__ == "__main__":
    main()

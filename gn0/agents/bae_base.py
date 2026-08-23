from pathlib import Path

import numpy as np
from GN_Bench.core.agent import Agent
from PIL import Image
from scipy.spatial.transform import Rotation as R

from gammanav.vln.model.bae import BAEOnlinePolicy
from gn0.media import EpisodeMediaRecorder

IMAGE_DIRS = {
    "rgb": "model_input/rgb",
    "hist": "model_input/rgb_history",
    "occ": "model_input/occ",
    "occ_traj": "trajectory_vis/occ_executed",
    "bev_traj": "trajectory_vis/bev_executed",
}


class BAEAgentBase(Agent):
    """Shared BAE agent utilities for evaluation and DAgger collection."""

    def __init__(
        self,
        model_path,
        result_path,
        prompt_type,
        action_num=1,
        dtype="bf16",
        save_media="none",
        video_fps=5.0,
        video_codec="h264",
    ):
        print("Initialize BAE")

        self.result_path = result_path
        self.prompt_type = prompt_type
        self.action_num = max(1, int(action_num))
        self.media = EpisodeMediaRecorder(
            result_path,
            mode=save_media,
            fps=video_fps,
            codec=video_codec,
        )

        self.policy = BAEOnlinePolicy(
            model_path=model_path,
            prompt_type=prompt_type,
            action_num=self.action_num,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            max_new_tokens=512,
        )
        # Compatibility attributes used by the DAgger adapter.  Both refer to
        # policy-owned state; no model implementation remains in the GN-Bench layer.
        self.inference = self.policy.inference
        self.history_action_list = self.policy.history_actions

        self.rgb_list = []
        self.episode_id = None
        self.image_path = None
        self.current_step = -1

        print("BAE Initialization Complete")

    def reset_episode_state(self, episode_ref) -> None:
        self.media.close()
        self.rgb_list = []
        self.current_step = -1
        self.episode_id = f"{episode_ref.parent.name}_{episode_ref.stem}"
        self.policy.reset(self.episode_id)

        self.image_path = Path(self.result_path) / self.episode_id / "image"
        self.media.start_episode(self.episode_id)
        if self.media.saves_images:
            self.image_path.mkdir(parents=True, exist_ok=True)

    def save_observation_images(self, sim, rgb):
        occ_map = sim.get_occ_map()
        bev_map = sim.get_bev_map() if self.prompt_type in {"V1", "V2"} else None
        occ_traj = sim.get_occ_map_with_trajectory()
        bev_traj = sim.get_bev_map_with_trajectory()

        occ_h, occ_w = occ_map.shape[:2]
        self.current_step += 1
        self.rgb_list.append(rgb)
        self.rgb_list = self.rgb_list[-17:]
        step = self.current_step

        paths = {
            "rgb": self.model_image_input(rgb, IMAGE_DIRS["rgb"]),
            "occ": self.model_image_input(occ_map, IMAGE_DIRS["occ"]),
            "bev": self.to_pil_image(bev_map) if bev_map is not None else None,
            "occ_traj": None,
            "bev_traj": None,
        }
        if self.media.saves_images:
            paths["occ_traj"] = self.media.record(
                occ_traj, IMAGE_DIRS["occ_traj"], step
            )
        paths["bev_traj"] = self.media.record(bev_traj, IMAGE_DIRS["bev_traj"], step)
        if self.media.saves_video:
            self.media.record(rgb, "rgb", step)
        return paths, occ_h, occ_w

    def build_prompt_image_paths(self, paths):
        if self.prompt_type in {"V1", "V3"}:
            hist = self.build_history_mosaic(
                self.rgb_list[:-1],
                self.history_action_list,
            )
            paths["hist"] = self.model_image_input(hist, IMAGE_DIRS["hist"])

        if self.prompt_type == "V1":
            self.require_bev_input(paths)
            return [paths["rgb"], paths["hist"], paths["bev"], paths["occ"]]
        if self.prompt_type == "V2":
            self.require_bev_input(paths)
            return [paths["bev"], paths["occ"]]
        if self.prompt_type == "V3":
            return [paths["rgb"], paths["hist"]]

        raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

    @staticmethod
    def to_pil_image(img: np.ndarray) -> Image.Image:
        """Convert an RGB numpy image to an in-memory PIL image for model input."""
        return BAEOnlinePolicy.to_pil_image(img)

    @staticmethod
    def require_bev_input(paths):
        if paths.get("bev") is None:
            raise ValueError(
                "Prompt V1/V2 requires a BEV image, but simulator returned none."
            )

    @staticmethod
    def get_current_pixel(sim):
        pixel_pos = sim.get_current_pixel_position()
        if pixel_pos is None or len(pixel_pos) < 2:
            return -1, -1
        return tuple(map(int, pixel_pos[:2]))

    @staticmethod
    def get_agent_pose(sim):
        agent_state = sim.get_agent_state()
        curr_pos = np.array(agent_state.position, dtype=np.float32)
        curr_rot = R.from_quat(agent_state.rotation)
        world_fwd = curr_rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        curr_yaw_deg = float(np.degrees(np.arctan2(world_fwd[1], world_fwd[0])))
        return [float(curr_pos[0]), float(curr_pos[1]), curr_yaw_deg]

    def build_history_mosaic(
        self,
        rgb_list,
        history_action_list,
        grid_size=4,
        target_size=(640, 480),
        only_moved=False,
    ):
        return BAEOnlinePolicy.build_history_mosaic(
            rgb_history=rgb_list,
            action_history=history_action_list,
            grid_size=grid_size,
            target_size=target_size,
            only_moved=only_moved,
        )

    def save_image(self, img: np.ndarray, prefix: str) -> str:
        """Persist an image required by the local DAgger data contract."""
        if img is None:
            raise ValueError(f"Cannot save empty image for prefix '{prefix}'.")
        path = self.media.record(img, prefix, self.current_step)
        if path is None:
            raise RuntimeError("save_image requires media mode 'images'")
        return path

    def model_image_input(self, img: np.ndarray, prefix: str):
        """Return a persisted path in image mode and a PIL image otherwise."""
        if self.media.saves_images:
            return self.save_image(img, prefix)
        return self.to_pil_image(img)

    def record_visualization(self, img: np.ndarray, prefix: str) -> None:
        self.media.record(img, prefix, self.current_step)

    def close_episode(self) -> None:
        self.media.close()

    def close(self) -> None:
        self.close_episode()

    def build_prev_actions_xml(self) -> str:
        return self.policy.build_prev_actions_xml()

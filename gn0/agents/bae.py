from gammanav.vln.model.bae.constants import ACTION_STOP, PIXEL_TO_METER

from .bae_base import BAEAgentBase


class BAEAgent(BAEAgentBase):
    def __init__(
        self,
        model_path,
        result_path,
        prompt_type,
        action_num=1,
        dtype="bf16",
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            result_path=result_path,
            prompt_type=prompt_type,
            action_num=action_num,
            dtype=dtype,
            **kwargs,
        )

    def reset(self, episode_ref, sim=None):
        self.reset_episode_state(episode_ref)
        print("BAE Reset Complete for Episode:", self.episode_id)

    def act(self, observations, info):
        sim = observations.get("sim")
        instruction = observations.get("instruction")["text"]
        rgb = observations.get("rgb")

        paths, occ_h, occ_w = self.save_observation_images(sim, rgb)

        if self.policy.has_pending_actions:
            response = self.policy.infer(
                {"instruction": instruction, "session_id": self.episode_id}
            )
            return {"action": int(response["actions"][0])}

        image_paths = self.build_prompt_image_paths(paths)
        cur_x, cur_y = self.get_current_pixel(sim)
        try:
            response = self.policy.infer(
                {
                    "instruction": instruction,
                    "session_id": self.episode_id,
                    "image_inputs": image_paths,
                    "cur_x": cur_x,
                    "cur_y": cur_y,
                    "occ_w": occ_w,
                    "occ_h": occ_h,
                    "occ_meter_per_px": PIXEL_TO_METER,
                    "occ_rot_deg": 0,
                }
            )
            metadata = response.get("metadata", {})
            predicted_actions = list(metadata.get("predicted_actions", []))
            action = (
                int(response["actions"][0]) if response.get("actions") else ACTION_STOP
            )

            if predicted_actions:
                action_img = sim.get_occ_map_with_actions(
                    predicted_actions, traj_color=(255, 165, 0)
                )
                self.record_visualization(action_img, "trajectory_vis/action_rollout")

            return {"action": action}

        except Exception as exc:
            raise RuntimeError(
                f"BAE inference failed for episode {self.episode_id}"
            ) from exc

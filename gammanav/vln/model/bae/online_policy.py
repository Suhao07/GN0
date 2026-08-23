"""Transport-neutral online policy wrapper for BAE."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image

from .constants import (
    ACTION_MOVE_FORWARD,
    ACTION_STOP,
    PIXEL_TO_METER,
)
from .inference import BAEInference

ACTION_TOKENS = {
    ACTION_STOP: "<STOP>",
    ACTION_MOVE_FORWARD: "<FWD>",
    2: "<LEFT>",
    3: "<RIGHT>",
}
HISTORY_ACTION_LIMIT = 5


class BAEInferenceBackend(Protocol):
    """Subset of :class:`BAEInference` used by the online policy."""

    def predict(self, **kwargs: Any) -> tuple[Any, ...]: ...


class BAEOnlinePolicy:
    """Stateful BAE policy independent of GN-Bench and simulator objects.

    The caller owns environment-specific observation extraction.  In particular,
    GN-Bench adapters may provide RGB/BEV/OCC images as paths or in-memory PIL
    images through ``observations['image_inputs']``.
    """

    def __init__(
        self,
        model_path: str | None = None,
        prompt_type: str = "V3",
        action_num: int = 1,
        dtype: str = "bf16",
        *,
        inference: BAEInferenceBackend | None = None,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        max_new_tokens: int = 512,
    ) -> None:
        if inference is None:
            if not model_path:
                raise ValueError(
                    "model_path is required when inference is not provided"
                )
            inference = BAEInference(
                model_path=model_path,
                prompt_type=prompt_type,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                max_new_tokens=max_new_tokens,
            )

        self.inference = inference
        self.prompt_type = str(prompt_type).upper()
        self.action_num = max(1, int(action_num))
        self.session_id: str | None = None
        self.pending_actions: list[int] = []
        self.history_actions: list[int] = []

    @property
    def has_pending_actions(self) -> bool:
        """Whether the next call can return a cached action without model inference."""

        return bool(self.pending_actions)

    def reset(self, session_id: str | None = None) -> None:
        self.session_id = None if session_id is None else str(session_id)
        self.pending_actions.clear()
        self.history_actions.clear()

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Run one BAE step using the same dict-in/dict-out style as WNM-3D.

        Simulator-specific RGB/BEV/OCC extraction remains in the GN-Bench
        adapter.  This policy only consumes the already prepared model inputs.
        """

        self._reset_for_new_session(observation.get("session_id"))

        if self.pending_actions:
            action = self.pending_actions.pop(0)
            self.history_actions.append(action)
            return self._response(
                action,
                metadata={"from_cache": True, "valid_prediction": True},
            )

        image_inputs = observation.get("image_inputs")
        if image_inputs is None:
            image_inputs = observation.get("image_paths")
        if image_inputs is None:
            raise KeyError("BAE observation requires 'image_inputs'")

        actions, pixels, raw_text, prompt_text = self.inference.predict(
            image_paths=image_inputs,
            instruction=str(observation.get("instruction", "")),
            cur_x=observation.get("cur_x"),
            cur_y=observation.get("cur_y"),
            occ_w=observation.get("occ_w"),
            occ_h=observation.get("occ_h"),
            occ_meter_per_px=float(observation.get("occ_meter_per_px", PIXEL_TO_METER)),
            occ_rot_deg=int(observation.get("occ_rot_deg", 0)),
            prev_actions=self.build_prev_actions_xml(),
        )

        predicted_actions = [int(action) for action in actions] if actions else []
        selected_actions = predicted_actions[: self.action_num]
        metadata = {
            "from_cache": False,
            "valid_prediction": bool(selected_actions),
            "predicted_actions": predicted_actions,
            "pixels": pixels,
            "raw_text": raw_text,
            "prompt_text": prompt_text,
        }

        if not selected_actions:
            return self._response(ACTION_STOP, metadata=metadata)

        self.pending_actions.extend(selected_actions)
        action = self.pending_actions.pop(0)
        self.history_actions.append(action)
        return self._response(action, metadata=metadata)

    @staticmethod
    def _response(action: int, *, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "action_type": "discrete",
            "actions": [int(action)],
            "action": int(action),
            "metadata": metadata,
        }

    def build_prev_actions_xml(self) -> str:
        recent = list(reversed(self.history_actions))[:HISTORY_ACTION_LIMIT]
        tokens = [ACTION_TOKENS.get(int(action), "<None>") for action in recent]
        while len(tokens) < HISTORY_ACTION_LIMIT:
            tokens.append("<None>")
        return "<action>" + ",".join(tokens) + "</action>"

    @staticmethod
    def build_history_mosaic(
        rgb_history: Sequence[np.ndarray],
        action_history: Sequence[int],
        grid_size: int = 4,
        target_size: tuple[int, int] = (640, 480),
        only_moved: bool = False,
    ) -> np.ndarray:
        """Build the BAE newest-to-oldest history mosaic."""

        target_width, target_height = target_size
        num_tiles = grid_size**2
        tile_width = target_width // grid_size
        tile_height = target_height // grid_size

        if only_moved:
            selected_frames = []
            for index, frame in enumerate(rgb_history):
                if index == 0:
                    selected_frames.append(frame)
                    continue
                previous_action_index = index - 1
                if previous_action_index >= len(action_history):
                    continue
                if int(action_history[previous_action_index]) == ACTION_MOVE_FORWARD:
                    selected_frames.append(frame)
        else:
            selected_frames = list(rgb_history)

        recent_frames = selected_frames[-num_tiles:][::-1]
        tiles = [
            cv2.resize(np.asarray(frame), (tile_width, tile_height))
            for frame in recent_frames
        ]
        tiles.extend(
            np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            for _ in range(num_tiles - len(tiles))
        )
        grid = np.asarray(tiles).reshape(
            grid_size,
            grid_size,
            tile_height,
            tile_width,
            3,
        )
        return grid.swapaxes(1, 2).reshape(target_height, target_width, 3)

    @staticmethod
    def to_pil_image(image: np.ndarray) -> Image.Image:
        if image is None:
            raise ValueError("Cannot convert an empty image to PIL")
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="RGB")

    def _reset_for_new_session(self, session_id: str | None) -> None:
        if session_id is None:
            return
        normalized = str(session_id)
        if self.session_id is None:
            self.session_id = normalized
        elif normalized != self.session_id:
            self.reset(normalized)

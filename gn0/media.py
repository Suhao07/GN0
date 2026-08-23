"""Streaming episode media output shared by GN0 agents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MEDIA_MODES = ("images", "video", "none")


class EpisodeMediaRecorder:
    """Write per-step images or streaming MP4 files without buffering episodes."""

    def __init__(
        self,
        result_root: str | Path,
        *,
        mode: str = "none",
        fps: float = 5.0,
        codec: str = "h264",
    ) -> None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in MEDIA_MODES:
            raise ValueError(f"mode must be one of {MEDIA_MODES}, got {mode!r}")
        if float(fps) <= 0:
            raise ValueError("fps must be positive")

        self.result_root = Path(result_root)
        self.mode = normalized_mode
        self.fps = float(fps)
        self.codec = str(codec or "h264").lower()
        self.episode_root: Path | None = None
        self._writers: dict[str, Any] = {}
        self._shapes: dict[str, tuple[int, int]] = {}
        self._tmp_paths: dict[str, Path] = {}
        self._final_paths: dict[str, Path] = {}

    @property
    def saves_images(self) -> bool:
        return self.mode == "images"

    @property
    def saves_video(self) -> bool:
        return self.mode == "video"

    def start_episode(self, episode_id: str) -> None:
        self.close()
        self.episode_root = self.result_root / str(episode_id)

    def record(self, image: np.ndarray, prefix: str, step: int) -> str | None:
        if image is None or self.episode_root is None or self.mode == "none":
            return None
        if self.saves_images:
            return self._save_image(image, prefix, step)
        self._append_video_frame(image, prefix)
        return None

    def close(self) -> None:
        for prefix, writer in list(self._writers.items()):
            if self._is_h264:
                writer.close()
            else:
                writer.release()
            tmp_path = self._tmp_paths.get(prefix)
            final_path = self._final_paths.get(prefix)
            if tmp_path is not None and final_path is not None and tmp_path.exists():
                os.replace(tmp_path, final_path)
        self._writers.clear()
        self._shapes.clear()
        self._tmp_paths.clear()
        self._final_paths.clear()

    def _save_image(self, image: np.ndarray, prefix: str, step: int) -> str:
        assert self.episode_root is not None
        save_dir = self.episode_root / "image" / prefix
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / f"{int(step)}.png"
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[-1] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        elif array.ndim == 3 and array.shape[-1] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
        if not cv2.imwrite(str(file_path), array):
            raise OSError(f"Failed to write image: {file_path}")
        return str(file_path)

    def _append_video_frame(self, image: np.ndarray, prefix: str) -> None:
        assert self.episode_root is not None
        rgb = self._to_rgb_frame(image)
        if self._is_h264:
            rgb = self._pad_to_even(rgb)

        height, width = rgb.shape[:2]
        writer = self._writers.get(prefix)
        if writer is None:
            video_dir = self.episode_root / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            final_path = video_dir / f"{prefix}.mp4"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = final_path.with_name(f".{final_path.stem}.tmp.{os.getpid()}.mp4")
            writer = self._open_writer(tmp_path, width=width, height=height)
            self._writers[prefix] = writer
            self._shapes[prefix] = (height, width)
            self._tmp_paths[prefix] = tmp_path
            self._final_paths[prefix] = final_path
        else:
            target_height, target_width = self._shapes[prefix]
            if (height, width) != (target_height, target_width):
                rgb = cv2.resize(
                    rgb,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )

        if self._is_h264:
            writer.append_data(rgb)
        else:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def _open_writer(self, path: Path, *, width: int, height: int):
        if self._is_h264:
            import imageio.v2 as imageio

            return imageio.get_writer(
                str(path),
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                pixelformat="yuv420p",
            )

        fourcc = self.codec[:4].ljust(4)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {path}")
        return writer

    @property
    def _is_h264(self) -> bool:
        return self.codec in {"h264", "x264", "avc1", "libx264"}

    @staticmethod
    def _to_rgb_frame(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        elif array.ndim == 3 and array.shape[-1] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
        elif array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Unsupported video frame shape: {array.shape}")
        return np.ascontiguousarray(array)

    @staticmethod
    def _pad_to_even(array: np.ndarray) -> np.ndarray:
        height, width = array.shape[:2]
        if height % 2 == 0 and width % 2 == 0:
            return array
        return cv2.copyMakeBorder(
            array,
            0,
            height % 2,
            0,
            width % 2,
            borderType=cv2.BORDER_REPLICATE,
        )

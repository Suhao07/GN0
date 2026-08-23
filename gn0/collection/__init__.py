"""Dataset and rollout collectors for GN0."""

from typing import Any

__all__ = ["InteriorGSDaggerCollector", "TrajectoryVideoCollector"]


def __getattr__(name: str) -> Any:
    """Keep optional media and Parquet dependencies lazy."""
    if name == "InteriorGSDaggerCollector":
        from .dagger import InteriorGSDaggerCollector

        return InteriorGSDaggerCollector
    if name == "TrajectoryVideoCollector":
        from .trajectory_video import TrajectoryVideoCollector

        return TrajectoryVideoCollector
    raise AttributeError(name)

"""Navigation agents that adapt policies or remote services to GN-Bench."""

from typing import Any

__all__ = ["BAEAgent", "RemoteNavAgent"]


def __getattr__(name: str) -> Any:
    """Load heavyweight policy dependencies only when an agent is requested."""
    if name == "BAEAgent":
        from .bae import BAEAgent

        return BAEAgent
    if name == "RemoteNavAgent":
        from .remote import RemoteNavAgent

        return RemoteNavAgent
    raise AttributeError(name)

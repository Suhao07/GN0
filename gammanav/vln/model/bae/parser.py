#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Robust parser for BAE model outputs."""

import json
import re
from typing import List, Optional

_RE_JSON_OBJ = re.compile(r"\{.*?\}", re.DOTALL)
_RE_ACTION_XML = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_RE_PIXEL_TOKEN_PAIR = re.compile(
    r"\[\s*<\s*(\d{1,4})\s*>\s*,\s*<\s*(\d{1,4})\s*>\s*\]"
)

ACTION_TOKEN_TO_ID = {
    "STOP": 0,
    "FWD": 1,
    "LEFT": 2,
    "RIGHT": 3,
}


def _parse_action_token(tok: str) -> Optional[int]:
    t = tok.strip()
    if not t:
        return None

    # Old numeric format: 0,1,2,3
    try:
        x = int(t)
        if 0 <= x <= 3:
            return x
    except Exception:
        pass

    # New token format: <FWD>/<LEFT>/<RIGHT>/<STOP> (or without <>)
    if t.startswith("<") and t.endswith(">"):
        t = t[1:-1]
    t = t.strip().upper()
    return ACTION_TOKEN_TO_ID.get(t)


def parse_actions6(text: str) -> Optional[List[int]]:
    """
    Parse exactly 6 VLN-CE actions from model output.

    Expected format examples:
    - JSON: {"vlnce":"<action><FWD>,<LEFT>,<FWD>,<RIGHT>,<FWD>,<STOP></action>"}
    - XML: <action><FWD>,<LEFT>,<FWD>,<RIGHT>,<FWD>,<STOP></action>
    - Backward-compatible numeric form is also supported.

    Returns:
        List of 6 integers [0-3], or None if parsing fails
    """
    s = text.strip()

    # Try JSON format first
    m = _RE_JSON_OBJ.search(s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("vlnce"), str):
                mm = _RE_ACTION_XML.search(obj["vlnce"])
                if mm:
                    actions = _parse_action_sequence(mm.group(1))
                    if actions is not None:
                        return actions
        except Exception:
            pass

    # Try direct XML format
    mm = _RE_ACTION_XML.search(s)
    if mm:
        try:
            actions = _parse_action_sequence(mm.group(1))
            if actions is not None:
                return actions
        except Exception:
            pass

    return None


def _parse_action_sequence(value: str) -> Optional[List[int]]:
    tokens = value.split(",")
    if len(tokens) != 6:
        return None
    actions = [_parse_action_token(token) for token in tokens]
    if any(action is None for action in actions):
        return None
    return [int(action) for action in actions]


def validate_actions(actions: List[int]) -> bool:
    """Validate an exact six-action VLN-CE prediction."""
    return len(actions) == 6 and all(
        isinstance(action, int) and not isinstance(action, bool) and 0 <= action <= 3
        for action in actions
    )


def parse_pixels(text: str) -> Optional[List[List[int]]]:
    """
    Parse pixel waypoints from model output.

    Expected format:
    - New: {"pixel":"[[<x0>,<y0>],[<x1>,<y1>],...]"}
    - Old: {"pixel":[[x0,y0],[x1,y1],...]}

    Only reads the FIRST JSON object in the text.

    Returns:
        List of [x, y] coordinate pairs, or None if parsing fails
    """
    s = text.strip()

    # Find the first JSON object
    m = _RE_JSON_OBJ.search(s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if not isinstance(obj, dict) or "pixel" not in obj:
                return None

            pixels = obj["pixel"]

            # New token-string format.
            if isinstance(pixels, str):
                result = []
                for x_str, y_str in _RE_PIXEL_TOKEN_PAIR.findall(pixels):
                    xi = int(x_str)
                    yi = int(y_str)
                    if not (0 <= xi <= 999 and 0 <= yi <= 999):
                        return None
                    result.append([xi, yi])
                return result if result else None

            # Old numeric list format (backward compatible).
            if isinstance(pixels, list):
                result = []
                for point in pixels:
                    if not (isinstance(point, list) and len(point) == 2):
                        return None
                    try:
                        x = int(point[0])
                        y = int(point[1])
                    except (TypeError, ValueError, OverflowError):
                        return None
                    if x != point[0] or y != point[1]:
                        return None
                    if not (0 <= x <= 999 and 0 <= y <= 999):
                        return None
                    result.append([x, y])
                return result if result else None
        except Exception:
            pass

    return None

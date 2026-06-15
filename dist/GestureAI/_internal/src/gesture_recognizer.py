"""
Gesture recognition module — finger counting and gesture classification.

Uses the 21 MediaPipe hand landmarks to determine which fingers are
raised and maps common finger combinations to named gestures.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from src.hand_tracker import HandResult

logger = logging.getLogger(__name__)

# ── Landmark indices (MediaPipe convention) ─────────────────────
#
#   Joint name abbreviations:
#     CMC  = Carpometacarpal   (thumb base, lm 1)
#     MCP  = Metacarpophalangeal (knuckle,  lm 2/5/9/13/17)
#     IP   = Interphalangeal   (thumb only, lm 3)
#     PIP  = Proximal IP       (fingers,    lm 6/10/14/18)
#     DIP  = Distal IP         (fingers,    lm 7/11/15/19)
#     TIP  = Fingertip         (lm 4/8/12/16/20)
#
#   Finger:   Thumb  Index  Middle  Ring  Pinky
#   TIP idx:    4      8      12     16     20
#   Key idx:    3      6      10     14     18   ← IP for thumb, PIP for rest
#
# For the thumb we compare the tip against the IP joint (lm 3) because
# the thumb abducts laterally instead of curling vertically.
# For all other fingers we compare the tip against the PIP joint.

_FINGER_TIP_IDS: list[int] = [4,  8,  12, 16, 20]
_FINGER_KEY_IDS: list[int] = [3,  6,  10, 14, 18]  # IP for thumb, PIP for fingers

# Thumb MCP landmark index — used to make Thumbs Down detection
# posture-independent (tip above/below its own MCP, not the wrist).
_THUMB_MCP_ID = 2

# ── Gesture look-up table ───────────────────────────────────────
# Maps a tuple of (thumb, index, middle, ring, pinky) booleans to a
# human-readable gesture name.  Only the most common / useful
# gestures are listed — anything else falls through to "Unknown".
_GESTURE_TABLE: dict[tuple[bool, ...], str] = {
    (False, False, False, False, False): "Fist",
    (True,  False, False, False, False): "Thumbs Up",   # refined below
    (False, True,  False, False, False): "Pointing",
    (False, True,  True,  False, False): "Peace / Victory",
    (False, True,  True,  True,  False): "Three",
    (False, True,  True,  True,  True):  "Four",
    (True,  True,  True,  True,  True):  "Open Palm",
    (True,  False, False, False, True):  "Rock On",
    (False, True,  False, False, True):  "Spider-Man",
    (True,  True,  False, False, False): "Gun / L",
    (True,  True,  False, False, True):  "I Love You",
    (False, False, True,  False, False): "Middle Finger",
    (True,  False, True,  True,  True):  "OK Reverse",
}


class GestureResult(NamedTuple):
    """Recognition output for a single hand."""
    fingers_up: list[bool]   # [thumb, index, middle, ring, pinky]
    finger_count: int         # Number of raised fingers (0–5)
    gesture_name: str         # Human-readable gesture label
    handedness: str           # "Left" or "Right"
    confidence: float         # 0.0–1.0; fraction of fingers in a clear state


def recognize(hand: HandResult) -> GestureResult:
    """
    Analyse a single hand's landmarks and return finger / gesture info.

    Thumb detection
    ~~~~~~~~~~~~~~~
    The thumb abducts laterally, so we compare x-coordinates:
    - **Right** hand (mirrored): tip is "up" when ``tip.x < ip.x``.
    - **Left** hand (mirrored):  tip is "up" when ``tip.x > ip.x``.

    Index → Pinky detection
    ~~~~~~~~~~~~~~~~~~~~~~~
    A finger is "up" when its tip y-coordinate is above (smaller y value)
    its PIP joint — i.e. ``tip.y < pip.y``.

    Thumbs Up / Down disambiguation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Both share the same boolean tuple.  We refine using the thumb-tip y
    relative to the thumb **MCP** (landmark 2) rather than the wrist,
    making the check posture-independent.

    Confidence
    ~~~~~~~~~~
    Estimated as the fraction of fingers whose tip-to-key-joint distance
    is large relative to the hand scale (index MCP → wrist), giving a
    rough measure of how cleanly each finger is extended or curled.

    Args:
        hand: A :class:`HandResult` from the tracker.

    Returns:
        A :class:`GestureResult` with finger states, count, name, and confidence.
    """
    lms = hand.landmarks
    if len(lms) < 21:
        logger.warning(
            "Incomplete landmarks (%d/21) — returning empty result.", len(lms)
        )
        return GestureResult(
            fingers_up=[False] * 5,
            finger_count=0,
            gesture_name="Unknown",
            handedness=hand.handedness,
            confidence=0.0,
        )

    fingers: list[bool] = []

    # ── Thumb (lateral comparison) ──────────────────────────────
    tip = lms[_FINGER_TIP_IDS[0]]
    key = lms[_FINGER_KEY_IDS[0]]   # thumb IP joint (lm 3)
    if hand.handedness == "Right":
        fingers.append(tip.x < key.x)
    else:
        fingers.append(tip.x > key.x)

    # ── Index → Pinky (vertical comparison) ─────────────────────
    for i in range(1, 5):
        tip = lms[_FINGER_TIP_IDS[i]]
        pip = lms[_FINGER_KEY_IDS[i]]
        fingers.append(tip.y < pip.y)

    count = sum(fingers)
    gesture = _GESTURE_TABLE.get(tuple(fingers), "Unknown")

    # ── Distinguish Thumbs Up vs Thumbs Down ────────────────────
    # Compare thumb tip against thumb MCP (lm 2) — not the wrist — so
    # the result is independent of how high the user holds their hand.
    if gesture == "Thumbs Up":
        thumb_tip = lms[_FINGER_TIP_IDS[0]]
        thumb_mcp = lms[_THUMB_MCP_ID]
        if thumb_tip.y > thumb_mcp.y:
            gesture = "Thumbs Down"

    # ── Confidence (hand-scale normalised) ──────────────────────
    # Use the distance from wrist (lm 0) to index MCP (lm 5) as a
    # scale reference so the metric is resolution-independent.
    wrist    = lms[0]
    idx_mcp  = lms[5]
    scale    = max(1, abs(idx_mcp.y - wrist.y))  # avoid division by zero

    clear_count = 0
    for i in range(5):
        tip_lm = lms[_FINGER_TIP_IDS[i]]
        key_lm = lms[_FINGER_KEY_IDS[i]]
        dist = abs(tip_lm.y - key_lm.y) if i > 0 else abs(tip_lm.x - key_lm.x)
        # "Clear" = tip is sufficiently far from its key joint
        if dist > scale * 0.3:
            clear_count += 1
    confidence = round(clear_count / 5, 2)

    return GestureResult(
        fingers_up=fingers,
        finger_count=count,
        gesture_name=gesture,
        handedness=hand.handedness,
        confidence=confidence,
    )

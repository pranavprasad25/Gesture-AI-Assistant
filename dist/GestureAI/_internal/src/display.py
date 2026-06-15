"""
Display utilities — drawing landmarks, connections, and HUD overlays.

Keeps all OpenCV drawing logic out of the main loop.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import (
    ACTION_HUD_COLOR,
    ACTION_HUD_FONT_SCALE,
    ACTION_HUD_POSITION,
    ACTION_HUD_THICKNESS,
    CONNECTION_COLOR,
    CONNECTION_THICKNESS,
    FINGER_COUNT_COLOR,
    FINGER_COUNT_FONT_SCALE,
    FINGER_COUNT_THICKNESS,
    FPS_FONT_SCALE,
    FPS_TEXT_COLOR,
    FPS_TEXT_POSITION,
    FPS_TEXT_THICKNESS,
    GESTURE_FONT_SCALE,
    GESTURE_TEXT_COLOR,
    GESTURE_TEXT_THICKNESS,
    LANDMARK_COLOR,
    LANDMARK_RADIUS,
)
from src.gesture_recognizer import GestureResult
from src.hand_tracker import HandResult

# MediaPipe hand-connection pairs (indices into the 21-landmark set).
# Hardcoded here for compatibility across all MediaPipe versions (0.9–0.10+).
# The 21-landmark hand topology is stable and will not change.
_HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),  (1, 2),   (2, 3),   (3, 4),    # Thumb
    (0, 5),  (5, 6),   (6, 7),   (7, 8),    # Index
    (0, 9),  (9, 10),  (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9),  (9, 13),  (13, 17),            # Palm cross-connections
]

# Pre-compute a font constant once rather than repeating the attribute lookup.
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_landmarks(frame: np.ndarray, hands: list[HandResult]) -> None:
    """
    Draw all 21 landmarks and their connections for every detected hand.

    Args:
        frame:  BGR image to draw on (modified in place).
        hands:  Detection results from :class:`HandTracker`.
    """
    for hand in hands:
        lms = hand.landmarks

        # ── Connections first (so dots sit on top) ───────────
        for start_idx, end_idx in _HAND_CONNECTIONS:
            pt1 = (lms[start_idx].x, lms[start_idx].y)
            pt2 = (lms[end_idx].x,   lms[end_idx].y)
            cv2.line(frame, pt1, pt2, CONNECTION_COLOR, CONNECTION_THICKNESS)

        # ── Landmark dots ────────────────────────────────────
        for lm in lms:
            cv2.circle(
                frame, (lm.x, lm.y), LANDMARK_RADIUS, LANDMARK_COLOR, cv2.FILLED,
            )

        # ── Handedness label near the wrist ──────────────────
        wrist = lms[0]
        cv2.putText(
            frame,
            hand.handedness,
            (wrist.x - 30, wrist.y - 20),
            _FONT,
            0.7,
            LANDMARK_COLOR,
            2,
        )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """
    Render the FPS counter on the frame.

    Args:
        frame: BGR image to draw on (modified in place).
        fps:   Current frames-per-second value.
    """
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        FPS_TEXT_POSITION,
        _FONT,
        FPS_FONT_SCALE,
        FPS_TEXT_COLOR,
        FPS_TEXT_THICKNESS,
    )


def draw_action_hud(frame: np.ndarray, action_label: str) -> None:
    """
    Render the last-triggered action below the FPS counter.

    Args:
        frame:        BGR image to draw on (modified in place).
        action_label: Text label of the last action (e.g. "Vol Up").
    """
    if not action_label:
        return
    cv2.putText(
        frame,
        f"Action: {action_label}",
        ACTION_HUD_POSITION,
        _FONT,
        ACTION_HUD_FONT_SCALE,
        ACTION_HUD_COLOR,
        ACTION_HUD_THICKNESS,
    )


def draw_gesture_info(
    frame: np.ndarray,
    hands: list[HandResult],
    gestures: list[GestureResult],
) -> None:
    """
    Render finger count and gesture name near each detected hand.

    The info is drawn below the wrist landmark so it stays anchored
    to the hand as it moves.

    Args:
        frame:    BGR image to draw on (modified in place).
        hands:    Detection results from :class:`HandTracker`.
        gestures: Recognition results from :func:`gesture_recognizer.recognize`.
    """
    for hand, gesture in zip(hands, gestures):
        wrist = hand.landmarks[0]
        x, y  = wrist.x - 30, wrist.y + 40

        cv2.putText(
            frame,
            f"Fingers: {gesture.finger_count}",
            (x, y),
            _FONT,
            FINGER_COUNT_FONT_SCALE,
            FINGER_COUNT_COLOR,
            FINGER_COUNT_THICKNESS,
        )
        cv2.putText(
            frame,
            gesture.gesture_name,
            (x, y + 30),
            _FONT,
            GESTURE_FONT_SCALE,
            GESTURE_TEXT_COLOR,
            GESTURE_TEXT_THICKNESS,
        )

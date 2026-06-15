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

# ── Debug HUD colours ────────────────────────────────────────────
_DEBUG_BG_COLOR    = (20, 20, 20)       # Near-black panel background
_DEBUG_TITLE_COLOR = (0, 220, 255)      # Amber — panel title
_DEBUG_KEY_COLOR   = (180, 180, 180)    # Light grey — key labels
_DEBUG_VAL_COLOR   = (80, 255, 120)     # Bright green — values
_DEBUG_ACTIVE_COLOR = (0, 100, 255)     # Orange-red — active state highlight

# ── Gesture reference card ───────────────────────────────────────
# Short labels shown in the bottom-left corner so the user always has
# a visible reminder of what each gesture does.
_GESTURE_LEGEND: list[tuple[str, str]] = [
    ("[1]  Pointing",      "Mouse Move"),
    ("[2]  Two Fingers",   "Left Click"),
    ("[3]  Three Fingers", "Scroll"),
    ("[P]  Open Palm",     "Screenshot"),
    ("[F]  Fist",          "Play/Pause"),
    ("[U]  Thumbs Up",     "Vol Up"),
    ("[D]  Thumbs Down",   "Vol Down"),
    ("[4]  Four Fingers",  "Chrome"),
]


def draw_landmarks(frame: np.ndarray, hands: list[HandResult]) -> None:
    """
    Draw all 21 landmarks and their connections for every detected hand.

    Highlights the three landmarks that drive gesture actions with
    distinctly colored rings:
    - lm 8  (index tip)  — yellow  — cursor movement & scroll
    - lm 12 (middle tip) — magenta — two/three-finger detection
    - lm 4  (thumb tip)  — blue    — thumb gestures

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

        # ── Highlight key action landmarks with colored rings ─
        # lm  8 = index tip   → mouse move / scroll reference
        # lm 12 = middle tip  → two-finger / three-finger gestures
        # lm  4 = thumb tip   → thumbs up/down
        for highlight_id, color in [
            (8,  (0,   255, 255)),   # Yellow  — index tip
            (12, (255,   0, 200)),   # Magenta — middle tip
            (4,  (255, 100,   0)),   # Blue    — thumb tip
        ]:
            lm = lms[highlight_id]
            cv2.circle(frame, (lm.x, lm.y), LANDMARK_RADIUS + 3, color, 2)

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
    Render finger count, gesture name, and confidence near each detected hand.

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
        # Confidence indicator — shown in grey, dims when low
        conf_color = (
            _DEBUG_ACTIVE_COLOR if gesture.confidence < 0.6 else (160, 160, 160)
        )
        cv2.putText(
            frame,
            f"conf: {gesture.confidence:.2f}",
            (x, y + 58),
            _FONT,
            0.45,
            conf_color,
            1,
        )


def draw_mouse_debug(
    frame: np.ndarray,
    gesture_name: str,
    debug_info: dict,
) -> None:
    """
    Draw a semi-transparent debug panel in the top-right corner showing
    virtual-mouse internals: gesture name, mouse coords, click state,
    and scroll direction.

    Args:
        frame:        BGR image to draw on (modified in place).
        gesture_name: Current recognised gesture (e.g. "Pointing").
        debug_info:   Dict from :meth:`MouseController.get_debug_info`.
                      Expected keys: ``mouse_x``, ``mouse_y``,
                      ``click_active``, ``scroll_dir``.
    """
    h, w = frame.shape[:2]

    # ── Panel geometry ───────────────────────────────────────────
    panel_w = 290
    panel_h = 150
    margin  = 12
    x0      = w - panel_w - 10
    y0      = 10
    x1      = min(w, x0 + panel_w)
    y1      = min(h, y0 + panel_h)

    # ── Semi-transparent background ──────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _DEBUG_BG_COLOR, cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # ── Border ───────────────────────────────────────────────────
    cv2.rectangle(frame, (x0, y0), (x1, y1), _DEBUG_TITLE_COLOR, 1)

    # ── Title ────────────────────────────────────────────────────
    tx = x0 + margin
    ty = y0 + 22
    cv2.putText(frame, "[ Mouse Debug ]", (tx, ty), _FONT, 0.52,
                _DEBUG_TITLE_COLOR, 1, cv2.LINE_AA)
    cv2.line(frame, (x0 + 4, ty + 5), (x1 - 4, ty + 5), _DEBUG_TITLE_COLOR, 1)

    # ── Data rows ────────────────────────────────────────────────
    click_active: bool = debug_info.get("click_active", False)

    click_val   = "● CLICK" if click_active else "—"
    click_color = _DEBUG_ACTIVE_COLOR if click_active else _DEBUG_VAL_COLOR

    rows: list[tuple[str, str, tuple]] = [
        ("Gesture",  gesture_name,                                              _DEBUG_VAL_COLOR),
        ("Mouse XY", f"({debug_info['mouse_x']}, {debug_info['mouse_y']})",    _DEBUG_VAL_COLOR),
        ("Click",    click_val,                                                 click_color),
        ("Scroll",   debug_info.get("scroll_dir", "—"),                        _DEBUG_VAL_COLOR),
    ]

    row_h = 26
    for i, (key, val, val_color) in enumerate(rows):
        ry = ty + 10 + (i + 1) * row_h
        cv2.putText(frame, f"{key}:", (tx, ry), _FONT, 0.45,
                    _DEBUG_KEY_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, val, (tx + 88, ry), _FONT, 0.45,
                    val_color, 1, cv2.LINE_AA)


def draw_gesture_legend(frame: np.ndarray) -> None:
    """
    Draw a compact gesture reference card in the bottom-left corner.

    Shows all active gesture → action mappings so the user always has
    an on-screen reminder without needing documentation.

    Args:
        frame: BGR image to draw on (modified in place).
    """
    h, w = frame.shape[:2]

    row_h    = 22
    panel_h  = len(_GESTURE_LEGEND) * row_h + 30
    panel_w  = 260
    margin   = 10
    x0       = 10
    y0       = h - panel_h - 10
    x1       = x0 + panel_w
    y1       = y0 + panel_h

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _DEBUG_BG_COLOR, cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), 1)

    # Title
    tx = x0 + margin
    ty = y0 + 18
    cv2.putText(frame, "Gestures", (tx, ty), _FONT, 0.5,
                _DEBUG_TITLE_COLOR, 1, cv2.LINE_AA)
    cv2.line(frame, (x0 + 4, ty + 4), (x1 - 4, ty + 4), (60, 60, 60), 1)

    # Rows — gesture  →  action
    for i, (gesture_label, action_label) in enumerate(_GESTURE_LEGEND):
        ry = ty + 6 + (i + 1) * row_h
        cv2.putText(frame, gesture_label, (tx, ry), _FONT, 0.38,
                    _DEBUG_KEY_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, f"→ {action_label}", (tx + 140, ry), _FONT, 0.38,
                    _DEBUG_VAL_COLOR, 1, cv2.LINE_AA)

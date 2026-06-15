"""
Mouse controller — maps hand landmarks to screen cursor movement.

Uses the index-finger tip (landmark 8) for cursor positioning with
exponential smoothing and a dead-zone to eliminate jitter.

Click is triggered explicitly by calling :meth:`click` — there is no
pinch detection in this module. The caller (actions.py / main.py)
decides when a click gesture has been recognised.

Scroll direction is derived from the raw index-finger tip y-coordinate
supplied by the caller each frame, not from the smoothed screen position.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import pyautogui

from src.config import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MOUSE_DEADZONE,
    MOUSE_FRAME_MARGIN,
    MOUSE_SMOOTHING,
    SCROLL_AMOUNT,
)
from src.hand_tracker import HandResult

logger = logging.getLogger(__name__)

# Disable PyAutoGUI's built-in pause & fail-safe for low-latency control.
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True          # Keep corner fail-safe as a safety net


class MouseController:
    """
    Translates hand landmarks into mouse actions.

    * **Move** — index-finger tip (lm 8) drives the cursor via exponential
      smoothing (``MOUSE_SMOOTHING``) and a pixel dead-zone
      (``MOUSE_DEADZONE``).  Only called when the Pointing gesture is active.
    * **Click** — call :meth:`click` directly; the gesture layer decides
      when a click should fire.  No pinch detection is performed here.
    * **Scroll** — call :meth:`scroll` with the raw index-finger tip y;
      direction is inferred from frame-to-frame delta.

    Args:
        frame_w: Actual camera frame width in pixels. Defaults to the
                 config value if not supplied.
        frame_h: Actual camera frame height in pixels. Defaults to the
                 config value if not supplied.

    Usage::

        mc = MouseController(frame_w=actual_w, frame_h=actual_h)
        # In main loop — only when gesture == "Pointing":
        mc.update(hand_result)
        # When gesture == "Two Fingers" fires a debounced click action:
        mc.click()
        # When gesture == "Three Fingers":
        mc.scroll(hand.landmarks[8].y)
    """

    def __init__(self, frame_w: int = FRAME_WIDTH, frame_h: int = FRAME_HEIGHT) -> None:
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._screen_w, self._screen_h = pyautogui.size()
        self._prev_x: float = self._screen_w / 2
        self._prev_y: float = self._screen_h / 2

        # Timestamp of last click — surfaced to get_debug_info for HUD.
        self._last_click_time: float = 0.0

        # Stores the raw finger y from the previous scroll frame.
        # Reset to None whenever the scroll gesture is released.
        self._prev_scroll_finger_y: Optional[int] = None

        # Debug state — populated each frame for the HUD overlay.
        self._debug_mouse_x: int = 0
        self._debug_mouse_y: int = 0
        self._debug_scroll_dir: str = "—"

        logger.info(
            "MouseController ready — screen %dx%d, frame %dx%d",
            self._screen_w, self._screen_h,
            frame_w, frame_h,
        )

    # ── Public API ──────────────────────────────────────────────

    def update(self, hand: HandResult) -> None:
        """
        Move the cursor using the index-finger tip (landmark 8).

        Should only be called when the hand is showing the **Pointing**
        gesture so that cursor movement does not interfere with other
        gesture actions.

        Args:
            hand: A :class:`HandResult` whose landmark 8 drives the cursor.
        """
        lms = hand.landmarks
        if len(lms) < 21:
            return
        self._move_cursor(lms[8].x, lms[8].y)

    def click(self) -> None:
        """
        Fire a single left click at the current cursor position.

        The caller is responsible for all debouncing and cooldowns.
        This method only executes the PyAutoGUI call.
        """
        try:
            pyautogui.click(_pause=False)
            self._last_click_time = time.perf_counter()
            logger.debug("Click fired at (%d, %d)", self._debug_mouse_x, self._debug_mouse_y)
        except pyautogui.FailSafeException:
            logger.warning("Click blocked — fail-safe triggered.")
        except Exception as exc:
            logger.debug("Click error: %s", exc)

    def scroll(self, finger_y: int) -> None:
        """
        Scroll based on the vertical movement of the index-finger tip.

        Direction is inferred by comparing *finger_y* (raw camera
        pixel coordinate) against the value from the previous call:

        - Hand moves **up**   (finger_y decreases) → scroll **up**.
        - Hand moves **down** (finger_y increases) → scroll **down**.

        Args:
            finger_y: Raw y pixel position of the index-finger tip (lm 8).
        """
        if self._prev_scroll_finger_y is not None:
            delta = self._prev_scroll_finger_y - finger_y   # positive → moved up
            if abs(delta) > 5:
                direction = 1 if delta > 0 else -1
                self._debug_scroll_dir = "↑ Up" if direction > 0 else "↓ Down"
                try:
                    # NOTE: pyautogui.scroll() does NOT accept _pause kwarg.
                    # pyautogui.PAUSE = 0 (set at module level) handles latency.
                    pyautogui.scroll(direction * SCROLL_AMOUNT)
                except pyautogui.FailSafeException:
                    pass
                except Exception as exc:
                    logger.debug("Scroll error: %s", exc)
            else:
                self._debug_scroll_dir = "— Still"
        self._prev_scroll_finger_y = finger_y

    def reset_scroll(self) -> None:
        """Reset scroll baseline (call when leaving the scroll gesture)."""
        self._prev_scroll_finger_y = None
        self._debug_scroll_dir = "—"

    def get_debug_info(self) -> dict:
        """
        Return a snapshot of internal state for the debug HUD overlay.

        Returns:
            A dict with keys:
            ``mouse_x``, ``mouse_y``, ``click_active``, ``scroll_dir``.
            ``click_active`` is True for 300 ms after each click.
        """
        click_active = (time.perf_counter() - self._last_click_time) < 0.3
        return {
            "mouse_x":     self._debug_mouse_x,
            "mouse_y":     self._debug_mouse_y,
            "click_active": click_active,
            "scroll_dir":  self._debug_scroll_dir,
        }

    # ── Internals ───────────────────────────────────────────────

    def _move_cursor(self, finger_x: int, finger_y: int) -> None:
        """Map the index-finger tip position to screen coordinates.

        Uses the *actual* frame dimensions (stored in ``self._frame_w``
        and ``self._frame_h``) rather than the config constants, so the
        mapping is correct even when the camera grants a resolution
        different from the requested one.
        """
        margin = MOUSE_FRAME_MARGIN

        # Clamp to the active region inside the frame margins
        clamped_x = max(margin, min(finger_x, self._frame_w - margin))
        clamped_y = max(margin, min(finger_y, self._frame_h - margin))

        # Normalise to [0, 1] within the active region
        norm_x = (clamped_x - margin) / max(1, self._frame_w  - 2 * margin)
        norm_y = (clamped_y - margin) / max(1, self._frame_h - 2 * margin)

        # Scale to screen
        target_x = norm_x * self._screen_w
        target_y = norm_y * self._screen_h

        # Exponential smoothing
        alpha    = MOUSE_SMOOTHING
        smooth_x = self._prev_x + alpha * (target_x - self._prev_x)
        smooth_y = self._prev_y + alpha * (target_y - self._prev_y)

        # Dead-zone: skip tiny movements to suppress jitter
        dx = smooth_x - self._prev_x
        dy = smooth_y - self._prev_y
        if math.hypot(dx, dy) < MOUSE_DEADZONE:
            return

        self._prev_x = smooth_x
        self._prev_y = smooth_y

        # Store for debug HUD
        self._debug_mouse_x = int(smooth_x)
        self._debug_mouse_y = int(smooth_y)

        try:
            pyautogui.moveTo(int(smooth_x), int(smooth_y), _pause=False)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI fail-safe triggered — cursor near corner.")

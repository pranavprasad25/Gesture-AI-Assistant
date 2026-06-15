"""
Mouse controller — maps hand landmarks to screen cursor movement.

Uses the index-finger tip (landmark 8) for positioning and a
thumb-index pinch gesture for left-click.  Applies exponential
smoothing and a dead-zone to eliminate jitter.

Scroll direction is derived from the raw index-finger tip y-coordinate
supplied by the caller each frame, not from the smoothed screen position.
"""

from __future__ import annotations

import logging
import math
import time

import pyautogui

from src.config import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MOUSE_CLICK_COOLDOWN,
    MOUSE_DEADZONE,
    MOUSE_FRAME_MARGIN,
    MOUSE_PINCH_THRESHOLD,
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

    * **Move** — index-finger tip drives the cursor via exponential
      smoothing (``MOUSE_SMOOTHING``) and a pixel dead-zone
      (``MOUSE_DEADZONE``).
    * **Click** — a thumb-to-index pinch closer than
      ``MOUSE_PINCH_THRESHOLD`` pixels triggers a left click,
      with a cooldown to prevent repeated firing.
    * **Scroll** — call :meth:`scroll` with the raw finger y-coordinate;
      direction is inferred from frame-to-frame delta.

    Usage::

        mc = MouseController()
        mc.update(hand_result)   # call once per frame
    """

    def __init__(self) -> None:
        self._screen_w, self._screen_h = pyautogui.size()
        self._prev_x: float = self._screen_w / 2
        self._prev_y: float = self._screen_h / 2
        self._last_click_time: float = 0.0
        self._was_pinching: bool = False
        # Stores the raw finger y from the previous scroll frame.
        # Reset to None whenever the scroll gesture is released.
        self._prev_scroll_finger_y: int | None = None
        logger.info(
            "MouseController ready — screen %dx%d", self._screen_w, self._screen_h,
        )

    # ── Public API ──────────────────────────────────────────────

    def update(self, hand: HandResult) -> None:
        """
        Process one frame's hand data: move cursor and optionally click.

        Args:
            hand: A :class:`HandResult` whose landmarks drive the mouse.
        """
        lms = hand.landmarks
        if len(lms) < 21:
            return

        self._move_cursor(lms[8].x, lms[8].y)
        self._handle_click(lms[4].x, lms[4].y, lms[8].x, lms[8].y)

    def scroll(self, finger_y: int) -> None:
        """
        Scroll based on the vertical movement of the index-finger tip.

        Direction is inferred by comparing *finger_y* (raw camera
        pixel coordinate) against the value from the previous call:
        - Hand moves **up** (finger_y decreases) → scroll **up**.
        - Hand moves **down** (finger_y increases) → scroll **down**.

        Args:
            finger_y: Raw y pixel position of the index-finger tip (lm 8).
        """
        if self._prev_scroll_finger_y is not None:
            delta = self._prev_scroll_finger_y - finger_y   # positive → moved up
            if abs(delta) > 5:
                direction = 1 if delta > 0 else -1
                try:
                    pyautogui.scroll(direction * SCROLL_AMOUNT, _pause=False)
                except pyautogui.FailSafeException:
                    pass
        self._prev_scroll_finger_y = finger_y

    def reset_scroll(self) -> None:
        """Reset scroll baseline (call when leaving the scroll gesture)."""
        self._prev_scroll_finger_y = None

    # ── Internals ───────────────────────────────────────────────

    def _move_cursor(self, finger_x: int, finger_y: int) -> None:
        """Map the index-finger tip position to screen coordinates."""
        margin = MOUSE_FRAME_MARGIN

        # Clamp to the active region inside the frame margins
        clamped_x = max(margin, min(finger_x, FRAME_WIDTH  - margin))
        clamped_y = max(margin, min(finger_y, FRAME_HEIGHT - margin))

        # Normalise to [0, 1] within the active region
        norm_x = (clamped_x - margin) / (FRAME_WIDTH  - 2 * margin)
        norm_y = (clamped_y - margin) / (FRAME_HEIGHT - 2 * margin)

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

        try:
            pyautogui.moveTo(int(smooth_x), int(smooth_y), _pause=False)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI fail-safe triggered — cursor near corner.")

    def _handle_click(
        self, thumb_x: int, thumb_y: int, index_x: int, index_y: int,
    ) -> None:
        """Detect a thumb-index pinch and fire a left click."""
        distance    = math.hypot(thumb_x - index_x, thumb_y - index_y)
        is_pinching = distance < MOUSE_PINCH_THRESHOLD

        now = time.perf_counter()

        # Trigger on the pinch *onset* (transition from open → pinched)
        if is_pinching and not self._was_pinching:
            if now - self._last_click_time > MOUSE_CLICK_COOLDOWN:
                try:
                    pyautogui.click(_pause=False)
                    logger.debug("Pinch click fired (dist=%.1f)", distance)
                except pyautogui.FailSafeException:
                    logger.warning("Click blocked — fail-safe triggered.")
                self._last_click_time = now

        self._was_pinching = is_pinching

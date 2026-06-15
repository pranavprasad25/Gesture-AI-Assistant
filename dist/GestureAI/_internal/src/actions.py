"""
Gesture-triggered actions — screenshot, media keys, volume, and scroll.

Each action has its own cooldown timer and a global debounce mechanism
that requires a gesture to be held for several consecutive frames before
it fires, preventing accidental triggers during transitions.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Optional

import pyautogui

import src.config as _cfg
from src.gesture_recognizer import GestureResult
from src.mouse_controller import MouseController

logger = logging.getLogger(__name__)

# Voice feedback phrases for each action label
_ACTION_PHRASES: dict[str, str] = {
    "Scroll":     "Scrolling",
    "Screenshot": "Screenshot taken",
    "Play/Pause": "Play pause toggled",
    "Vol Up":     "Volume up",
    "Vol Down":   "Volume down",
}


class ActionDispatcher:
    """
    Maps recognised gestures to system actions with debouncing.

    A gesture must be seen for ``ACTION_DEBOUNCE_FRAMES`` consecutive
    frames before the action fires.  After firing, a per-action cooldown
    prevents the same action from firing again too quickly.

    Usage::

        dispatcher = ActionDispatcher(mouse_controller)
        action = dispatcher.update(gesture)   # call once per frame
    """

    # Gestures that should NOT trigger an action (used for mouse control
    # or intentionally inert).
    _PASSTHROUGH_GESTURES = frozenset({
        "Pointing",       # Mouse movement
        "Unknown",
        "Three",
        "Four",
        "Rock On",
        "Spider-Man",
        "Gun / L",
        "I Love You",
        "Middle Finger",
        "OK Reverse",
    })

    def __init__(
        self,
        mouse: MouseController,
        voice: object | None = None,
    ) -> None:
        self._mouse = mouse
        self._voice = voice               # VoiceAssistant (optional)
        self._gesture_streak: str = ""
        self._streak_count: int = 0
        self._last_action_times: dict[str, float] = {}
        self._last_action_label: str = ""

        # Ensure screenshot directory exists
        os.makedirs(_cfg.SCREENSHOT_DIR, exist_ok=True)
        logger.info(
            "ActionDispatcher ready (debounce=%d frames)",
            _cfg.ACTION_DEBOUNCE_FRAMES,
        )

    # ── Public API ──────────────────────────────────────────────

    @property
    def last_action(self) -> str:
        """The label of the most recently fired action (for HUD display)."""
        return self._last_action_label

    def update(self, gesture: GestureResult) -> Optional[str]:
        """
        Process one frame's gesture and potentially fire an action.

        Args:
            gesture: Current-frame recognition result.

        Returns:
            The name of the fired action, or ``None``.
        """
        name = gesture.gesture_name

        # ── Debounce: count consecutive frames of the same gesture ──
        if name == self._gesture_streak:
            self._streak_count += 1
        else:
            self._gesture_streak = name
            self._streak_count = 1
            # Reset scroll baseline when switching away from scroll gesture
            if name != "Peace / Victory":
                self._mouse.reset_scroll()

        # Not enough consecutive frames yet, or gesture is pass-through
        if self._streak_count < _cfg.ACTION_DEBOUNCE_FRAMES:
            return None
        if name in self._PASSTHROUGH_GESTURES:
            return None

        # ── Per-action cooldown ─────────────────────────────────
        now = time.perf_counter()
        cooldown = _cfg.ACTION_COOLDOWNS.get(name, 1.0)
        last_time = self._last_action_times.get(name, 0.0)
        if now - last_time < cooldown:
            return None

        # ── Dispatch ────────────────────────────────────────────
        action = self._dispatch(name, gesture)
        if action:
            self._last_action_times[name] = now
            self._last_action_label = action
            self._announce(action)
            # Reset streak so the action doesn't re-fire every frame
            # (except Scroll, which should repeat every cooldown interval)
            if name != "Peace / Victory":
                self._streak_count = 0
        return action

    def _announce(self, action: str) -> None:
        """Speak the action label via the voice assistant (if enabled).

        Reads ``VOICE_FEEDBACK_ENABLED`` at call time so runtime changes
        (e.g. toggling via env reload) are respected.
        """
        if not _cfg.VOICE_FEEDBACK_ENABLED or self._voice is None:
            return
        phrase = _ACTION_PHRASES.get(action, action)
        try:
            self._voice.speak(phrase)
        except Exception:
            logger.debug("Voice announcement skipped for '%s'", action)

    # ── Internals ───────────────────────────────────────────────

    def _dispatch(self, name: str, gesture: GestureResult) -> Optional[str]:
        """Route a debounced gesture to the correct system action."""
        try:
            if name == "Peace / Victory":
                return self._scroll(gesture)
            if name == "Open Palm":
                return self._screenshot()
            if name == "Fist":
                return self._play_pause()
            if name == "Thumbs Up":
                return self._volume_up()
            if name == "Thumbs Down":
                return self._volume_down()
        except Exception:
            logger.exception("Action dispatch error for gesture '%s'", name)
        return None

    # ── Action implementations ──────────────────────────────────

    def _scroll(self, gesture: GestureResult) -> str:
        """Scroll using the two-finger (Peace) gesture.

        Passes the raw index-finger tip y-coordinate so the mouse
        controller can derive direction from the true hand movement.
        """
        # Index finger tip = landmark 8; fingers_up list is 0-indexed
        # but we don't have direct access to landmarks here.
        # The mouse controller tracks its own prev_scroll_finger_y via
        # successive calls; we source the y from the gesture's hand via
        # the voice assistant's last known hand — instead, the caller
        # (main loop) passes finger_y directly.  For now we delegate to
        # mouse.scroll() which is called by main with the landmark y.
        #
        # This method is kept for the return label; actual scroll call
        # happens in main.py after update() returns "Scroll".
        return "Scroll"

    def _screenshot(self) -> str:
        """Capture the screen and save to SCREENSHOT_DIR."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_cfg.SCREENSHOT_DIR, f"screenshot_{timestamp}.png")
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(path)
            logger.info("Screenshot saved → %s", path)
        except Exception:
            logger.exception("Screenshot failed")
        return "Screenshot"

    def _play_pause(self) -> str:
        """Send media play/pause key."""
        try:
            pyautogui.press("playpause", _pause=False)
            logger.info("Play/Pause toggled")
        except pyautogui.FailSafeException:
            logger.warning("Play/Pause blocked — fail-safe triggered.")
        return "Play/Pause"

    def _volume_up(self) -> str:
        """Increase system volume."""
        try:
            pyautogui.press("volumeup", _pause=False)
            logger.info("Volume Up")
        except pyautogui.FailSafeException:
            logger.warning("Volume Up blocked — fail-safe triggered.")
        return "Vol Up"

    def _volume_down(self) -> str:
        """Decrease system volume."""
        try:
            pyautogui.press("volumedown", _pause=False)
            logger.info("Volume Down")
        except pyautogui.FailSafeException:
            logger.warning("Volume Down blocked — fail-safe triggered.")
        return "Vol Down"

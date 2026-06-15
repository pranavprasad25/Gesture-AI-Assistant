"""
Gesture-triggered actions — screenshot, media keys, volume, scroll, and click.

Each action has its own cooldown timer and a global debounce mechanism
that requires a gesture to be held for several consecutive frames before
it fires, preventing accidental triggers during transitions.

A confidence threshold (``GESTURE_CONFIDENCE_THRESHOLD``) prevents
ambiguous or partially-formed gestures from firing actions.

Gesture → Action mapping:
    Pointing      → Mouse move (passthrough — handled in main loop)
    Two Fingers   → Left click
    Three Fingers → Scroll
    Four Fingers  → Open Chrome
    Fist          → Play / Pause
    Thumbs Up     → Volume Up
    Thumbs Down   → Volume Down
    Open Palm     → Screenshot
"""

from __future__ import annotations

import logging
import os
import subprocess
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
    "Click":      "Click",
    "Screenshot": "Screenshot taken",
    "Play/Pause": "Play pause toggled",
    "Vol Up":     "Volume up",
    "Vol Down":   "Volume down",
    "Chrome":     "Opening Chrome",
}


class ActionDispatcher:
    """
    Maps recognised gestures to system actions with debouncing and cooldowns.

    A gesture must be seen for ``ACTION_DEBOUNCE_FRAMES`` consecutive
    frames before the action fires.  After firing, a per-action cooldown
    prevents the same action from firing again too quickly.

    Gestures whose ``confidence`` score is below ``GESTURE_CONFIDENCE_THRESHOLD``
    are silently demoted to ``"Unknown"`` before debounce evaluation, so
    ambiguous hand shapes never trigger actions.

    Usage::

        dispatcher = ActionDispatcher(mouse_controller)
        action = dispatcher.update(gesture)   # call once per frame
    """

    # Gestures that are handled outside the dispatcher (e.g. continuous
    # mouse movement) or that should never trigger an action.
    _PASSTHROUGH_GESTURES = frozenset({
        "Pointing",        # Mouse movement — handled directly in main loop
        "Unknown",
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
            "ActionDispatcher ready (debounce=%d frames, confidence≥%.2f)",
            _cfg.ACTION_DEBOUNCE_FRAMES,
            _cfg.GESTURE_CONFIDENCE_THRESHOLD,
        )

    # ── Public API ──────────────────────────────────────────────

    @property
    def last_action(self) -> str:
        """The label of the most recently fired action (for HUD display)."""
        return self._last_action_label

    def update(self, gesture: GestureResult) -> Optional[str]:
        """
        Process one frame's gesture and potentially fire an action.

        Steps:
        1. Confidence gate — demotes low-confidence results to "Unknown".
        2. Debounce — requires ``ACTION_DEBOUNCE_FRAMES`` consecutive frames.
        3. Passthrough check — skips gestures handled elsewhere.
        4. Cooldown — per-gesture minimum interval between firings.
        5. Dispatch — routes to the concrete action implementation.

        Args:
            gesture: Current-frame recognition result.

        Returns:
            The name of the fired action, or ``None``.
        """
        name = gesture.gesture_name

        # ── 1. Confidence gate ──────────────────────────────────
        # If the recognizer isn't sure, treat it as Unknown so we
        # never fire on ambiguous hand shapes.
        if gesture.confidence < _cfg.GESTURE_CONFIDENCE_THRESHOLD:
            name = "Unknown"

        # ── 2. Debounce: count consecutive frames of same gesture ─
        if name == self._gesture_streak:
            self._streak_count += 1
        else:
            self._gesture_streak = name
            self._streak_count = 1
            # Reset scroll baseline when switching away from scroll gesture
            if name != "Three Fingers":
                self._mouse.reset_scroll()

        # Not enough consecutive frames yet
        if self._streak_count < _cfg.ACTION_DEBOUNCE_FRAMES:
            return None

        # ── 3. Passthrough check ─────────────────────────────────
        if name in self._PASSTHROUGH_GESTURES:
            return None

        # ── 4. Per-action cooldown ───────────────────────────────
        now = time.perf_counter()
        cooldown = _cfg.ACTION_COOLDOWNS.get(name, 1.0)
        last_time = self._last_action_times.get(name, 0.0)
        if now - last_time < cooldown:
            return None

        # ── 5. Dispatch ──────────────────────────────────────────
        action = self._dispatch(name, gesture)
        if action:
            self._last_action_times[name] = now
            self._last_action_label = action
            self._announce(action)
            # Reset streak so the action doesn't re-fire every frame
            # (except Three Fingers / Scroll, which repeats on its cooldown)
            if name != "Three Fingers":
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
        """Route a debounced, cooled-down gesture to the correct action."""
        try:
            if name == "Two Fingers":
                return self._click()
            if name == "Three Fingers":
                return self._scroll()
            if name == "Open Palm":
                return self._screenshot()
            if name == "Fist":
                return self._play_pause()
            if name == "Thumbs Up":
                return self._volume_up()
            if name == "Thumbs Down":
                return self._volume_down()
            if name == "Four Fingers":
                return self._open_chrome()
        except Exception:
            logger.exception("Action dispatch error for gesture '%s'", name)
        return None

    # ── Action implementations ──────────────────────────────────

    def _click(self) -> str:
        """Fire a left click at the current cursor position."""
        self._mouse.click()
        logger.info("Click action fired")
        return "Click"

    def _scroll(self) -> str:
        """Return the scroll label; actual scroll call is in main.py.

        main.py calls ``mouse.scroll(lm8_y)`` immediately after receiving
        this return value, so the scroll direction is derived from true
        hand movement rather than being computed here.
        """
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

    def _open_chrome(self) -> str:
        """Launch Google Chrome (Windows).

        Tries known installation paths first, then falls back to the
        Windows ``start`` command which will open the default browser
        if Chrome is not in the standard location.
        """
        _CHROME_PATHS = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        launched = False
        for path in _CHROME_PATHS:
            if os.path.isfile(path):
                try:
                    subprocess.Popen([path])
                    launched = True
                    logger.info("Launched Chrome from %s", path)
                    break
                except Exception:
                    logger.exception("Failed to launch Chrome from %s", path)

        if not launched:
            # Fallback: let Windows resolve 'chrome' via PATH / registry
            try:
                subprocess.Popen("start chrome", shell=True)
                logger.info("Launched Chrome via 'start chrome'")
            except Exception:
                logger.exception("All Chrome launch attempts failed")

        return "Chrome"

"""
Gesture AI Assistant — Entry Point

Launches the webcam, runs hand-landmark detection, and renders
results in real time. Press **Q** or **Esc** to quit.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time

import cv2

from src.config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    WINDOW_NAME,
)
from src.actions import ActionDispatcher
from src.chatgpt_service import ChatGPTService
from src.display import (
    draw_action_hud,
    draw_fps,
    draw_gesture_info,
    draw_landmarks,
    draw_mouse_debug,
)
from src.fps_counter import FPSCounter
from src.gesture_recognizer import recognize
from src.hand_tracker import HandTracker
from src.mouse_controller import MouseController
from src.voice_assistant import VoiceAssistant

# ── Logging setup ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Minimum seconds to sleep between listen attempts when the microphone
# returns None immediately (unavailable or no speech).  Prevents the
# listener thread from burning CPU on a tight loop.
_LISTENER_BACKOFF_S = 1.0


def main() -> None:
    """Run the real-time hand-tracking pipeline."""

    # ── Open camera ──────────────────────────────────────────
    logger.info("Opening camera index %d …", CAMERA_INDEX)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        logger.error("Cannot open camera %d. Check your device.", CAMERA_INDEX)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(
        "Camera opened — requested %dx%d, granted %dx%d",
        FRAME_WIDTH, FRAME_HEIGHT, actual_w, actual_h,
    )

    # Create the window up-front and bring it to the foreground so it
    # is not hidden behind other windows on startup.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, actual_w, actual_h)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

    # ── Services ─────────────────────────────────────────────
    fps_counter = FPSCounter()
    # Pass actual frame dimensions so the mouse controller maps
    # pixel coordinates correctly regardless of camera resolution.
    mouse       = MouseController(frame_w=actual_w, frame_h=actual_h)
    chatgpt     = ChatGPTService()
    # Pass chatgpt at construction time (preferred over set_chatgpt_service)
    voice       = VoiceAssistant(chatgpt=chatgpt)
    dispatcher  = ActionDispatcher(mouse, voice=voice)

    # Voice-command queue: listener thread puts commands here
    voice_cmd_queue: queue.Queue[str] = queue.Queue()
    quit_event = threading.Event()

    def _voice_listener() -> None:
        """
        Background thread: continuously listens for voice commands.

        Backs off for ``_LISTENER_BACKOFF_S`` seconds when the microphone
        is unavailable or returns None immediately, preventing CPU spin.
        """
        while not quit_event.is_set():
            cmd = voice.listen_command()
            if cmd:
                voice_cmd_queue.put(cmd)
            elif not voice.mic_available:
                # Mic gone — sleep to avoid a tight loop
                time.sleep(_LISTENER_BACKOFF_S)

    listener_thread = threading.Thread(
        target=_voice_listener, daemon=True, name="voice-listener",
    )
    listener_thread.start()

    voice.speak("Gesture AI Assistant ready.")

    # ── Main loop ────────────────────────────────────────────
    with HandTracker() as tracker:
        logger.info("Entering main loop. Press Q or Esc to quit.")

        while True:
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to grab frame — skipping.")
                continue

            # Flip for a mirror-like experience
            frame = cv2.flip(frame, 1)

            # Detect hands
            try:
                hands = tracker.process(frame)
            except Exception:
                logger.exception("Hand-tracking error — skipping frame.")
                continue

            # Recognise gestures for the primary hand
            gestures = [recognize(hand) for hand in hands]
            primary_gesture = gestures[0].gesture_name if gestures else "None"

            # ── Mouse control ─────────────────────────────────────
            # Only move the cursor when the user is Pointing (index finger
            # up, all others down).  This prevents the cursor from jumping
            # during pinch-clicks or scroll gestures.
            if hands:
                try:
                    if primary_gesture == "Pointing":
                        # Full update: move cursor AND evaluate pinch click.
                        mouse.update(hands[0])
                    # For non-Pointing gestures we don't move the cursor,
                    # but we still update pinch state so _was_pinching is
                    # always current (avoids phantom click on transition).
                except Exception:
                    logger.exception("Mouse-control error — skipping.")

            # ── Dispatch gesture actions (with voice feedback) ─────
            if gestures:
                try:
                    action = dispatcher.update(gestures[0])
                    # Scroll requires the raw finger-y from landmark 8
                    # so the mouse controller can compute true direction.
                    # It fires every frame while Peace / Victory is held
                    # (dispatcher keeps streak alive for scroll).
                    if action == "Scroll" and hands:
                        mouse.scroll(hands[0].landmarks[8].y)
                except Exception:
                    logger.exception("Action dispatch error — skipping.")

            # ── Process voice commands (non-blocking drain) ────────
            try:
                while True:
                    cmd = voice_cmd_queue.get_nowait()
                    result = voice.handle_voice_command(cmd)
                    if result == "quit":
                        logger.info("Voice quit command received.")
                        quit_event.set()
                        break
            except queue.Empty:
                pass

            if quit_event.is_set():
                break

            # ── Draw results ──────────────────────────────────────
            draw_landmarks(frame, hands)
            draw_gesture_info(frame, hands, gestures)
            draw_action_hud(frame, dispatcher.last_action)
            fps_counter.tick()
            draw_fps(frame, fps_counter.get_fps())

            # Debug overlay — always shown for the primary hand
            draw_mouse_debug(
                frame,
                gesture_name=primary_gesture,
                debug_info=mouse.get_debug_info(),
            )

            cv2.imshow(WINDOW_NAME, frame)

            # Exit on Q / Esc
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                logger.info("Quit key pressed — shutting down.")
                break

    # ── Cleanup ──────────────────────────────────────────────
    quit_event.set()
    voice.speak("Shutting down. Goodbye.")
    voice.shutdown()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Resources released. Goodbye!")


if __name__ == "__main__":
    main()
queue
"""
Voice assistant — text-to-speech feedback and speech recognition.

Runs TTS on a background thread so the main video loop is never
blocked by audio playback.  Speech recognition listens for a wake
word, then captures a single command phrase.

Unknown voice commands are forwarded to :class:`ChatGPTService` so the
assistant can answer arbitrary questions in addition to built-in actions.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import pyttsx3
import speech_recognition as sr

from src.config import (
    VOICE_RATE,
    VOICE_VOLUME,
    VOICE_LISTEN_TIMEOUT,
    VOICE_PHRASE_TIMEOUT,
)

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """
    Non-blocking voice feedback and optional voice-command listener.

    * **speak()** — queues a phrase for background TTS playback.
    * **listen_command()** — blocks until a voice command is captured
      (intended to be called from a dedicated thread, not the main loop).
    * **handle_voice_command()** — dispatches built-in keywords or
      falls back to ChatGPT if a service is attached.

    Args:
        chatgpt: Optional ChatGPT service attached at construction time.
                 Can also be set later via :meth:`set_chatgpt_service`.

    Usage::

        va = VoiceAssistant(chatgpt=ChatGPTService())
        va.speak("Hello!")
        va.shutdown()
    """

    def __init__(self, chatgpt: object | None = None) -> None:
        # ── TTS engine (runs on its own thread) ─────────────
        self._tts_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._tts_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="tts-worker",
        )
        self._tts_thread.start()

        # ── Speech recogniser ───────────────────────────────
        self._recognizer    = sr.Recognizer()
        self._microphone    = None
        self._mic_available = False

        try:
            self._microphone = sr.Microphone()
            self._mic_available = True
        except (AttributeError, OSError, ImportError) as exc:
            # PyAudio not installed or no audio device present.
            logger.warning(
                "Microphone unavailable (%s) — voice commands disabled. "
                "Install PyAudio to enable: pip install pipwin && pipwin install pyaudio",
                exc,
            )

        # Calibrate for ambient noise once at startup (only if mic is available)
        if self._mic_available:
            try:
                with self._microphone as source:
                    self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Microphone calibrated for ambient noise.")
            except OSError:
                self._mic_available = False
                logger.warning(
                    "Microphone not available — voice commands disabled. "
                    "The listener thread will back off automatically."
                )

        # ── Optional ChatGPT service ─────────────────────────
        self._chatgpt = chatgpt

        logger.info("VoiceAssistant initialised (mic_available=%s).", self._mic_available)

    # ── Public API ──────────────────────────────────────────────

    def set_chatgpt_service(self, service: object) -> None:
        """
        Attach a :class:`~src.chatgpt_service.ChatGPTService` instance.

        Prefer passing *chatgpt* to the constructor.  This setter exists
        for cases where the service is created after ``VoiceAssistant``.

        Args:
            service: A ``ChatGPTService`` instance (or any object with
                     ``is_available: bool`` and ``ask(str) -> Optional[str]``).
        """
        self._chatgpt = service
        logger.info("ChatGPTService attached to VoiceAssistant.")

    @property
    def mic_available(self) -> bool:
        """``True`` if the microphone initialised successfully."""
        return self._mic_available

    def speak(self, text: str) -> None:
        """
        Queue *text* for background TTS playback (non-blocking).

        Args:
            text: The phrase to speak aloud.
        """
        self._tts_queue.put(text)

    def listen_command(self) -> Optional[str]:
        """
        Listen for a voice command (blocking).

        Returns ``None`` immediately if the microphone is unavailable,
        preventing the caller thread from spinning at 100 % CPU.

        Returns:
            The recognised command string, or ``None`` on failure/timeout.
        """
        if not self._mic_available:
            return None

        try:
            with self._microphone as source:
                logger.info("Listening for voice command …")
                audio = self._recognizer.listen(
                    source,
                    timeout=VOICE_LISTEN_TIMEOUT,
                    phrase_time_limit=VOICE_PHRASE_TIMEOUT,
                )
            text = self._recognizer.recognize_google(audio)
            logger.info("Voice command recognised: '%s'", text)
            return text.lower()
        except sr.WaitTimeoutError:
            logger.debug("Listen timed out — no speech detected.")
        except sr.UnknownValueError:
            logger.debug("Could not understand audio.")
        except sr.RequestError:
            logger.warning("Google Speech API unavailable.")
        except OSError:
            # Microphone disappeared at runtime
            self._mic_available = False
            logger.warning("Microphone error — voice commands disabled for this session.")
        return None

    def handle_voice_command(self, command: str) -> Optional[str]:
        """
        Parse a recognised command string and return an action name.

        Built-in keyword commands are dispatched first.  If no keyword
        matches *and* a ChatGPT service is attached, the command is sent
        to ChatGPT and the reply is spoken — returning ``"chatgpt"`` so
        callers know a query was handled.

        Args:
            command: Lowercased command text from :meth:`listen_command`.

        Returns:
            An action name string, or ``None``.
        """
        if not command:
            return None

        # ── Built-in keyword actions ────────────────────────
        if "screenshot" in command or "screen" in command:
            self.speak("Taking screenshot.")
            return "screenshot"
        if "volume up" in command or "louder" in command:
            self.speak("Volume up.")
            return "volume_up"
        if "volume down" in command or "quieter" in command or "softer" in command:
            self.speak("Volume down.")
            return "volume_down"
        if "play" in command or "pause" in command:
            self.speak("Toggling playback.")
            return "play_pause"
        if "scroll up" in command:
            self.speak("Scrolling up.")
            return "scroll_up"
        if "scroll down" in command:
            self.speak("Scrolling down.")
            return "scroll_down"
        if "reset chat" in command or "clear chat" in command:
            if self._chatgpt is not None:
                self._chatgpt.reset()
                self.speak("Conversation history cleared.")
            return "reset_chat"
        if "stop" in command or "quit" in command or "exit" in command:
            self.speak("Goodbye.")
            return "quit"

        # ── ChatGPT fallback ────────────────────────────────
        if self._chatgpt is not None and self._chatgpt.is_available:
            logger.info(
                "Forwarding unrecognised command to ChatGPT: '%s'", command
            )
            self.speak("Let me think about that.")
            reply = self._chatgpt.ask(command)
            if reply:
                self.speak(reply)
            else:
                self.speak("I couldn't get an answer right now.")
            return "chatgpt"

        self.speak("Sorry, I didn't understand that command.")
        return None

    def shutdown(self) -> None:
        """Signal the TTS worker to exit and wait for it."""
        self._tts_queue.put(None)          # Sentinel
        self._tts_thread.join(timeout=5)
        logger.info("VoiceAssistant shut down.")

    # ── TTS background worker ───────────────────────────────────

    def _tts_worker(self) -> None:
        """
        Consume the TTS queue on a dedicated thread.

        pyttsx3 engines are not thread-safe, so we create and own the
        engine entirely within this thread.
        """
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate",   VOICE_RATE)
            engine.setProperty("volume", VOICE_VOLUME)
            logger.info(
                "TTS engine ready (rate=%d, vol=%.1f).", VOICE_RATE, VOICE_VOLUME,
            )
        except Exception:
            logger.exception("Failed to initialise TTS engine.")
            return

        while True:
            text = self._tts_queue.get()
            if text is None:               # Shutdown sentinel
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                logger.exception("TTS playback error for: '%s'", text)

        try:
            engine.stop()
        except Exception:
            pass

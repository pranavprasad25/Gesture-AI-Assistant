"""
Configuration constants for the Gesture AI Assistant.

Centralizes all tunable parameters so they can be adjusted
without modifying application logic.

Environment variables (loaded from .env) take precedence over
the hard-coded defaults below.
"""

import os

from dotenv import load_dotenv

# Load .env at import time so every module picks up the values.
# This must happen before any os.getenv() call below.
load_dotenv()


# ── Camera Settings ──────────────────────────────────────────────
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_WIDTH  = int(os.getenv("FRAME_WIDTH",  "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))

# ── MediaPipe Hand Tracking (Tasks API) ─────────────────────────
MAX_NUM_HANDS                  = 2
MIN_DETECTION_CONFIDENCE       = 0.7
MIN_HAND_PRESENCE_CONFIDENCE   = 0.7
MIN_TRACKING_CONFIDENCE        = 0.5

# Path to the downloaded hand_landmarker.task model file.
# Override via env: HAND_LANDMARKER_MODEL=path/to/hand_landmarker.task
HAND_LANDMARKER_MODEL = os.getenv(
    "HAND_LANDMARKER_MODEL",
    "models/hand_landmarker.task",
)

# ── Display / Drawing ───────────────────────────────────────────
LANDMARK_COLOR       = (0, 255, 128)       # BGR — bright green
LANDMARK_RADIUS      = 5
CONNECTION_COLOR     = (255, 180, 0)       # BGR — cyan-blue
CONNECTION_THICKNESS = 2

FPS_TEXT_POSITION  = (20, 45)
FPS_FONT_SCALE     = 1.1
FPS_TEXT_COLOR     = (0, 255, 255)         # BGR — yellow
FPS_TEXT_THICKNESS = 2

GESTURE_TEXT_COLOR      = (128, 255, 0)   # BGR — lime green
GESTURE_FONT_SCALE      = 0.9
GESTURE_TEXT_THICKNESS  = 2
FINGER_COUNT_COLOR      = (255, 100, 255) # BGR — pink-magenta
FINGER_COUNT_FONT_SCALE = 0.9
FINGER_COUNT_THICKNESS  = 2

# ── Mouse Control ───────────────────────────────────────────────
MOUSE_SMOOTHING    = 0.35              # EMA alpha (0=frozen, 1=raw)
MOUSE_DEADZONE     = 3                 # Pixels — ignore moves smaller than this
MOUSE_FRAME_MARGIN = 100              # Pixels — dead border around frame edge

# ── Gesture Recognition ─────────────────────────────────────────
# Minimum recognizer confidence (0.0–1.0) below which a gesture is
# treated as "Unknown" regardless of the finger pattern.
GESTURE_CONFIDENCE_THRESHOLD = 0.6

# ── Gesture Actions ─────────────────────────────────────────────
# A gesture must appear for this many consecutive frames before its
# action fires.  Balances responsiveness vs. false-trigger resistance.
ACTION_DEBOUNCE_FRAMES = 5               # Consecutive frames before action fires
SCROLL_AMOUNT          = 5               # Lines per scroll tick
SCREENSHOT_DIR         = "screenshots"   # Folder for saved screenshots

# Per-gesture cooldowns (seconds) — prevents rapid re-firing.
# Gesture names must match the keys in gesture_recognizer._GESTURE_TABLE.
ACTION_COOLDOWNS: dict[str, float] = {
    "Three Fingers":  0.12,   # Scroll — fast repeat keeps it smooth
    "Two Fingers":    0.5,    # Click  — half-second between clicks
    "Open Palm":      2.0,    # Screenshot — long cooldown to avoid spam
    "Fist":           1.5,    # Play/Pause
    "Thumbs Up":      0.4,    # Volume Up
    "Thumbs Down":    0.4,    # Volume Down
    "Four Fingers":   3.0,    # Open Chrome — very long to prevent accidents
}

# ── Display / HUD ───────────────────────────────────────────────
ACTION_HUD_COLOR      = (0, 200, 255)    # BGR — orange
ACTION_HUD_FONT_SCALE = 0.85
ACTION_HUD_THICKNESS  = 2
ACTION_HUD_POSITION   = (20, 80)         # Below FPS counter

# ── Voice Assistant ─────────────────────────────────────────────
VOICE_RATE             = 175             # Words per minute for TTS
VOICE_VOLUME           = 0.9            # TTS volume (0.0 – 1.0)
VOICE_WAKE_WORD        = "assistant"    # Say this to activate listening
VOICE_LISTEN_TIMEOUT   = 5             # Seconds to wait for speech
VOICE_PHRASE_TIMEOUT   = 5            # Max seconds for a single phrase
VOICE_FEEDBACK_ENABLED = True          # Announce gesture actions via TTS

# ── OpenAI / ChatGPT ────────────────────────────────────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
CHATGPT_MODEL       = os.getenv("CHATGPT_MODEL",       "gpt-4o-mini")
CHATGPT_MAX_TOKENS  = int(os.getenv("CHATGPT_MAX_TOKENS",  "256"))
CHATGPT_TEMPERATURE = float(os.getenv("CHATGPT_TEMPERATURE", "0.7"))
# Maximum number of past user/assistant *turn pairs* kept in memory
CHATGPT_MAX_HISTORY = int(os.getenv("CHATGPT_MAX_HISTORY", "10"))

# System prompt sent to ChatGPT on every request
CHATGPT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant embedded in a gesture-controlled desktop "
    "application. Keep answers concise (1-3 sentences) and spoken-word friendly "
    "since your response will be read aloud via text-to-speech."
)

WINDOW_NAME = "Gesture AI Assistant"

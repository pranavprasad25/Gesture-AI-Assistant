"""
Hand tracking module powered by MediaPipe Tasks API (mediapipe >= 0.10.30).

Wraps the MediaPipe HandLandmarker solution behind a clean interface so
the rest of the application never imports mediapipe directly.

The model file ``models/hand_landmarker.task`` is required at runtime.
It is downloaded automatically by ``scripts/download_models.py``.
"""

from __future__ import annotations

import os

# Suppress MediaPipe / TFLite internal C++ log noise before the import.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import logging
import time
from typing import Any, NamedTuple

import cv2
import mediapipe as mp
import numpy as np

from src.config import (
    HAND_LANDMARKER_MODEL,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)

logger = logging.getLogger(__name__)

# Shorthand aliases for the Tasks API
_vision      = mp.tasks.vision
_BaseOptions = mp.tasks.BaseOptions
_RunningMode = _vision.RunningMode


class Landmark(NamedTuple):
    """A single hand landmark in pixel coordinates."""
    id: int
    x: int
    y: int
    z: float   # relative depth (normalized, from MediaPipe)


class HandResult(NamedTuple):
    """Detection result for one hand."""
    landmarks: list[Landmark]
    handedness: str   # "Left" or "Right"


class HandTracker:
    """
    Real-time hand-landmark detector using MediaPipe Tasks HandLandmarker.

    Uses ``RunningMode.VIDEO`` for synchronous per-frame detection with
    inter-frame tracking — no callbacks required.

    Usage::

        with HandTracker() as tracker:
            results = tracker.process(frame)   # list[HandResult]
    """

    def __init__(
        self,
        model_path: str = HAND_LANDMARKER_MODEL,
        max_hands: int = MAX_NUM_HANDS,
        detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        presence_confidence: float = MIN_HAND_PRESENCE_CONFIDENCE,
        tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        logger.info(
            "Initialising HandTracker "
            "(model=%s, max_hands=%d, det=%.2f, presence=%.2f, track=%.2f)",
            model_path, max_hands,
            detection_confidence, presence_confidence, tracking_confidence,
        )

        options = _vision.HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=model_path),
            running_mode=_RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = _vision.HandLandmarker.create_from_options(options)
        # Monotonic timestamp in milliseconds, required by VIDEO mode.
        self._start_ms = int(time.monotonic() * 1000)

    # ── Public API ──────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> list[HandResult]:
        """
        Detect hands in *frame* (BGR) and return landmark data.

        Args:
            frame: A BGR image from OpenCV.

        Returns:
            A list of :class:`HandResult` objects (one per detected hand).
        """
        # MediaPipe Tasks expects an SRGB mp.Image (RGB channel order).
        # Marking the frame non-writeable avoids an internal copy.
        frame.flags.writeable = False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = True

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000) - self._start_ms

        detection = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not detection.hand_landmarks:
            return []

        h, w, _ = frame.shape
        hands: list[HandResult] = []

        for hand_lms, handedness_list in zip(
            detection.hand_landmarks,
            detection.handedness,
        ):
            # Convert normalized [0,1] coords → pixel coords
            landmarks = [
                Landmark(
                    id=idx,
                    x=int(lm.x * w),
                    y=int(lm.y * h),
                    z=lm.z,
                )
                for idx, lm in enumerate(hand_lms)
            ]
            # category_name is "Left" or "Right"
            label: str = handedness_list[0].category_name
            hands.append(HandResult(landmarks=landmarks, handedness=label))

        return hands

    def close(self) -> None:
        """Release MediaPipe resources."""
        logger.info("Closing HandTracker")
        self._landmarker.close()

    # Context-manager support
    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

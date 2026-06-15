"""
Lightweight FPS counter that smooths readings over a short window.
"""

import time
from collections import deque


class FPSCounter:
    """Tracks frames-per-second using a sliding-window average."""

    def __init__(self, window_size: int = 30) -> None:
        """
        Args:
            window_size: Number of recent frame timestamps to keep.
                         A larger window gives a smoother (but slower-reacting) value.
        """
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> None:
        """Record the current timestamp. Call once per frame."""
        self._timestamps.append(time.perf_counter())

    def get_fps(self) -> float:
        """
        Return the current smoothed FPS.

        Returns:
            0.0 if fewer than two frames have been recorded, otherwise the
            average FPS over the sliding window.
        """
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

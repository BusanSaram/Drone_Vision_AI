"""Lightweight rolling timing statistics for the real-time loop.

Plain Python, no threads. `StageTimer` keeps a rolling window per named stage
(for live display) plus run totals (for the end-of-run summary). `FpsMeter`
reports FPS from the mean of recent frame intervals, which is steadier than
the instantaneous 1/dt.
"""

import time
from collections import deque
from contextlib import contextmanager


class RollingAverage:
    """Mean of the most recent `window` values."""

    def __init__(self, window: int = 30):
        self._values = deque(maxlen=window)

    def add(self, value: float) -> None:
        self._values.append(value)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float | None:
        return sum(self._values) / len(self._values) if self._values else None


class StageTimer:
    """Per-stage durations: rolling means and whole-run means, in milliseconds."""

    def __init__(self, window: int = 30, clock=time.perf_counter):
        self._window = window
        self._clock = clock
        self._rolling: dict[str, RollingAverage] = {}
        self._totals: dict[str, list[float]] = {}  # name -> [sum_seconds, count]

    @contextmanager
    def measure(self, name: str):
        start = self._clock()
        try:
            yield
        finally:
            self.add(name, self._clock() - start)

    def add(self, name: str, seconds: float) -> None:
        self._rolling.setdefault(name, RollingAverage(self._window)).add(seconds)
        total = self._totals.setdefault(name, [0.0, 0])
        total[0] += seconds
        total[1] += 1

    def reset_totals(self) -> None:
        """Forget run totals (e.g. after warm-up); rolling windows are kept."""
        self._totals.clear()

    def rolling_ms(self) -> dict[str, float]:
        return {name: avg.mean * 1000 for name, avg in self._rolling.items() if avg.mean is not None}

    def totals_ms(self) -> dict[str, float]:
        return {name: s / n * 1000 for name, (s, n) in self._totals.items() if n}


class FpsMeter:
    """FPS from the mean interval between the last `window` ticks."""

    def __init__(self, window: int = 30, clock=time.perf_counter):
        self._clock = clock
        self._intervals = RollingAverage(window)
        self._last: float | None = None

    def tick(self, now: float | None = None) -> float | None:
        """Record one frame; returns the smoothed FPS, or None before the second tick."""
        now = now if now is not None else self._clock()
        if self._last is not None:
            self._intervals.add(now - self._last)
        self._last = now
        mean = self._intervals.mean
        return 1.0 / mean if mean else None

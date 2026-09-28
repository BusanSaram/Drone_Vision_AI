"""Unit tests for rolling_stats (RollingAverage, StageTimer, FpsMeter) using a fake clock.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rolling_stats import FpsMeter, RollingAverage, StageTimer  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class RollingAverageTests(unittest.TestCase):
    def test_empty_is_none(self):
        self.assertIsNone(RollingAverage(3).mean)

    def test_window_drops_old_values(self):
        avg = RollingAverage(3)
        for v in (10, 1, 2, 3):
            avg.add(v)
        self.assertEqual(len(avg), 3)
        self.assertAlmostEqual(avg.mean, 2.0)


class StageTimerTests(unittest.TestCase):
    def test_measure_uses_clock(self):
        clock = FakeClock()
        timer = StageTimer(window=2, clock=clock)
        for duration in (0.010, 0.020, 0.030):
            with timer.measure("stage"):
                clock.now += duration
        self.assertAlmostEqual(timer.rolling_ms()["stage"], 25.0)  # last two
        self.assertAlmostEqual(timer.totals_ms()["stage"], 20.0)  # all three

    def test_measure_records_even_on_exception(self):
        clock = FakeClock()
        timer = StageTimer(clock=clock)
        with self.assertRaises(RuntimeError):
            with timer.measure("stage"):
                clock.now += 0.005
                raise RuntimeError
        self.assertAlmostEqual(timer.totals_ms()["stage"], 5.0)

    def test_reset_totals_keeps_rolling(self):
        timer = StageTimer(window=5)
        timer.add("stage", 1.0)
        timer.reset_totals()
        timer.add("stage", 0.002)
        self.assertAlmostEqual(timer.totals_ms()["stage"], 2.0)
        self.assertAlmostEqual(timer.rolling_ms()["stage"], 501.0)


class FpsMeterTests(unittest.TestCase):
    def test_first_tick_has_no_fps(self):
        self.assertIsNone(FpsMeter().tick(0.0))

    def test_steady_rate(self):
        meter = FpsMeter(window=10)
        fps = None
        for i in range(11):
            fps = meter.tick(i / 30)
        self.assertAlmostEqual(fps, 30.0, places=6)

    def test_smooths_single_slow_frame(self):
        meter = FpsMeter(window=4)
        times = [0.0, 0.1, 0.2, 0.3, 0.7]  # intervals 0.1, 0.1, 0.1, 0.4
        for t in times:
            fps = meter.tick(t)
        self.assertAlmostEqual(fps, 1 / 0.175)


if __name__ == "__main__":
    unittest.main()

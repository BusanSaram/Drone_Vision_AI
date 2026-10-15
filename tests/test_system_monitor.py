"""Unit tests for the /proc parsing in raspberry_pi/system_monitor.py (sample text, no /proc needed).

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "raspberry_pi"))

import system_monitor as sm  # noqa: E402


class ParseTest(unittest.TestCase):
    def test_total_cpu(self):
        text = "cpu  100 5 50 800 20 3 2 0 0 0\ncpu0 25 1 12 200 5 1 0 0 0 0\n"
        busy, total = sm.parse_total_cpu(text)
        self.assertEqual(total, 980)
        self.assertEqual(busy, 980 - 800 - 20)

    def test_total_cpu_rejects_wrong_line(self):
        with self.assertRaises(ValueError):
            sm.parse_total_cpu("cpu0 1 2 3 4 5 6 7 8\n")

    def test_process_ticks_with_spaces_in_name(self):
        # pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime ...
        text = "4321 (python3 my app) S 1 4321 4321 0 -1 4194304 500 0 0 0 150 30 0 0 20 0 3 0 100"
        self.assertEqual(sm.parse_process_ticks(text), 180)

    def test_percent(self):
        self.assertAlmostEqual(sm.percent(25, 100), 25.0)
        self.assertEqual(sm.percent(5, 0), 0.0)

    def test_format_sample(self):
        s = sm.format_sample({"total": 12.4, "self": None, "ffmpeg": 31.0, "temp_c": 52.25})
        self.assertEqual(s, "cpu_total=12% cpu_self=- cpu_ffmpeg=31% temp=52.2C")


if __name__ == "__main__":
    unittest.main()

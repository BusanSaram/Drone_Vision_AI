"""Unit tests for TrackValidator.is_tracked (used to detect a lost target).

Explicit `now` timestamps - no sleeping.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from person_tracker import TrackedPerson  # noqa: E402
from track_validator import TrackValidator  # noqa: E402


def person(track_id):
    return TrackedPerson(track_id=track_id, bbox=(0, 0, 10, 10), confidence=0.9)


class IsTrackedTests(unittest.TestCase):
    def setUp(self):
        self.validator = TrackValidator(confirmation_time=0.5, lost_grace_time=2.0)

    def test_unknown_track_is_not_tracked(self):
        self.assertFalse(self.validator.is_tracked(1))

    def test_visible_candidate_is_tracked(self):
        self.validator.update([person(1)], now=0.0)
        self.assertTrue(self.validator.is_tracked(1))

    def test_missing_candidate_is_dropped(self):
        self.validator.update([person(1)], now=0.0)
        self.validator.update([], now=0.1)
        self.assertFalse(self.validator.is_tracked(1))

    def test_confirmed_track_tracked_during_grace_then_expires(self):
        v = self.validator
        v.update([person(1)], now=0.0)
        v.update([person(1)], now=0.5)  # CONFIRMED
        v.update([], now=1.0)
        self.assertTrue(v.is_tracked(1))
        v.update([], now=2.4)  # 1.9 s missing
        self.assertTrue(v.is_tracked(1))
        v.update([], now=2.5)  # 2.0 s missing -> expired
        self.assertFalse(v.is_tracked(1))


if __name__ == "__main__":
    unittest.main()

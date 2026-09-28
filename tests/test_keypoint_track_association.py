"""Unit tests for keypoint_track_association.match_tracks_to_detections.

Hand-built tracks and boxes - checks the matching rules only.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keypoint_track_association import match_tracks_to_detections  # noqa: E402
from person_tracker import TrackedPerson  # noqa: E402
from pose_detections import PoseDetections  # noqa: E402

BOX_A = (50, 50, 150, 400)
BOX_B = (400, 60, 500, 410)


def detections(*boxes) -> PoseDetections:
    n = len(boxes)
    return PoseDetections.from_arrays(boxes, np.full(n, 0.9), np.zeros(n), np.ones((n, 17, 3)))


def track(track_id, bbox, detection_index) -> TrackedPerson:
    return TrackedPerson(track_id=track_id, bbox=bbox, confidence=0.9, detection_index=detection_index)


class MatchTests(unittest.TestCase):
    def test_two_people_matched_by_index(self):
        dets = detections(BOX_A, BOX_B)
        tracks = [track(1, BOX_A, 0), track(2, BOX_B, 1)]
        self.assertEqual(match_tracks_to_detections(tracks, dets), {1: 0, 2: 1})

    def test_index_used_not_position_order(self):
        dets = detections(BOX_B, BOX_A)
        tracks = [track(1, BOX_A, 1), track(2, BOX_B, 0)]
        self.assertEqual(match_tracks_to_detections(tracks, dets), {1: 1, 2: 0})

    def test_small_kalman_offset_still_matches(self):
        dets = detections(BOX_A)
        shifted = (55, 55, 155, 405)
        self.assertEqual(match_tracks_to_detections([track(1, shifted, 0)], dets), {1: 0})

    def test_missing_index_gets_nothing(self):
        dets = detections(BOX_A)
        self.assertEqual(match_tracks_to_detections([track(1, BOX_A, None)], dets), {})

    def test_out_of_range_index_gets_nothing(self):
        dets = detections(BOX_A)
        self.assertEqual(match_tracks_to_detections([track(1, BOX_A, 1), track(2, BOX_A, -1)], dets), {})

    def test_low_iou_withheld_instead_of_reassigned(self):
        # Track 1 points at detection 1, but its box sits on detection 0:
        # inconsistent, so no keypoints - and never a fallback to detection 0.
        dets = detections(BOX_A, BOX_B)
        self.assertEqual(match_tracks_to_detections([track(1, BOX_A, 1)], dets), {})

    def test_min_iou_is_configurable(self):
        dets = detections(BOX_A)
        partly = (100, 50, 200, 400)  # IoU with BOX_A = 1/3
        self.assertEqual(match_tracks_to_detections([track(1, partly, 0)], dets), {})
        self.assertEqual(match_tracks_to_detections([track(1, partly, 0)], dets, min_iou=0.3), {1: 0})

    def test_detection_claimed_twice_goes_to_nobody(self):
        dets = detections(BOX_A, BOX_B)
        tracks = [track(1, BOX_A, 0), track(2, (52, 52, 152, 402), 0), track(3, BOX_B, 1)]
        self.assertEqual(match_tracks_to_detections(tracks, dets), {3: 1})

    def test_no_tracks_or_no_detections(self):
        self.assertEqual(match_tracks_to_detections([], detections(BOX_A)), {})
        self.assertEqual(match_tracks_to_detections([track(1, BOX_A, 0)], PoseDetections.empty()), {})


if __name__ == "__main__":
    unittest.main()

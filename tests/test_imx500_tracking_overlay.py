"""Unit tests for raspberry_pi/imx500_tracking_overlay.py (drawing only, synthetic results).

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "raspberry_pi"))

import imx500_tracking_overlay as ov  # noqa: E402
from coco_pose_recognizer import BodyPose  # noqa: E402
from imx500_tracking_pipeline import PipelineResult, TargetStatus  # noqa: E402
from person_tracker import TrackedPerson  # noqa: E402
from pose_command_stabilizer import StabilizerState  # noqa: E402
from pose_detections import PoseDetections  # noqa: E402
from target_selector import SelectionState, SelectorResult  # noqa: E402
from track_validator import TrackState, ValidatedTrack  # noqa: E402

W, H = 640, 480


def result(tracks, target_id=None, pose_states=None):
    """tracks: [(track_id, state, bbox)]."""
    validated = [ValidatedTrack(TrackedPerson(tid, box, 0.73, 0), state) for tid, state, box in tracks]
    detections = PoseDetections.from_arrays([tracks[0][2]] if tracks else np.zeros((0, 4)), [0.73] if tracks else [],
                                            [0] if tracks else [], np.zeros((len(tracks[:1]), 17, 3)))
    state = SelectionState.FOLLOWING if target_id is not None else SelectionState.SEARCHING
    return PipelineResult(0.0, detections, [v.tracked_person for v in validated], validated, {},
                          pose_states or {}, SelectorResult(state, target_id))


def status(target_id=None, usable=False, reason="no target", state=SelectionState.SEARCHING):
    return TargetStatus(state, target_id, None, None, usable, reason)


class DrawTracksTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((H, W, 3), dtype=np.uint8)

    def test_colors_match_windows_demo(self):
        self.assertEqual(ov.CONFIRMED_COLOR, (255, 255, 0))
        self.assertEqual(ov.CANDIDATE_COLOR, (0, 165, 255))
        self.assertEqual(ov.TARGET_COLOR, (255, 0, 255))
        r = result([(1, TrackState.CONFIRMED, (100, 100, 200, 300)),
                    (2, TrackState.CANDIDATE, (300, 100, 400, 300)),
                    (3, TrackState.CONFIRMED, (450, 100, 600, 300))], target_id=3)
        ov.draw_tracks(self.image, r, hold_time=0.7)
        self.assertEqual(tuple(self.image[200, 100]), ov.CONFIRMED_COLOR)
        self.assertEqual(tuple(self.image[200, 300]), ov.CANDIDATE_COLOR)
        self.assertEqual(tuple(self.image[200, 450]), ov.TARGET_COLOR)

    def test_boxes_outside_image_are_clipped(self):
        self.assertEqual(ov.clip_box((-20, 300, 700, 520), W, H), (0, 300, W - 1, H - 1))
        ov.draw_tracks(self.image, result([(1, TrackState.CONFIRMED, (-20, 300, 700, 520))]), 0.7)
        self.assertEqual(tuple(self.image[400, 0]), ov.CONFIRMED_COLOR)

    def test_none_result(self):
        ov.draw_tracks(self.image, None, 0.7)
        self.assertFalse(self.image.any())

    def test_pose_label_lines_same_text_as_windows(self):
        holding = StabilizerState(BodyPose.ONE_ARM_UP, BodyPose.ONE_ARM_UP, 0.25, BodyPose.NONE, False)
        self.assertEqual(ov.pose_label_lines(holding, 0.7),
                         ["Raw Pose: ONE_ARM_UP", "Command: NONE (hold 0.25/0.70s)"])


class StatusTest(unittest.TestCase):
    def test_warning_priority(self):
        self.assertIsNone(ov.warning_text(0.06, status()))
        self.assertIn("STALE", ov.warning_text(0.8, status(3, False, "result stale", SelectionState.FOLLOWING)))
        self.assertEqual(ov.warning_text(0.06, status(3, False, "target not in latest result", SelectionState.FOLLOWING)),
                         "TARGET NOT USABLE: target not in latest result")
        self.assertIsNone(ov.warning_text(0.06, status(3, True, "ok", SelectionState.FOLLOWING)))

    def test_status_lines(self):
        r = result([(1, TrackState.CONFIRMED, (100, 100, 200, 300)), (2, TrackState.CANDIDATE, (300, 100, 400, 300))])
        lines = ov.status_lines(16.0, 8.0, r, status(3, True, "ok", SelectionState.FOLLOWING))
        self.assertEqual(lines, ["FPS 16.0 | DPS  8.0 | det 1 | confirmed 1", "State: FOLLOWING | Target: 3"])
        with_clock = ov.status_lines(16.0, 8.0, r, status(), elapsed=12.34)
        self.assertEqual(with_clock[1], "State: SEARCHING | Target: - | t= 12.3s")

    def test_warning_drawn_in_red(self):
        image = np.zeros((H, W, 3), dtype=np.uint8)
        ov.draw_tracking_status(image, ["a", "b"], None)
        self.assertFalse(np.any(np.all(image == ov.WARNING_COLOR, axis=2)))
        ov.draw_tracking_status(image, ["a", "b"], "STALE 0.8s")
        self.assertTrue(np.any(np.all(image == ov.WARNING_COLOR, axis=2)))


if __name__ == "__main__":
    unittest.main()

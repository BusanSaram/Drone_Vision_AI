"""Unit tests for the read-only IMX500 detection diagnostics (raspberry_pi/).

Synthetic arrays only - no picamera2, no camera.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "raspberry_pi"))

import imx500_detection_diagnostics as diag  # noqa: E402


def keypoints_with_scores(scores):
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[:, 2] = scores
    return kps


class BoxMetricsTest(unittest.TestCase):
    def test_identical_boxes(self):
        box = (0, 0, 10, 10)
        self.assertAlmostEqual(diag.box_iou(box, box), 1.0)
        self.assertAlmostEqual(diag.containment(box, box), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(diag.box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertEqual(diag.containment((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_small_box_inside_large_box(self):
        large, small = (0, 0, 100, 200), (10, 10, 60, 60)
        self.assertAlmostEqual(diag.containment(small, large), 1.0)
        self.assertAlmostEqual(diag.containment(large, small), 2500 / 20000)
        self.assertAlmostEqual(diag.box_iou(small, large), 2500 / 20000)

    def test_zero_area_box(self):
        self.assertEqual(diag.containment((5, 5, 5, 5), (0, 0, 10, 10)), 0.0)
        self.assertEqual(diag.box_iou((5, 5, 5, 5), (5, 5, 5, 5)), 0.0)


class KeypointTest(unittest.TestCase):
    def test_valid_indices_use_threshold_and_finiteness(self):
        scores = np.full(17, 0.2)
        scores[[5, 6, 9]] = 0.5
        kps = keypoints_with_scores(scores)
        kps[9, 0] = np.nan  # high score but non-finite position
        self.assertEqual(diag.valid_keypoint_indices(kps, 0.5), [5, 6])
        self.assertEqual(len(diag.valid_keypoint_indices(kps, 0.1)), 16)


class LinesTest(unittest.TestCase):
    def test_split_person_pair(self):
        # One person split into an upper-body and a lower-body fragment.
        upper = np.zeros(17)
        upper[:11] = 0.9
        lower = np.zeros(17)
        lower[11:] = 0.9
        xyxy = np.array([[0, 0, 100, 100], [0, 0, 100, 200]], dtype=np.float32)
        kps = np.stack([keypoints_with_scores(upper), keypoints_with_scores(lower)])
        conf = np.array([0.6, 0.4], dtype=np.float32)

        det = diag.detection_lines(xyxy, conf, kps, 0.5)
        self.assertEqual(len(det), 2)
        self.assertIn("valid_kp=11/17", det[0])
        self.assertIn("valid_kp=6/17", det[1])

        pairs = diag.pair_lines(xyxy, kps, 0.5)
        self.assertEqual(len(pairs), 1)
        self.assertIn("iou=0.500", pairs[0])
        self.assertIn("0_in_1=1.000", pairs[0])
        self.assertIn("shared_valid_kp=0", pairs[0])

    def test_no_pairs_for_single_or_empty(self):
        self.assertEqual(diag.pair_lines(np.zeros((1, 4)), np.zeros((1, 17, 3)), 0.5), [])
        self.assertEqual(diag.pair_lines(np.zeros((0, 4)), np.zeros((0, 17, 3)), 0.5), [])
        self.assertEqual(diag.detection_lines(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))), [])


class CountersTest(unittest.TestCase):
    def test_summary_counts(self):
        counters = diag.DiagnosticCounters()
        counters.record_no_output()
        counters.record_fresh(np.array([]))
        counters.record_fresh(np.array([0.7]))
        counters.record_fresh(np.array([0.35, 0.9]))
        text = "\n".join(counters.summary_lines())
        self.assertIn("callbacks: 4, no tensor output (cached reused): 1", text)
        self.assertIn("0 people: 1, 1 people: 1, 2 people: 1", text)
        self.assertIn("0.350 .. 0.900", text)


if __name__ == "__main__":
    unittest.main()

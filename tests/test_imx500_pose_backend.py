"""Unit tests for raspberry_pi/imx500_pose_backend.py (modlib Poses -> PoseDetections).

A small stand-in replaces modlib's Poses - no modlib, no camera.

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

from coco_keypoints import NUM_KEYPOINTS  # noqa: E402
from imx500_pose_backend import MIN_DETECTION_SCORE, poses_to_detections  # noqa: E402
from pose_detections import PoseDetections  # noqa: E402

W, H = 640, 480


class FakePoses:
    """Same attributes modlib's Poses exposes (normalized coordinates)."""

    def __init__(self, bbox, keypoints, keypoint_scores, confidence):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.keypoints = np.asarray(keypoints, dtype=np.float32)
        self.keypoint_scores = np.asarray(keypoint_scores, dtype=np.float32)
        self.confidence = np.asarray(confidence, dtype=np.float32)
        self.roi_seen = None

    def compensate_for_roi(self, roi):
        self.roi_seen = roi


def fake(confidences, bbox=(0.25, 0.25, 0.75, 1.016), kp=(0.5, 0.5)):
    n = len(confidences)
    return FakePoses([bbox] * n, np.tile(np.asarray(kp, dtype=np.float32), (n, NUM_KEYPOINTS, 1)),
                     np.full((n, NUM_KEYPOINTS), 0.9), confidences)


class PosesToDetectionsTest(unittest.TestCase):
    def test_pixel_coordinates_clipped_and_layout(self):
        d = poses_to_detections(fake([0.73]), (0, 0, 1, 1), W, H)
        self.assertIsInstance(d, PoseDetections)
        np.testing.assert_allclose(d.xyxy[0], [160, 120, 480, H - 1])
        self.assertEqual(d.keypoints.shape, (1, NUM_KEYPOINTS, 3))
        np.testing.assert_allclose(d.keypoints[0, :, :2], np.tile([320, 240], (NUM_KEYPOINTS, 1)))
        np.testing.assert_allclose(d.keypoints[0, :, 2], 0.9)
        np.testing.assert_allclose(d.cls, [0])
        np.testing.assert_allclose(d.xywh[0], [320, 299.5, 320, 359])  # what BoT-SORT reads

    def test_windows_predict_floor_applied(self):
        self.assertEqual(MIN_DETECTION_SCORE, 0.25)
        d = poses_to_detections(fake([0.73, 0.27, 0.24, 0.0]), None, W, H)
        np.testing.assert_allclose(d.conf, [0.73, 0.27])

    def test_roi_compensation_is_requested(self):
        poses = fake([0.73])
        poses_to_detections(poses, (0, 0, 1, 1), W, H)
        self.assertEqual(poses.roi_seen, (0, 0, 1, 1))

    def test_empty(self):
        d = poses_to_detections(fake([]), None, W, H)
        self.assertEqual(len(d), 0)
        self.assertEqual(d.keypoints.shape, (0, NUM_KEYPOINTS, 3))


if __name__ == "__main__":
    unittest.main()

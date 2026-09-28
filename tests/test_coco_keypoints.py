"""Unit tests for coco_keypoints (keypoint validation and skeleton selection).

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coco_keypoints import (  # noqa: E402
    COCO_SKELETON, KEYPOINT_NAMES, LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, NUM_KEYPOINTS,
    reliable_mask, skeleton_segments, validate_keypoints,
)


def keypoints_with_conf(conf: float) -> np.ndarray:
    kpts = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    kpts[:, 0] = np.arange(NUM_KEYPOINTS) * 10 + 100
    kpts[:, 1] = 200
    kpts[:, 2] = conf
    return kpts


class CocoModelTests(unittest.TestCase):
    def test_seventeen_standard_names(self):
        self.assertEqual(NUM_KEYPOINTS, 17)
        self.assertEqual(KEYPOINT_NAMES[0], "nose")
        self.assertEqual(KEYPOINT_NAMES[5], "left_shoulder")
        self.assertEqual(KEYPOINT_NAMES[10], "right_wrist")
        self.assertEqual(KEYPOINT_NAMES[16], "right_ankle")

    def test_skeleton_indices_valid_and_unique(self):
        for a, b in COCO_SKELETON:
            self.assertTrue(0 <= a < NUM_KEYPOINTS and 0 <= b < NUM_KEYPOINTS)
            self.assertNotEqual(a, b)
        self.assertEqual(len(COCO_SKELETON), len({frozenset(c) for c in COCO_SKELETON}))
        self.assertEqual(len(COCO_SKELETON), 19)


class ValidateKeypointsTests(unittest.TestCase):
    def test_accepts_17x3(self):
        self.assertEqual(validate_keypoints(keypoints_with_conf(1.0)).shape, (17, 3))

    def test_rejects_missing_confidence_column(self):
        with self.assertRaises(ValueError):
            validate_keypoints(np.zeros((17, 2)))

    def test_rejects_wrong_keypoint_count(self):
        with self.assertRaises(ValueError):
            validate_keypoints(np.zeros((33, 3)))


class ReliabilityTests(unittest.TestCase):
    def test_threshold_is_inclusive(self):
        self.assertTrue(reliable_mask(keypoints_with_conf(0.5), min_conf=0.5).all())
        self.assertFalse(reliable_mask(keypoints_with_conf(0.49), min_conf=0.5).any())

    def test_non_finite_coordinates_are_unreliable(self):
        kpts = keypoints_with_conf(0.9)
        kpts[LEFT_WRIST, 0] = np.nan
        mask = reliable_mask(kpts)
        self.assertFalse(mask[LEFT_WRIST])
        self.assertEqual(mask.sum(), 16)

    def test_all_connections_when_all_reliable(self):
        self.assertEqual(skeleton_segments(keypoints_with_conf(0.9)), list(COCO_SKELETON))

    def test_no_connections_when_none_reliable(self):
        self.assertEqual(skeleton_segments(keypoints_with_conf(0.1)), [])

    def test_connection_dropped_if_either_endpoint_unreliable(self):
        kpts = keypoints_with_conf(0.9)
        kpts[LEFT_ELBOW, 2] = 0.2
        segments = skeleton_segments(kpts)
        self.assertNotIn((LEFT_SHOULDER, LEFT_ELBOW), segments)
        self.assertNotIn((LEFT_ELBOW, LEFT_WRIST), segments)
        self.assertEqual(len(segments), len(COCO_SKELETON) - 2)

    def test_custom_threshold(self):
        kpts = keypoints_with_conf(0.3)
        self.assertEqual(skeleton_segments(kpts, min_conf=0.5), [])
        self.assertEqual(len(skeleton_segments(kpts, min_conf=0.25)), len(COCO_SKELETON))


if __name__ == "__main__":
    unittest.main()

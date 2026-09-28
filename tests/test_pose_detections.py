"""Unit tests for PoseDetections: keypoint extraction and row alignment.

Also checks that PoseDetections works as BoT-SORT input through the real
PersonTracker and that each track's `detection_index` points back at the
detection (and therefore the keypoints) that produced it. Synthetic boxes on
a blank frame - no model, no webcam.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from person_tracker import PersonTracker  # noqa: E402
from pose_detections import PoseDetections  # noqa: E402


def make_keypoints(n: int) -> np.ndarray:
    """Keypoints whose values encode their detection row, so misalignment is detectable."""
    kpts = np.zeros((n, 17, 3), dtype=np.float32)
    for i in range(n):
        kpts[i, :, 0] = 1000 + i
        kpts[i, :, 1] = np.arange(17)
        kpts[i, :, 2] = 0.9
    return kpts


def make_detections(boxes, confs) -> PoseDetections:
    n = len(boxes)
    return PoseDetections.from_arrays(boxes, confs, np.zeros(n), make_keypoints(n))


class FromArraysTests(unittest.TestCase):
    def test_shapes_and_dtypes(self):
        dets = make_detections([[10, 20, 50, 120], [200, 30, 260, 150]], [0.9, 0.8])
        self.assertEqual(len(dets), 2)
        self.assertEqual(dets.xyxy.shape, (2, 4))
        self.assertEqual(dets.keypoints.shape, (2, 17, 3))
        self.assertEqual(dets.conf.dtype, np.float32)

    def test_xywh(self):
        dets = make_detections([[10, 20, 50, 120]], [0.9])
        np.testing.assert_allclose(dets.xywh, [[30, 70, 40, 100]])

    def test_empty(self):
        for dets in (PoseDetections.empty(), PoseDetections.from_arrays(np.zeros((0, 4)), [], [], None)):
            self.assertEqual(len(dets), 0)
            self.assertEqual(dets.keypoints.shape, (0, 17, 3))
            self.assertEqual(dets.xywh.shape, (0, 4))

    def test_missing_keypoints_rejected(self):
        with self.assertRaises(ValueError):
            PoseDetections.from_arrays([[0, 0, 10, 10]], [0.9], [0], None)

    def test_keypoints_without_confidence_rejected(self):
        with self.assertRaises(ValueError):
            PoseDetections.from_arrays([[0, 0, 10, 10]], [0.9], [0], np.zeros((1, 17, 2)))

    def test_row_count_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            PoseDetections.from_arrays([[0, 0, 10, 10]], [0.9, 0.8], [0, 0], make_keypoints(1))
        with self.assertRaises(ValueError):
            PoseDetections.from_arrays([[0, 0, 10, 10]], [0.9], [0], make_keypoints(2))


class IndexingTests(unittest.TestCase):
    def test_boolean_mask_keeps_rows_aligned(self):
        dets = make_detections([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], [0.9, 0.1, 0.8])
        subset = dets[dets.conf >= 0.5]
        self.assertEqual(len(subset), 2)
        np.testing.assert_allclose(subset.xyxy[:, 0], [0, 40])
        np.testing.assert_allclose(subset.keypoints[:, 0, 0], [1000, 1002])  # rows 0 and 2

    def test_all_false_mask(self):
        dets = make_detections([[0, 0, 10, 10]], [0.9])
        self.assertEqual(len(dets[np.array([False])]), 0)


class TrackerCompatibilityTests(unittest.TestCase):
    """PoseDetections fed through the real BoT-SORT wrapper (default config, new_track_thresh 0.8)."""

    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.tracker = PersonTracker(new_track_thresh=0.8)

    def test_detection_index_points_back_to_source_detection(self):
        boxes = [[50, 50, 150, 400], [400, 60, 500, 410]]
        tracked = []
        for step in range(5):
            # Detection order reversed every other frame, so a stale or
            # order-based index would point at the wrong person.
            ordered = boxes if step % 2 == 0 else boxes[::-1]
            # Each person's keypoints sit at their own box's left edge.
            kpts = np.zeros((2, 17, 3), dtype=np.float32)
            for i, box in enumerate(ordered):
                kpts[i, :, 0] = box[0]
                kpts[i, :, 2] = 0.9
            dets = PoseDetections.from_arrays(ordered, [0.9, 0.9], [0, 0], kpts)
            tracked = self.tracker.update(dets, self.frame)
            for person in tracked:
                index = person.detection_index
                self.assertIsNotNone(index)
                self.assertLess(abs(dets.xyxy[index, 0] - person.bbox[0]), 5)
                self.assertLess(abs(dets.keypoints[index, 0, 0] - person.bbox[0]), 5)
        self.assertEqual(len(tracked), 2)
        self.assertEqual(len({p.track_id for p in tracked}), 2)

    def test_empty_detections(self):
        self.assertEqual(self.tracker.update(PoseDetections.empty(), self.frame), [])


if __name__ == "__main__":
    unittest.main()

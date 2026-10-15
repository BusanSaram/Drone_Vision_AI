"""Unit tests for the IMX500 pose overlay (raspberry_pi/imx500_pose_overlay.py).

Synthetic arrays only - no modlib, no camera.

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

from coco_keypoints import KEYPOINT_CONF_THRESHOLD, LEFT_SHOULDER, LEFT_WRIST, NUM_KEYPOINTS  # noqa: E402
import imx500_pose_overlay as overlay  # noqa: E402

W, H = 640, 480


def one_person(bbox, kp_xy=(0.5, 0.5), kp_score=0.9, conf=0.73):
    """Normalized modlib-style arrays for a single detection."""
    keypoints = np.tile(np.asarray(kp_xy, dtype=np.float32), (1, NUM_KEYPOINTS, 1))
    scores = np.full((1, NUM_KEYPOINTS), kp_score, dtype=np.float32)
    return np.asarray([bbox], dtype=np.float32), keypoints, scores, np.asarray([conf], dtype=np.float32)


class ToPixelPosesTest(unittest.TestCase):
    def test_scales_x_by_width_and_y_by_height(self):
        bbox, kps, scores, conf = one_person([0.25, 0.5, 0.75, 1.0], kp_xy=(0.5, 0.25))
        p = overlay.to_pixel_poses(bbox, kps, scores, conf, W, H)
        np.testing.assert_allclose(p.boxes[0], [160, 240, 480, H - 1])  # y2 = 1.0 -> clipped to last row
        np.testing.assert_allclose(p.keypoints[0, :, 0], 320)
        np.testing.assert_allclose(p.keypoints[0, :, 1], 120)
        np.testing.assert_allclose(p.keypoints[0, :, 2], 0.9)

    def test_clips_boxes_and_keypoints_to_image(self):
        # run1.log showed y2 = 650/640 = 1.016 (person extends below the frame)
        bbox, kps, scores, conf = one_person([-0.1, 0.4375, 1.2, 1.016], kp_xy=(1.5, -0.2))
        p = overlay.to_pixel_poses(bbox, kps, scores, conf, W, H)
        np.testing.assert_allclose(p.boxes[0], [0, 210, W - 1, H - 1])
        np.testing.assert_allclose(p.keypoints[0, :, 0], W - 1)
        np.testing.assert_allclose(p.keypoints[0, :, 1], 0)

    def test_people_below_score_threshold_are_dropped(self):
        bbox = np.asarray([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6], [0.3, 0.3, 0.4, 0.4]], dtype=np.float32)
        kps = np.full((3, NUM_KEYPOINTS, 2), 0.5, dtype=np.float32)
        scores = np.full((3, NUM_KEYPOINTS), 0.9, dtype=np.float32)
        conf = np.asarray([0.73, 0.27, overlay.PERSON_SCORE_THRESHOLD], dtype=np.float32)
        p = overlay.to_pixel_poses(bbox, kps, scores, conf, W, H)
        self.assertEqual(len(p), 2)
        np.testing.assert_allclose(p.confidence, [0.73, overlay.PERSON_SCORE_THRESHOLD])
        np.testing.assert_allclose(p.boxes[1], [192, 144, 256, 192])

    def test_no_detections(self):
        p = overlay.to_pixel_poses(np.zeros((0, 4)), np.zeros((0, NUM_KEYPOINTS, 2)),
                                   np.zeros((0, NUM_KEYPOINTS)), np.zeros((0,)), W, H)
        self.assertEqual(len(p), 0)
        self.assertEqual(p.boxes.shape, (0, 4))
        self.assertEqual(p.keypoints.shape, (0, NUM_KEYPOINTS, 3))

    def test_all_300_slots_from_the_model(self):
        conf = np.zeros(300, dtype=np.float32)
        conf[7] = 0.62
        p = overlay.to_pixel_poses(np.full((300, 4), 0.5), np.full((300, NUM_KEYPOINTS, 2), 0.5),
                                   np.full((300, NUM_KEYPOINTS), 0.9), conf, W, H)
        self.assertEqual(len(p), 1)


class DrawTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((H, W, 3), dtype=np.uint8)

    def pixel_poses(self, kp_score=0.9):
        bbox, kps, scores, conf = one_person([0.25, 0.25, 0.75, 0.75], kp_score=kp_score)
        kps[0, LEFT_SHOULDER] = (0.4, 0.4)
        kps[0, LEFT_WRIST] = (0.6, 0.6)
        return overlay.to_pixel_poses(bbox, kps, scores, conf, W, H)

    def test_box_is_always_green(self):
        overlay.draw_poses(self.image, self.pixel_poses())
        self.assertEqual(overlay.BOX_COLOR, (0, 255, 0))
        self.assertEqual(tuple(self.image[240, 160]), overlay.BOX_COLOR)  # left edge of the box

    def test_keypoints_drawn_only_when_reliable(self):
        overlay.draw_poses(self.image, self.pixel_poses(kp_score=0.9))
        self.assertEqual(tuple(self.image[192, 256]), overlay.KEYPOINT_COLOR)  # left shoulder (0.4, 0.4)

        low = np.zeros_like(self.image)
        overlay.draw_poses(low, self.pixel_poses(kp_score=KEYPOINT_CONF_THRESHOLD - 0.01))
        self.assertEqual(tuple(low[192, 256]), (0, 0, 0))
        self.assertFalse(np.any(np.all(low == overlay.SKELETON_COLOR, axis=2)))

    def test_skeleton_drawn_between_reliable_keypoints(self):
        overlay.draw_poses(self.image, self.pixel_poses())
        self.assertTrue(np.any(np.all(self.image == overlay.SKELETON_COLOR, axis=2)))


class StatusTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((H, W, 3), dtype=np.uint8)

    def has_color(self, color):
        return bool(np.any(np.all(self.image[:25] == color, axis=2)))

    def test_status_text_has_fps_dps_people_and_no_new_reused(self):
        text = overlay.status_text(16.0, 8.0, 1)
        self.assertEqual(text, "FPS 16.0 | DPS  8.0 | people 1")
        self.assertNotIn("NEW", text)
        self.assertNotIn("REUSED", text)

    def test_is_stale_only_above_threshold(self):
        self.assertEqual(overlay.STALE_AFTER_S, 0.5)
        self.assertFalse(overlay.is_stale(0.0))
        self.assertFalse(overlay.is_stale(0.0625))  # one repeated frame at 16 FPS / 8 DPS
        self.assertFalse(overlay.is_stale(0.5))  # "more than 0.5 s" - exactly 0.5 is not stale
        self.assertTrue(overlay.is_stale(0.51))

    def test_no_stale_warning_for_normal_alternation(self):
        for age in (0.0, 0.0625):
            overlay.draw_status(self.image, 16.0, 8.0, 1, result_age_s=age)
            self.assertTrue(self.has_color(overlay.STATUS_COLOR))
            self.assertFalse(self.has_color(overlay.STALE_COLOR))

    def test_stale_warning_when_results_stop(self):
        overlay.draw_status(self.image, 16.0, 0.0, 1, result_age_s=0.8)
        self.assertTrue(self.has_color(overlay.STATUS_COLOR))  # FPS / DPS / people stay visible
        self.assertTrue(self.has_color(overlay.STALE_COLOR))


if __name__ == "__main__":
    unittest.main()

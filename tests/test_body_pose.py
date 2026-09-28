"""Synthetic unit tests for body_pose_recognizer and pose_command_stabilizer.

These use hand-built stick-figure skeletons, not real MediaPipe output, so
they check the geometry and timing logic only - not real-world recognition.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from body_pose_recognizer import (  # noqa: E402
    LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST,
    BodyPose, classify_pose, elevation, joint_angle,
)
from pose_command_stabilizer import PoseCommandStabilizer  # noqa: E402

FRAME_W, FRAME_H = 640, 480  # non-square on purpose, to exercise the pixel conversion

# Person facing the camera, shoulder width 100 px. Their RIGHT side appears
# on the image LEFT (smaller x), as in an unmirrored webcam frame.
NOSE_PT = (320, 140)
R_SHOULDER, L_SHOULDER = (270, 200), (370, 200)
ARMS_DOWN = {"r_elbow": (260, 300), "r_wrist": (255, 390), "l_elbow": (380, 300), "l_wrist": (385, 390)}


def skeleton(r_elbow, r_wrist, l_elbow, l_wrist, scale=1.0, offset=(0, 0), visibility=None):
    """Build 33 normalized landmarks from pixel joint positions.

    `scale`/`offset` shrink and move the whole person (simulates distance).
    `visibility` optionally overrides visibility per landmark index.
    """
    pixel_points = {
        NOSE: NOSE_PT,
        RIGHT_SHOULDER: R_SHOULDER, LEFT_SHOULDER: L_SHOULDER,
        RIGHT_ELBOW: r_elbow, LEFT_ELBOW: l_elbow,
        RIGHT_WRIST: r_wrist, LEFT_WRIST: l_wrist,
    }
    cx, cy = 320, 240
    landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    for index, (x, y) in pixel_points.items():
        x = cx + (x - cx) * scale + offset[0]
        y = cy + (y - cy) * scale + offset[1]
        landmarks[index] = SimpleNamespace(x=x / FRAME_W, y=y / FRAME_H, visibility=1.0)
    for index, value in (visibility or {}).items():
        landmarks[index].visibility = value
    return landmarks


def classify(**joints):
    extra = {k: joints.pop(k) for k in ("scale", "offset", "visibility") if k in joints}
    return classify_pose(skeleton(**{**ARMS_DOWN, **joints}, **extra), FRAME_W, FRAME_H)


T_POSE_ARMS = {"r_elbow": (190, 200), "r_wrist": (110, 200), "l_elbow": (450, 200), "l_wrist": (530, 200)}
DOUBLE_BICEPS_ARMS = {"r_elbow": (190, 195), "r_wrist": (185, 115), "l_elbow": (450, 195), "l_wrist": (455, 115)}
RIGHT_ARM_UP = {"r_elbow": (265, 110), "r_wrist": (262, 30)}
LEFT_ARM_UP = {"l_elbow": (375, 110), "l_wrist": (378, 30)}


class GeometryHelperTests(unittest.TestCase):
    def test_joint_angle(self):
        self.assertAlmostEqual(joint_angle((0, 0), (1, 0), (2, 0)), 180.0)
        self.assertAlmostEqual(joint_angle((0, 0), (1, 0), (1, 1)), 90.0)

    def test_elevation_positive_is_up(self):
        self.assertAlmostEqual(elevation((0, 10), (10, 0)), 45.0)  # image y grows downward
        self.assertAlmostEqual(elevation((0, 0), (-10, 0)), 0.0)  # direction-independent in x


class ClassifyPoseTests(unittest.TestCase):
    def test_arms_down_is_none(self):
        self.assertIs(classify(), BodyPose.NONE)

    def test_t_pose(self):
        self.assertIs(classify(**T_POSE_ARMS), BodyPose.T_POSE)

    def test_t_pose_tolerates_drooped_arms(self):
        # ~15 degrees below horizontal on both sides
        self.assertIs(
            classify(r_elbow=(190, 221), r_wrist=(110, 243), l_elbow=(450, 221), l_wrist=(530, 243)),
            BodyPose.T_POSE,
        )

    def test_arms_pointing_at_camera_is_not_t_pose(self):
        # Straight, level arms but foreshortened (little outward extent)
        self.assertIs(
            classify(r_elbow=(245, 200), r_wrist=(225, 200), l_elbow=(395, 200), l_wrist=(415, 200)),
            BodyPose.NONE,
        )

    def test_double_biceps(self):
        self.assertIs(classify(**DOUBLE_BICEPS_ARMS), BodyPose.DOUBLE_BICEPS)

    def test_one_arm_up_either_side(self):
        self.assertIs(classify(**RIGHT_ARM_UP), BodyPose.ONE_ARM_UP)
        self.assertIs(classify(**LEFT_ARM_UP), BodyPose.ONE_ARM_UP)

    def test_both_arms_up_is_none(self):
        self.assertIs(classify(**RIGHT_ARM_UP, **LEFT_ARM_UP), BodyPose.NONE)

    def test_one_arm_up_other_arm_horizontal_is_none(self):
        self.assertIs(classify(**RIGHT_ARM_UP, l_elbow=(450, 200), l_wrist=(530, 200)), BodyPose.NONE)

    def test_one_arm_bent_up_is_not_one_arm_up(self):
        # Single-arm biceps flex: wrist above the head but elbow bent ~90 degrees
        self.assertIs(classify(r_elbow=(190, 150), r_wrist=(190, 60)), BodyPose.NONE)

    def test_only_one_arm_in_t_is_none(self):
        self.assertIs(classify(r_elbow=(190, 200), r_wrist=(110, 200)), BodyPose.NONE)

    def test_distance_independent(self):
        # Same poses with the person 40% the size and off-center
        far = {"scale": 0.4, "offset": (150, -60)}
        self.assertIs(classify(**T_POSE_ARMS, **far), BodyPose.T_POSE)
        self.assertIs(classify(**DOUBLE_BICEPS_ARMS, **far), BodyPose.DOUBLE_BICEPS)
        self.assertIs(classify(**RIGHT_ARM_UP, **far), BodyPose.ONE_ARM_UP)
        self.assertIs(classify(**far), BodyPose.NONE)

    def test_low_visibility_blocks_commands(self):
        self.assertIs(classify(**T_POSE_ARMS, visibility={LEFT_WRIST: 0.3}), BodyPose.NONE)
        self.assertIs(classify(**DOUBLE_BICEPS_ARMS, visibility={RIGHT_ELBOW: 0.2}), BodyPose.NONE)
        self.assertIs(classify(**RIGHT_ARM_UP, visibility={NOSE: 0.1}), BodyPose.NONE)
        # Down arm not visible -> can't confirm "exactly one arm up"
        self.assertIs(classify(**RIGHT_ARM_UP, visibility={LEFT_WRIST: 0.1}), BodyPose.NONE)


class StabilizerTests(unittest.TestCase):
    def setUp(self):
        self.stabilizer = PoseCommandStabilizer(hold_time=0.7)

    def test_not_active_before_hold_time(self):
        s = self.stabilizer
        s.update(BodyPose.T_POSE, now=0.0)
        state = s.update(BodyPose.T_POSE, now=0.69)
        self.assertIs(state.candidate, BodyPose.T_POSE)
        self.assertIs(state.active, BodyPose.NONE)
        self.assertAlmostEqual(state.held_for, 0.69)

    def test_activates_once_after_hold_time(self):
        s = self.stabilizer
        s.update(BodyPose.T_POSE, now=0.0)
        first = s.update(BodyPose.T_POSE, now=0.7)
        later = s.update(BodyPose.T_POSE, now=2.0)
        self.assertIs(first.active, BodyPose.T_POSE)
        self.assertTrue(first.newly_activated)
        self.assertIs(later.active, BodyPose.T_POSE)
        self.assertFalse(later.newly_activated)  # no repeated firing while held

    def test_change_before_hold_resets_candidate(self):
        s = self.stabilizer
        s.update(BodyPose.T_POSE, now=0.0)
        s.update(BodyPose.DOUBLE_BICEPS, now=0.5)
        state = s.update(BodyPose.DOUBLE_BICEPS, now=1.0)  # only 0.5 s of DOUBLE_BICEPS
        self.assertIs(state.active, BodyPose.NONE)
        self.assertIs(s.update(BodyPose.DOUBLE_BICEPS, now=1.2).active, BodyPose.DOUBLE_BICEPS)

    def test_dropout_resets_timer(self):
        s = self.stabilizer
        s.update(BodyPose.ONE_ARM_UP, now=0.0)
        s.update(BodyPose.NONE, now=0.4)
        s.update(BodyPose.ONE_ARM_UP, now=0.5)
        self.assertIs(s.update(BodyPose.ONE_ARM_UP, now=1.1).active, BodyPose.NONE)
        self.assertIs(s.update(BodyPose.ONE_ARM_UP, now=1.2).active, BodyPose.ONE_ARM_UP)

    def test_release_clears_and_rehold_fires_again(self):
        s = self.stabilizer
        s.update(BodyPose.DOUBLE_BICEPS, now=0.0)
        self.assertTrue(s.update(BodyPose.DOUBLE_BICEPS, now=0.8).newly_activated)
        released = s.update(BodyPose.NONE, now=1.0)
        self.assertIs(released.active, BodyPose.NONE)
        self.assertEqual(released.held_for, 0.0)
        s.update(BodyPose.DOUBLE_BICEPS, now=1.5)
        self.assertTrue(s.update(BodyPose.DOUBLE_BICEPS, now=2.2).newly_activated)

    def test_none_never_activates(self):
        s = self.stabilizer
        s.update(BodyPose.NONE, now=0.0)
        state = s.update(BodyPose.NONE, now=5.0)
        self.assertIs(state.active, BodyPose.NONE)
        self.assertFalse(state.newly_activated)


if __name__ == "__main__":
    unittest.main()

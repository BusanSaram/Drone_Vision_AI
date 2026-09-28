"""Synthetic unit tests for coco_pose_recognizer (COCO 17-keypoint body poses).

Hand-built stick figures, not real YOLO output: these check the geometry
rules only, not real-world recognition. Face and hip keypoints are left at
confidence 0 on purpose - the rules must not need them.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import coco_pose_recognizer  # noqa: E402
from coco_keypoints import (  # noqa: E402
    LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST,
)
from coco_pose_recognizer import (  # noqa: E402
    POSE_TO_COMMAND, BodyPose, PoseCommand, classify_pose, elevation, joint_angle,
)

# Person facing the camera, shoulder width 100 px. Their RIGHT side appears
# on the image LEFT (smaller x), as in an unmirrored webcam frame.
R_SHOULDER, L_SHOULDER = (270, 200), (370, 200)
ARMS_DOWN = {"r_elbow": (260, 300), "r_wrist": (255, 390), "l_elbow": (380, 300), "l_wrist": (385, 390)}

T_POSE_ARMS = {"r_elbow": (190, 200), "r_wrist": (110, 200), "l_elbow": (450, 200), "l_wrist": (530, 200)}
DOUBLE_BICEPS_ARMS = {"r_elbow": (190, 195), "r_wrist": (185, 115), "l_elbow": (450, 195), "l_wrist": (455, 115)}
RIGHT_ARM_UP = {"r_elbow": (265, 110), "r_wrist": (262, 30)}
LEFT_ARM_UP = {"l_elbow": (375, 110), "l_wrist": (378, 30)}


def keypoints(r_elbow, r_wrist, l_elbow, l_wrist, scale=1.0, offset=(0, 0), conf=None, mirror=False):
    """Build a (17, 3) COCO keypoint array from pixel arm-joint positions.

    `scale`/`offset` shrink and move the whole person (simulates distance).
    `conf` optionally overrides confidence per keypoint index.
    `mirror` flips the figure horizontally (person's left side on image left).
    """
    points = {
        RIGHT_SHOULDER: R_SHOULDER, LEFT_SHOULDER: L_SHOULDER,
        RIGHT_ELBOW: r_elbow, LEFT_ELBOW: l_elbow,
        RIGHT_WRIST: r_wrist, LEFT_WRIST: l_wrist,
    }
    cx, cy = 320, 240
    kpts = np.zeros((17, 3), dtype=np.float32)  # face/hips/legs: conf 0
    for index, (x, y) in points.items():
        if mirror:
            x = 2 * cx - x
        kpts[index] = (cx + (x - cx) * scale + offset[0], cy + (y - cy) * scale + offset[1], 0.9)
    for index, value in (conf or {}).items():
        kpts[index, 2] = value
    return kpts


def classify(**joints):
    extra = {k: joints.pop(k) for k in ("scale", "offset", "conf", "mirror") if k in joints}
    return classify_pose(keypoints(**{**ARMS_DOWN, **joints}, **extra))


class GeometryHelperTests(unittest.TestCase):
    def test_joint_angle(self):
        self.assertAlmostEqual(joint_angle((0, 0), (1, 0), (2, 0)), 180.0)
        self.assertAlmostEqual(joint_angle((0, 0), (1, 0), (1, 1)), 90.0)

    def test_elevation_positive_is_up(self):
        self.assertAlmostEqual(elevation((0, 10), (10, 0)), 45.0)  # image y grows downward
        self.assertAlmostEqual(elevation((0, 0), (-10, 0)), 0.0)  # direction-independent in x


class CommandMappingTests(unittest.TestCase):
    def test_mapping(self):
        self.assertIs(POSE_TO_COMMAND[BodyPose.ONE_ARM_UP], PoseCommand.TARGET_SELECT)
        self.assertIs(POSE_TO_COMMAND[BodyPose.T_POSE], PoseCommand.HOVER)
        self.assertIs(POSE_TO_COMMAND[BodyPose.DOUBLE_BICEPS], PoseCommand.LAND)
        self.assertIs(POSE_TO_COMMAND[BodyPose.NONE], PoseCommand.NONE)


class OneArmUpTests(unittest.TestCase):
    def test_right_arm_up(self):
        self.assertIs(classify(**RIGHT_ARM_UP), BodyPose.ONE_ARM_UP)

    def test_left_arm_up(self):
        self.assertIs(classify(**LEFT_ARM_UP), BodyPose.ONE_ARM_UP)

    def test_both_arms_up_is_none(self):
        self.assertIs(classify(**RIGHT_ARM_UP, **LEFT_ARM_UP), BodyPose.NONE)

    def test_other_arm_horizontal_is_none(self):
        self.assertIs(classify(**RIGHT_ARM_UP, l_elbow=(450, 200), l_wrist=(530, 200)), BodyPose.NONE)

    def test_bent_raised_arm_is_none(self):
        # Single-arm flex: wrist high but elbow bent ~90 degrees
        self.assertIs(classify(r_elbow=(190, 150), r_wrist=(190, 60)), BodyPose.NONE)

    def test_shallow_diagonal_arm_is_none(self):
        # Straight arm, wrist 1.2 S above the shoulder, but only ~37 degrees up:
        # neither clearly raised nor horizontal.
        self.assertIs(classify(r_elbow=(190, 140), r_wrist=(110, 80)), BodyPose.NONE)

    def test_wrist_just_above_shoulder_is_none(self):
        # Steep, straight arm, but wrist only 0.5 S above the shoulder line
        self.assertIs(classify(r_elbow=(268, 175), r_wrist=(266, 150)), BodyPose.NONE)

    def test_face_keypoints_not_needed(self):
        self.assertIs(classify(**RIGHT_ARM_UP, conf={NOSE: 0.0}), BodyPose.ONE_ARM_UP)


class TPoseTests(unittest.TestCase):
    def test_t_pose(self):
        self.assertIs(classify(**T_POSE_ARMS), BodyPose.T_POSE)

    def test_tolerates_drooped_arms(self):
        # ~15 degrees below horizontal on both sides
        self.assertIs(
            classify(r_elbow=(190, 221), r_wrist=(110, 243), l_elbow=(450, 221), l_wrist=(530, 243)),
            BodyPose.T_POSE,
        )

    def test_arms_pointing_at_camera_is_none(self):
        # Straight, level arms but foreshortened (little outward extent)
        self.assertIs(
            classify(r_elbow=(245, 200), r_wrist=(225, 200), l_elbow=(395, 200), l_wrist=(415, 200)),
            BodyPose.NONE,
        )

    def test_only_one_arm_out_is_none(self):
        self.assertIs(classify(r_elbow=(190, 200), r_wrist=(110, 200)), BodyPose.NONE)


class DoubleBicepsTests(unittest.TestCase):
    def test_double_biceps(self):
        self.assertIs(classify(**DOUBLE_BICEPS_ARMS), BodyPose.DOUBLE_BICEPS)

    def test_one_arm_flexed_is_none(self):
        self.assertIs(classify(r_elbow=(190, 195), r_wrist=(185, 115)), BodyPose.NONE)

    def test_forearms_pointing_down_is_none(self):
        self.assertIs(
            classify(r_elbow=(190, 195), r_wrist=(185, 275), l_elbow=(450, 195), l_wrist=(455, 275)),
            BodyPose.NONE,
        )


class NoneAndAmbiguityTests(unittest.TestCase):
    def test_neutral_arms_down_is_none(self):
        self.assertIs(classify(), BodyPose.NONE)

    def test_two_rules_matching_is_none(self):
        # The geometry keeps the rules disjoint, so force an overlap to check
        # the "more than one match -> NONE" guard itself.
        with mock.patch.object(coco_pose_recognizer, "is_t_pose_arm", return_value=True), \
                mock.patch.object(coco_pose_recognizer, "is_double_biceps_arm", return_value=True):
            self.assertIs(classify(**T_POSE_ARMS), BodyPose.NONE)

    def test_fixtures_match_exactly_one_rule(self):
        for arms, expected in ((T_POSE_ARMS, BodyPose.T_POSE), (DOUBLE_BICEPS_ARMS, BodyPose.DOUBLE_BICEPS),
                               (RIGHT_ARM_UP, BodyPose.ONE_ARM_UP), (LEFT_ARM_UP, BodyPose.ONE_ARM_UP)):
            self.assertIs(classify(**arms), expected)

    def test_zero_shoulder_width_is_none(self):
        kpts = keypoints(**T_POSE_ARMS)
        kpts[LEFT_SHOULDER, :2] = kpts[RIGHT_SHOULDER, :2]
        self.assertIs(classify_pose(kpts), BodyPose.NONE)

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            classify_pose(np.zeros((33, 3)))


class LowConfidenceTests(unittest.TestCase):
    def test_low_confidence_arm_keypoint_blocks_each_pose(self):
        self.assertIs(classify(**T_POSE_ARMS, conf={LEFT_WRIST: 0.3}), BodyPose.NONE)
        self.assertIs(classify(**DOUBLE_BICEPS_ARMS, conf={RIGHT_ELBOW: 0.2}), BodyPose.NONE)
        self.assertIs(classify(**RIGHT_ARM_UP, conf={RIGHT_SHOULDER: 0.1}), BodyPose.NONE)

    def test_unreliable_down_arm_blocks_one_arm_up(self):
        # Can't confirm "exactly one arm up" without seeing the other arm
        self.assertIs(classify(**RIGHT_ARM_UP, conf={LEFT_WRIST: 0.1}), BodyPose.NONE)

    def test_non_finite_keypoint_is_none(self):
        kpts = keypoints(**{**ARMS_DOWN, **T_POSE_ARMS})
        kpts[LEFT_ELBOW, 0] = np.nan
        self.assertIs(classify_pose(kpts), BodyPose.NONE)

    def test_min_conf_is_configurable(self):
        kpts = keypoints(**{**ARMS_DOWN, **T_POSE_ARMS}, conf={LEFT_WRIST: 0.4})
        self.assertIs(classify_pose(kpts), BodyPose.NONE)
        self.assertIs(classify_pose(kpts, min_conf=0.3), BodyPose.T_POSE)


class InvarianceTests(unittest.TestCase):
    def test_distance_independent(self):
        far = {"scale": 0.4, "offset": (150, -60)}
        self.assertIs(classify(**T_POSE_ARMS, **far), BodyPose.T_POSE)
        self.assertIs(classify(**DOUBLE_BICEPS_ARMS, **far), BodyPose.DOUBLE_BICEPS)
        self.assertIs(classify(**RIGHT_ARM_UP, **far), BodyPose.ONE_ARM_UP)
        self.assertIs(classify(**far), BodyPose.NONE)

    def test_mirrored_person(self):
        # Mirrored webcam / person facing away: left and right swap sides in the image
        self.assertIs(classify(**T_POSE_ARMS, mirror=True), BodyPose.T_POSE)
        self.assertIs(classify(**DOUBLE_BICEPS_ARMS, mirror=True), BodyPose.DOUBLE_BICEPS)
        self.assertIs(classify(**LEFT_ARM_UP, mirror=True), BodyPose.ONE_ARM_UP)
        self.assertIs(classify(mirror=True), BodyPose.NONE)


if __name__ == "__main__":
    unittest.main()

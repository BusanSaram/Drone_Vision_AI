"""Synthetic unit tests for pose_track_associator.

Hand-built stick-figure poses and boxes, not real MediaPipe/YOLO output: these
check the association logic only, not real-world multi-person behavior.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pose_track_associator import (  # noqa: E402
    LEFT_SHOULDER, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_WRIST, associate_poses, pose_region,
)

FRAME_W, FRAME_H = 640, 480

# Standing person, relative to their center, at scale 1 (~335 px tall bbox).
BODY = {
    0: (0, -120),                      # nose
    11: (40, -90), 12: (-40, -90),     # shoulders
    13: (50, -30), 14: (-50, -30),     # elbows
    15: (55, 20), 16: (-55, 20),       # wrists
    23: (25, 20), 24: (-25, 20),       # hips
    25: (25, 100), 26: (-25, 100),     # knees
    27: (25, 170), 28: (-25, 170),     # ankles
}


def pose(cx, cy, scale=1.0, joints=None, visibility=None):
    """33 normalized landmarks for a person centered at (cx, cy) pixels."""
    landmarks = [SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]
    for index, (dx, dy) in {**BODY, **(joints or {})}.items():
        x, y = cx + dx * scale, cy + dy * scale
        landmarks[index] = SimpleNamespace(x=x / FRAME_W, y=y / FRAME_H, visibility=0.99)
    for index, value in (visibility or {}).items():
        landmarks[index].visibility = value
    return landmarks


def track(track_id, cx, cy, scale=1.0):
    """Tracked person whose bbox surrounds `pose(cx, cy, scale)` like a YOLO box would."""
    bbox = (int(cx - 70 * scale), int(cy - 150 * scale), int(cx + 70 * scale), int(cy + 185 * scale))
    return SimpleNamespace(track_id=track_id, bbox=bbox)


def associate(poses, tracks):
    return associate_poses(poses, tracks, FRAME_W, FRAME_H)


class AssociationTests(unittest.TestCase):
    def test_one_person_one_pose(self):
        self.assertEqual(associate([pose(320, 250, 0.6)], [track(1, 320, 250, 0.6)]), {1: 0})

    def test_two_people_one_pose_goes_to_correct_person(self):
        tracks = [track(1, 180, 250, 0.6), track(2, 460, 250, 0.6)]
        self.assertEqual(associate([pose(460, 250, 0.6)], tracks), {2: 0})
        self.assertEqual(associate([pose(180, 250, 0.6)], tracks), {1: 0})

    def test_two_people_two_poses(self):
        tracks = [track(7, 180, 250, 0.6), track(9, 460, 250, 0.6)]
        # Pose order from MediaPipe is unrelated to track order.
        poses = [pose(460, 250, 0.6), pose(180, 250, 0.6)]
        self.assertEqual(associate(poses, tracks), {9: 0, 7: 1})

    def test_adjacent_people_with_arm_reaching_into_neighbor_box(self):
        # Person 1 in a T-pose: their right wrist reaches into person 2's box.
        t_pose = {13: (100, -90), 14: (-100, -90), 15: (160, -90), 16: (-160, -90)}
        tracks = [track(1, 250, 250, 0.6), track(2, 350, 250, 0.6)]
        self.assertEqual(associate([pose(250, 250, 0.6, joints=t_pose)], tracks), {1: 0})

    def test_raised_wrist_slightly_outside_bbox_still_matches(self):
        arm_up = {13: (45, -150), 15: (50, -165)}  # wrist a little above the bbox top
        self.assertEqual(associate([pose(320, 250, 0.6, joints=arm_up)], [track(1, 320, 250, 0.6)]), {1: 0})

    def test_person_in_front_of_larger_box(self):
        # A small (farther) person partly in front of a bigger (closer) person's box.
        tracks = [track(1, 320, 240, 1.3), track(2, 330, 250, 0.6)]
        self.assertEqual(associate([pose(330, 250, 0.6)], tracks), {2: 0})

    def test_pose_mostly_outside_bbox_is_not_matched(self):
        # Anchor inside the box, but most landmarks far below it.
        tracks = [SimpleNamespace(track_id=1, bbox=(250, 100, 390, 230))]
        self.assertEqual(associate([pose(320, 250, 0.9)], tracks), {})

    def test_unreliable_shoulders_are_not_associated(self):
        poses = [pose(320, 250, 0.6, visibility={LEFT_SHOULDER: 0.2, RIGHT_SHOULDER: 0.3})]
        self.assertEqual(associate(poses, [track(1, 320, 250, 0.6)]), {})

    def test_too_few_reliable_landmarks_are_not_associated(self):
        hidden = {i: 0.1 for i in (0, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)}  # only shoulders left
        self.assertIsNone(pose_region(pose(320, 250, 0.6, visibility=hidden), FRAME_W, FRAME_H))
        self.assertEqual(associate([pose(320, 250, 0.6, visibility=hidden)], [track(1, 320, 250, 0.6)]), {})

    def test_low_visibility_wrists_are_ignored_not_fatal(self):
        poses = [pose(320, 250, 0.6, visibility={LEFT_WRIST: 0.1, RIGHT_WRIST: 0.1})]
        self.assertEqual(associate(poses, [track(1, 320, 250, 0.6)]), {1: 0})

    def test_ambiguous_overlapping_boxes_leave_pose_unassociated(self):
        # Two near-identical boxes (e.g. duplicate tracks on one person).
        tracks = [track(1, 320, 250, 0.6), track(2, 322, 251, 0.6)]
        self.assertEqual(associate([pose(320, 250, 0.6)], tracks), {})

    def test_no_pose(self):
        self.assertEqual(associate([], [track(1, 320, 250, 0.6)]), {})

    def test_no_tracked_people(self):
        self.assertEqual(associate([pose(320, 250, 0.6)], []), {})

    def test_two_equal_poses_for_one_track_assign_neither(self):
        # e.g. MediaPipe returning a duplicate of the same person
        poses = [pose(320, 250, 0.6), pose(321, 250, 0.6)]
        self.assertEqual(associate(poses, [track(1, 320, 250, 0.6)]), {})

    def test_two_poses_for_one_track_better_one_wins_other_stays_unassociated(self):
        # Both poses are valid candidates for track 1 (IoU ~0.30 and ~0.68), so this
        # exercises the one-pose-per-track rule, not the MIN_IOU cutoff.
        poses = [pose(320, 250, 0.4), pose(320, 250, 0.6)]
        self.assertEqual(associate(poses, [track(1, 320, 250, 0.6)]), {1: 1})

    def test_each_track_gets_at_most_one_pose(self):
        tracks = [track(1, 180, 250, 0.6), track(2, 460, 250, 0.6)]
        poses = [pose(180, 250, 0.6), pose(181, 250, 0.6), pose(460, 250, 0.6)]
        result = associate(poses, tracks)
        self.assertEqual(result, {2: 2})  # track 1 is contested by two equal poses -> none
        self.assertEqual(len(set(result.values())), len(result))


if __name__ == "__main__":
    unittest.main()

"""Unit tests for pose_command_stabilizer: the 0.7 s hold, the NONE-only
dropout grace (2 frames / 0.2 s), and per-track separation.

Explicit `now` timestamps / a fake clock - no sleeping.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coco_pose_recognizer import BodyPose  # noqa: E402
from pose_command_stabilizer import (  # noqa: E402
    HOLD_TIME, NONE_GRACE_FRAMES, NONE_GRACE_TIME, PoseCommandStabilizer, TrackPoseStabilizers,
)

UP = BodyPose.ONE_ARM_UP
NONE = BodyPose.NONE


class StabilizerTests(unittest.TestCase):
    def setUp(self):
        self.stabilizer = PoseCommandStabilizer(hold_time=0.7)

    def test_default_hold_time(self):
        self.assertEqual(HOLD_TIME, 0.7)

    def test_not_active_before_hold_time(self):
        s = self.stabilizer
        s.update(BodyPose.T_POSE, now=0.0)
        state = s.update(BodyPose.T_POSE, now=0.69)
        self.assertIs(state.raw_pose, BodyPose.T_POSE)
        self.assertIs(state.candidate, BodyPose.T_POSE)
        self.assertIs(state.active, BodyPose.NONE)
        self.assertAlmostEqual(state.held_for, 0.69)

    def test_activates_once_after_hold_time(self):
        s = self.stabilizer
        s.update(BodyPose.ONE_ARM_UP, now=0.0)
        first = s.update(BodyPose.ONE_ARM_UP, now=0.7)
        later = s.update(BodyPose.ONE_ARM_UP, now=2.0)
        self.assertIs(first.active, BodyPose.ONE_ARM_UP)
        self.assertTrue(first.newly_activated)
        self.assertIs(later.active, BodyPose.ONE_ARM_UP)
        self.assertFalse(later.newly_activated)  # no repeated firing while held

    def test_change_before_hold_resets_candidate(self):
        s = self.stabilizer
        s.update(BodyPose.T_POSE, now=0.0)
        s.update(BodyPose.DOUBLE_BICEPS, now=0.5)
        self.assertIs(s.update(BodyPose.DOUBLE_BICEPS, now=1.0).active, BodyPose.NONE)  # only 0.5 s
        self.assertIs(s.update(BodyPose.DOUBLE_BICEPS, now=1.2).active, BodyPose.DOUBLE_BICEPS)

    def test_single_wrong_frame_does_not_fire_and_restarts_hold(self):
        s = self.stabilizer
        s.update(BodyPose.ONE_ARM_UP, now=0.0)
        glitch = s.update(BodyPose.T_POSE, now=0.4)
        self.assertIs(glitch.active, BodyPose.NONE)
        self.assertFalse(glitch.newly_activated)
        s.update(BodyPose.ONE_ARM_UP, now=0.5)
        self.assertIs(s.update(BodyPose.ONE_ARM_UP, now=1.1).active, BodyPose.NONE)
        self.assertIs(s.update(BodyPose.ONE_ARM_UP, now=1.2).active, BodyPose.ONE_ARM_UP)

    def test_active_command_clears_immediately_when_pose_changes(self):
        s = self.stabilizer
        s.update(BodyPose.T_POSE, now=0.0)
        s.update(BodyPose.T_POSE, now=0.8)
        changed = s.update(BodyPose.DOUBLE_BICEPS, now=0.9)
        self.assertIs(changed.active, BodyPose.NONE)
        self.assertIs(changed.candidate, BodyPose.DOUBLE_BICEPS)

    def test_release_clears_and_rehold_fires_again(self):
        # A real release = NONE beyond the grace limits (here 0.5 s after the
        # last matching frame), unlike a short dropout (see GraceTests).
        s = self.stabilizer
        s.update(BodyPose.DOUBLE_BICEPS, now=0.0)
        self.assertTrue(s.update(BodyPose.DOUBLE_BICEPS, now=0.8).newly_activated)
        released = s.update(BodyPose.NONE, now=1.3)
        self.assertIs(released.active, BodyPose.NONE)
        self.assertIs(released.candidate, BodyPose.NONE)
        self.assertEqual(released.held_for, 0.0)
        self.assertFalse(released.in_grace)
        s.update(BodyPose.DOUBLE_BICEPS, now=1.5)
        self.assertTrue(s.update(BodyPose.DOUBLE_BICEPS, now=2.2).newly_activated)

    def test_none_never_activates(self):
        s = self.stabilizer
        s.update(BodyPose.NONE, now=0.0)
        state = s.update(BodyPose.NONE, now=5.0)
        self.assertIs(state.active, BodyPose.NONE)
        self.assertFalse(state.newly_activated)

    def test_uses_injected_clock(self):
        clock = [10.0]
        s = PoseCommandStabilizer(hold_time=0.7, time_fn=lambda: clock[0])
        s.update(BodyPose.T_POSE)
        clock[0] = 10.6
        self.assertIs(s.update(BodyPose.T_POSE).active, BodyPose.NONE)
        clock[0] = 10.75
        self.assertIs(s.update(BodyPose.T_POSE).active, BodyPose.T_POSE)


class GraceTests(unittest.TestCase):
    """NONE-only dropout grace: <= 2 consecutive NONE frames AND <= 0.2 s since last match."""

    def setUp(self):
        self.stabilizer = PoseCommandStabilizer(hold_time=0.7, none_grace_frames=2, none_grace_time=0.2)

    def test_default_grace_values(self):
        self.assertEqual(NONE_GRACE_FRAMES, 2)
        self.assertEqual(NONE_GRACE_TIME, 0.2)

    def test_one_none_frame_preserves_hold(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        s.update(UP, now=0.1)
        dropout = s.update(NONE, now=0.15)
        self.assertTrue(dropout.in_grace)
        self.assertIs(dropout.raw_pose, NONE)
        self.assertIs(dropout.candidate, UP)
        self.assertAlmostEqual(dropout.held_for, 0.15)  # timer keeps running
        s.update(UP, now=0.2)
        self.assertIs(s.update(UP, now=0.6).active, NONE)
        confirmed = s.update(UP, now=0.7)  # 0.7 s from the ORIGINAL start
        self.assertIs(confirmed.active, UP)
        self.assertTrue(confirmed.newly_activated)

    def test_two_none_frames_within_limits_preserve_hold(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        s.update(UP, now=0.3)
        self.assertTrue(s.update(NONE, now=0.35).in_grace)
        self.assertTrue(s.update(NONE, now=0.4).in_grace)
        back = s.update(UP, now=0.45)
        self.assertIs(back.candidate, UP)
        self.assertAlmostEqual(back.held_for, 0.45)
        self.assertTrue(s.update(UP, now=0.7).newly_activated)

    def test_third_consecutive_none_resets(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        s.update(NONE, now=0.03)
        s.update(NONE, now=0.06)
        third = s.update(NONE, now=0.09)  # frame limit exceeded, time still < 0.2 s
        self.assertFalse(third.in_grace)
        self.assertIs(third.candidate, NONE)
        self.assertEqual(third.held_for, 0.0)
        # The pose must now be held a full 0.7 s again.
        s.update(UP, now=0.12)
        self.assertIs(s.update(UP, now=0.8).active, NONE)  # only 0.68 s
        self.assertTrue(s.update(UP, now=0.83).newly_activated)

    def test_none_beyond_grace_time_resets_within_frame_limit(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        late = s.update(NONE, now=0.25)  # 1st NONE frame, but 0.25 s > 0.2 s (e.g. low FPS)
        self.assertFalse(late.in_grace)
        self.assertIs(late.candidate, NONE)

    def test_second_none_beyond_grace_time_resets(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        self.assertTrue(s.update(NONE, now=0.1).in_grace)
        late = s.update(NONE, now=0.3)  # 2 frames allowed, but 0.3 s since last UP
        self.assertFalse(late.in_grace)
        self.assertIs(late.candidate, NONE)

    def test_matching_frame_resets_none_count(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        s.update(NONE, now=0.03)
        s.update(NONE, now=0.06)
        s.update(UP, now=0.09)  # streak back to 0
        self.assertTrue(s.update(NONE, now=0.12).in_grace)
        self.assertTrue(s.update(NONE, now=0.15).in_grace)
        state = s.update(UP, now=0.18)
        self.assertIs(state.candidate, UP)
        self.assertAlmostEqual(state.held_for, 0.18)  # still the original hold

    def test_one_arm_up_to_t_pose_resets_immediately(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        s.update(UP, now=0.5)
        changed = s.update(BodyPose.T_POSE, now=0.55)
        self.assertFalse(changed.in_grace)
        self.assertIs(changed.candidate, BodyPose.T_POSE)
        self.assertEqual(changed.held_for, 0.0)
        self.assertIs(s.update(BodyPose.T_POSE, now=1.2).active, NONE)  # own hold: 0.65 s
        self.assertTrue(s.update(BodyPose.T_POSE, now=1.25).newly_activated)

    def test_one_arm_up_to_double_biceps_resets_immediately(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        s.update(UP, now=0.5)
        changed = s.update(BodyPose.DOUBLE_BICEPS, now=0.55)
        self.assertFalse(changed.in_grace)
        self.assertIs(changed.candidate, BodyPose.DOUBLE_BICEPS)
        self.assertEqual(changed.held_for, 0.0)
        self.assertIs(s.update(BodyPose.DOUBLE_BICEPS, now=1.2).active, NONE)
        self.assertTrue(s.update(BodyPose.DOUBLE_BICEPS, now=1.25).newly_activated)

    def test_different_pose_after_none_is_not_bridged(self):
        # UP, NONE (grace), T_POSE: T_POSE must still reset, not continue UP.
        s = self.stabilizer
        s.update(UP, now=0.0)
        s.update(UP, now=0.5)
        s.update(NONE, now=0.55)
        changed = s.update(BodyPose.T_POSE, now=0.6)
        self.assertIs(changed.candidate, BodyPose.T_POSE)
        self.assertEqual(changed.held_for, 0.0)

    def test_cannot_activate_on_none_frame(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        s.update(UP, now=0.6)
        dropout = s.update(NONE, now=0.75)  # hold time passes during the dropout
        self.assertTrue(dropout.in_grace)
        self.assertGreaterEqual(dropout.held_for, 0.7)
        self.assertIs(dropout.active, NONE)
        self.assertFalse(dropout.newly_activated)
        back = s.update(UP, now=0.78)  # confirms on the first real frame
        self.assertIs(back.active, UP)
        self.assertTrue(back.newly_activated)

    def test_confirmed_command_survives_short_dropout(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        self.assertTrue(s.update(UP, now=0.7).newly_activated)
        dropout = s.update(NONE, now=0.75)
        self.assertTrue(dropout.in_grace)
        self.assertIs(dropout.active, UP)
        self.assertFalse(dropout.newly_activated)

    def test_no_duplicate_activation_after_dropout(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        self.assertTrue(s.update(UP, now=0.7).newly_activated)
        s.update(NONE, now=0.75)
        back = s.update(UP, now=0.8)
        self.assertIs(back.active, UP)
        self.assertFalse(back.newly_activated)
        self.assertFalse(s.update(UP, now=2.0).newly_activated)

    def test_confirmed_command_clears_immediately_on_t_pose(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        s.update(UP, now=0.7)
        changed = s.update(BodyPose.T_POSE, now=0.75)
        self.assertIs(changed.active, NONE)
        self.assertIs(changed.candidate, BodyPose.T_POSE)
        self.assertFalse(changed.in_grace)

    def test_confirmed_command_clears_when_dropout_exceeds_grace(self):
        s = self.stabilizer
        s.update(UP, now=0.0)
        s.update(UP, now=0.7)
        s.update(NONE, now=0.73)
        s.update(NONE, now=0.76)
        gone = s.update(NONE, now=0.79)
        self.assertIs(gone.active, NONE)
        s.update(UP, now=0.82)
        self.assertTrue(s.update(UP, now=1.52).newly_activated)  # fresh hold, new event

    def test_none_without_candidate_is_not_grace(self):
        state = self.stabilizer.update(NONE, now=0.0)
        self.assertFalse(state.in_grace)
        self.assertIs(state.candidate, NONE)


class TrackPoseStabilizersTests(unittest.TestCase):
    def setUp(self):
        self.stabilizers = TrackPoseStabilizers(hold_time=0.7)

    def test_tracks_are_independent(self):
        s = self.stabilizers
        s.update(1, BodyPose.ONE_ARM_UP, now=0.0)
        s.update(2, BodyPose.T_POSE, now=0.3)
        one = s.update(1, BodyPose.ONE_ARM_UP, now=0.7)
        two = s.update(2, BodyPose.T_POSE, now=0.7)
        self.assertIs(one.active, BodyPose.ONE_ARM_UP)  # held 0.7 s
        self.assertIs(two.active, BodyPose.NONE)  # held only 0.4 s
        self.assertIs(s.update(2, BodyPose.T_POSE, now=1.0).active, BodyPose.T_POSE)

    def test_one_track_changing_does_not_reset_another(self):
        s = self.stabilizers
        s.update(1, BodyPose.DOUBLE_BICEPS, now=0.0)
        s.update(2, BodyPose.DOUBLE_BICEPS, now=0.0)
        s.update(2, BodyPose.NONE, now=0.4)  # track 2 drops the pose
        self.assertIs(s.update(1, BodyPose.DOUBLE_BICEPS, now=0.7).active, BodyPose.DOUBLE_BICEPS)
        self.assertIs(s.update(2, BodyPose.DOUBLE_BICEPS, now=0.7).active, BodyPose.NONE)

    def test_prune_removes_stale_tracks(self):
        s = self.stabilizers
        s.update(1, BodyPose.T_POSE, now=0.0)
        s.update(2, BodyPose.T_POSE, now=0.0)
        s.prune({2})
        self.assertEqual(s.track_ids, {2})
        s.prune([])
        self.assertEqual(s.track_ids, set())

    def test_returning_track_restarts_hold(self):
        s = self.stabilizers
        s.update(1, BodyPose.T_POSE, now=0.0)
        self.assertIs(s.update(1, BodyPose.T_POSE, now=0.8).active, BodyPose.T_POSE)
        s.prune(set())  # track 1 lost / no longer eligible
        state = s.update(1, BodyPose.T_POSE, now=1.0)
        self.assertIs(state.active, BodyPose.NONE)
        self.assertEqual(state.held_for, 0.0)
        self.assertTrue(s.update(1, BodyPose.T_POSE, now=1.7).newly_activated)

    def test_grace_state_is_independent_per_track(self):
        s = self.stabilizers
        s.update(3, UP, now=0.0)
        s.update(7, UP, now=0.0)
        self.assertTrue(s.update(3, NONE, now=0.05).in_grace)  # track 3: one dropout
        s.update(7, NONE, now=0.03)  # track 7: three dropouts -> reset
        s.update(7, NONE, now=0.05)
        self.assertIs(s.update(7, NONE, now=0.07).candidate, NONE)
        self.assertIs(s.update(3, UP, now=0.1).candidate, UP)  # unaffected by track 7
        self.assertTrue(s.update(3, UP, now=0.7).newly_activated)
        s.update(7, UP, now=0.1)
        self.assertIs(s.update(7, UP, now=0.7).active, NONE)  # restarted at 0.1

    def test_pruned_track_in_grace_starts_fresh(self):
        s = self.stabilizers
        s.update(1, UP, now=0.0)
        self.assertTrue(s.update(1, NONE, now=0.05).in_grace)
        s.prune(set())  # track dropped by the validator this frame
        state = s.update(1, UP, now=0.1)
        self.assertEqual(state.held_for, 0.0)
        self.assertIs(s.update(1, UP, now=0.7).active, NONE)  # new hold from 0.1
        self.assertTrue(s.update(1, UP, now=0.8).newly_activated)

    def test_grace_settings_passed_to_each_track(self):
        s = TrackPoseStabilizers(hold_time=0.7, none_grace_frames=0, none_grace_time=0.2)
        s.update(1, UP, now=0.0)
        self.assertFalse(s.update(1, NONE, now=0.03).in_grace)  # grace disabled


if __name__ == "__main__":
    unittest.main()

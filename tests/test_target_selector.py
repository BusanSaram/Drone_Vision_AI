"""Unit tests for target_selector: SEARCHING / FOLLOWING / LAND_REQUESTED.

`SelectorTests` drive the state machine directly with per-frame activations.
`WithStabilizerTests` run real TrackPoseStabilizers in front of it (explicit
timestamps, no sleeping) to check the edge-triggered behavior end to end,
in particular that an ambiguous selection is not resolved by someone simply
continuing to hold ONE_ARM_UP.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coco_pose_recognizer import BodyPose  # noqa: E402
from pose_command_stabilizer import TrackPoseStabilizers  # noqa: E402
from target_selector import SelectionEvent, SelectionState, TargetSelector  # noqa: E402

UP = BodyPose.ONE_ARM_UP
T = BodyPose.T_POSE
BICEPS = BodyPose.DOUBLE_BICEPS
NONE = BodyPose.NONE


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.selector = TargetSelector()

    def select(self, track_id):
        result = self.selector.update({track_id: UP}, target_alive=False)
        self.assertIs(result.event, SelectionEvent.TARGET_SELECTED)
        return result

    def test_starts_searching_without_target(self):
        self.assertIs(self.selector.state, SelectionState.SEARCHING)
        self.assertIsNone(self.selector.target_id)

    def test_no_activations_stays_searching(self):
        result = self.selector.update({}, target_alive=False)
        self.assertIs(result.state, SelectionState.SEARCHING)
        self.assertIsNone(result.event)

    def test_single_one_arm_up_selects_target(self):
        result = self.select(4)
        self.assertIs(result.state, SelectionState.FOLLOWING)
        self.assertEqual(result.target_id, 4)
        self.assertEqual(result.event_track_ids, (4,))

    def test_other_commands_ignored_while_searching(self):
        for pose in (T, BICEPS):
            result = self.selector.update({1: pose}, target_alive=False)
            self.assertIs(result.state, SelectionState.SEARCHING)
            self.assertIsNone(result.event)

    def test_one_arm_up_selected_even_if_others_issue_other_commands(self):
        result = self.selector.update({1: T, 2: UP, 3: BICEPS}, target_alive=False)
        self.assertIs(result.event, SelectionEvent.TARGET_SELECTED)
        self.assertEqual(result.target_id, 2)

    def test_simultaneous_one_arm_up_selects_nobody(self):
        result = self.selector.update({5: UP, 2: UP}, target_alive=False)
        self.assertIs(result.event, SelectionEvent.SELECTION_AMBIGUOUS)
        self.assertIs(result.state, SelectionState.SEARCHING)
        self.assertIsNone(result.target_id)
        self.assertEqual(result.event_track_ids, (2, 5))

    def test_later_single_activation_after_ambiguity_selects(self):
        self.selector.update({1: UP, 2: UP}, target_alive=False)
        self.selector.update({}, target_alive=False)
        result = self.selector.update({2: UP}, target_alive=False)
        self.assertIs(result.event, SelectionEvent.TARGET_SELECTED)
        self.assertEqual(result.target_id, 2)

    def test_following_stays_when_nothing_happens(self):
        self.select(1)
        result = self.selector.update({}, target_alive=True)
        self.assertIs(result.state, SelectionState.FOLLOWING)
        self.assertEqual(result.target_id, 1)
        self.assertIsNone(result.event)

    def test_other_people_ignored_while_following(self):
        self.select(1)
        for pose in (UP, T, BICEPS):
            result = self.selector.update({2: pose}, target_alive=True)
            self.assertIs(result.state, SelectionState.FOLLOWING)
            self.assertEqual(result.target_id, 1)
            self.assertIsNone(result.event)

    def test_target_one_arm_up_while_following_is_noop(self):
        self.select(1)
        result = self.selector.update({1: UP}, target_alive=True)
        self.assertIs(result.state, SelectionState.FOLLOWING)
        self.assertIsNone(result.event)

    def test_target_t_pose_releases_to_searching(self):
        self.select(1)
        result = self.selector.update({1: T}, target_alive=True)
        self.assertIs(result.event, SelectionEvent.TARGET_RELEASED)
        self.assertIs(result.state, SelectionState.SEARCHING)
        self.assertIsNone(result.target_id)

    def test_target_double_biceps_requests_land(self):
        self.select(1)
        result = self.selector.update({1: BICEPS, 2: T}, target_alive=True)
        self.assertIs(result.event, SelectionEvent.LAND_REQUESTED)
        self.assertIs(result.state, SelectionState.LAND_REQUESTED)
        self.assertEqual(result.target_id, 1)

    def test_land_requested_is_terminal_within_session(self):
        self.select(1)
        self.selector.update({1: BICEPS}, target_alive=True)
        for activations, alive in (({1: T}, True), ({1: UP, 2: UP}, True), ({3: UP}, False), ({}, False)):
            result = self.selector.update(activations, target_alive=alive)
            self.assertIs(result.state, SelectionState.LAND_REQUESTED)
            self.assertEqual(result.target_id, 1)
            self.assertIsNone(result.event)

    def test_target_lost_returns_to_searching(self):
        self.select(1)
        result = self.selector.update({}, target_alive=False)
        self.assertIs(result.event, SelectionEvent.TARGET_LOST)
        self.assertEqual(result.event_track_ids, (1,))
        self.assertIs(result.state, SelectionState.SEARCHING)
        self.assertIsNone(result.target_id)

    def test_target_lost_does_not_switch_to_other_person(self):
        self.select(1)
        # Someone else completes ONE_ARM_UP on the very frame the target is lost.
        result = self.selector.update({2: UP}, target_alive=False)
        self.assertIs(result.event, SelectionEvent.TARGET_LOST)
        self.assertIsNone(result.target_id)
        # Their activation was consumed in FOLLOWING; nothing selects them now.
        self.assertIsNone(self.selector.update({}, target_alive=False).target_id)

    def test_lost_takes_priority_over_target_command(self):
        self.select(1)
        result = self.selector.update({1: BICEPS}, target_alive=False)
        self.assertIs(result.event, SelectionEvent.TARGET_LOST)
        self.assertIs(result.state, SelectionState.SEARCHING)

    def test_reselect_after_release(self):
        self.select(1)
        self.selector.update({1: T}, target_alive=True)
        result = self.selector.update({3: UP}, target_alive=False)
        self.assertIs(result.event, SelectionEvent.TARGET_SELECTED)
        self.assertEqual(result.target_id, 3)


class ResetTests(unittest.TestCase):
    """reset() = new autonomous session: SEARCHING, no target, from any state."""

    def setUp(self):
        self.selector = TargetSelector()

    def land(self, track_id):
        self.selector.update({track_id: UP}, target_alive=False)
        self.selector.update({track_id: BICEPS}, target_alive=True)
        self.assertIs(self.selector.state, SelectionState.LAND_REQUESTED)

    def test_reset_from_land_requested(self):
        self.land(1)
        self.selector.reset()
        self.assertIs(self.selector.state, SelectionState.SEARCHING)
        self.assertIsNone(self.selector.target_id)

    def test_select_after_reset_from_land_requested(self):
        self.land(1)
        self.selector.reset()
        result = self.selector.update({2: UP}, target_alive=False)
        self.assertIs(result.event, SelectionEvent.TARGET_SELECTED)
        self.assertEqual(result.target_id, 2)

    def test_reset_from_following_clears_target(self):
        self.selector.update({1: UP}, target_alive=False)
        self.selector.reset()
        self.assertIs(self.selector.state, SelectionState.SEARCHING)
        self.assertIsNone(self.selector.target_id)
        # The old target's commands no longer matter.
        result = self.selector.update({1: BICEPS}, target_alive=True)
        self.assertIs(result.state, SelectionState.SEARCHING)
        self.assertIsNone(result.event)

    def test_reset_while_searching_is_noop(self):
        self.selector.reset()
        self.assertIs(self.selector.state, SelectionState.SEARCHING)
        self.assertIsNone(self.selector.target_id)
        self.assertIs(self.selector.update({4: UP}, target_alive=False).event, SelectionEvent.TARGET_SELECTED)


class WithStabilizerTests(unittest.TestCase):
    """Real 0.7 s stabilizers feeding the selector, as in yolo_pose_demo.py."""

    def setUp(self):
        self.stabilizers = TrackPoseStabilizers(hold_time=0.7)
        self.selector = TargetSelector()
        self.alive = set()  # stands in for TrackValidator.is_tracked

    def frame(self, now, poses):
        """One frame: `poses` = {track_id: raw pose} for visible CONFIRMED tracks."""
        activations = {}
        for track_id, raw in poses.items():
            state = self.stabilizers.update(track_id, raw, now)
            if state.newly_activated:
                activations[track_id] = state.active
        self.stabilizers.prune(poses)
        target = self.selector.target_id
        return self.selector.update(activations, target_alive=target is not None and target in self.alive)

    def test_hold_shorter_than_0_7_s_does_not_select(self):
        self.frame(0.0, {1: UP})
        result = self.frame(0.69, {1: UP})
        self.assertIs(result.state, SelectionState.SEARCHING)
        self.assertIs(self.frame(0.7, {1: UP}).event, SelectionEvent.TARGET_SELECTED)

    def test_ambiguous_then_other_releases_does_not_select_holder(self):
        self.frame(0.0, {1: UP, 2: UP})
        self.assertIs(self.frame(0.7, {1: UP, 2: UP}).event, SelectionEvent.SELECTION_AMBIGUOUS)
        # Person 2 lowers their arm; person 1 keeps holding.
        for t in (0.8, 1.0, 1.5, 2.0, 3.0):
            result = self.frame(t, {1: UP, 2: NONE})
            self.assertIs(result.state, SelectionState.SEARCHING)
            self.assertIsNone(result.event)

    def test_after_ambiguity_new_hold_selects(self):
        self.frame(0.0, {1: UP, 2: UP})
        self.frame(0.7, {1: UP, 2: UP})  # ambiguous
        self.frame(1.0, {1: NONE, 2: NONE})  # both release (beyond NONE grace)
        self.frame(1.1, {1: UP, 2: NONE})  # person 1 starts a new hold
        self.assertIsNone(self.frame(1.79, {1: UP, 2: NONE}).event)
        result = self.frame(1.8, {1: UP, 2: NONE})
        self.assertIs(result.event, SelectionEvent.TARGET_SELECTED)
        self.assertEqual(result.target_id, 1)

    def test_one_arm_up_held_during_following_does_not_select_on_return(self):
        self.alive = {1, 2}
        self.frame(0.0, {1: UP})
        self.frame(0.7, {1: UP})  # target 1
        self.frame(1.0, {1: NONE, 2: UP})  # person 2 raises during FOLLOWING
        self.frame(1.8, {1: T, 2: UP})  # person 2's hold completes: ignored
        self.assertEqual(self.selector.target_id, 1)
        result = self.frame(2.5, {1: T, 2: UP})  # target 1's T_POSE completes
        self.assertIs(result.event, SelectionEvent.TARGET_RELEASED)
        # Person 2 is still holding, but their activation was already consumed.
        for t in (2.6, 3.0, 4.0):
            self.assertIsNone(self.frame(t, {1: NONE, 2: UP}).target_id)

    def test_target_hidden_within_validator_grace_keeps_target(self):
        self.alive = {1}
        self.frame(0.0, {1: UP})
        self.frame(0.7, {1: UP})
        # Track 1 missing from this frame but still known to the validator.
        result = self.frame(1.0, {})
        self.assertIs(result.state, SelectionState.FOLLOWING)
        self.assertEqual(result.target_id, 1)
        self.alive = set()  # validator expired it
        self.assertIs(self.frame(3.0, {}).event, SelectionEvent.TARGET_LOST)

    # ---- new autonomous session (selector + stabilizers reset, as start_new_session() does) ----

    def new_session(self):
        self.selector.reset()
        self.stabilizers.reset()

    def test_hold_started_before_session_does_not_count(self):
        self.frame(0.0, {1: UP})
        self.frame(0.6, {1: UP})  # 0.6 s held in the old session
        self.new_session()
        self.frame(0.6, {1: UP})  # first frame of the new session: hold restarts
        self.assertIsNone(self.frame(0.7, {1: UP}).event)  # would have fired without the reset
        self.assertIsNone(self.frame(1.29, {1: UP}).event)
        result = self.frame(1.3, {1: UP})  # 0.7 s after the session started
        self.assertIs(result.event, SelectionEvent.TARGET_SELECTED)
        self.assertEqual(result.target_id, 1)

    def test_completed_hold_is_not_carried_into_new_session(self):
        self.frame(0.0, {1: UP, 2: UP})
        self.frame(0.7, {1: UP, 2: UP})  # ambiguous: both activations consumed
        self.new_session()
        self.frame(1.0, {1: UP, 2: NONE})  # person 1 still holding
        self.assertIsNone(self.frame(1.69, {1: UP, 2: NONE}).event)
        result = self.frame(1.7, {1: UP, 2: NONE})  # fresh 0.7 s hold in the new session
        self.assertIs(result.event, SelectionEvent.TARGET_SELECTED)
        self.assertEqual(result.target_id, 1)

    def test_land_then_new_session_then_select(self):
        self.alive = {1, 2}
        self.frame(0.0, {1: UP})
        self.frame(0.7, {1: UP})  # target 1
        self.frame(1.0, {1: BICEPS})
        self.assertIs(self.frame(1.7, {1: BICEPS}).event, SelectionEvent.LAND_REQUESTED)
        # Same session: every pose is ignored.
        self.frame(2.0, {1: T, 2: UP})
        for t in (2.7, 3.0, 4.0):
            result = self.frame(t, {1: T, 2: UP})
            self.assertIs(result.state, SelectionState.LAND_REQUESTED)
            self.assertIsNone(result.event)
        self.new_session()
        self.frame(5.0, {1: NONE, 2: UP})
        result = self.frame(5.7, {1: NONE, 2: UP})
        self.assertIs(result.event, SelectionEvent.TARGET_SELECTED)
        self.assertEqual(result.target_id, 2)


if __name__ == "__main__":
    unittest.main()

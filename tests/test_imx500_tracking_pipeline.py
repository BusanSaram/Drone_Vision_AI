"""Unit tests for raspberry_pi/imx500_tracking_pipeline.py.

Runs the REAL reused modules (Ultralytics BoT-SORT via PersonTracker, TrackValidator,
pose recognition, stabilizer, TargetSelector) on synthetic detections arriving at
the IMX500's 8 results/s. No modlib, no camera. Pose stick figures come from
test_coco_pose_recognizer (same tests/ folder).

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "raspberry_pi"))

from ultralytics.trackers.basetrack import BaseTrack  # noqa: E402

import imx500_tracking_pipeline as tp  # noqa: E402
from person_tracker import TrackedPerson  # noqa: E402
from pose_detections import PoseDetections  # noqa: E402
from target_selector import SelectionEvent, SelectionState  # noqa: E402
from test_coco_pose_recognizer import ARMS_DOWN, RIGHT_ARM_UP, T_POSE_ARMS, keypoints  # noqa: E402
from track_validator import TrackState, ValidatedTrack  # noqa: E402

DT = 0.125  # 8 results/s
PERSON_BOX = (100, 20, 540, 470)  # encloses the test stick figure


def person(score=0.9, arms=None):
    kpts = keypoints(**{**ARMS_DOWN, **(arms or {})})
    return PoseDetections.from_arrays([PERSON_BOX], [score], [0], kpts[None])


def nobody():
    return PoseDetections.empty()


def state_of(result, track_id=None):
    states = [v.state for v in result.validated_tracks
              if track_id is None or v.tracked_person.track_id == track_id]
    return states[0] if states else None


class PipelineTestBase(unittest.TestCase):
    def setUp(self):
        BaseTrack.reset_id()
        self.t = 0.0

    def feed(self, pipeline, seconds, make, frame=None):
        """Feed NEW results every DT for `seconds`; return (results, selector events)."""
        results, events = [], []
        for _ in range(round(seconds / DT)):
            r = pipeline.update_new(make(), frame, self.t)
            results.append(r)
            if r.selection.event is not None:
                events.append((self.t, r.selection.event))
            self.t += DT
        return results, events

    def select_target(self, pipeline):
        """Confirm one person, then hold ONE_ARM_UP until TargetSelector selects them."""
        self.feed(pipeline, 0.625, person)  # CONFIRMED at 0.5 s
        _, events = self.feed(pipeline, 1.0, lambda: person(arms=RIGHT_ARM_UP))
        self.assertIn(SelectionEvent.TARGET_SELECTED, [e for _, e in events])
        self.assertIs(pipeline.selector.state, SelectionState.FOLLOWING)
        return pipeline.selector.target_id


class ConfirmationTest(PipelineTestBase):
    def test_confirms_after_half_second_at_8_results_per_second(self):
        pipeline = tp.ImxTrackingPipeline()
        results, _ = self.feed(pipeline, 0.625, person)
        self.assertIs(state_of(results[3]), TrackState.CANDIDATE)  # t = 0.375
        self.assertIs(state_of(results[4]), TrackState.CONFIRMED)  # t = 0.5 -> 5 observations

    def test_new_track_thresh_against_imx500_scores(self):
        # 0.730 / 0.621 are the two person scores seen in the first IMX500 test
        for score, expected in ((0.73, {0.8: 0, 0.7: 1, 0.6: 1}), (0.62, {0.8: 0, 0.7: 0, 0.6: 1})):
            BaseTrack.reset_id()
            self.t = 0.0
            pipeline = tp.ImxTrackingPipeline(
                new_track_thresh=0.8, comparisons=((0.7, tp.DEFAULT_TRACKER_CONFIG), (0.6, tp.DEFAULT_TRACKER_CONFIG)))
            self.feed(pipeline, 1.0, lambda: person(score))
            got = {v.new_track_thresh: len(v.stats.confirmed_ids) for v in [pipeline.primary, *pipeline.comparisons]}
            self.assertEqual(got, expected, f"score {score}")

    def test_comparison_trackers_never_drive_selection(self):
        pipeline = tp.ImxTrackingPipeline(new_track_thresh=0.8, comparisons=((0.6, tp.DEFAULT_TRACKER_CONFIG),))
        self.feed(pipeline, 0.625, lambda: person(0.73))
        _, events = self.feed(pipeline, 1.0, lambda: person(0.73, RIGHT_ARM_UP))
        self.assertEqual(events, [])  # 0.6 tracker confirmed the person, but only the primary (0.8) counts
        self.assertEqual(len(pipeline.comparisons[0].stats.confirmed_ids), 1)
        self.assertIs(pipeline.selector.state, SelectionState.SEARCHING)

    def test_gmc_frame_only_for_primary(self):
        pipeline = tp.ImxTrackingPipeline(comparisons=((0.7, tp.DEFAULT_TRACKER_CONFIG),))
        seen = {}
        for name, variant in (("primary", pipeline.primary), ("compare", pipeline.comparisons[0])):
            original = variant.tracker.update
            variant.tracker.update = lambda d, f, _n=name, _o=original: (seen.setdefault(_n, f), _o(d, f))[1]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.update_new(person(), frame, 0.0)
        self.assertIs(seen["primary"], frame)
        self.assertIsNone(seen["compare"])
        self.assertEqual(pipeline.comparisons[0].label, "thr=0.7 buf=30 gmc=off")

    def test_reused_frames_are_not_observations(self):
        pipeline = tp.ImxTrackingPipeline()
        for i in range(5):  # NEW at 0 .. 0.5 s with REUSED frames in between
            r = pipeline.update_new(person(), None, i * DT)
            self.assertIsNone(pipeline.check_stall(i * DT + DT / 2))
        self.assertIs(state_of(r), TrackState.CONFIRMED)
        self.assertEqual(pipeline.primary.stats.updates, 5)
        self.assertEqual(pipeline.primary.stats.confirmation_resets, 0)

    def test_frame_for_gmc_is_accepted(self):
        pipeline = tp.ImxTrackingPipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results, _ = self.feed(pipeline, 0.625, person, frame=frame)
        self.assertIs(state_of(results[-1]), TrackState.CONFIRMED)


class TargetLossTest(PipelineTestBase):
    def test_person_leaves_while_results_continue(self):
        pipeline = tp.ImxTrackingPipeline()
        self.select_target(pipeline)
        self.assertTrue(pipeline.target_status(self.t - DT).usable)

        left_at = self.t - DT  # last update that saw the target
        _, events = self.feed(pipeline, 2.5, nobody)
        self.assertEqual([e for _, e in events], [SelectionEvent.TARGET_LOST])
        lost_at = events[0][0]
        self.assertGreaterEqual(lost_at - left_at, 2.0)  # LOST_GRACE_TIME, measured on NEW results
        self.assertLess(lost_at - left_at, 2.0 + DT + 1e-9)
        self.assertIs(pipeline.selector.state, SelectionState.SEARCHING)
        self.assertIsNone(pipeline.selector.target_id)
        self.assertFalse(pipeline.target_status(self.t).usable)

    def test_target_not_usable_during_grace_period(self):
        pipeline = tp.ImxTrackingPipeline()
        self.select_target(pipeline)
        self.feed(pipeline, 0.25, nobody)  # still FOLLOWING (inside the 2 s grace) but not seen
        status = pipeline.target_status(self.t - DT)
        self.assertIs(status.state, SelectionState.FOLLOWING)
        self.assertFalse(status.usable)
        self.assertEqual(status.reason, "target not in latest result")

    def test_inference_stall_expires_target(self):
        pipeline = tp.ImxTrackingPipeline()
        self.select_target(pipeline)
        last = self.t - DT  # last NEW result

        status = pipeline.target_status(last + 0.6)
        self.assertFalse(status.usable)
        self.assertEqual(status.reason, "result stale")
        self.assertIsNone(pipeline.check_stall(last + 1.9))
        self.assertIs(pipeline.selector.state, SelectionState.FOLLOWING)

        selection = pipeline.check_stall(last + 2.0)
        self.assertIs(selection.event, SelectionEvent.TARGET_LOST)
        self.assertIs(pipeline.selector.state, SelectionState.SEARCHING)
        self.assertEqual(pipeline.last_result.validated_tracks, [])
        self.assertIsNone(pipeline.check_stall(last + 2.5))  # only once per stall

        self.t = last + 3.0  # results resume: the person must be confirmed again
        results, events = self.feed(pipeline, 0.25, person)
        self.assertIs(state_of(results[-1]), TrackState.CANDIDATE)
        self.assertEqual(events, [])

    def test_t_pose_releases_target(self):
        pipeline = tp.ImxTrackingPipeline()
        self.select_target(pipeline)
        self.feed(pipeline, 0.25, person)  # release the raised arm
        _, events = self.feed(pipeline, 1.0, lambda: person(arms=T_POSE_ARMS))
        self.assertEqual([e for _, e in events], [SelectionEvent.TARGET_RELEASED])
        self.assertIs(pipeline.selector.state, SelectionState.SEARCHING)


class TrackerSettingsTest(unittest.TestCase):
    def test_buffer_is_counted_in_updates(self):
        self.assertEqual(tp.tracker_buffer_updates(tp.DEFAULT_TRACKER_CONFIG), 30)
        self.assertAlmostEqual(tp.buffer_seconds(30, 8.0), 3.75)
        self.assertAlmostEqual(tp.buffer_seconds(30, 30.0), 1.0)

    def test_config_copy_changes_only_track_buffer(self):
        import yaml
        with tempfile.TemporaryDirectory() as d:
            path = tp.tracker_config_with_buffer(8, d)
            self.assertEqual(tp.tracker_buffer_updates(path), 8)
            with open(tp.check_yaml(tp.DEFAULT_TRACKER_CONFIG), encoding="utf-8") as f:
                original = yaml.safe_load(f)
            with open(path, encoding="utf-8") as f:
                copy = yaml.safe_load(f)
            self.assertEqual({k: v for k, v in copy.items() if k != "track_buffer"},
                             {k: v for k, v in original.items() if k != "track_buffer"})
            self.assertEqual(tp.TrackerVariant(0.8, path).label, "thr=0.8 buf=8")

    def test_constants(self):
        self.assertEqual(tp.NEW_TRACK_THRESH, 0.8)
        self.assertEqual(tp.COMPARISON_NEW_TRACK_THRESHOLDS, (0.7, 0.6))
        self.assertEqual(tp.CONFIRMATION_TIME, 0.5)
        self.assertEqual(tp.LOST_GRACE_TIME, 2.0)
        self.assertEqual(tp.NO_RESULT_TIMEOUT, 2.0)
        self.assertEqual(tp.TARGET_MAX_AGE_S, 0.5)


class ValidationStatsTest(unittest.TestCase):
    @staticmethod
    def v(track_id, state, box=(0, 0, 100, 200)):
        return ValidatedTrack(TrackedPerson(track_id, box, 0.7), state)

    def test_resets_and_possible_duplicates(self):
        s = tp.ValidationStats()
        s.observe([self.v(1, TrackState.CANDIDATE)])
        s.observe([])  # candidate vanished -> reset
        s.observe([self.v(2, TrackState.CANDIDATE)])
        s.observe([self.v(2, TrackState.CONFIRMED)])  # confirmed -> not a reset
        s.observe([self.v(2, TrackState.CONFIRMED), self.v(3, TrackState.CONFIRMED, (5, 5, 100, 200))])
        self.assertEqual(s.confirmation_resets, 1)
        self.assertEqual(s.confirmed_ids, {2, 3})
        self.assertEqual(s.max_confirmed, 2)
        self.assertEqual(s.possible_duplicate_updates, 1)
        self.assertEqual(s.track_ids, {1, 2, 3})


if __name__ == "__main__":
    unittest.main()

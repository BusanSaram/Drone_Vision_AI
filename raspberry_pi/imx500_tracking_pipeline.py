"""Tracking / confirmation / gesture / target-selection on IMX500 results, reusing the Windows modules.

Per NEW inference result, the same steps in the same order as src/yolo_pose_demo.py:
    PersonTracker (BoT-SORT) -> TrackValidator -> match_tracks_to_detections
    -> classify_pose (CONFIRMED tracks only) -> TrackPoseStabilizers -> TargetSelector
All of those are imported unchanged. Only NEW results are fed in (`update_new`);
frames on which modlib repeated the previous result must not call it, because
that would present one observation to BoT-SORT and the validator twice.

Rates: the Windows pipeline ran these modules at ~30 FPS; here they run at the
IMX500's ~8 results/s. Everything time-based keeps its meaning
(CONFIRMATION_TIME 0.5 s, LOST_GRACE_TIME 2.0 s, HOLD_TIME 0.7 s, NONE grace
0.2 s) but sees fewer observations. BoT-SORT's `track_buffer` is counted in
UPDATES, not seconds: the default 30 is ~1 s at 30 FPS but ~3.75 s at 8/s.
`buffer_seconds()` reports that, and `tracker_config_with_buffer()` builds a
copy of botsort.yaml with another value for comparison runs.

Two ways the target can go away, both end in TargetSelector's TARGET_LOST:
    1. NEW results keep arriving but the target is not in them: the reused
       TrackValidator expires the track after LOST_GRACE_TIME, as on Windows.
    2. NEW results stop entirely (the IMX500 output stalls): nothing would ever
       call the validator again, so the target would never expire. `check_stall()`
       (call it on every frame) feeds the validators ONE empty update once no
       NEW result has arrived for NO_RESULT_TIMEOUT (= LOST_GRACE_TIME), which
       expires every track and triggers TARGET_LOST through the normal path.
       BoT-SORT itself is not updated - an empty update is not an observation.

`target_status()` is the gate any future consumer (following control) must use:
the target is `usable` only while FOLLOWING, seen in the LATEST NEW result, and
that result is at most TARGET_MAX_AGE_S old. A target that is missing from the
latest result, or whose result is stale, is never reported as usable - even
while the validator's 2 s grace still keeps its track alive. No flight control
lives here.

Comparison trackers: extra PersonTracker + TrackValidator pairs (e.g. with
NEW_TRACK_THRESH 0.7 / 0.6) see the same NEW detections - by default without
the frame, i.e. without GMC, to save Pi CPU - and only collect
`ValidationStats` for the log. They never drive the display, gestures
or target selection. Note: Ultralytics numbers tracks with one process-wide
counter, so with comparison trackers running the primary's track IDs are not
consecutive. Behaviour is unaffected.
"""

import os
from dataclasses import dataclass, replace

import yaml
from ultralytics.utils.checks import check_yaml

from coco_pose_recognizer import BodyPose, classify_pose
from geometry_utils import as_geom_box, iou
from imx500_pose_overlay import STALE_AFTER_S
from keypoint_track_association import match_tracks_to_detections
from person_tracker import DEFAULT_TRACKER_CONFIG, PersonTracker, TrackedPerson
from pose_command_stabilizer import StabilizerState, TrackPoseStabilizers
from pose_detections import PoseDetections
from target_selector import SelectionState, SelectorResult, TargetSelector
from track_validator import CONFIRMATION_TIME, LOST_GRACE_TIME, TrackState, TrackValidator, ValidatedTrack

# Original Windows value (src/yolo_person_detection.py, "experimental, not tuned").
# Not imported from there: that module pulls in the CPU/GPU YOLO detector.
NEW_TRACK_THRESH = 0.8
COMPARISON_NEW_TRACK_THRESHOLDS = (0.7, 0.6)

NO_RESULT_TIMEOUT = LOST_GRACE_TIME  # no NEW result for this long -> expire all tracks (case 2 above)
TARGET_MAX_AGE_S = STALE_AFTER_S  # same 0.5 s as the on-screen STALE warning

# Log-only: two CONFIRMED boxes overlapping this much in one update are counted
# as a possible duplicate of one person. Not used by any decision.
POSSIBLE_DUPLICATE_IOU = 0.5


def tracker_config_with_buffer(track_buffer: int, directory: str) -> str:
    """Write a copy of the default botsort.yaml with only `track_buffer` changed; return its path."""
    with open(check_yaml(DEFAULT_TRACKER_CONFIG), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["track_buffer"] = int(track_buffer)
    path = os.path.join(directory, f"botsort_track_buffer_{int(track_buffer)}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def tracker_buffer_updates(tracker_config: str) -> int:
    with open(check_yaml(tracker_config), encoding="utf-8") as f:
        return int(yaml.safe_load(f)["track_buffer"])


def buffer_seconds(track_buffer: int, results_per_second: float) -> float:
    """How long BoT-SORT keeps a lost track when it is updated `results_per_second` times a second."""
    return track_buffer / results_per_second if results_per_second > 0 else float("inf")


class ValidationStats:
    """What a TrackValidator did over a run, derived only from its per-update output."""

    def __init__(self):
        self.updates = 0
        self.track_ids: set[int] = set()
        self.confirmed_ids: set[int] = set()
        self.confirmation_resets = 0  # CANDIDATE that vanished before reaching CONFIRMED
        self.max_confirmed = 0  # most CONFIRMED tracks visible in one update
        self.possible_duplicate_updates = 0  # updates with two CONFIRMED boxes at IoU >= POSSIBLE_DUPLICATE_IOU
        self._candidates: set[int] = set()

    def observe(self, validated: list[ValidatedTrack]) -> None:
        self.updates += 1
        states = {v.tracked_person.track_id: v.state for v in validated}
        self.confirmation_resets += sum(1 for tid in self._candidates if tid not in states)
        self._candidates = {tid for tid, s in states.items() if s is TrackState.CANDIDATE}
        self.track_ids |= set(states)

        confirmed = [v.tracked_person for v in validated if v.state is TrackState.CONFIRMED]
        self.confirmed_ids |= {p.track_id for p in confirmed}
        self.max_confirmed = max(self.max_confirmed, len(confirmed))
        boxes = [as_geom_box(p.bbox) for p in confirmed]
        if any(iou(boxes[i], boxes[j]) >= POSSIBLE_DUPLICATE_IOU
               for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
            self.possible_duplicate_updates += 1

    def summary(self) -> str:
        return (f"ids={len(self.track_ids)} confirmed_ids={len(self.confirmed_ids)} "
                f"resets={self.confirmation_resets} max_confirmed={self.max_confirmed} "
                f"possible_dup_updates={self.possible_duplicate_updates}")


class TrackerVariant:
    """One PersonTracker + TrackValidator pair (Windows timing constants) with its stats.

    `use_gmc=False` passes no frame to BoT-SORT, which skips its global motion
    compensation (~14 ms per update on the Pi 5) - used for comparison trackers.
    """

    def __init__(self, new_track_thresh: float, tracker_config: str = DEFAULT_TRACKER_CONFIG, use_gmc: bool = True):
        self.new_track_thresh = new_track_thresh
        self.track_buffer = tracker_buffer_updates(tracker_config)
        self.use_gmc = use_gmc
        self.label = f"thr={new_track_thresh:g} buf={self.track_buffer}" + ("" if use_gmc else " gmc=off")
        self.tracker = PersonTracker(tracker_config=tracker_config, new_track_thresh=new_track_thresh)
        self.validator = TrackValidator(confirmation_time=CONFIRMATION_TIME, lost_grace_time=LOST_GRACE_TIME)
        self.stats = ValidationStats()

    def update(self, detections: PoseDetections, frame, now: float):
        tracked = self.tracker.update(detections, frame if self.use_gmc else None)
        validated = self.validator.update(tracked, now)
        self.stats.observe(validated)
        return tracked, validated

    def expire(self, now: float) -> None:
        """Stall handling: one empty validator update (BoT-SORT is left untouched)."""
        self.stats.observe(self.validator.update([], now))


@dataclass
class PipelineResult:
    timestamp: float  # time.monotonic() of the NEW result this was computed from
    detections: PoseDetections
    tracked_people: list[TrackedPerson]
    validated_tracks: list[ValidatedTrack]
    track_to_detection: dict[int, int]
    pose_states: dict[int, StabilizerState]  # CONFIRMED tracks only
    selection: SelectorResult


@dataclass(frozen=True)
class TargetStatus:
    state: SelectionState
    target_id: int | None
    bbox: tuple[int, int, int, int] | None  # target box from the latest NEW result it was seen in
    age: float | None  # seconds since the target was last seen in a NEW result
    usable: bool  # True only if a consumer may act on `bbox` right now
    reason: str


class ImxTrackingPipeline:
    def __init__(self, new_track_thresh: float = NEW_TRACK_THRESH,
                 tracker_config: str = DEFAULT_TRACKER_CONFIG,
                 comparisons: tuple[tuple[float, str], ...] = (),
                 comparison_gmc: bool = False,
                 no_result_timeout: float = NO_RESULT_TIMEOUT,
                 target_max_age: float = TARGET_MAX_AGE_S):
        """`comparisons`: (new_track_thresh, tracker_config) pairs run as log-only comparison trackers,
        without GMC unless `comparison_gmc`. The primary tracker always gets the frame (GMC on)."""
        self.primary = TrackerVariant(new_track_thresh, tracker_config, use_gmc=True)
        self.comparisons = [TrackerVariant(thr, cfg, use_gmc=comparison_gmc) for thr, cfg in comparisons]
        self.stabilizers = TrackPoseStabilizers()
        self.selector = TargetSelector()
        self.no_result_timeout = no_result_timeout
        self.target_max_age = target_max_age
        self.last_result: PipelineResult | None = None
        self.last_new_time: float | None = None
        self.stalled = False
        self._target_seen_at: float | None = None
        self._target_bbox: tuple[int, int, int, int] | None = None

    def update_new(self, detections: PoseDetections, frame, now: float) -> PipelineResult:
        """Process one NEW inference result. `frame` is the clean BGR image (BoT-SORT GMC) or None."""
        tracked, validated = self.primary.update(detections, frame, now)
        track_to_detection = match_tracks_to_detections(tracked, detections)

        # Same as yolo_pose_demo.py: only CONFIRMED tracks are eligible; a CONFIRMED
        # track without keypoints this result counts as NONE.
        pose_states, activations = {}, {}
        for v in validated:
            if v.state is not TrackState.CONFIRMED:
                continue
            track_id = v.tracked_person.track_id
            index = track_to_detection.get(track_id)
            raw_pose = BodyPose.NONE if index is None else classify_pose(detections.keypoints[index])
            state = self.stabilizers.update(track_id, raw_pose, now)
            pose_states[track_id] = state
            if state.newly_activated:
                activations[track_id] = state.active
        self.stabilizers.prune(pose_states)

        selection = self._select(activations)
        for variant in self.comparisons:
            variant.update(detections, frame, now)

        self.last_new_time = now
        self.stalled = False
        self._update_target_observation(selection.target_id, tracked, now)
        self.last_result = PipelineResult(now, detections, tracked, validated, track_to_detection,
                                          pose_states, selection)
        return self.last_result

    def check_stall(self, now: float) -> SelectorResult | None:
        """Call on every frame. Once, after NO_RESULT_TIMEOUT without a NEW result,
        expires all tracks via an empty validator update; returns that selector result."""
        if self.last_new_time is None or self.stalled or now - self.last_new_time < self.no_result_timeout:
            return None
        self.stalled = True
        self.primary.expire(now)
        for variant in self.comparisons:
            variant.expire(now)
        self.stabilizers.prune(())
        selection = self._select({})
        self._update_target_observation(selection.target_id, [], now)
        self.last_result = replace(self.last_result, detections=PoseDetections.empty(), tracked_people=[],
                                   validated_tracks=[], track_to_detection={}, pose_states={},
                                   selection=selection)
        return selection

    def target_status(self, now: float) -> TargetStatus:
        state, target_id = self.selector.state, self.selector.target_id
        age = None if self._target_seen_at is None else now - self._target_seen_at

        def status(usable, reason):
            return TargetStatus(state, target_id, self._target_bbox, age, usable, reason)

        if target_id is None:
            return status(False, "no target")
        if state is not SelectionState.FOLLOWING:
            return status(False, f"state {state.value}")
        if self._target_seen_at is None or self._target_seen_at != self.last_new_time:
            return status(False, "target not in latest result")
        if age > self.target_max_age:
            return status(False, "result stale")
        return status(True, "ok")

    def result_age(self, now: float) -> float | None:
        return None if self.last_new_time is None else now - self.last_new_time

    def reset_session(self) -> None:
        """New autonomous session, as yolo_pose_demo.start_new_session(): selector + pose holds only."""
        self.selector.reset()
        self.stabilizers.reset()
        self._target_seen_at = None
        self._target_bbox = None

    def _select(self, activations: dict[int, BodyPose]) -> SelectorResult:
        target_id = self.selector.target_id
        target_alive = target_id is not None and self.primary.validator.is_tracked(target_id)
        return self.selector.update(activations, target_alive)

    def _update_target_observation(self, target_id, tracked, now: float) -> None:
        if target_id is None:
            self._target_seen_at = None
            self._target_bbox = None
            return
        for person in tracked:
            if person.track_id == target_id:
                self._target_seen_at = now
                self._target_bbox = person.bbox
                return

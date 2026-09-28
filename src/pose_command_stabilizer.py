"""Hold-time stabilizer for per-frame body-pose classifications, one per track.

A raw per-frame pose from `classify_pose()` only becomes an ACTIVE command
after the same pose has been seen continuously for `hold_time` seconds:

    raw pose == candidate       -> hold continues; confirms once held >= hold_time
    raw pose is a different pose -> it becomes the new candidate, timer restarts,
                                   active command clears at once (no grace)
    raw pose is NONE             -> short dropout grace (see below)

NONE grace: YOLO Pose keypoints sometimes flicker for a frame or two while a
pose is really still held, and classify_pose() then returns NONE. A NONE
frame while a real pose is the candidate is tolerated - candidate, its hold
timer and any active command are kept unchanged - only while BOTH hold:

    consecutive NONE frames      <= none_grace_frames
    time since last matching frame <= none_grace_time

As soon as either limit is exceeded, the state fully resets (candidate and
active command cleared), so the pose must be held again for `hold_time`.
The frame limit matches the cause (per-inference flicker); the time limit
caps how long a stale candidate/command can survive when FPS is low.
Grace applies ONLY to NONE - a different recognized pose always resets
immediately. A command is never confirmed on a NONE frame, only on a frame
where the candidate pose is actually recognized.

`newly_activated` is True on exactly one update per hold, so a consumer can
treat it as a one-shot event; a dropout tolerated by the grace does not end
the hold, so it does not cause a second event. Uses time.monotonic by default
(not wall clock) for hold and grace time, since FPS is not constant.

`TrackPoseStabilizers` keeps one independent stabilizer per track ID.
Neither knows anything about drone control.
"""

import time
from dataclasses import dataclass

from coco_pose_recognizer import BodyPose

HOLD_TIME = 0.7  # seconds; experimental starting value, not tuned
NONE_GRACE_FRAMES = 2  # max consecutive NONE frames tolerated; experimental starting value
NONE_GRACE_TIME = 0.2  # seconds since last matching frame; experimental starting value


@dataclass
class StabilizerState:
    raw_pose: BodyPose
    candidate: BodyPose
    held_for: float  # seconds the current candidate has been held (0 for NONE)
    active: BodyPose
    newly_activated: bool  # True only on the update where `active` was set
    in_grace: bool = False  # True when this NONE frame was tolerated as a dropout


class PoseCommandStabilizer:
    def __init__(
        self,
        hold_time: float = HOLD_TIME,
        none_grace_frames: int = NONE_GRACE_FRAMES,
        none_grace_time: float = NONE_GRACE_TIME,
        time_fn=time.monotonic,
    ):
        self._hold_time = hold_time
        self._none_grace_frames = none_grace_frames
        self._none_grace_time = none_grace_time
        self._time_fn = time_fn
        self._candidate = BodyPose.NONE
        self._candidate_since = 0.0
        self._last_match_time = 0.0  # last frame where raw pose == candidate
        self._none_streak = 0  # consecutive tolerated/counted NONE frames
        self._active = BodyPose.NONE

    @property
    def hold_time(self) -> float:
        return self._hold_time

    def _reset(self, candidate: BodyPose, now: float) -> None:
        """Start over with `candidate` (NONE = no candidate); clears any active command."""
        self._candidate = candidate
        self._candidate_since = now
        self._last_match_time = now
        self._none_streak = 0
        self._active = BodyPose.NONE

    def update(self, raw_pose: BodyPose, now: float | None = None) -> StabilizerState:
        now = now if now is not None else self._time_fn()
        in_grace = False

        if raw_pose is BodyPose.NONE and self._candidate is not BodyPose.NONE:
            # Possible keypoint dropout: keep everything only within both limits.
            self._none_streak += 1
            if (
                self._none_streak <= self._none_grace_frames
                and now - self._last_match_time <= self._none_grace_time
            ):
                in_grace = True
            else:
                self._reset(BodyPose.NONE, now)
        elif raw_pose is not self._candidate:
            # A different real pose (or the first pose after NONE): no grace.
            self._reset(raw_pose, now)
        elif raw_pose is not BodyPose.NONE:
            # Same real pose as the candidate: the hold continues.
            self._last_match_time = now
            self._none_streak = 0

        held_for = 0.0 if self._candidate is BodyPose.NONE else now - self._candidate_since

        # Confirm only on a frame where the candidate pose is actually seen.
        newly_activated = False
        if (
            raw_pose is self._candidate
            and self._candidate is not BodyPose.NONE
            and self._active is BodyPose.NONE
            and held_for >= self._hold_time
        ):
            self._active = self._candidate
            newly_activated = True

        return StabilizerState(
            raw_pose=raw_pose,
            candidate=self._candidate,
            held_for=held_for,
            active=self._active,
            newly_activated=newly_activated,
            in_grace=in_grace,
        )


class TrackPoseStabilizers:
    """One `PoseCommandStabilizer` per track ID, created on first use.

    Call `update()` for each relevant track every frame, then `prune()` with
    the IDs that were relevant this frame: any other track's state (including
    its NONE grace state) is dropped, so a track that disappears (or stops
    being eligible) starts its hold from scratch if it comes back.
    """

    def __init__(
        self,
        hold_time: float = HOLD_TIME,
        none_grace_frames: int = NONE_GRACE_FRAMES,
        none_grace_time: float = NONE_GRACE_TIME,
        time_fn=time.monotonic,
    ):
        self._hold_time = hold_time
        self._none_grace_frames = none_grace_frames
        self._none_grace_time = none_grace_time
        self._time_fn = time_fn
        self._stabilizers: dict[int, PoseCommandStabilizer] = {}

    @property
    def hold_time(self) -> float:
        return self._hold_time

    @property
    def track_ids(self) -> set[int]:
        return set(self._stabilizers)

    def update(self, track_id: int, raw_pose: BodyPose, now: float | None = None) -> StabilizerState:
        stabilizer = self._stabilizers.get(track_id)
        if stabilizer is None:
            stabilizer = PoseCommandStabilizer(
                hold_time=self._hold_time,
                none_grace_frames=self._none_grace_frames,
                none_grace_time=self._none_grace_time,
                time_fn=self._time_fn,
            )
            self._stabilizers[track_id] = stabilizer
        return stabilizer.update(raw_pose, now)

    def prune(self, keep_track_ids) -> None:
        keep = set(keep_track_ids)
        for track_id in list(self._stabilizers):
            if track_id not in keep:
                del self._stabilizers[track_id]

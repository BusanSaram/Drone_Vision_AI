"""Hold-time stabilizer for per-frame body-pose classifications.

A raw per-frame pose from `classify_pose()` only becomes an ACTIVE command
after the same pose has been seen continuously for `hold_time` seconds:

    raw pose changes           -> it becomes the new candidate, timer restarts
    candidate held >= hold_time -> candidate becomes the active command
    raw pose changes again     -> active command clears, new candidate starts

`newly_activated` is True on exactly one update per hold, so a consumer can
treat it as a one-shot event; the pose must be released and held again to
fire again. Uses wall-clock timestamps (like TrackValidator), not frame
counts, since FPS is not constant.

Standalone at this stage: it tracks a single pose stream and knows nothing
about person track IDs or drone control.
"""

import time
from dataclasses import dataclass

from body_pose_recognizer import BodyPose

HOLD_TIME = 0.7  # seconds; experimental starting value, not tuned


@dataclass
class StabilizerState:
    raw_pose: BodyPose
    candidate: BodyPose
    held_for: float  # seconds the current candidate has been held (0 for NONE)
    active: BodyPose
    newly_activated: bool  # True only on the update where `active` was set


class PoseCommandStabilizer:
    def __init__(self, hold_time: float = HOLD_TIME, time_fn=time.monotonic):
        self._hold_time = hold_time
        self._time_fn = time_fn
        self._candidate = BodyPose.NONE
        self._candidate_since = 0.0
        self._active = BodyPose.NONE

    @property
    def hold_time(self) -> float:
        return self._hold_time

    def update(self, raw_pose: BodyPose, now: float | None = None) -> StabilizerState:
        now = now if now is not None else self._time_fn()

        if raw_pose is not self._candidate:
            self._candidate = raw_pose
            self._candidate_since = now
            self._active = BodyPose.NONE

        held_for = 0.0 if self._candidate is BodyPose.NONE else now - self._candidate_since

        newly_activated = False
        if (
            self._candidate is not BodyPose.NONE
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
        )

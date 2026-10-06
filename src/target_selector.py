"""Target-selection state machine: which tracked person (if any) is the target.

Pipeline position:
    TrackPoseStabilizers (newly_activated poses, CONFIRMED tracks only)
        + TrackValidator.is_tracked(target_id)
            -> TargetSelector -> state, target track ID, one-shot event

States:
    SEARCHING       - no target. Only ONE_ARM_UP is considered: exactly one
                      track completing a ONE_ARM_UP hold this frame becomes
                      the target (-> FOLLOWING). T_POSE / DOUBLE_BICEPS are
                      ignored.
    FOLLOWING       - one target track ID. Only that track's commands count:
                        T_POSE        -> clear target, back to SEARCHING
                        DOUBLE_BICEPS -> LAND_REQUESTED
                      Commands from every other track are ignored. If the
                      target is lost (validator expired its track), clear the
                      target and go back to SEARCHING - never switch to
                      another person automatically.
    LAND_REQUESTED  - terminal for the current autonomous session. The target
                      ID is kept for display; every pose command is ignored.
                      Only an external `reset()` (new session) leaves it.

Sessions: `reset()` starts a fresh autonomous session - SEARCHING, no
target - from any state. It is called by the integration layer (eventually
on a MANUAL -> AUTO flight-mode transition), never by a body pose. The
caller must also clear the pose stabilizers (`TrackPoseStabilizers.reset()`)
so holds started before the session cannot complete into commands in it.
The tracker and TrackValidator keep running across sessions.

Input is edge-triggered: `activations` holds only the poses whose 0.7 s hold
completed THIS frame (`StabilizerState.newly_activated`), never poses that
are merely still being held. So a person who keeps holding a pose never fires
twice; to act again they must release and hold again. In particular:

    Ambiguous selection: if two or more tracks complete a ONE_ARM_UP hold on
    the same frame, nobody is selected (no tie-break by ID, bbox size or
    proximity) and the state stays SEARCHING. Those activations are consumed:
    a person who keeps holding ONE_ARM_UP is NOT selected later just because
    the others release - they must release and complete a new 0.7 s hold.

This module only decides the selection state. It knows nothing about
detection, OpenCV, flight control or MAVLink; nothing here moves a drone.
"""

from dataclasses import dataclass
from enum import Enum

from coco_pose_recognizer import BodyPose


class SelectionState(Enum):
    SEARCHING = "SEARCHING"
    FOLLOWING = "FOLLOWING"
    LAND_REQUESTED = "LAND_REQUESTED"


class SelectionEvent(Enum):
    TARGET_SELECTED = "TARGET_SELECTED"
    SELECTION_AMBIGUOUS = "SELECTION_AMBIGUOUS"
    TARGET_RELEASED = "TARGET_RELEASED"  # target's T_POSE
    TARGET_LOST = "TARGET_LOST"
    LAND_REQUESTED = "LAND_REQUESTED"  # target's DOUBLE_BICEPS


@dataclass
class SelectorResult:
    state: SelectionState
    target_id: int | None
    event: SelectionEvent | None = None  # what happened on this update, if anything
    event_track_ids: tuple[int, ...] = ()  # track(s) the event refers to


class TargetSelector:
    def __init__(self):
        self._state = SelectionState.SEARCHING
        self._target_id: int | None = None

    @property
    def state(self) -> SelectionState:
        return self._state

    @property
    def target_id(self) -> int | None:
        return self._target_id

    def update(self, activations: dict[int, BodyPose], target_alive: bool) -> SelectorResult:
        """Advance by one frame.

        Args:
            activations: `{track_id: pose}` for tracks whose stabilizer reported
                `newly_activated` this frame (CONFIRMED tracks only).
            target_alive: Whether the current target track is still known to
                the TrackValidator (`is_tracked(target_id)`). Ignored when
                there is no target.
        """
        if self._state is SelectionState.SEARCHING:
            return self._update_searching(activations)
        if self._state is SelectionState.FOLLOWING:
            return self._update_following(activations, target_alive)
        return self._result()  # LAND_REQUESTED: ignore everything until reset()

    def reset(self) -> None:
        """Start a new autonomous session: SEARCHING with no target, from any state."""
        self._clear()

    def _update_searching(self, activations) -> SelectorResult:
        raised = sorted(tid for tid, pose in activations.items() if pose is BodyPose.ONE_ARM_UP)
        if len(raised) == 1:
            self._state = SelectionState.FOLLOWING
            self._target_id = raised[0]
            return self._result(SelectionEvent.TARGET_SELECTED, raised)
        if len(raised) > 1:
            return self._result(SelectionEvent.SELECTION_AMBIGUOUS, raised)
        return self._result()

    def _update_following(self, activations, target_alive) -> SelectorResult:
        target_id = self._target_id
        if not target_alive:
            self._clear()
            return self._result(SelectionEvent.TARGET_LOST, [target_id])

        pose = activations.get(target_id)  # other tracks' commands are ignored
        if pose is BodyPose.T_POSE:
            self._clear()
            return self._result(SelectionEvent.TARGET_RELEASED, [target_id])
        if pose is BodyPose.DOUBLE_BICEPS:
            self._state = SelectionState.LAND_REQUESTED
            return self._result(SelectionEvent.LAND_REQUESTED, [target_id])
        return self._result()

    def _clear(self) -> None:
        self._state = SelectionState.SEARCHING
        self._target_id = None

    def _result(self, event: SelectionEvent | None = None, track_ids=()) -> SelectorResult:
        return SelectorResult(self._state, self._target_id, event, tuple(track_ids))

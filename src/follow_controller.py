"""Forward-only person-following controller (standalone; no flight-controller I/O).

Pipeline position (planned, not connected yet):
    ImxTrackingPipeline (TargetSelector decides WHO the target is)
        -> FollowInput (one per NEW AI result, see imx500_follow_adapter.py)
        -> FollowController.update()  -> FollowCommand (forward m/s >= 0, yaw rate, hold, land)
        -> command publisher calls FollowController.command_at(now) at its own rate
           (MAVLink / ArduPilot SITL - not implemented here)

What it may command: FORWARD, YAW LEFT / RIGHT, HOLD. Never backward, never
lateral, never altitude. Selection stays with TargetSelector: this module never
selects, switches or releases a target - it only follows `target_id` while the
selector is FOLLOWING, and only on measurements tagged with that same track ID.

Per NEW AI result (deduplicated by `result_timestamp`):
    1. Selection gate   - SEARCHING / no target -> INACTIVE; LAND_REQUESTED -> land request.
    2. Data gate        - usable, target observed, measurement from the selected track,
                          result fresh (age <= max_result_age_s).
    3. Full-body check  - head, both shoulders, both hips, a knee and an ankle reliable;
                          box not touching the image edge; keypoints span the box;
                          upright (vertical order, torso tilt, leg/thigh vs torso length);
                          roughly facing the camera (shoulder width vs torso length).
    3b. Occlusion       - another detection's box covering the target box, or the lower-body
                          keypoint confidence dropping vs the last accepted result -> HOLD with
                          the label OCCLUSION_SUSPECTED. Unseen body parts (inside the frame) and
                          a size jump with one box edge fixed get the same label. It is a
                          suspicion, never reported as a confirmed occlusion.
    4. Box-height check - confidence, h_norm = box height / frame height within
                          [min_box_height, max_box_height], change vs the last ACCEPTED
                          height limited by real elapsed time. A rejected size never
                          becomes the baseline; after a rejected jump the only way back
                          is a size close to the old baseline, a new selection, or an
                          explicit rebaseline() - a new size is never trusted just
                          because it stays stable.
    5. Revalidation     - after any invalid result, resume_valid_count consecutive valid
                          results are required before motion resumes.
    6. Control          - yaw from horizontal center error (deadband, gain, max rate,
                          rate limit); forward only while the box is smaller than the
                          target band (hysteresis, gain, max speed, acceleration limit);
                          inside or above the band forward = 0.
Any failed gate sets forward AND yaw to 0 immediately (no decay of the last command).

Conventions: forward_mps >= 0 along the body x axis; yaw_rate_dps > 0 = yaw RIGHT
(clockwise seen from above, the sign ArduPilot uses for body yaw rate). Image x
grows to the right, y downward. Every numeric default in FollowParams is an
initial, untuned value for SITL - not for physical flight.
"""

import math
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from coco_keypoints import (
    KEYPOINT_CONF_THRESHOLD, LEFT_ANKLE, LEFT_EAR, LEFT_EYE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, NOSE,
    NUM_KEYPOINTS, RIGHT_ANKLE, RIGHT_EAR, RIGHT_EYE, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER,
)
from target_selector import SelectionState

HEAD = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
SHOULDERS = (LEFT_SHOULDER, RIGHT_SHOULDER)
HIPS = (LEFT_HIP, RIGHT_HIP)
KNEES = (LEFT_KNEE, RIGHT_KNEE)
ANKLES = (LEFT_ANKLE, RIGHT_ANKLE)


@dataclass(frozen=True)
class FollowParams:
    """All values are INITIAL, UNTUNED defaults chosen to be conservative in SITL."""

    # --- full-body validation ---
    keypoint_min_conf: float = KEYPOINT_CONF_THRESHOLD  # 0.5, same as gesture recognition
    min_head_keypoints: int = 1  # of nose / eyes / ears
    edge_margin_px: float = 2.0  # box within this of any image edge = clipped
    min_keypoint_span_ratio: float = 0.7  # (lowest ankle - highest head point) / box height
    max_torso_tilt_deg: float = 25.0  # hip-center -> shoulder-center vs vertical
    min_leg_to_torso_ratio: float = 1.2  # (ankle_y - hip_y) / (hip_y - shoulder_y); standing ~1.5+
    min_thigh_to_torso_ratio: float = 0.5  # (knee_y - hip_y) / (hip_y - shoulder_y); sitting ~0
    min_shoulder_to_torso_ratio: float = 0.45  # shoulder width / torso length; sideways is small

    # --- box-height measurement validation ---
    min_confidence: float = 0.6  # detection score (IMX500 scores seen: 0.5, 0.621, 0.73, 0.816, 0.879)
    min_box_height: float = 0.25  # h_norm below this: too far / too small to trust
    max_box_height: float = 0.95  # h_norm above this: too close to measure
    max_result_age_s: float = 0.3  # AI result older than this is stale (NEW results arrive ~every 0.125 s)
    max_change_rate_per_s: float = 0.6  # allowed |h/h_ref - 1| per second of real elapsed time
    change_noise_allowance: float = 0.08  # extra relative change allowed per comparison (box jitter)
    max_change_after_gap: float = 0.3  # cap on allowed relative change however long the gap

    # --- occlusion indicators (HOLD + OCCLUSION_SUSPECTED; never reported as a confirmed occlusion) ---
    max_other_overlap_ratio: float = 0.15  # part of the target box covered by another detection's box
    max_lower_body_conf_drop: float = 0.35  # drop of mean knee/ankle confidence vs the last accepted result
    edge_shift_stable: float = 0.05  # |edge shift| / reference box height counted as "edge did not move"
    edge_shift_moved: float = 0.10  # edge shift / reference box height counted as "edge moved"

    # --- revalidation ---
    resume_valid_count: int = 4  # consecutive valid NEW results (~0.5 s at 8/s) before motion

    # --- forward (distance) control ---
    target_box_height_low: float = 0.55  # band [low, high]: hold inside it
    target_box_height_high: float = 0.70  # above it: also hold (never backward)
    start_forward_hysteresis: float = 0.03  # start moving only below low - this; stop at low
    forward_gain: float = 2.0  # m/s per unit of (low - h_norm)
    max_forward_mps: float = 0.5
    max_forward_accel: float = 0.5  # m/s^2, applied to increases only (decrease is immediate)

    # --- yaw control ---
    yaw_deadband: float = 0.08  # normalized center error in [-1, 1]
    yaw_gain_dps: float = 30.0  # deg/s per unit of center error
    max_yaw_rate_dps: float = 20.0
    max_yaw_accel_dps2: float = 40.0

    # --- communication ---
    command_timeout_s: float = 0.3  # command_at(): no update() for this long -> HOLD


class FollowState(Enum):
    INACTIVE = "INACTIVE"  # no target selected (TargetSelector SEARCHING)
    ACTIVE = "ACTIVE"  # valid data, moving forward (yaw may also be nonzero)
    HOLD_DISTANCE = "HOLD_DISTANCE"  # valid data, box in / above the target band: forward 0, yaw allowed
    HOLD_INVALID = "HOLD_INVALID"  # target selected but its data is not trusted: forward 0, yaw 0
    HOLD_TARGET_LOST = "HOLD_TARGET_LOST"  # selected target not seen in the latest result: forward 0, yaw 0
    HOLD_TIMEOUT = "HOLD_TIMEOUT"  # controller not updated within command_timeout_s
    LAND_REQUESTED = "LAND_REQUESTED"  # TargetSelector LAND_REQUESTED: forward 0, yaw 0, land=True


@dataclass(frozen=True)
class FollowInput:
    """One AI result for the controller. Pixel coordinates in a frame_width x frame_height image."""

    now: float  # controller clock (time.monotonic) when this input is built
    result_timestamp: float | None  # time.monotonic() of the AI (NEW) result the data comes from
    selection_state: SelectionState
    target_id: int | None  # TargetSelector's target
    usable: bool  # ImxTrackingPipeline.target_status().usable
    target_observed: bool  # target_id present among the tracks of that result
    measurement_track_id: int | None  # track ID the bbox / keypoints below belong to
    bbox: tuple[float, float, float, float] | None  # x1, y1, x2, y2 (raw detection box of the target)
    confidence: float | None  # detection score of that box
    keypoints: np.ndarray | None  # (17, 3) x, y, confidence
    frame_width: int
    frame_height: int
    other_boxes: tuple = ()  # x1, y1, x2, y2 of the OTHER detections in the same result (occlusion check)


@dataclass(frozen=True)
class FollowCommand:
    state: FollowState
    reason: str
    forward_mps: float
    yaw_rate_dps: float
    hold: bool  # True = no translation (forward 0). Yaw may still be nonzero in HOLD_DISTANCE.
    land: bool
    target_id: int | None
    box_height: float | None = None  # h_norm of the measurement, for logging
    center_error: float | None = None  # [-1, 1], + = target right of center
    stage: str = ""  # gate that decided: selection / data / full_body / occlusion / measurement /
    #                  revalidation / control / timeout
    diagnostics: tuple = ()  # labels such as "OCCLUSION_SUSPECTED" (a suspicion, never a confirmation)
    indicators: tuple = ()  # evidence behind the labels / the decision, for the log

    def __post_init__(self):
        if not (self.forward_mps >= 0.0) or not math.isfinite(self.forward_mps):
            raise ValueError(f"forward_mps must be finite and >= 0, got {self.forward_mps}")
        if not math.isfinite(self.yaw_rate_dps):
            raise ValueError(f"yaw_rate_dps must be finite, got {self.yaw_rate_dps}")


def _stop(state: FollowState, reason: str, target_id, land: bool = False, **extra) -> FollowCommand:
    return FollowCommand(state, reason, 0.0, 0.0, True, land, target_id, **extra)


# ---------- validation (pure functions) ----------

# Full-body failures that come from body parts not being seen (as opposed to posture or framing).
VISIBILITY_FAILURES = ("head not visible", "shoulders not visible", "hips not visible", "legs not visible",
                       "keypoints do not span the box")


def body_metrics(keypoints, bbox, frame_width: int, frame_height: int,
                 params: FollowParams = FollowParams()) -> dict:
    """Every full-body check, its measured value and the list of failed checks (in check order).

    `reason` is the first failed check ("ok" if none) - exactly what validate_full_body() returns.
    Ratios are None when the keypoints they need are not reliable.
    """
    if keypoints is None or bbox is None:
        return {"reason": "no keypoints", "failures": ["no keypoints"]}
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.shape != (NUM_KEYPOINTS, 3):
        return {"reason": f"keypoints shape {kp.shape}", "failures": [f"keypoints shape {kp.shape}"]}
    ok = np.isfinite(kp).all(axis=1) & (kp[:, 2] >= params.keypoint_min_conf)
    conf = np.nan_to_num(kp[:, 2])

    x1, y1, x2, y2 = (float(v) for v in bbox)
    edge = params.edge_margin_px
    head = [i for i in HEAD if ok[i]]
    knees = [i for i in KNEES if ok[i]]
    ankles = [i for i in ANKLES if ok[i]]
    m = {
        "clipped": x1 <= edge or y1 <= edge or x2 >= frame_width - 1 - edge or y2 >= frame_height - 1 - edge,
        "head": len(head), "shoulders": int(sum(ok[i] for i in SHOULDERS)), "hips": int(sum(ok[i] for i in HIPS)),
        "knees": len(knees), "ankles": len(ankles),
        "upper_conf": float(conf[list(SHOULDERS + HIPS)].mean()),
        "lower_conf": float(conf[list(KNEES + ANKLES)].mean()),
        "span": None, "upright_order": None, "tilt_deg": None, "leg_ratio": None, "thigh_ratio": None,
        "shoulder_ratio": None,
    }
    failures = []
    if m["clipped"]:
        failures.append("box clipped by image edge")
    if m["head"] < params.min_head_keypoints:
        failures.append("head not visible")
    if m["shoulders"] < 2:
        failures.append("shoulders not visible")
    if m["hips"] < 2:
        failures.append("hips not visible")
    if not knees or not ankles:
        failures.append("legs not visible")

    if head and m["shoulders"] == 2 and m["hips"] == 2 and knees and ankles:
        head_y = min(kp[i, 1] for i in head)
        shoulder = kp[list(SHOULDERS), :2].mean(axis=0)
        hip = kp[list(HIPS), :2].mean(axis=0)
        knee_y = float(np.mean([kp[i, 1] for i in knees]))
        ankle_y = float(np.mean([kp[i, 1] for i in ankles]))
        lowest_ankle = max(kp[i, 1] for i in ankles)
        m["span"] = (lowest_ankle - head_y) / (y2 - y1) if y2 > y1 else 0.0
        m["upright_order"] = bool(head_y < shoulder[1] < hip[1] < knee_y < ankle_y)
        if m["span"] < params.min_keypoint_span_ratio:
            failures.append("keypoints do not span the box")
        if not m["upright_order"]:
            failures.append("not upright (vertical order)")
        torso = hip[1] - shoulder[1]
        if torso > 0:
            m["tilt_deg"] = math.degrees(math.atan2(abs(shoulder[0] - hip[0]), torso))
            m["leg_ratio"] = (ankle_y - hip[1]) / torso
            m["thigh_ratio"] = (knee_y - hip[1]) / torso
            m["shoulder_ratio"] = abs(kp[LEFT_SHOULDER, 0] - kp[RIGHT_SHOULDER, 0]) / torso
            if m["tilt_deg"] > params.max_torso_tilt_deg:
                failures.append("torso tilted")
            if m["leg_ratio"] < params.min_leg_to_torso_ratio:
                failures.append("legs folded (crouch / sit)")
            if m["thigh_ratio"] < params.min_thigh_to_torso_ratio:
                failures.append("thighs not vertical (sitting)")
            if m["shoulder_ratio"] < params.min_shoulder_to_torso_ratio:
                failures.append("sideways to camera")
    m["failures"] = failures
    m["reason"] = failures[0] if failures else "ok"
    return m


def validate_full_body(keypoints, bbox, frame_width: int, frame_height: int,
                       params: FollowParams = FollowParams()) -> tuple[bool, str]:
    """Is the person visible enough, upright and facing the camera for box height to mean distance?"""
    reason = body_metrics(keypoints, bbox, frame_width, frame_height, params)["reason"]
    return reason == "ok", reason


def overlap_ratio(target, other) -> float:
    """Part of the target box's area covered by `other` (0..1)."""
    tx1, ty1, tx2, ty2 = target
    ox1, oy1, ox2, oy2 = other
    w = min(tx2, ox2) - max(tx1, ox1)
    h = min(ty2, oy2) - max(ty1, oy1)
    area = (tx2 - tx1) * (ty2 - ty1)
    return max(0.0, w) * max(0.0, h) / area if area > 0 else 0.0


def allowed_relative_change(dt: float, params: FollowParams, after_rejected_jump: bool) -> float:
    """Largest |h / h_ref - 1| accepted after `dt` real seconds since the last accepted measurement."""
    if after_rejected_jump:
        return params.change_noise_allowance  # only a return to the old size is accepted
    return min(params.max_change_after_gap, params.max_change_rate_per_s * max(dt, 0.0) + params.change_noise_allowance)


# ---------- controller ----------

class FollowController:
    def __init__(self, params: FollowParams = FollowParams()):
        self.params = params
        self._target_id: int | None = None
        self._last_update_time: float | None = None
        self._last_command: FollowCommand = _stop(FollowState.INACTIVE, "not started", None)
        self._reset_target(None)

    @property
    def last_command(self) -> FollowCommand:
        return self._last_command

    @property
    def valid_streak(self) -> int:
        return self._valid_streak

    @property
    def baseline_height(self) -> float | None:
        return self._baseline_h

    @property
    def last_body_metrics(self) -> dict | None:
        """body_metrics() of the last NEW result that reached the full-body check (for logging)."""
        return self._last_body

    def rebaseline(self) -> None:
        """Operator action: drop the accepted box-height baseline so the next valid streak defines it."""
        self._clear_baseline()
        self._invalidate()

    def update(self, inp: FollowInput) -> FollowCommand:
        """Process one input (call once per AI result; repeated result timestamps are not re-measured)."""
        self._last_update_time = inp.now
        self._last_command = self._decide(inp)
        return self._last_command

    def command_at(self, now: float) -> FollowCommand:
        """What a command publisher should send at time `now` (communication / freshness watchdog)."""
        if self._last_update_time is None or now - self._last_update_time > self.params.command_timeout_s:
            self._invalidate()
            return _stop(FollowState.HOLD_TIMEOUT, "no controller update within command timeout", self._target_id,
                         stage="timeout")
        cmd = self._last_command
        moving = cmd.forward_mps > 0 or cmd.yaw_rate_dps != 0
        if moving and (self._last_result_t is None or now - self._last_result_t > self.params.max_result_age_s):
            self._invalidate()
            return _stop(FollowState.HOLD_INVALID, "AI result stale", self._target_id, stage="timeout")
        return cmd

    # --- internals ---

    def _clear_baseline(self) -> None:
        self._baseline_h = None
        self._baseline_t = None
        self._baseline_bbox = None
        self._baseline_lower_conf = None
        self._jump_rejected = False

    def _reset_target(self, target_id) -> None:
        self._target_id = target_id
        self._clear_baseline()
        self._last_result_t = None
        self._last_body = None
        self._invalidate()

    def _invalidate(self) -> None:
        self._valid_streak = 0
        self._forward = 0.0
        self._yaw = 0.0
        self._moving = False
        self._last_motion_time = None

    def _hold(self, state, reason, **extra) -> FollowCommand:
        self._invalidate()
        return _stop(state, reason, self._target_id, **extra)

    def _decide(self, inp: FollowInput) -> FollowCommand:
        p = self.params
        if inp.selection_state is SelectionState.LAND_REQUESTED:
            self._invalidate()
            return _stop(FollowState.LAND_REQUESTED, "landing requested by target", inp.target_id, land=True,
                         stage="selection")
        if inp.selection_state is not SelectionState.FOLLOWING or inp.target_id is None:
            self._reset_target(None)
            return _stop(FollowState.INACTIVE, "no target selected", None, stage="selection")
        if inp.target_id != self._target_id:
            self._reset_target(inp.target_id)  # new selection: nothing carried over

        if not inp.target_observed:
            return self._hold(FollowState.HOLD_TARGET_LOST, "target not in latest result", stage="data")
        if inp.result_timestamp is None or inp.now - inp.result_timestamp > p.max_result_age_s:
            return self._hold(FollowState.HOLD_INVALID, "AI result stale", stage="data")
        if not inp.usable:
            return self._hold(FollowState.HOLD_INVALID, "target status not usable", stage="data")
        if inp.measurement_track_id != inp.target_id or inp.bbox is None:
            return self._hold(FollowState.HOLD_INVALID, "no measurement for the selected track", stage="data")

        if self._last_result_t is not None and inp.result_timestamp <= self._last_result_t:
            return self._last_command  # same AI result again (e.g. REUSED frame): no new measurement
        self._last_result_t = inp.result_timestamp

        bbox = tuple(float(v) for v in inp.bbox)
        x1, y1, x2, y2 = bbox
        h = (y2 - y1) / inp.frame_height
        err = ((x1 + x2) / 2 / inp.frame_width - 0.5) * 2
        meas = {"box_height": h, "center_error": err}

        # 1. full body (unchanged checks; adds an OCCLUSION_SUSPECTED label when body parts are unseen
        #    while the box is inside the frame - a suspicion only, the parts may simply be out of view)
        body = self._last_body = body_metrics(inp.keypoints, bbox, inp.frame_width, inp.frame_height, p)
        if body["reason"] != "ok":
            unseen = [f for f in body["failures"] if f in VISIBILITY_FAILURES]
            suspected = bool(unseen) and not body.get("clipped", False)
            return self._hold(FollowState.HOLD_INVALID, f"full body: {body['reason']}", stage="full_body",
                              diagnostics=("OCCLUSION_SUSPECTED",) if suspected else (),
                              indicators=tuple(body["failures"]), **meas)

        # 2. occlusion indicators on an otherwise valid body
        overlaps = [overlap_ratio(bbox, tuple(float(v) for v in o)) for o in inp.other_boxes]
        worst = max(overlaps, default=0.0)
        if worst >= p.max_other_overlap_ratio:
            return self._hold(FollowState.HOLD_INVALID, "another detection overlaps the target box",
                              stage="occlusion", diagnostics=("OCCLUSION_SUSPECTED",),
                              indicators=(f"other_overlap={worst:.2f}",), **meas)
        if self._baseline_lower_conf is not None and \
                self._baseline_lower_conf - body["lower_conf"] >= p.max_lower_body_conf_drop:
            return self._hold(FollowState.HOLD_INVALID, "lower-body keypoint confidence dropped",
                              stage="occlusion", diagnostics=("OCCLUSION_SUSPECTED",),
                              indicators=(f"lower_conf {self._baseline_lower_conf:.2f}->{body['lower_conf']:.2f}",),
                              **meas)

        # 3. box-height measurement
        if inp.confidence is None or inp.confidence < p.min_confidence:
            return self._hold(FollowState.HOLD_INVALID, "detection confidence low", stage="measurement", **meas)
        if not p.min_box_height <= h <= p.max_box_height:
            return self._hold(FollowState.HOLD_INVALID, "box height outside usable range", stage="measurement", **meas)
        if self._baseline_h is not None:
            dt = inp.result_timestamp - self._baseline_t
            change = abs(h / self._baseline_h - 1)
            if dt <= 0 or change > allowed_relative_change(dt, p, self._jump_rejected):
                self._jump_rejected = True  # rejected size is NOT adopted as the baseline
                edges = self._edge_indicators(bbox)
                one_edge = any(e in ("bottom edge rose, top stable", "top edge dropped, bottom stable") for e in edges)
                return self._hold(FollowState.HOLD_INVALID,
                                  f"box height change {change:.0%} not explained (ref {self._baseline_h:.2f})",
                                  stage="measurement", diagnostics=("OCCLUSION_SUSPECTED",) if one_edge else (),
                                  indicators=edges, **meas)

        self._baseline_h, self._baseline_t, self._jump_rejected = h, inp.result_timestamp, False
        self._baseline_bbox, self._baseline_lower_conf = bbox, body["lower_conf"]
        self._valid_streak += 1
        if self._valid_streak < p.resume_valid_count:
            self._forward = self._yaw = 0.0
            return _stop(FollowState.HOLD_INVALID, f"revalidating {self._valid_streak}/{p.resume_valid_count}",
                         self._target_id, stage="revalidation", **meas)
        return self._control(inp.now, h, err, meas)

    def _edge_indicators(self, bbox) -> tuple:
        """Which box edges moved vs the last accepted box (a shrink with one edge fixed hints at occlusion)."""
        if self._baseline_bbox is None:
            return ()
        p = self.params
        ref_h = self._baseline_bbox[3] - self._baseline_bbox[1]
        top = (bbox[1] - self._baseline_bbox[1]) / ref_h
        bottom = (bbox[3] - self._baseline_bbox[3]) / ref_h
        out = [f"top_shift={top:+.2f}", f"bottom_shift={bottom:+.2f}"]
        if abs(top) <= p.edge_shift_stable and bottom <= -p.edge_shift_moved:
            out.append("bottom edge rose, top stable")
        elif abs(bottom) <= p.edge_shift_stable and top >= p.edge_shift_moved:
            out.append("top edge dropped, bottom stable")
        return tuple(out)

    def _control(self, now: float, h: float, err: float, meas) -> FollowCommand:
        p = self.params
        dt = 0.0 if self._last_motion_time is None else max(0.0, now - self._last_motion_time)
        self._last_motion_time = now

        yaw_target = 0.0 if abs(err) <= p.yaw_deadband else max(-p.max_yaw_rate_dps,
                                                                  min(p.max_yaw_rate_dps, p.yaw_gain_dps * err))
        step = p.max_yaw_accel_dps2 * dt
        self._yaw = self._yaw + max(-step, min(step, yaw_target - self._yaw))

        self._moving = h < p.target_box_height_low if self._moving else \
            h < p.target_box_height_low - p.start_forward_hysteresis
        v_target = min(p.max_forward_mps, p.forward_gain * (p.target_box_height_low - h)) if self._moving else 0.0
        self._forward = max(0.0, min(v_target, self._forward + p.max_forward_accel * dt))

        if self._moving:
            state, reason = FollowState.ACTIVE, "following"
        elif h > p.target_box_height_high:
            state, reason = FollowState.HOLD_DISTANCE, "too close (never backward)"
        else:
            state, reason = FollowState.HOLD_DISTANCE, "in distance band" if h >= p.target_box_height_low \
                else "below band, inside hysteresis"
        return FollowCommand(state, reason, self._forward, self._yaw, self._forward == 0.0, False,
                             self._target_id, stage="control", **meas)

"""Single-frame body-pose classification from one person's 17 COCO keypoints.

Classifies one person's keypoints (a (17, 3) array of pixel x, y and
confidence, e.g. one row of `PoseDetections.keypoints`) into one of:

    ONE_ARM_UP     - exactly one arm clearly raised above the head, other arm down
    T_POSE         - both arms extended outward, roughly horizontal
    DOUBLE_BICEPS  - both upper arms out, elbows bent, forearms pointing up
    NONE           - none of the above, ambiguous, or keypoints not reliable

Ported from the MediaPipe-based `body_pose_recognizer.py` on main (same rules
and thresholds), adapted to COCO keypoints:
    - Keypoints are already in pixels, so no frame-size conversion is needed.
    - Keypoint confidence replaces MediaPipe landmark visibility.
    - No face keypoints: "arm above the head" is measured from the shoulder
      line (the MediaPipe version measured from the nose). ONE_ARM_UP also
      requires the raised arm to point steeply upward, which keeps it
      disjoint from T_POSE.

Only the six arm keypoints (shoulders, elbows, wrists) are used, so the
person's hips/legs may be out of frame.

Geometry conventions:
    - Image y grows downward. "Elevation" is the angle of a segment above
      horizontal, so positive = pointing up.
    - Distance thresholds are multiples of the shoulder width in the image
      (S), so the rules don't depend on distance from the camera.
    - "Outward" is measured away from the body's center line, so the rules
      work the same for both arms and either facing direction.

Hold-time stabilization is separate (pose_command_stabilizer.py). This
module knows nothing about tracking or drone control - the command names are
display labels only at this stage. All thresholds are experimental starting
values, not tuned.
"""

import math
from dataclasses import dataclass
from enum import Enum

from coco_keypoints import (
    KEYPOINT_CONF_THRESHOLD, LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST,
    reliable_mask, validate_keypoints,
)


class BodyPose(Enum):
    NONE = "NONE"
    ONE_ARM_UP = "ONE_ARM_UP"
    T_POSE = "T_POSE"
    DOUBLE_BICEPS = "DOUBLE_BICEPS"


class PoseCommand(Enum):
    NONE = "NONE"
    TARGET_SELECT = "TARGET_SELECT"
    HOVER = "HOVER"
    LAND = "LAND"


# Display mapping only - no drone behavior is triggered.
POSE_TO_COMMAND = {
    BodyPose.NONE: PoseCommand.NONE,
    BodyPose.ONE_ARM_UP: PoseCommand.TARGET_SELECT,
    BodyPose.T_POSE: PoseCommand.HOVER,
    BodyPose.DOUBLE_BICEPS: PoseCommand.LAND,
}

ARM_KEYPOINTS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)

# ONE_ARM_UP: raised arm
ARM_UP_MIN_WRIST_ABOVE_SHOULDER = 0.9  # x S; roughly the old "0.3 S above the nose"
ARM_UP_MIN_ELEVATION = 45.0  # degrees, shoulder -> wrist; clearly upward, never T-pose-like
ARM_UP_MIN_ELBOW_ANGLE = 130.0  # degrees; raised arm must be mostly straight
# ONE_ARM_UP: the other arm must be clearly down
ARM_DOWN_MIN_WRIST_BELOW_SHOULDER = 0.3  # x S

# T_POSE (each arm)
T_POSE_MIN_ELBOW_ANGLE = 150.0  # degrees; nearly straight
T_POSE_MAX_ARM_TILT = 25.0  # degrees from horizontal, shoulder -> wrist
T_POSE_MIN_WRIST_OUTWARD = 1.0  # x S; wrist horizontally out from its shoulder

# DOUBLE_BICEPS (each arm)
DOUBLE_BICEPS_MIN_ELBOW_ANGLE = 45.0  # degrees
DOUBLE_BICEPS_MAX_ELBOW_ANGLE = 120.0  # degrees; clearly bent
DOUBLE_BICEPS_MIN_UPPER_ARM_ELEVATION = -30.0  # degrees, shoulder -> elbow
DOUBLE_BICEPS_MAX_UPPER_ARM_ELEVATION = 50.0  # degrees, shoulder -> elbow
DOUBLE_BICEPS_MIN_ELBOW_OUTWARD = 0.3  # x S
DOUBLE_BICEPS_MIN_FOREARM_ELEVATION = 45.0  # degrees, elbow -> wrist


@dataclass
class Arm:
    """One arm's joints in pixel coordinates, plus its outward direction (+1/-1 in x)."""

    shoulder: tuple[float, float]
    elbow: tuple[float, float]
    wrist: tuple[float, float]
    outward: int


# ---------- geometry helpers (pure functions, pixel coordinates) ----------

def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def joint_angle(a, b, c) -> float:
    """Angle ABC at vertex b, in degrees (180 = a, b, c in a straight line)."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))


def elevation(start, end) -> float:
    """Angle of segment start->end above horizontal, in degrees (+90 = straight up)."""
    return math.degrees(math.atan2(start[1] - end[1], abs(end[0] - start[0])))


def height_above(a, b) -> float:
    """How far point a is above point b, in pixels (image y grows downward)."""
    return b[1] - a[1]


# ---------- per-arm checks ----------

def is_arm_raised(arm: Arm, scale: float) -> bool:
    return (
        joint_angle(arm.shoulder, arm.elbow, arm.wrist) >= ARM_UP_MIN_ELBOW_ANGLE
        and height_above(arm.elbow, arm.shoulder) > 0
        and height_above(arm.wrist, arm.shoulder) >= ARM_UP_MIN_WRIST_ABOVE_SHOULDER * scale
        and elevation(arm.shoulder, arm.wrist) >= ARM_UP_MIN_ELEVATION
    )


def is_arm_down(arm: Arm, scale: float) -> bool:
    return height_above(arm.shoulder, arm.wrist) >= ARM_DOWN_MIN_WRIST_BELOW_SHOULDER * scale


def is_t_pose_arm(arm: Arm, scale: float) -> bool:
    wrist_outward = (arm.wrist[0] - arm.shoulder[0]) * arm.outward
    return (
        joint_angle(arm.shoulder, arm.elbow, arm.wrist) >= T_POSE_MIN_ELBOW_ANGLE
        and abs(elevation(arm.shoulder, arm.wrist)) <= T_POSE_MAX_ARM_TILT
        and wrist_outward >= T_POSE_MIN_WRIST_OUTWARD * scale
    )


def is_double_biceps_arm(arm: Arm, scale: float) -> bool:
    elbow_angle = joint_angle(arm.shoulder, arm.elbow, arm.wrist)
    upper_arm_elevation = elevation(arm.shoulder, arm.elbow)
    elbow_outward = (arm.elbow[0] - arm.shoulder[0]) * arm.outward
    return (
        DOUBLE_BICEPS_MIN_ELBOW_ANGLE <= elbow_angle <= DOUBLE_BICEPS_MAX_ELBOW_ANGLE
        and DOUBLE_BICEPS_MIN_UPPER_ARM_ELEVATION <= upper_arm_elevation <= DOUBLE_BICEPS_MAX_UPPER_ARM_ELEVATION
        and elbow_outward >= DOUBLE_BICEPS_MIN_ELBOW_OUTWARD * scale
        and elevation(arm.elbow, arm.wrist) >= DOUBLE_BICEPS_MIN_FOREARM_ELEVATION
    )


# ---------- classification ----------

def classify_pose(keypoints, min_conf: float = KEYPOINT_CONF_THRESHOLD) -> BodyPose:
    """Classify one person's pose in one frame.

    Args:
        keypoints: (17, 3) COCO keypoints for one person - pixel x, y, confidence.
        min_conf: Minimum confidence for each of the six arm keypoints.

    Returns:
        The single matching `BodyPose`, or `BodyPose.NONE` if no rule matches,
        more than one matches (ambiguous), or any arm keypoint is unreliable.
    """
    keypoints = validate_keypoints(keypoints)
    if not reliable_mask(keypoints, min_conf)[list(ARM_KEYPOINTS)].all():
        return BodyPose.NONE

    def px(i):
        return (float(keypoints[i, 0]), float(keypoints[i, 1]))

    left_shoulder, right_shoulder = px(LEFT_SHOULDER), px(RIGHT_SHOULDER)
    scale = distance(left_shoulder, right_shoulder)
    if scale <= 0:
        return BodyPose.NONE

    left_outward = 1 if left_shoulder[0] >= right_shoulder[0] else -1
    arms = (
        Arm(left_shoulder, px(LEFT_ELBOW), px(LEFT_WRIST), left_outward),
        Arm(right_shoulder, px(RIGHT_ELBOW), px(RIGHT_WRIST), -left_outward),
    )

    matches = []

    if all(is_t_pose_arm(arm, scale) for arm in arms):
        matches.append(BodyPose.T_POSE)

    if all(is_double_biceps_arm(arm, scale) for arm in arms):
        matches.append(BodyPose.DOUBLE_BICEPS)

    a, b = arms
    if (is_arm_raised(a, scale) and is_arm_down(b, scale)) or (is_arm_raised(b, scale) and is_arm_down(a, scale)):
        matches.append(BodyPose.ONE_ARM_UP)

    # The rules are designed to be mutually exclusive; if more than one still
    # matches, refuse to guess - NONE is the safe answer for a drone command.
    return matches[0] if len(matches) == 1 else BodyPose.NONE

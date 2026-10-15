"""Draw the tracking pipeline's state onto the streamed frame.

Same colours and labels as the Windows demo (src/yolo_pose_demo.py: draw_track,
draw_skeleton, pose_label_lines, draw_status), redrawn here because importing
that module would pull the CPU/GPU YOLO detector onto the Pi:
    CONFIRMED cyan, CANDIDATE orange, TARGET magenta (thicker box, "TARGET |"),
    "ID n | STATE | conf x.xx", then "Raw Pose" / "Command" for CONFIRMED tracks,
    skeleton (reliable keypoints only) for CONFIRMED tracks.
Smaller text than the Windows window because the stream is 640x480.

Boxes come from BoT-SORT's Kalman state and can leave the image, so they are
clipped. Status: FPS / DPS / detections / CONFIRMED count, selection state and
target, plus one red warning line - STALE while the newest inference result is
older than STALE_AFTER_S, otherwise the reason the target is not usable.
"""

import cv2
import numpy as np

from coco_keypoints import reliable_mask, skeleton_segments
from coco_pose_recognizer import POSE_TO_COMMAND, BodyPose
from imx500_pose_overlay import is_stale
from track_validator import TrackState

# BGR, same values as src/yolo_pose_demo.py
CONFIRMED_COLOR = (255, 255, 0)
CANDIDATE_COLOR = (0, 165, 255)
TARGET_COLOR = (255, 0, 255)
KEYPOINT_COLOR = (0, 255, 0)
WARNING_COLOR = (0, 0, 255)
STATUS_COLOR = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.45
LINE_HEIGHT = 16


def clip_box(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (min(max(int(x1), 0), width - 1), min(max(int(y1), 0), height - 1),
            min(max(int(x2), 0), width - 1), min(max(int(y2), 0), height - 1))


def pose_label_lines(state, hold_time: float) -> list[str]:
    """'Raw Pose' / 'Command' lines for one track's StabilizerState (same text as the Windows demo)."""
    command = POSE_TO_COMMAND[state.active].value
    if state.active is BodyPose.NONE and state.candidate is not BodyPose.NONE:
        command += f" (hold {min(state.held_for, hold_time):.2f}/{hold_time:.2f}s)"
    if state.in_grace:
        command += " [NONE grace]"
    return [f"Raw Pose: {state.raw_pose.value}", f"Command: {command}"]


def track_color(state: TrackState, is_target: bool):
    if is_target:
        return TARGET_COLOR
    return CONFIRMED_COLOR if state is TrackState.CONFIRMED else CANDIDATE_COLOR


def draw_skeleton(image, keypoints, color) -> None:
    points = [(int(round(x)), int(round(y))) for x, y, _ in keypoints]
    for a, b in skeleton_segments(keypoints):
        cv2.line(image, points[a], points[b], color, 2, cv2.LINE_AA)
    for point, reliable in zip(points, reliable_mask(keypoints)):
        if reliable:
            cv2.circle(image, point, 3, KEYPOINT_COLOR, -1, cv2.LINE_AA)


def draw_tracks(image: np.ndarray, result, hold_time: float) -> np.ndarray:
    """Draw every validated track of a PipelineResult. Modifies `image` in place."""
    if result is None:
        return image
    height, width = image.shape[:2]
    target_id = result.selection.target_id
    for validated in result.validated_tracks:
        person = validated.tracked_person
        is_target = person.track_id == target_id
        color = track_color(validated.state, is_target)
        x1, y1, x2, y2 = clip_box(person.bbox, width, height)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3 if is_target else 2)

        header = f"ID {person.track_id} | {validated.state.value} | conf {person.confidence:.2f}"
        lines = [("TARGET | " if is_target else "") + header]
        state = result.pose_states.get(person.track_id)
        if state is not None:
            lines += pose_label_lines(state, hold_time)
        top = max(y1 - 6 - LINE_HEIGHT * (len(lines) - 1), 40 + LINE_HEIGHT)  # keep clear of the status band
        for i, line in enumerate(lines):
            cv2.putText(image, line, (x1, top + LINE_HEIGHT * i), FONT, TEXT_SCALE, color, 1, cv2.LINE_AA)

        index = result.track_to_detection.get(person.track_id)
        if validated.state is TrackState.CONFIRMED and index is not None:
            draw_skeleton(image, result.detections.keypoints[index], color)
    return image


def status_lines(fps: float, dps: float, result, target_status, elapsed: float | None = None) -> list[str]:
    """`elapsed`: optional test clock (s) shown so a tester can follow a scenario timetable on screen."""
    detections = 0 if result is None else len(result.detections)
    confirmed = 0 if result is None else sum(v.state is TrackState.CONFIRMED for v in result.validated_tracks)
    target = "-" if target_status.target_id is None else str(target_status.target_id)
    second = f"State: {target_status.state.value} | Target: {target}"
    if elapsed is not None:
        second += f" | t={elapsed:5.1f}s"
    return [f"FPS {fps:4.1f} | DPS {dps:4.1f} | det {detections} | confirmed {confirmed}", second]


def warning_text(result_age, target_status) -> str | None:
    if result_age is not None and is_stale(result_age):
        return f"STALE {result_age:.1f}s - no new inference result"
    if target_status.target_id is not None and not target_status.usable:
        return f"TARGET NOT USABLE: {target_status.reason}"
    return None


def draw_tracking_status(image: np.ndarray, lines: list[str], warning: str | None) -> np.ndarray:
    """Dark status band with `lines`, plus a red `warning` line under it. Modifies `image` in place."""
    band = LINE_HEIGHT * len(lines) + 8
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, band), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (8, 16 + LINE_HEIGHT * i), FONT, 0.5, STATUS_COLOR, 1, cv2.LINE_AA)
    if warning:
        cv2.rectangle(image, (0, band), (image.shape[1] - 1, band + LINE_HEIGHT + 4), (0, 0, 0), -1)
        cv2.putText(image, warning, (8, band + LINE_HEIGHT), FONT, 0.5, WARNING_COLOR, 1, cv2.LINE_AA)
    return image

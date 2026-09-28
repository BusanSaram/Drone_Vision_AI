"""Standalone webcam test for MediaPipe Pose and body-pose command recognition.

Pipeline for this test only: Webcam -> MediaPipe PoseLandmarker ->
classify_pose() -> PoseCommandStabilizer -> OpenCV visualization. Completely
independent of the YOLO/BoT-SORT person tracking pipeline
(`yolo_person_detection.py`) - no shared state, no imports from it.

Uses the MediaPipe Tasks `PoseLandmarker` API. The legacy `mp.solutions.pose`
API does not exist in the installed MediaPipe version (1.0.1).

The displayed commands (TARGET_SELECT / HOVER / LAND) are visual labels only;
nothing is sent anywhere. Not implemented here (by design, for this
milestone): pose-to-person association, target selection, or drone control.

Model: `models/pose_landmarker_lite.task`, downloaded separately from
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)

from body_pose_recognizer import POSE_TO_COMMAND, BodyPose, classify_pose
from pose_command_stabilizer import PoseCommandStabilizer

# Resolved relative to this file (not the caller's cwd): MediaPipe has no
# auto-download fallback and hard-fails if the path doesn't resolve.
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task")

# Experimental starting values, not tuned. These match MediaPipe's defaults.
NUM_POSES = 1
MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Landmarks with visibility below this are not drawn (e.g. legs out of
# frame), so the skeleton doesn't show guessed off-screen points.
MIN_LANDMARK_VISIBILITY = 0.5

LANDMARK_COLOR = (0, 255, 0)  # green (BGR)
CONNECTION_COLOR = (255, 255, 0)  # cyan (BGR)


def create_landmarker(num_poses: int = NUM_POSES) -> PoseLandmarker:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return PoseLandmarker.create_from_options(options)


def draw_pose(frame, landmarks, frame_width, frame_height):
    """Draw one pose's 33 landmarks and skeleton connections.

    `landmarks` are MediaPipe's normalized (x, y) in [0, 1] relative to frame
    width/height, converted to pixels here.
    """
    points = [(int(lm.x * frame_width), int(lm.y * frame_height)) for lm in landmarks]
    visible = [(lm.visibility or 0.0) >= MIN_LANDMARK_VISIBILITY for lm in landmarks]

    for connection in PoseLandmarksConnections.POSE_LANDMARKS:
        if visible[connection.start] and visible[connection.end]:
            cv2.line(frame, points[connection.start], points[connection.end], CONNECTION_COLOR, 2)
    for point, is_visible in zip(points, visible):
        if is_visible:
            cv2.circle(frame, point, 4, LANDMARK_COLOR, -1)


def draw_status(frame, state, hold_time, fps):
    """Top-left text: raw pose, candidate, active command, hold progress, FPS."""
    if state.candidate is BodyPose.NONE:
        hold_text = "Hold: -"
    else:
        hold_text = f"Hold: {min(state.held_for, hold_time):.2f} / {hold_time:.2f} s"

    lines = [
        f"Raw Pose: {state.raw_pose.value}",
        f"Candidate: {state.candidate.value}",
        f"Command: {POSE_TO_COMMAND[state.active].value}",
        hold_text,
        f"FPS: {fps:.1f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (10, 30 + 30 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )


def main():
    landmarker = create_landmarker()
    stabilizer = PoseCommandStabilizer()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: could not open webcam.")
        landmarker.close()
        return

    prev_time = time.time()
    last_timestamp_ms = -1

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: could not read frame from webcam.")
            break

        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(int(time.monotonic() * 1000), last_timestamp_ms + 1)
        last_timestamp_ms = timestamp_ms

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        frame_height, frame_width = frame.shape[:2]
        raw_pose = BodyPose.NONE
        for landmarks in result.pose_landmarks:
            draw_pose(frame, landmarks, frame_width, frame_height)
        if result.pose_landmarks:  # NUM_POSES = 1, so at most one person
            raw_pose = classify_pose(result.pose_landmarks[0], frame_width, frame_height)

        state = stabilizer.update(raw_pose)
        if state.newly_activated:
            print(f"Command activated: {POSE_TO_COMMAND[state.active].value} ({state.active.value})")

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        draw_status(frame, state, stabilizer.hold_time, fps)

        cv2.imshow("MediaPipe Pose Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()

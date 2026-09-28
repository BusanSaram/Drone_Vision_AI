"""Webcam integration demo: which tracked person is doing which body pose.

Pipeline:
    Webcam -> PersonDetector (YOLOv8n) -> PersonTracker (BoT-SORT) -> TrackValidator
           -> MediaPipe PoseLandmarker (up to MAX_POSES people, full frame)
           -> associate_poses() against CONFIRMED tracks -> classify_pose() per matched pose
           -> OpenCV visualization

For each tracked person the box shows "ID n | STATE" and "Pose: <raw pose>"
("Pose: -" if no pose is associated with that track this frame). At each
associated pose's torso anchor, "ID n" is drawn, so you can see which track
the skeleton was attached to. Usable poses that matched no track are marked
"UNASSOCIATED".

The pose shown is the raw per-frame classification. Nothing here selects a
target or triggers any command - this demo only verifies association.
"""

import time

import cv2
import mediapipe as mp

from body_pose_recognizer import classify_pose
from person_detector import PersonDetector, get_device
from person_tracker import PersonTracker
from pose_test import create_landmarker, draw_pose
from pose_track_associator import associate_poses, pose_region
from track_validator import TrackState, TrackValidator, confirmed_only
from yolo_person_detection import NEW_TRACK_THRESH

# Maximum number of people MediaPipe looks for in one frame. Experimental
# starting value; more poses cost more CPU time.
MAX_POSES = 3

CONFIRMED_COLOR = (255, 255, 0)  # cyan (BGR)
CANDIDATE_COLOR = (0, 165, 255)  # orange (BGR)
UNASSOCIATED_COLOR = (0, 0, 255)  # red (BGR)


def draw_track(frame, validated, pose_label):
    person = validated.tracked_person
    x1, y1, x2, y2 = person.bbox
    color = CONFIRMED_COLOR if validated.state is TrackState.CONFIRMED else CANDIDATE_COLOR

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, f"ID {person.track_id} | {validated.state.value}", (x1, max(y1 - 30, 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )
    cv2.putText(
        frame, f"Pose: {pose_label}", (x1, max(y1 - 8, 35)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )


def draw_anchor(frame, anchor, text, color):
    point = (int(anchor[0]), int(anchor[1]))
    cv2.circle(frame, point, 6, color, -1)
    cv2.putText(frame, text, (point[0] + 8, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_fps(frame, fps):
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )


def main():
    device = get_device()
    print(f"Using device: {device}")

    detector = PersonDetector(device=device)
    tracker = PersonTracker(new_track_thresh=NEW_TRACK_THRESH)
    validator = TrackValidator()
    landmarker = create_landmarker(num_poses=MAX_POSES)

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

        frame_height, frame_width = frame.shape[:2]

        # Detection / tracking (unchanged pipeline).
        boxes = detector.detect(frame)
        tracked_people = tracker.update(boxes, frame)
        validated_tracks = validator.update(tracked_people)

        # Pose on the full frame. VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(int(time.monotonic() * 1000), last_timestamp_ms + 1)
        last_timestamp_ms = timestamp_ms
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        poses = landmarker.detect_for_video(mp_image, timestamp_ms).pose_landmarks

        # Association: only CONFIRMED tracks are eligible.
        track_to_pose = associate_poses(poses, confirmed_only(validated_tracks), frame_width, frame_height)
        pose_to_track = {pose_index: track_id for track_id, pose_index in track_to_pose.items()}

        # Drawing happens after all inference so overlays never feed back into it.
        for landmarks in poses:
            draw_pose(frame, landmarks, frame_width, frame_height)

        for validated in validated_tracks:
            pose_index = track_to_pose.get(validated.tracked_person.track_id)
            if pose_index is None:
                label = "-"
            else:
                label = classify_pose(poses[pose_index], frame_width, frame_height).value
            draw_track(frame, validated, label)

        for pose_index, landmarks in enumerate(poses):
            region = pose_region(landmarks, frame_width, frame_height)
            if region is None:
                continue
            if pose_index in pose_to_track:
                draw_anchor(frame, region.anchor, f"ID {pose_to_track[pose_index]}", CONFIRMED_COLOR)
            else:
                draw_anchor(frame, region.anchor, "UNASSOCIATED", UNASSOCIATED_COLOR)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        draw_fps(frame, fps)

        cv2.imshow("Pose-Track Association Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()

"""Webcam experiment: YOLOv8n-Pose -> BoT-SORT -> TrackValidator, no MediaPipe.

Pipeline:
    Webcam -> YoloPoseDetector (YOLOv8n-Pose, CUDA if available)
                 -> PoseDetections (bboxes + 17 COCO keypoints per person)
           -> PersonTracker (BoT-SORT, same config as yolo_person_detection.py)
           -> TrackValidator (CANDIDATE / CONFIRMED)
           -> match_tracks_to_detections() (track_id -> this frame's keypoints)
           -> OpenCV visualization

Tracker and validator settings are identical to the detection-only baseline
(`yolo_person_detection.py`) so the two can be compared directly. Only the
detector model differs.

Display: CONFIRMED tracks get bbox, "ID n | CONFIRMED | conf x.xx" and their
skeleton; CANDIDATE tracks get bbox and label only. Top-left shows smoothed
FPS, the number of YOLO Pose detections and the number of CONFIRMED tracks.
Per-stage rolling timings are printed to the console every
PRINT_INTERVAL seconds; run-average timings and FPS are printed on exit.

No body-pose commands, target selection or drone control here.
"""

import time

import cv2

from coco_keypoints import KEYPOINT_CONF_THRESHOLD, reliable_mask, skeleton_segments
from keypoint_track_association import match_tracks_to_detections
from person_detector import get_device
from person_tracker import PersonTracker
from rolling_stats import FpsMeter, StageTimer
from track_validator import CONFIRMATION_TIME, LOST_GRACE_TIME, TrackState, TrackValidator
from yolo_person_detection import NEW_TRACK_THRESH, WARMUP_FRAMES
from yolo_pose_detector import YoloPoseDetector

ROLLING_WINDOW = 30  # frames
PRINT_INTERVAL = 2.0  # seconds between console timing lines

STAGES = ("pose_inference", "tracking", "validation", "association", "drawing", "total")

CONFIRMED_COLOR = (255, 255, 0)  # cyan (BGR)
CANDIDATE_COLOR = (0, 165, 255)  # orange (BGR)
KEYPOINT_COLOR = (0, 255, 0)  # green (BGR)
STATUS_COLOR = (0, 0, 255)  # red (BGR)


def draw_skeleton(frame, keypoints, color, min_conf=KEYPOINT_CONF_THRESHOLD):
    """Draw reliable COCO connections and keypoints for one person."""
    points = [(int(x), int(y)) for x, y, _conf in keypoints]
    for a, b in skeleton_segments(keypoints, min_conf):
        cv2.line(frame, points[a], points[b], color, 2)
    for point, reliable in zip(points, reliable_mask(keypoints, min_conf)):
        if reliable:
            cv2.circle(frame, point, 4, KEYPOINT_COLOR, -1)


def draw_track(frame, validated):
    person = validated.tracked_person
    x1, y1, x2, y2 = person.bbox
    color = CONFIRMED_COLOR if validated.state is TrackState.CONFIRMED else CANDIDATE_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {person.track_id} | {validated.state.value} | conf {person.confidence:.2f}"
    cv2.putText(frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_status(frame, fps, num_detections, num_confirmed):
    fps_text = "FPS: -" if fps is None else f"FPS: {fps:.1f}"
    lines = [fps_text, f"Pose detections: {num_detections}", f"Confirmed tracks: {num_confirmed}"]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, STATUS_COLOR, 2)


def format_timings(timings_ms) -> str:
    return " | ".join(f"{name} {timings_ms[name]:.1f}ms" for name in STAGES if name in timings_ms)


def main():
    device = get_device()
    print(f"Using device: {device}")

    detector = YoloPoseDetector(device=device)
    tracker = PersonTracker(new_track_thresh=NEW_TRACK_THRESH)
    validator = TrackValidator(confirmation_time=CONFIRMATION_TIME, lost_grace_time=LOST_GRACE_TIME)
    timer = StageTimer(window=ROLLING_WINDOW)
    fps_meter = FpsMeter(window=ROLLING_WINDOW)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    frame_index = 0
    last_print = time.perf_counter()
    prev_frame_time = None
    start_time = None
    instant_fps_values = []  # same metric as the baseline's "Average FPS"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame from webcam.")
            break

        # FPS covers the whole loop (capture, processing, display).
        now = time.perf_counter()
        fps = fps_meter.tick(now)
        frame_index += 1
        if frame_index == WARMUP_FRAMES + 1:
            timer.reset_totals()  # exclude model/CUDA warm-up from the run averages
            start_time = now
        if frame_index > WARMUP_FRAMES and prev_frame_time is not None:
            instant_fps_values.append(1.0 / (now - prev_frame_time))
        prev_frame_time = now

        # "total" = processing only: inference through drawing, excluding capture and imshow/waitKey.
        with timer.measure("total"):
            with timer.measure("pose_inference"):
                detections = detector.detect(frame)
            with timer.measure("tracking"):
                tracked_people = tracker.update(detections, frame)
            with timer.measure("validation"):
                validated_tracks = validator.update(tracked_people)
            with timer.measure("association"):
                track_to_detection = match_tracks_to_detections(tracked_people, detections)

            with timer.measure("drawing"):
                num_confirmed = 0
                for validated in validated_tracks:
                    draw_track(frame, validated)
                    if validated.state is not TrackState.CONFIRMED:
                        continue
                    num_confirmed += 1
                    index = track_to_detection.get(validated.tracked_person.track_id)
                    if index is not None:
                        draw_skeleton(frame, detections.keypoints[index], CONFIRMED_COLOR)
                draw_status(frame, fps, len(detections), num_confirmed)

        if now - last_print >= PRINT_INTERVAL:
            fps_text = "-" if fps is None else f"{fps:.1f}"
            print(f"[rolling {ROLLING_WINDOW}f] FPS {fps_text} | {format_timings(timer.rolling_ms())}", flush=True)
            last_print = now

        cv2.imshow("YOLOv8n-Pose Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    end_time = time.perf_counter()
    cap.release()
    cv2.destroyAllWindows()

    print("--- Performance Summary (after warm-up) ---")
    if not instant_fps_values:
        print(f"No frames were measured (fewer than {WARMUP_FRAMES} warm-up frames processed).")
    else:
        duration = end_time - start_time
        print(f"Test duration: {duration:.2f} s")
        print(f"Total frames: {len(instant_fps_values)}")
        print(f"Average FPS (mean of per-frame 1/dt, same metric as baseline): "
              f"{sum(instant_fps_values) / len(instant_fps_values):.2f}")
        print(f"Effective FPS (frames / duration): {len(instant_fps_values) / duration:.2f}")
        print(f"Average stage times: {format_timings(timer.totals_ms())}")
    validator.print_summary()


if __name__ == "__main__":
    main()

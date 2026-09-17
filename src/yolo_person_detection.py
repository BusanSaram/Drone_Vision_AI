"""Real-time YOLO person detection with persistent BoT-SORT IDs on the live webcam feed."""

import time

import cv2

from person_detector import PersonDetector, get_device
from person_tracker import PersonTracker
from tracking_diagnostics import TrackingDiagnostics
from track_validator import CONFIRMATION_TIME, LOST_GRACE_TIME, TrackState, TrackValidator
from proximity_diagnostics import ProximityDiagnostics, nearest_confirmed_relationship

WARMUP_FRAMES = 10

# Verbose per-frame/per-event diagnostic logging (multi-detection geometry,
# bbox/IoU dumps, tracking fragmentation events, proximity diagnostics, FPS
# min/max, first-appearance summary). Off by default so normal runs only
# print the end-of-run summaries; set True to bring it back for debugging.
DEBUG = False

# BoT-SORT: minimum confidence required to START a new track (does not
# affect association with existing tracks). None = Ultralytics default
# (0.25, from botsort.yaml). Experimenting with a higher value to test
# whether it suppresses false tracks from low-confidence spurious
# detections while a single real person is in frame. Not a final tuned
# value - still experimental.
NEW_TRACK_THRESH = 0.8

CANDIDATE_COLOR = (0, 165, 255)  # orange (BGR)
CONFIRMED_COLOR = (255, 255, 0)  # cyan (BGR)


def draw_tracks(frame, tracked_people):
    for person in tracked_people:
        x1, y1, x2, y2 = person.bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Person ID {person.track_id} | {person.confidence:.2f}"
        cv2.putText(
            frame, label, (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )


def draw_raw_detections(frame, boxes):
    """Diagnostic-only overlay of raw YOLO boxes, separate from tracked boxes.

    Purely visual - does not read from or affect the tracker.
    """
    for bbox, confidence in zip(boxes.xyxy, boxes.conf):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
        label = f"RAW YOLO {float(confidence):.2f}"
        cv2.putText(
            frame, label, (x1, min(y2 + 15, frame.shape[0] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )


def draw_validation_labels(frame, validated_tracks):
    """Diagnostic-only overlay showing each track's TrackValidator state.

    Drawn in addition to (not replacing) the raw BoT-SORT box/label from
    `draw_tracks`, so raw tracker output stays visible for debugging.
    """
    for validated in validated_tracks:
        x1, y1, _x2, _y2 = validated.tracked_person.bbox
        track_id = validated.tracked_person.track_id

        if validated.state is TrackState.CONFIRMED:
            label = f"Person ID {track_id} | CONFIRMED"
            color = CONFIRMED_COLOR
        else:
            label = f"ID {track_id} | CANDIDATE"
            color = CANDIDATE_COLOR

        cv2.putText(
            frame, label, (x1, max(y1 - 30, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )


def draw_proximity_hints(frame, validated_tracks):
    """Diagnostic-only overlay: for each CANDIDATE, show its distance/IoU to the
    nearest CONFIRMED person this frame, if any.

    Purely informational - never changes CANDIDATE/CONFIRMED state, and never
    moves or modifies any bounding box.
    """
    for validated in validated_tracks:
        if validated.state is not TrackState.CANDIDATE:
            continue
        person = validated.tracked_person
        nearest = nearest_confirmed_relationship(person.bbox, person.track_id, validated_tracks)
        if nearest is None:
            continue
        x1, _y1, _x2, y2 = person.bbox
        label = f"near ID {nearest.confirmed_id} | gap_h={nearest.horizontal_gap:.0f}px | IoU={nearest.iou:.2f}"
        cv2.putText(
            frame, label, (x1, min(y2 + 35, frame.shape[0] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CANDIDATE_COLOR, 1,
        )


def draw_fps(frame, fps):
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )


def print_performance_summary(start_time, end_time, fps_values):
    frame_count = len(fps_values)

    print("--- Performance Summary ---")

    if frame_count == 0:
        print("No frames were measured (fewer than "
              f"{WARMUP_FRAMES} warm-up frames processed).")
        return

    duration = end_time - start_time
    print(f"Test duration: {duration:.2f} s")
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {sum(fps_values) / frame_count:.2f}")
    if DEBUG:
        print(f"Min FPS: {min(fps_values):.2f}")
        print(f"Max FPS: {max(fps_values):.2f}")


def main():
    device = get_device()
    print(f"Using device: {device}")

    detector = PersonDetector(device=device)
    tracker = PersonTracker(new_track_thresh=NEW_TRACK_THRESH)
    diagnostics = TrackingDiagnostics(new_track_thresh=NEW_TRACK_THRESH)
    validator = TrackValidator(confirmation_time=CONFIRMATION_TIME, lost_grace_time=LOST_GRACE_TIME, debug=DEBUG)
    proximity_diagnostics = ProximityDiagnostics()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    prev_time = time.time()
    start_time = None
    frame_index = 0
    fps_values = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: could not read frame from webcam.")
            break

        boxes = detector.detect(frame)
        tracked_people = tracker.update(boxes, frame)
        now = time.time()
        validated_tracks = validator.update(tracked_people, now)
        if DEBUG:
            proximity_diagnostics.observe(now, boxes, validated_tracks, frame.shape[1], frame.shape[0])

        draw_tracks(frame, tracked_people)
        draw_raw_detections(frame, boxes)
        draw_validation_labels(frame, validated_tracks)
        draw_proximity_hints(frame, validated_tracks)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        frame_index += 1
        if frame_index > WARMUP_FRAMES:
            if start_time is None:
                start_time = current_time
            fps_values.append(fps)

        if DEBUG:
            diagnostics.observe(frame_index, boxes, tracked_people, frame.shape[0])

        cv2.imshow("YOLO Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    end_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

    print_performance_summary(start_time, end_time, fps_values)
    if DEBUG:
        diagnostics.print_first_appearance_summary()
    validator.print_summary()


if __name__ == "__main__":
    main()

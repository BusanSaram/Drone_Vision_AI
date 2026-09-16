"""Real-time YOLO person detection with persistent BoT-SORT IDs on the live webcam feed."""

import time

import cv2

from person_detector import PersonDetector, get_device
from person_tracker import PersonTracker

WARMUP_FRAMES = 10


def draw_tracks(frame, tracked_people):
    for person in tracked_people:
        x1, y1, x2, y2 = person.bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Person ID {person.track_id} | {person.confidence:.2f}"
        cv2.putText(
            frame, label, (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )


def draw_fps(frame, fps):
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )


def print_performance_summary(start_time, end_time, fps_values):
    frame_count = len(fps_values)

    print("\n--- Performance Summary (post warm-up) ---")

    if frame_count == 0:
        print("No frames were measured (fewer than "
              f"{WARMUP_FRAMES} warm-up frames processed).")
        return

    duration = end_time - start_time
    print(f"Test duration: {duration:.2f} s")
    print(f"Total measured frames: {frame_count}")
    print(f"Average FPS: {sum(fps_values) / frame_count:.2f}")
    print(f"Min FPS: {min(fps_values):.2f}")
    print(f"Max FPS: {max(fps_values):.2f}")


def main():
    device = get_device()
    print(f"Using device: {device}")

    detector = PersonDetector(device=device)
    tracker = PersonTracker()

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

        draw_tracks(frame, tracked_people)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        frame_index += 1
        if frame_index > WARMUP_FRAMES:
            if start_time is None:
                start_time = current_time
            fps_values.append(fps)

        cv2.imshow("YOLO Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    end_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

    print_performance_summary(start_time, end_time, fps_values)


if __name__ == "__main__":
    main()

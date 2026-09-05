"""Minimal YOLO person detection test on the live webcam feed."""

import time

import cv2
import torch
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class index for "person"
MODEL_PATH = "yolov8n.pt"
WARMUP_FRAMES = 10


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def draw_detections(frame, results):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {confidence:.2f}"
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

    model = YOLO(MODEL_PATH)

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

        results = model.predict(
            frame, classes=[PERSON_CLASS_ID], device=device, verbose=False,
        )[0]

        draw_detections(frame, results)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        draw_fps(frame, fps)

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

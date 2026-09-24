"""Standalone webcam test for hand gesture recognition.

Pipeline for this test only: Webcam -> MediaPipe GestureRecognizer -> OpenCV
visualization. Completely independent of the YOLO/BoT-SORT person
tracking pipeline (`yolo_person_detection.py`) - no shared state, no
imports from it, and it does not modify that pipeline in any way.

Not implemented here (by design, for this milestone): gesture-to-person
association, target selection, or drone control.
"""

import time

import cv2

from gesture_recognizer import GestureRecognizer

# Standard 21-point MediaPipe hand landmark skeleton connections, for
# drawing only - not used by recognition itself.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)

LANDMARK_COLOR = (0, 255, 0)  # green (BGR)
CONNECTION_COLOR = (255, 255, 0)  # cyan (BGR)
LABEL_COLOR = (0, 165, 255)  # orange (BGR)


def draw_hand(frame, hand, frame_width, frame_height):
    points = [(int(x * frame_width), int(y * frame_height)) for x, y in hand.landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], CONNECTION_COLOR, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, LANDMARK_COLOR, -1)

    x1 = min(x for x, _ in points)
    y1 = min(y for _, y in points)
    label = f"{hand.handedness} | {hand.gesture_name} | {hand.confidence:.2f} | {hand.source}"
    cv2.putText(
        frame, label, (x1, max(y1 - 10, 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_COLOR, 2,
    )


def draw_fps(frame, fps):
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )


def main():
    recognizer = GestureRecognizer()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    prev_time = time.time()
    start_ms = time.time() * 1000

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: could not read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror view, easier to test gestures against
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(time.time() * 1000 - start_ms)

        hands = recognizer.recognize(frame_rgb, timestamp_ms)

        frame_height, frame_width = frame.shape[:2]
        for hand in hands:
            draw_hand(frame, hand, frame_width, frame_height)

        current_time = time.time()
        prev_time = current_time

        cv2.imshow("Gesture Recognition Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()


if __name__ == "__main__":
    main()

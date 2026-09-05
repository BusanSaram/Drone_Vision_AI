"""Minimal webcam test: opens the default camera and shows the live feed."""

import cv2


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: could not read frame from webcam.")
            break

        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

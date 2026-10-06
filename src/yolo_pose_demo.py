"""Webcam experiment: YOLOv8n-Pose -> BoT-SORT -> TrackValidator, no MediaPipe.

Pipeline:
    Webcam -> YoloPoseDetector (YOLOv8n-Pose, CUDA if available)
                 -> PoseDetections (bboxes + 17 COCO keypoints per person)
           -> PersonTracker (BoT-SORT, same config as yolo_person_detection.py)
           -> TrackValidator (CANDIDATE / CONFIRMED)
           -> match_tracks_to_detections() (track_id -> this frame's keypoints)
           -> classify_pose() per CONFIRMED track -> raw pose
           -> TrackPoseStabilizers (0.7 s hold + short NONE grace, one per track ID) -> command
           -> TargetSelector (SEARCHING / FOLLOWING / LAND_REQUESTED, selected track ID)
           -> OpenCV visualization

Tracker and validator settings are identical to the detection-only baseline
(`yolo_person_detection.py`) so the two can be compared directly. Only the
detector model differs.

Display: CONFIRMED tracks get bbox, "ID n | CONFIRMED | conf x.xx", their
skeleton, "Raw Pose: ..." and "Command: ..." (with hold progress while a pose
is being held, and "[NONE grace]" while a NONE dropout is tolerated);
CANDIDATE tracks get bbox and label only. A CONFIRMED track with no
keypoints this frame counts as raw pose NONE. The selected target is drawn
in magenta with a "TARGET" label. Top-left shows smoothed FPS, the number of
YOLO Pose detections, the number of CONFIRMED tracks and the selection
state / target ID. Selection events are printed to the console.
Per-stage rolling timings are printed to the console every
PRINT_INTERVAL seconds; run-average timings and FPS are printed on exit.

TargetSelector only decides who the target is: nothing follows anyone or
talks to a drone. A LAND request is a state/console message only.

Keys: q = quit. r = DEV/DEMO ONLY - simulate a new autonomous session
(selector -> SEARCHING with no target, all pose hold history cleared;
tracker and validator keep running). Stand-in for the real MANUAL -> AUTO
flight-mode transition, which will call start_new_session() instead.
"""

import time

import cv2
import numpy as np

from coco_keypoints import KEYPOINT_CONF_THRESHOLD, reliable_mask, skeleton_segments
from coco_pose_recognizer import POSE_TO_COMMAND, BodyPose, classify_pose
from keypoint_track_association import match_tracks_to_detections
from person_detector import get_device
from person_tracker import PersonTracker
from pose_command_stabilizer import TrackPoseStabilizers
from rolling_stats import FpsMeter, StageTimer
from target_selector import TargetSelector
from track_validator import CONFIRMATION_TIME, LOST_GRACE_TIME, TrackState, TrackValidator
from yolo_person_detection import NEW_TRACK_THRESH, WARMUP_FRAMES
from yolo_pose_detector import YoloPoseDetector

ROLLING_WINDOW = 30  # frames
PRINT_INTERVAL = 2.0  # seconds between console timing lines

STAGES = (
    "pose_inference", "tracking", "validation", "association", "pose_commands", "selection", "drawing", "total",
)
LABEL_LINE_HEIGHT = 22  # px between stacked per-track label lines

CONFIRMED_COLOR = (255, 255, 0)  # cyan (BGR)
CANDIDATE_COLOR = (0, 165, 255)  # orange (BGR)
TARGET_COLOR = (255, 0, 255)  # magenta (BGR)
KEYPOINT_COLOR = (0, 255, 0)  # green (BGR)
STATUS_COLOR = (0, 0, 255)  # red (BGR)

WINDOW_NAME = "YOLOv8n-Pose Tracking"
INITIAL_WINDOW_SIZE = (1280, 720)  # display only; camera and inference resolution are unchanged


def draw_skeleton(frame, keypoints, color, min_conf=KEYPOINT_CONF_THRESHOLD):
    """Draw reliable COCO connections and keypoints for one person."""
    points = [(int(x), int(y)) for x, y, _conf in keypoints]
    for a, b in skeleton_segments(keypoints, min_conf):
        cv2.line(frame, points[a], points[b], color, 2)
    for point, reliable in zip(points, reliable_mask(keypoints, min_conf)):
        if reliable:
            cv2.circle(frame, point, 4, KEYPOINT_COLOR, -1)


def pose_label_lines(state, hold_time) -> list[str]:
    """'Raw Pose' / 'Command' lines for one track's StabilizerState."""
    command = POSE_TO_COMMAND[state.active].value
    if state.active is BodyPose.NONE and state.candidate is not BodyPose.NONE:
        command += f" (hold {min(state.held_for, hold_time):.2f}/{hold_time:.2f}s)"
    if state.in_grace:
        command += " [NONE grace]"
    return [f"Raw Pose: {state.raw_pose.value}", f"Command: {command}"]


def draw_track(frame, validated, extra_lines=(), is_target=False):
    """Bbox plus stacked label lines above it: ID/state/conf first, then `extra_lines`."""
    person = validated.tracked_person
    x1, y1, x2, y2 = person.bbox
    if is_target:
        color = TARGET_COLOR
    else:
        color = CONFIRMED_COLOR if validated.state is TrackState.CONFIRMED else CANDIDATE_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_target else 2)
    header = f"ID {person.track_id} | {validated.state.value} | conf {person.confidence:.2f}"
    if is_target:
        header = "TARGET | " + header
    lines = [header, *extra_lines]
    top = max(y1 - 10 - LABEL_LINE_HEIGHT * (len(lines) - 1), 15)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x1, top + LABEL_LINE_HEIGHT * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_status(frame, fps, num_detections, num_confirmed, selection):
    fps_text = "FPS: -" if fps is None else f"FPS: {fps:.1f}"
    target_text = "-" if selection.target_id is None else str(selection.target_id)
    lines = [
        fps_text,
        f"Pose detections: {num_detections}",
        f"Confirmed tracks: {num_confirmed}",
        f"State: {selection.state.value} | Target: {target_text}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, STATUS_COLOR, 2)


def start_new_session(selector, stabilizers):
    """Begin a fresh autonomous session: SEARCHING, no target, no pose hold history.

    Holds started before the session must not complete into commands in it,
    so the stabilizers are cleared together with the selector. The tracker
    and TrackValidator are deliberately left running (track IDs stay valid).
    """
    selector.reset()
    stabilizers.reset()


def fit_to_window(frame, window_size):
    """Scale `frame` to fit `window_size` (w, h) without distortion, padding with black.

    Display only: the window may have any shape, so the frame is scaled by the
    smaller of the two ratios and centered (letterboxed) instead of stretched.
    """
    win_w, win_h = window_size
    if win_w <= 0 or win_h <= 0:  # minimized / window not available
        return frame
    h, w = frame.shape[:2]
    scale = min(win_w / w, win_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = np.zeros((win_h, win_w, 3), dtype=frame.dtype)
    x, y = (win_w - new_w) // 2, (win_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return canvas


def format_timings(timings_ms) -> str:
    return " | ".join(f"{name} {timings_ms[name]:.1f}ms" for name in STAGES if name in timings_ms)


def main():
    device = get_device()
    print(f"Using device: {device}")

    detector = YoloPoseDetector(device=device)
    tracker = PersonTracker(new_track_thresh=NEW_TRACK_THRESH)
    validator = TrackValidator(confirmation_time=CONFIRMATION_TIME, lost_grace_time=LOST_GRACE_TIME)
    stabilizers = TrackPoseStabilizers()
    selector = TargetSelector()
    timer = StageTimer(window=ROLLING_WINDOW)
    fps_meter = FpsMeter(window=ROLLING_WINDOW)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    # Resizable display window; the user can resize/maximize it freely.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *INITIAL_WINDOW_SIZE)

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

            # Per-track raw pose + hold. Only CONFIRMED tracks are eligible; a
            # confirmed track without keypoints this frame counts as NONE.
            with timer.measure("pose_commands"):
                pose_now = time.monotonic()
                pose_states = {}
                activations = {}  # track_id -> pose whose hold completed this frame
                for validated in validated_tracks:
                    if validated.state is not TrackState.CONFIRMED:
                        continue
                    track_id = validated.tracked_person.track_id
                    index = track_to_detection.get(track_id)
                    raw_pose = BodyPose.NONE if index is None else classify_pose(detections.keypoints[index])
                    state = stabilizers.update(track_id, raw_pose, pose_now)
                    pose_states[track_id] = state
                    if state.newly_activated:
                        activations[track_id] = state.active
                        print(f"Track {track_id} command: {POSE_TO_COMMAND[state.active].value} "
                              f"({state.active.value})", flush=True)
                stabilizers.prune(pose_states)

            # Target selection: only the selected track's commands count while FOLLOWING;
            # the target is lost only once the validator expires its track.
            with timer.measure("selection"):
                target_id = selector.target_id
                target_alive = target_id is not None and validator.is_tracked(target_id)
                selection = selector.update(activations, target_alive)
                if selection.event is not None:
                    ids = ", ".join(str(i) for i in selection.event_track_ids)
                    print(f"Selection: {selection.event.value} (track {ids}) -> {selection.state.value}",
                          flush=True)

            with timer.measure("drawing"):
                for validated in validated_tracks:
                    track_id = validated.tracked_person.track_id
                    is_target = track_id == selection.target_id
                    state = pose_states.get(track_id)
                    if state is None:  # CANDIDATE
                        draw_track(frame, validated, is_target=is_target)
                        continue
                    draw_track(frame, validated, pose_label_lines(state, stabilizers.hold_time), is_target)
                    index = track_to_detection.get(track_id)
                    if index is not None:
                        draw_skeleton(frame, detections.keypoints[index], TARGET_COLOR if is_target else CONFIRMED_COLOR)
                draw_status(frame, fps, len(detections), len(pose_states), selection)

        if now - last_print >= PRINT_INTERVAL:
            fps_text = "-" if fps is None else f"{fps:.1f}"
            print(f"[rolling {ROLLING_WINDOW}f] FPS {fps_text} | {format_timings(timer.rolling_ms())}", flush=True)
            last_print = now

        _x, _y, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
        cv2.imshow(WINDOW_NAME, fit_to_window(frame, (win_w, win_h)))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            # DEV/DEMO ONLY: replace with the real MANUAL -> AUTO mode transition (RC/MAVLink).
            start_new_session(selector, stabilizers)
            print("[DEV] Simulated new autonomous session (r key): SEARCHING, target cleared, "
                  "pose hold history reset; tracker/validator unchanged", flush=True)

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

"""Minimal headless test: YOLOv8n-Pose running ON the IMX500 via Sony modlib 1.5.0.

Pipeline (all inference on the camera's AI accelerator, nothing on the Pi CPU
except tensor parsing and this logging):
    packerOut.zip (Ultralytics `format=imx` export)
        -> modlib AiCamera.deploy()  (imx500-package -> pack/network.rpk -> IMX500 firmware)
        -> output tensors -> pp_yolo_pose_ultralytics -> Poses
        -> console log: detection counts, boxes, keypoint scores, FPS / DPS

Expected output tensors for our export (from dnnParams.xml):
    0: boxes      (300, 4)   x1 y1 x2 y2 in input-tensor pixels (0..640)
    1: scores     (300,)
    2: labels     (300,)     always 0 (person)
    3: keypoints  (300, 51)  17 x (x, y, score)
The first tensor set is checked against this and the result printed.

This is a test only: it does not track, select targets, stream video or touch
the HigherHRNet scripts. SCORE_THRESHOLD only decides what is COUNTED in the
log - nothing is filtered or changed in the model output.

Run on the Pi (inside /home/pi/yolo_pose_imx):
    .venv/bin/python yolo_pose_modlib_test.py --duration 30
"""

import argparse
import os
import time
from typing import List

import numpy as np

from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_yolo_pose_ultralytics
from modlib.models.results import Poses

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "packerOut.zip")

INPUT_SIZE = 640
NUM_KEYPOINTS = 17
EXPECTED_SHAPES = [(300, 4), (300,), (300,), (300, NUM_KEYPOINTS * 3)]

# Log-only thresholds (counting, not filtering).
SCORE_THRESHOLD = 0.5
KEYPOINT_THRESHOLD = 0.5  # same value as KEYPOINT_CONF_THRESHOLD in src/coco_keypoints.py


class YoloPoseImx(Model):
    """Our Ultralytics YOLOv8n-Pose export, loaded as a CONVERTED (packerOut.zip) model."""

    def __init__(self, model_file: str):
        super().__init__(
            model_file=model_file,
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.first_tensor_report = None  # filled once, from the camera's processing thread

    # The return annotation is required: modlib infers the task type from it.
    def post_process(self, output_tensors: List[np.ndarray]) -> Poses:
        if self.first_tensor_report is None:
            self.first_tensor_report = check_tensor_format(output_tensors)
        return pp_yolo_pose_ultralytics(output_tensors, input_tensor_sz=INPUT_SIZE, num_keypoints=NUM_KEYPOINTS)


def check_tensor_format(output_tensors) -> List[str]:
    """Compare the raw tensors against our export's expected layout. Called before
    pp_yolo_pose_ultralytics, which divides the keypoint tensor in place."""
    shapes = [tuple(t.shape) for t in output_tensors]
    lines = [f"tensor count={len(output_tensors)} shapes={shapes} expected={EXPECTED_SHAPES}"]
    if shapes != EXPECTED_SHAPES:
        lines.append("MISMATCH: tensor shapes/order differ from the export - results below are not trustworthy")
        return lines

    boxes, scores, labels, kpts = output_tensors
    valid = scores >= SCORE_THRESHOLD
    lines.append(f"scores min/max={scores.min():.3f}/{scores.max():.3f}, above {SCORE_THRESHOLD}: {int(valid.sum())}")
    if valid.any():
        b = boxes[valid]
        ordered = np.mean((b[:, 2] >= b[:, 0]) & (b[:, 3] >= b[:, 1]))
        in_range = np.mean((b >= -1) & (b <= INPUT_SIZE + 1))
        k = kpts[valid].reshape(-1, NUM_KEYPOINTS, 3)
        lines.append(f"boxes x2>=x1 & y2>=y1: {ordered:.0%}, within 0..{INPUT_SIZE}: {in_range:.0%}")
        lines.append(f"labels of counted detections: {sorted(set(labels[valid].astype(int).tolist()))} (expect [0])")
        lines.append(
            f"keypoint xy range: {k[:, :, :2].min():.1f}..{k[:, :, :2].max():.1f} (expect ~0..{INPUT_SIZE}), "
            f"keypoint score range: {k[:, :, 2].min():.3f}..{k[:, :, 2].max():.3f} (expect 0..1)"
        )
    else:
        lines.append("no detection above threshold in the first tensor set - stand in view to check box/keypoint layout")
    return lines


def detection_line(poses: Poses) -> str:
    counted = np.flatnonzero(poses.confidence >= SCORE_THRESHOLD)
    if len(counted) == 0:
        return f"people={0}"
    top = counted[np.argmax(poses.confidence[counted])]
    x1, y1, x2, y2 = poses.bbox[top] * INPUT_SIZE  # normalized -> input-tensor pixels
    kp = poses.keypoint_scores[top]
    good = int(np.count_nonzero(kp >= KEYPOINT_THRESHOLD))
    return (
        f"people={len(counted)} top: score={poses.confidence[top]:.3f} "
        f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
        f"kp>={KEYPOINT_THRESHOLD}: {good}/{NUM_KEYPOINTS} "
        f"kp_score min/med/max={kp.min():.2f}/{np.median(kp):.2f}/{kp.max():.2f}"
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to packerOut.zip")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to run after the camera starts")
    parser.add_argument("--frame-rate", type=int, default=16, help="Camera frame rate (Ultralytics docs example: 16)")
    parser.add_argument("--log-interval", type=float, default=1.0, help="Seconds between console lines")
    return parser.parse_args()


def main():
    args = get_args()
    model = YoloPoseImx(args.model)

    # headless=True: no frame.image is produced (SSH, no display).
    # The duration is timed here instead of AiCamera(timeout=...): that timer starts
    # at construction (so it would include the firmware upload) and stops the device
    # from inside the iterator, before the `with` block stops it again.
    device = AiCamera(headless=True, frame_rate=args.frame_rate)

    # overwrite=False: package packerOut.zip -> pack/network.rpk on the first run, reuse it
    # afterwards, and never block on modlib's interactive "overwrite?" prompt.
    device.deploy(model, overwrite=False)

    new_frames = 0
    reused_frames = 0
    count_hist: dict[int, int] = {}
    last_frame = None

    with device as stream:  # __exit__ always calls device.stop(), also on Ctrl+C / errors
        start = time.monotonic()
        next_log = start
        print(f"[TEST] camera started, running {args.duration:.0f}s", flush=True)
        try:
            for frame in stream:
                now = time.monotonic()
                if now - start >= args.duration:
                    break
                last_frame = frame
                if not frame.new_detection:  # modlib repeats the last result when no new tensor arrived
                    reused_frames += 1
                    continue
                new_frames += 1
                poses = frame.detections
                n = int(np.count_nonzero(poses.confidence >= SCORE_THRESHOLD))
                count_hist[n] = count_hist.get(n, 0) + 1

                if model.first_tensor_report is not None and new_frames == 1:
                    for line in model.first_tensor_report:
                        print(f"[TENSOR] {line}", flush=True)
                if now >= next_log:
                    next_log = now + args.log_interval
                    print(
                        f"[{now - start:5.1f}s] FPS={frame.fps:5.1f} DPS={frame.dps:5.1f} {detection_line(poses)}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("[TEST] interrupted", flush=True)

    print("[TEST] --- Summary ---")
    print(f"[TEST] new-result frames: {new_frames}, frames reusing the previous result: {reused_frames}")
    hist = ", ".join(f"{n} people: {c}" for n, c in sorted(count_hist.items())) or "-"
    print(f"[TEST] new-result frames by people count (score >= {SCORE_THRESHOLD}): {hist}")
    if last_frame is not None:
        print(f"[TEST] last reported FPS={last_frame.fps:.1f} DPS={last_frame.dps:.1f}")


if __name__ == "__main__":
    main()

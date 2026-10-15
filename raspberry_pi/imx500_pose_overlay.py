"""Draw IMX500 YOLOv8n-Pose results onto the camera frame.

Pure numpy + OpenCV (no modlib, no camera), so it can be unit-tested on Windows.

Coordinate system:
    modlib's `pp_yolo_pose_ultralytics` returns boxes and keypoints normalized to
    0..1 over the model input tensor. With `preserve_aspect_ratio=False` the input
    tensor covers the full field of view of `frame.image` (ROI 0,0,1,1; the caller
    applies `Poses.compensate_for_roi` for any other ROI), so pixel = normalized *
    image width/height - x horizontally, y vertically, origin top-left, no flip or
    rotation. Boxes and keypoints are clipped to the image.

Thresholds come from the existing pipeline: keypoints/skeleton use
`KEYPOINT_CONF_THRESHOLD` via `reliable_mask` / `skeleton_segments`
(src/coco_keypoints.py); people are drawn from PERSON_SCORE_THRESHOLD, the same
value the IMX500 test script counts with. Display only - nothing is tuned.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from coco_keypoints import NUM_KEYPOINTS, reliable_mask, skeleton_segments

PERSON_SCORE_THRESHOLD = 0.5

# Result age above which the overlay shows a STALE warning. At 16 FPS / 8 DPS a
# healthy stream alternates between a fresh result (age 0) and one repeated
# result (age ~0.06 s), so only a real stall of the inference output - several
# missed results in a row - crosses this.
STALE_AFTER_S = 0.5

# BGR
BOX_COLOR = (0, 255, 0)
SKELETON_COLOR = (0, 255, 255)
KEYPOINT_COLOR = (0, 0, 255)
STATUS_COLOR = (255, 255, 255)
STALE_COLOR = (0, 0, 255)


@dataclass(frozen=True, eq=False)
class PixelPoses:
    """People above PERSON_SCORE_THRESHOLD, in frame pixels."""

    boxes: np.ndarray  # (N, 4) float32 x1 y1 x2 y2, clipped to the image
    keypoints: np.ndarray  # (N, 17, 3) float32 (x, y, score), x/y clipped - same layout as coco_keypoints
    confidence: np.ndarray  # (N,) float32

    def __len__(self) -> int:
        return len(self.confidence)


def to_pixel_poses(bbox, keypoints, keypoint_scores, confidence, width: int, height: int,
                   min_score: float = PERSON_SCORE_THRESHOLD) -> PixelPoses:
    """Normalized modlib Poses arrays -> clipped pixel coordinates for an image of width x height.

    bbox (N, 4) x1 y1 x2 y2, keypoints (N, 17, 2) x y, keypoint_scores (N, 17), confidence (N,).
    """
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    keep = confidence >= min_score
    n = int(keep.sum())

    boxes = np.asarray(bbox, dtype=np.float32).reshape(-1, 4)[keep] * [width, height, width, height]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)

    xy = np.asarray(keypoints, dtype=np.float32).reshape(-1, NUM_KEYPOINTS, 2)[keep] * [width, height]
    xy[:, :, 0] = np.clip(xy[:, :, 0], 0, width - 1)
    xy[:, :, 1] = np.clip(xy[:, :, 1], 0, height - 1)
    scores = np.asarray(keypoint_scores, dtype=np.float32).reshape(-1, NUM_KEYPOINTS)[keep]

    return PixelPoses(
        boxes=boxes.reshape(n, 4).astype(np.float32),
        keypoints=np.concatenate([xy, scores[:, :, None]], axis=2).reshape(n, NUM_KEYPOINTS, 3).astype(np.float32),
        confidence=confidence[keep],
    )


def draw_poses(image: np.ndarray, poses: PixelPoses) -> np.ndarray:
    """Draw box, score, reliable keypoints and skeleton for each person. Modifies `image` in place."""
    for box, kpts, conf in zip(poses.boxes, poses.keypoints, poses.confidence):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(image, f"person {conf:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)

        for a, b in skeleton_segments(kpts):
            pa = (int(round(kpts[a, 0])), int(round(kpts[a, 1])))
            pb = (int(round(kpts[b, 0])), int(round(kpts[b, 1])))
            cv2.line(image, pa, pb, SKELETON_COLOR, 2, cv2.LINE_AA)
        for x, y, _ in kpts[reliable_mask(kpts)]:
            cv2.circle(image, (int(round(x)), int(round(y))), 3, KEYPOINT_COLOR, -1, cv2.LINE_AA)
    return image


def is_stale(result_age_s: float, threshold_s: float = STALE_AFTER_S) -> bool:
    """True when the newest inference result is MORE than `threshold_s` seconds old."""
    return result_age_s > threshold_s


def status_text(fps: float, dps: float, people: int) -> str:
    return f"FPS {fps:4.1f} | DPS {dps:4.1f} | people {people}"


def draw_status(image: np.ndarray, fps: float, dps: float, people: int, result_age_s: float) -> np.ndarray:
    """Status band at the top: FPS / DPS / people, plus a STALE warning only while the
    newest inference result is older than STALE_AFTER_S. Modifies `image` in place."""
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 24), (0, 0, 0), -1)
    cv2.putText(image, status_text(fps, dps, people), (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                STATUS_COLOR, 1, cv2.LINE_AA)
    if is_stale(result_age_s):
        cv2.putText(image, f"STALE {result_age_s:.1f}s", (image.shape[1] - 130, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, STALE_COLOR, 2, cv2.LINE_AA)
    return image

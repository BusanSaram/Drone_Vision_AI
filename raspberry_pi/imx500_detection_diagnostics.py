"""Read-only diagnostics for IMX500 HigherHRNet person detections.

Used by `higherhrnet_pose_test.py --diag` to investigate:
    - why one person sometimes produces more than one detection, and
    - what range HigherHRNet person / keypoint scores actually fall in
      (they are not on the same scale as YOLO confidences).

Pure numpy, no picamera2 import, so it can be unit-tested on Windows. Nothing
here filters, modifies or reorders detections - it only formats log lines.

Boxes are xyxy in frame pixels (the output of `convert_imx500_output`);
keypoints are (N, 17, 3) with (x, y, score).
"""

import numpy as np

# Default keypoint score used only to COUNT "valid" keypoints in the log.
# Matches KEYPOINT_CONF_THRESHOLD in src/coco_keypoints.py so the counts show
# what the Windows pose recognizer would accept. Diagnostic only - it does
# not filter anything.
DEFAULT_KEYPOINT_THRESHOLD = 0.5


def box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def intersection_area(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    w = min(ax2, bx2) - max(ax1, bx1)
    h = min(ay2, by2) - max(ay1, by1)
    return max(0.0, float(w)) * max(0.0, float(h))


def box_iou(box_a, box_b) -> float:
    inter = intersection_area(box_a, box_b)
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner, outer) -> float:
    """Fraction of `inner`'s area that lies inside `outer` (1.0 = fully inside)."""
    area = box_area(inner)
    return intersection_area(inner, outer) / area if area > 0 else 0.0


def valid_keypoint_indices(keypoints, threshold: float) -> list[int]:
    """COCO indices (0-16) whose score >= threshold and whose x/y are finite."""
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
    ok = np.isfinite(keypoints).all(axis=1) & (keypoints[:, 2] >= threshold)
    return [int(i) for i in np.flatnonzero(ok)]


def detection_lines(xyxy, conf, keypoints, threshold: float = DEFAULT_KEYPOINT_THRESHOLD) -> list[str]:
    """One line per detection: score, box, valid keypoint count/indices, keypoint score range."""
    lines = []
    for i in range(len(conf)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        kp_scores = np.asarray(keypoints[i], dtype=np.float32)[:, 2]
        valid = valid_keypoint_indices(keypoints[i], threshold)
        lines.append(
            f"  det {i}: score={float(conf[i]):.3f} "
            f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) {x2 - x1:.0f}x{y2 - y1:.0f} "
            f"valid_kp={len(valid)}/17 idx={valid} "
            f"kp_score min/med/max={kp_scores.min():.3f}/{np.median(kp_scores):.3f}/{kp_scores.max():.3f}"
        )
    return lines


def pair_lines(xyxy, keypoints, threshold: float = DEFAULT_KEYPOINT_THRESHOLD) -> list[str]:
    """One line per detection pair: IoU, containment both ways, shared valid keypoint types.

    A small box mostly contained in a larger one, with few shared keypoint
    types, points to one person split into partial skeletons by HigherHRNet's
    keypoint grouping rather than two real people.
    """
    lines = []
    n = len(xyxy)
    for i in range(n):
        for j in range(i + 1, n):
            shared = sorted(
                set(valid_keypoint_indices(keypoints[i], threshold))
                & set(valid_keypoint_indices(keypoints[j], threshold))
            )
            lines.append(
                f"  pair {i}-{j}: iou={box_iou(xyxy[i], xyxy[j]):.3f} "
                f"{i}_in_{j}={containment(xyxy[i], xyxy[j]):.3f} "
                f"{j}_in_{i}={containment(xyxy[j], xyxy[i]):.3f} "
                f"shared_valid_kp={len(shared)} idx={shared}"
            )
    return lines


class DiagnosticCounters:
    """Run totals for the end-of-run summary."""

    def __init__(self):
        self.callbacks = 0
        self.no_output = 0  # get_outputs() returned None: script reuses cached detections
        self.by_count: dict[int, int] = {}  # fresh people-count -> number of frames
        self.min_score: float | None = None
        self.max_score: float | None = None

    def record_no_output(self) -> None:
        self.callbacks += 1
        self.no_output += 1

    def record_fresh(self, conf) -> None:
        self.callbacks += 1
        n = len(conf)
        self.by_count[n] = self.by_count.get(n, 0) + 1
        if n:
            lo, hi = float(np.min(conf)), float(np.max(conf))
            self.min_score = lo if self.min_score is None else min(self.min_score, lo)
            self.max_score = hi if self.max_score is None else max(self.max_score, hi)

    def summary_lines(self) -> list[str]:
        counts = ", ".join(f"{n} people: {frames}" for n, frames in sorted(self.by_count.items())) or "-"
        score_range = (
            "-" if self.min_score is None else f"{self.min_score:.3f} .. {self.max_score:.3f}"
        )
        return [
            "[DIAG] --- Summary ---",
            f"[DIAG] callbacks: {self.callbacks}, no tensor output (cached reused): {self.no_output}",
            f"[DIAG] fresh frames by detection count: {counts}",
            f"[DIAG] person score range over fresh detections: {score_range}",
        ]

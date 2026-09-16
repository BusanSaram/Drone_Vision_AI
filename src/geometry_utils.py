"""Shared pure bbox-geometry helpers used by the diagnostic and validation layers.

All functions operate on `(x1, y1, x2, y2, width, height)` tuples and have no
side effects - they never read or write any detector/tracker/validator state.
"""

import math


def iou(box_a, box_b) -> float:
    """Standard bounding-box IoU: intersection_area / (area_a + area_b - intersection_area)."""
    ax1, ay1, ax2, ay2, aw, ah = box_a
    bx1, by1, bx2, by2, bw, bh = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def edge_gaps(box_a, box_b) -> tuple[float, float]:
    """Signed horizontal/vertical edge gap; negative = overlapping by that many px."""
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    return max(ax1, bx1) - min(ax2, bx2), max(ay1, by1) - min(ay2, by2)


def center(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box[:4]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(box_a, box_b) -> float:
    ax, ay = center(box_a)
    bx, by = center(box_b)
    return math.hypot(ax - bx, ay - by)


def as_geom_box(bbox) -> tuple[float, float, float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1), float(y1), float(x2), float(y2), float(x2 - x1), float(y2 - y1))

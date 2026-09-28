"""Links each BoT-SORT track to the current-frame pose detection it came from.

Pipeline position:
    PoseDetections ──> PersonTracker (BoT-SORT) ──> tracked people (track_id, bbox, detection_index)
          └──────────────────────────┬─────────────────────┘
                         match_tracks_to_detections() -> {track_id: detection row}

BoT-SORT already knows which detection updated each track this frame: its
output carries that detection's row index (`TrackedPerson.detection_index`).
Ultralytics' own `model.track()` uses the same index to carry keypoints over
to tracks. So no geometric guessing is needed - but the index is still
checked conservatively before it is trusted:

    1. The index must be present and within this frame's detections.
    2. The track's bbox (BoT-SORT's Kalman-filtered box) must overlap the
       indexed detection's bbox with IoU >= MIN_BOX_IOU. A track that was
       just updated by that detection should overlap it heavily; a low IoU
       means something is inconsistent, so the keypoints are withheld.
    3. A detection claimed by more than one track is given to none of them.

A track that fails any check gets no keypoints this frame. Nothing ever falls
back to a "nearest" detection.
"""

from geometry_utils import as_geom_box, iou

# Experimental starting value, not tuned.
MIN_BOX_IOU = 0.5


def match_tracks_to_detections(tracked_people, detections, min_iou: float = MIN_BOX_IOU) -> dict[int, int]:
    """Map track_id -> detection row index for confident, unambiguous matches only.

    Args:
        tracked_people: Objects with `track_id`, pixel `bbox` (x1, y1, x2, y2)
            and `detection_index` (e.g. `TrackedPerson`).
        detections: This frame's `PoseDetections` (anything with `len()` and
            an `xyxy` array works).
        min_iou: Minimum IoU between the track bbox and the indexed detection bbox.

    Returns:
        `{track_id: detection_index}`. Tracks missing from the result have no
        keypoints this frame.
    """
    claims: dict[int, list[int]] = {}
    for person in tracked_people:
        index = person.detection_index
        if index is None or not 0 <= index < len(detections):
            continue
        if iou(as_geom_box(person.bbox), as_geom_box(detections.xyxy[index])) < min_iou:
            continue
        claims.setdefault(index, []).append(person.track_id)

    return {track_ids[0]: index for index, track_ids in claims.items() if len(track_ids) == 1}

"""Associates MediaPipe Pose results with tracked people (BoT-SORT track IDs).

Pipeline position:
    PersonTracker/TrackValidator -> tracked people (track_id, bbox)  ┐
    MediaPipe PoseLandmarker     -> per-person pose landmarks        ┴-> associate_poses() -> {track_id: pose_index}

Independent of how detection is performed: it only needs objects with
`track_id` and `bbox` (x1, y1, x2, y2 in pixels), and pose landmarks with
normalized `.x`, `.y` and `.visibility`. It works the same whether the boxes
came from local YOLO or, later, from IMX500 on-camera detections.

Method (per pose):
    1. Keep only reliable body landmarks (visibility >= MIN_VISIBILITY; face
       points excluded). The pose is unusable without both shoulders or with
       fewer than MIN_RELIABLE_LANDMARKS points.
    2. Anchor = torso center (mean of reliable shoulders/hips).
       Pose box = bounding box of the reliable landmarks.
    3. A track is a candidate only if the anchor lies inside its bbox AND at
       least MIN_INSIDE_FRACTION of the reliable landmarks lie inside its bbox
       expanded by BBOX_MARGIN_RATIO.
    4. Candidates are ranked by IoU(pose box, track bbox). If the best IoU is
       below MIN_IOU, or the top two are within AMBIGUITY_MARGIN, the pose is
       left unassociated.
Then, across poses: a track claimed as best by more than one pose goes to the
top pose only if it wins by AMBIGUITY_MARGIN, otherwise to none of them. A
pose that loses its best track is left unassociated rather than falling back
to a second choice.

All thresholds are experimental starting values, not tuned.
"""

from dataclasses import dataclass

# MediaPipe Pose landmark indices used for association (body only, no face points).
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

BODY_LANDMARKS = (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)
TORSO_LANDMARKS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

MIN_VISIBILITY = 0.5
MIN_RELIABLE_LANDMARKS = 4
BBOX_MARGIN_RATIO = 0.1  # of bbox width/height, added on each side for the inside-fraction check
MIN_INSIDE_FRACTION = 0.6
MIN_IOU = 0.2
AMBIGUITY_MARGIN = 0.1  # IoU difference required to prefer one match over another


@dataclass
class PoseRegion:
    """Pixel-space summary of one pose's reliable landmarks."""

    anchor: tuple[float, float]  # torso center
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 around reliable landmarks
    points: list[tuple[float, float]]


def pose_region(landmarks, frame_width: int, frame_height: int) -> PoseRegion | None:
    """Reliable-landmark summary of one pose, or None if the pose is unusable for association."""

    def reliable(i):
        return (landmarks[i].visibility or 0.0) >= MIN_VISIBILITY

    if not (reliable(LEFT_SHOULDER) and reliable(RIGHT_SHOULDER)):
        return None

    def px(i):
        return (landmarks[i].x * frame_width, landmarks[i].y * frame_height)

    points = [px(i) for i in BODY_LANDMARKS if reliable(i)]
    if len(points) < MIN_RELIABLE_LANDMARKS:
        return None

    torso = [px(i) for i in TORSO_LANDMARKS if reliable(i)]
    anchor = (sum(p[0] for p in torso) / len(torso), sum(p[1] for p in torso) / len(torso))
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return PoseRegion(anchor=anchor, box=(min(xs), min(ys), max(xs), max(ys)), points=points)


def _inside(point, box) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def _expanded(box, ratio: float):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * ratio, (y2 - y1) * ratio
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_score(region: PoseRegion, bbox) -> float | None:
    """IoU score of a pose against one track bbox, or None if the track is not a candidate."""
    if not _inside(region.anchor, bbox):
        return None
    wide = _expanded(bbox, BBOX_MARGIN_RATIO)
    inside_fraction = sum(_inside(p, wide) for p in region.points) / len(region.points)
    if inside_fraction < MIN_INSIDE_FRACTION:
        return None
    return _iou(region.box, bbox)


def _best_track(region: PoseRegion, tracked_people) -> tuple[int, float] | None:
    """Best (track_id, score) for one pose, or None if no confident, unambiguous match."""
    scored = []
    for person in tracked_people:
        score = match_score(region, person.bbox)
        if score is not None:
            scored.append((score, person.track_id))
    if not scored:
        return None
    scored.sort(reverse=True)
    best_score, best_id = scored[0]
    if best_score < MIN_IOU:
        return None
    if len(scored) > 1 and best_score - scored[1][0] < AMBIGUITY_MARGIN:
        return None
    return best_id, best_score


def associate_poses(poses, tracked_people, frame_width: int, frame_height: int) -> dict[int, int]:
    """Match poses to tracked people, at most one pose per track and one track per pose.

    Args:
        poses: List of per-person landmark lists (e.g. `PoseLandmarkerResult.pose_landmarks`).
        tracked_people: Objects with `track_id` and pixel `bbox` (e.g. `TrackedPerson`).
        frame_width, frame_height: Frame size in pixels.

    Returns:
        `{track_id: pose_index}` for confident matches only. Tracks and poses
        missing from the result are unassociated this frame.
    """
    claims: dict[int, list[tuple[float, int]]] = {}
    for pose_index, landmarks in enumerate(poses):
        region = pose_region(landmarks, frame_width, frame_height)
        if region is None:
            continue
        best = _best_track(region, tracked_people)
        if best is None:
            continue
        track_id, score = best
        claims.setdefault(track_id, []).append((score, pose_index))

    result = {}
    for track_id, claimants in claims.items():
        claimants.sort(reverse=True)
        if len(claimants) > 1 and claimants[0][0] - claimants[1][0] < AMBIGUITY_MARGIN:
            continue  # two poses fit this track about equally well - don't guess
        result[track_id] = claimants[0][1]
    return result

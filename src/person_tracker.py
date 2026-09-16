"""Assigns persistent IDs to per-frame person detections.

Wraps Ultralytics' built-in BOTSORT tracker (BoT-SORT with ReID disabled;
`with_reid: False` in the bundled config), used directly here instead of
`model.track()` so that detection and tracking stay as separate,
independently swappable stages. This module only depends on a
`Boxes`-like object (conf, xywh, xyxy, cls, boolean indexing) - not on how
the detections were produced - so it keeps working unchanged if the
detector backend changes later.
"""

from dataclasses import dataclass

from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

DEFAULT_TRACKER_CONFIG = "botsort.yaml"  # bundled with Ultralytics; with_reid is False by default


@dataclass
class TrackedPerson:
    track_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


class PersonTracker:
    """Wraps Ultralytics' BOTSORT tracker to produce persistent person IDs."""

    def __init__(self, tracker_config: str = DEFAULT_TRACKER_CONFIG, new_track_thresh: float | None = None):
        cfg_path = check_yaml(tracker_config)
        cfg = IterableSimpleNamespace(**YAML.load(cfg_path))
        if new_track_thresh is not None:
            cfg.new_track_thresh = new_track_thresh
        self._tracker = BOTSORT(args=cfg)

    def update(self, boxes, frame) -> list[TrackedPerson]:
        """Match this frame's detections against existing tracks.

        Args:
            boxes: `Boxes`-like detections for the current frame (see
                `PersonDetector.detect`).
            frame: The current BGR frame. BoT-SORT uses this for its
                global motion compensation (GMC) step, which the previous
                ByteTrack backend did not use.

        Returns:
            One `TrackedPerson` per confirmed track this frame.
        """
        tracks = self._tracker.update(boxes, frame)

        tracked_people = []
        for x1, y1, x2, y2, track_id, confidence, _cls, _det_idx in tracks:
            tracked_people.append(
                TrackedPerson(
                    track_id=int(track_id),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(confidence),
                )
            )
        return tracked_people

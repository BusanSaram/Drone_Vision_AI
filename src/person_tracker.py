"""Assigns persistent IDs to per-frame person detections.

Wraps Ultralytics' built-in BYTETracker (the same tracker used internally
by `model.track()`), used directly here instead so that detection and
tracking stay as separate, independently swappable stages. This module
only depends on a `Boxes`-like object (conf, xywh, xyxy, cls, boolean
indexing) - not on how the detections were produced - so it keeps working
unchanged if the detector backend changes later.
"""

from dataclasses import dataclass

from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

DEFAULT_TRACKER_CONFIG = "bytetrack.yaml"  # bundled with Ultralytics


@dataclass
class TrackedPerson:
    track_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


class PersonTracker:
    """Wraps Ultralytics' BYTETracker to produce persistent person IDs."""

    def __init__(self, tracker_config: str = DEFAULT_TRACKER_CONFIG, track_buffer: int | None = None):
        cfg_path = check_yaml(tracker_config)
        cfg = IterableSimpleNamespace(**YAML.load(cfg_path))
        if track_buffer is not None:
            cfg.track_buffer = track_buffer
        self._tracker = BYTETracker(args=cfg)

    def update(self, boxes, frame) -> list[TrackedPerson]:
        """Match this frame's detections against existing tracks.

        Args:
            boxes: `Boxes`-like detections for the current frame (see
                `PersonDetector.detect`).
            frame: The current BGR frame (used by some trackers for motion
                compensation; unused by plain ByteTrack).

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

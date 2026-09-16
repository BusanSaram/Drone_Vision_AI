"""Read-only diagnostic instrumentation for the detector -> tracker pipeline.

Purpose: determine whether track-ID fragmentation (e.g. during distance
changes) originates in YOLO detection becoming unreliable, or in the
tracker itself losing/reassigning IDs.

This module only *observes* per-frame detector/tracker output and prints
event-based diagnostics. It never modifies detections, tracks, or frames,
and none of its thresholds are passed into `PersonDetector` or
`PersonTracker` - they only decide when to print.
"""

from collections import deque
from dataclasses import dataclass, field

HISTORY_LEN = 20  # rolling window of recent frames kept for event dumps

# Diagnostic-only thresholds: used solely to decide when to print an event.
# They do not affect detection or tracking decisions in any way.
CONFIDENCE_DROP_THRESHOLD = 0.15
BBOX_SCALE_CHANGE_THRESHOLD = 0.15


@dataclass
class FrameRecord:
    frame_index: int
    yolo_confidences: list[float] = field(default_factory=list)
    yolo_bbox_scales: list[float] = field(default_factory=list)
    track_ids: list[int] = field(default_factory=list)


class TrackingDiagnostics:
    """Observes raw YOLO detections and tracked output, logs fragmentation-relevant events."""

    def __init__(
        self,
        history_len: int = HISTORY_LEN,
        confidence_drop_threshold: float = CONFIDENCE_DROP_THRESHOLD,
        bbox_scale_change_threshold: float = BBOX_SCALE_CHANGE_THRESHOLD,
    ):
        self._history: deque[FrameRecord] = deque(maxlen=history_len)
        self._confidence_drop_threshold = confidence_drop_threshold
        self._bbox_scale_change_threshold = bbox_scale_change_threshold

        self._prev_yolo_count = 0
        self._prev_track_ids: set[int] = set()
        self._prev_track_confidence: dict[int, float] = {}
        self._prev_track_bbox_scale: dict[int, float] = {}
        self._ever_seen_ids: set[int] = set()

    def observe(self, frame_index, boxes, tracked_people, frame_height: int) -> None:
        """Record one frame of detector + tracker output and log any notable events.

        Args:
            frame_index: Current frame counter from the main loop.
            boxes: Raw `Boxes`-like YOLO detections for this frame (see
                `PersonDetector.detect`) - read-only here.
            tracked_people: `list[TrackedPerson]` returned by `PersonTracker.update`
                for this frame - read-only here.
            frame_height: Height of the current frame, for normalizing bbox size.
        """
        yolo_count = len(boxes)
        yolo_confidences = [float(c) for c in boxes.conf] if yolo_count else []
        yolo_bbox_scales = (
            [float((y2 - y1) / frame_height) for _, y1, _, y2 in boxes.xyxy] if yolo_count else []
        )

        track_ids = [p.track_id for p in tracked_people]
        track_confidence = {p.track_id: p.confidence for p in tracked_people}
        track_bbox_scale = {
            p.track_id: (p.bbox[3] - p.bbox[1]) / frame_height for p in tracked_people
        }

        self._history.append(FrameRecord(frame_index, yolo_confidences, yolo_bbox_scales, track_ids))

        events = self._detect_events(
            yolo_count, track_ids, track_confidence, track_bbox_scale,
        )

        if events:
            self._print_history(events)

        self._prev_yolo_count = yolo_count
        self._prev_track_ids = set(track_ids)
        self._prev_track_confidence = track_confidence
        self._prev_track_bbox_scale = track_bbox_scale
        self._ever_seen_ids |= set(track_ids)

    def _detect_events(self, yolo_count, track_ids, track_confidence, track_bbox_scale) -> list[str]:
        events = []

        if yolo_count == 0 and self._prev_yolo_count > 0:
            events.append("YOLO person detection disappeared")
        elif yolo_count > 0 and self._prev_yolo_count == 0:
            events.append("YOLO person detection returned")

        current_ids = set(track_ids)
        appeared_ids = current_ids - self._prev_track_ids
        disappeared_ids = self._prev_track_ids - current_ids

        for track_id in sorted(disappeared_ids):
            events.append(f"track ID {track_id} disappeared")

        for track_id in sorted(appeared_ids):
            if track_id in self._ever_seen_ids:
                events.append(f"track ID {track_id} re-appeared (previously seen this session)")
            else:
                events.append(
                    f"NEW track ID {track_id} appeared (never seen before this session -> likely fragmentation)"
                )

        for track_id, confidence in track_confidence.items():
            prev_confidence = self._prev_track_confidence.get(track_id)
            if prev_confidence is not None and (prev_confidence - confidence) >= self._confidence_drop_threshold:
                events.append(
                    f"track {track_id} confidence dropped substantially: "
                    f"{prev_confidence:.2f} -> {confidence:.2f}"
                )

        for track_id, scale in track_bbox_scale.items():
            prev_scale = self._prev_track_bbox_scale.get(track_id)
            if prev_scale is not None and abs(prev_scale - scale) >= self._bbox_scale_change_threshold:
                events.append(
                    f"track {track_id} bbox scale changed substantially: {prev_scale:.2f} -> {scale:.2f}"
                )

        return events

    def _print_history(self, events: list[str]) -> None:
        print("\n--- Tracking diagnostic event ---")
        print("Trigger: " + "; ".join(events))
        for record in self._history:
            yolo_desc = "yes" if record.yolo_confidences else "no"
            conf_desc = ", ".join(f"{c:.2f}" for c in record.yolo_confidences) or "-"
            scale_desc = ", ".join(f"{s:.2f}" for s in record.yolo_bbox_scales) or "-"
            track_desc = ", ".join(str(t) for t in record.track_ids) or "none"
            print(
                f"Frame {record.frame_index} | "
                f"YOLO: {yolo_desc} (n={len(record.yolo_confidences)}, conf=[{conf_desc}], scale=[{scale_desc}]) | "
                f"Track IDs: {track_desc}"
            )
        print("--- end diagnostic event ---\n")

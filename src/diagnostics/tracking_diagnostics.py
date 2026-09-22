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

NEW_TRACK_CONTEXT_FRAMES = 10  # previous frames to dump alongside the triggering frame


@dataclass
class FrameRecord:
    frame_index: int
    yolo_confidences: list[float] = field(default_factory=list)
    yolo_bbox_scales: list[float] = field(default_factory=list)
    track_ids: list[int] = field(default_factory=list)
    # Raw per-detection geometry: (x1, y1, x2, y2, width, height). Kept so a
    # brand-new-track event can reprint full historical geometry on demand
    # without needing to have retained the original `Boxes` objects.
    yolo_boxes: list[tuple[float, float, float, float, float, float]] = field(default_factory=list)


class TrackingDiagnostics:
    """Observes raw YOLO detections and tracked output, logs fragmentation-relevant events."""

    def __init__(
        self,
        history_len: int = HISTORY_LEN,
        confidence_drop_threshold: float = CONFIDENCE_DROP_THRESHOLD,
        bbox_scale_change_threshold: float = BBOX_SCALE_CHANGE_THRESHOLD,
        new_track_thresh: float | None = None,
    ):
        self._history: deque[FrameRecord] = deque(maxlen=history_len)
        self._confidence_drop_threshold = confidence_drop_threshold
        self._bbox_scale_change_threshold = bbox_scale_change_threshold
        # For display only, in the "NEW TRACK CREATED" event - not used in any
        # calculation. Lets the log state which new_track_thresh was running
        # when a given track was first created.
        self._new_track_thresh_label = new_track_thresh

        self._prev_yolo_count = 0
        self._prev_track_ids: set[int] = set()
        self._prev_track_confidence: dict[int, float] = {}
        self._prev_track_bbox_scale: dict[int, float] = {}
        self._ever_seen_ids: set[int] = set()
        # Persistent, always-populated record of every ID's first-observed frame,
        # independent of terminal scrollback - queryable/printable at any time,
        # e.g. at program exit, so a NEW TRACK CREATED event is never unrecoverable
        # even if its live print scrolled out of view mid-session.
        self._first_seen_frame: dict[int, int] = {}

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
        yolo_boxes = []
        for (x1, y1, x2, y2) in (boxes.xyxy if yolo_count else []):
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            yolo_boxes.append((x1, y1, x2, y2, x2 - x1, y2 - y1))
        yolo_bbox_scales = [box[5] / frame_height for box in yolo_boxes]

        track_ids = [p.track_id for p in tracked_people]
        track_confidence = {p.track_id: p.confidence for p in tracked_people}
        track_bbox_scale = {
            p.track_id: (p.bbox[3] - p.bbox[1]) / frame_height for p in tracked_people
        }

        self._history.append(
            FrameRecord(frame_index, yolo_confidences, yolo_bbox_scales, track_ids, yolo_boxes)
        )

        events = self._detect_events(
            yolo_count, track_ids, track_confidence, track_bbox_scale,
        )

        if events:
            self._print_history(events)

        if yolo_count >= 2:
            self._print_multi_detection_geometry(frame_index, boxes, frame_height, track_ids)

        # First-ever appearance of each track ID this session: captured here
        # (before _ever_seen_ids is updated below) so we can log exactly what
        # raw detection(s) were on screen at the moment a brand new ID was created.
        brand_new_ids = set(track_ids) - self._ever_seen_ids
        if brand_new_ids:
            people_by_id = {p.track_id: p for p in tracked_people}
            for track_id in sorted(brand_new_ids):
                self._first_seen_frame[track_id] = frame_index
                self._print_new_track_created(
                    frame_index, people_by_id[track_id], boxes, frame_height, track_ids,
                )

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

    def _print_multi_detection_geometry(self, frame_index, boxes, frame_height, track_ids) -> None:
        """Dump raw per-detection bbox geometry and pairwise IoU when YOLO reports >= 2 person detections.

        Diagnostic-only: reads `boxes` (a `Boxes`-like object) without modifying it,
        and does not feed its output back into detection or tracking in any way.
        """
        print("\n--- Multiple YOLO person detections ---")
        print(f"Frame: {frame_index}")

        detections = []
        for i, (bbox, confidence) in enumerate(zip(boxes.xyxy, boxes.conf)):
            x1, y1, x2, y2 = (float(v) for v in bbox)
            width = x2 - x1
            height = y2 - y1
            bbox_height_ratio = height / frame_height
            detections.append((x1, y1, x2, y2, width, height))
            print(
                f"\nDetection {i}\n"
                f"confidence: {float(confidence):.2f}\n"
                f"bbox: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})\n"
                f"width: {width:.1f}\n"
                f"height: {height:.1f}\n"
                f"bbox_height_ratio: {bbox_height_ratio:.2f}"
            )

        print()
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                iou = self._iou(detections[i], detections[j])
                print(f"IoU(det{i}, det{j}): {iou:.2f}")

        track_desc = ", ".join(str(t) for t in track_ids) or "none"
        print(f"\nActive Track IDs: {track_desc}")
        print("----------------------------------------\n", flush=True)

    def _print_frame_geometry(self, record: FrameRecord) -> None:
        """Print one frame's raw YOLO detections plus pairwise IoU/edge-gap/axis-overlap.

        Diagnostic-only: reads a `FrameRecord` snapshot, computes nothing that
        feeds back into detection or tracking.
        """
        print(f"\nFrame {record.frame_index}")
        if not record.yolo_boxes:
            print("  No RAW YOLO person detections.")

        for i, (x1, y1, x2, y2, width, height) in enumerate(record.yolo_boxes):
            print(
                f"  Detection {i}: confidence={record.yolo_confidences[i]:.2f}, "
                f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), width={width:.1f}, height={height:.1f}"
            )

        for i in range(len(record.yolo_boxes)):
            for j in range(i + 1, len(record.yolo_boxes)):
                box_i, box_j = record.yolo_boxes[i], record.yolo_boxes[j]
                iou = self._iou(box_i, box_j)
                h_gap, v_gap = self._edge_gaps(box_i, box_j)
                print(
                    f"  IoU(det{i}, det{j})={iou:.2f} | "
                    f"horizontal edge gap={h_gap:.1f}px | vertical edge gap={v_gap:.1f}px "
                    f"(negative = overlapping by that many px, positive = separated by that many px) | "
                    f"overlap on X axis={h_gap < 0} | overlap on Y axis={v_gap < 0}"
                )

        track_desc = ", ".join(str(t) for t in record.track_ids) or "none"
        print(f"  Active Track IDs: {track_desc}")

    @staticmethod
    def _edge_gaps(box_a, box_b) -> tuple[float, float]:
        """Signed horizontal/vertical edge gap between two (x1,y1,x2,y2,w,h) boxes.

        Negative = the boxes overlap on that axis by that many pixels.
        Positive = the boxes are separated on that axis by that many pixels.
        Zero = the edges exactly touch. Diagnostic-only; not used by tracking logic.
        """
        ax1, ay1, ax2, ay2 = box_a[:4]
        bx1, by1, bx2, by2 = box_b[:4]
        horizontal_gap = max(ax1, bx1) - min(ax2, bx2)
        vertical_gap = max(ay1, by1) - min(ay2, by2)
        return horizontal_gap, vertical_gap

    def _print_new_track_created(self, frame_index, new_person, boxes, frame_height, track_ids) -> None:
        """Log the first-ever appearance of a track ID: the previous
        `NEW_TRACK_CONTEXT_FRAMES` frames plus the current one (raw detections,
        pairwise IoU, pairwise edge gaps, axis overlap, active track IDs), then
        the triggering frame's new-track-specific analysis.

        This identifies the best *geometric* match only - the tracker API does not expose
        which raw detection it actually used internally, so the result is reported as a
        "likely source detection", not a confirmed one.
        """
        context = list(self._history)[-(NEW_TRACK_CONTEXT_FRAMES + 1):]
        print(
            f"\n=== NEW TRACK CREATED: {len(context)}-frame context "
            f"(up to {NEW_TRACK_CONTEXT_FRAMES} previous + current) ==="
        )
        for record in context:
            self._print_frame_geometry(record)

        x1, y1, x2, y2 = new_person.bbox
        track_box = (float(x1), float(y1), float(x2), float(y2), float(x2 - x1), float(y2 - y1))

        print("\n--- NEW TRACK CREATED ---")
        print(f"Frame: {frame_index}")
        print(f"New Track ID: {new_person.track_id}")
        print(f"Track confidence: {new_person.confidence:.2f}")
        print(f"Track bbox: ({x1}, {y1}, {x2}, {y2})")

        print("\nCandidate source detections (edge distance is between this detection and the new track's bbox):")
        best_index, best_iou, best_confidence = None, -1.0, None
        for i, (bbox, confidence) in enumerate(zip(boxes.xyxy, boxes.conf)):
            dx1, dy1, dx2, dy2 = (float(v) for v in bbox)
            width, height = dx2 - dx1, dy2 - dy1
            bbox_height_ratio = height / frame_height
            det_box = (dx1, dy1, dx2, dy2, width, height)
            iou = self._iou(track_box, det_box)
            h_gap, v_gap = self._edge_gaps(track_box, det_box)
            print(
                f"Detection {i}: confidence={float(confidence):.2f}, "
                f"bbox=({dx1:.1f}, {dy1:.1f}, {dx2:.1f}, {dy2:.1f}), "
                f"width={width:.1f}, height={height:.1f}, bbox_height_ratio={bbox_height_ratio:.2f}, "
                f"IoU with track={iou:.2f}, horizontal edge gap={h_gap:.1f}px, vertical edge gap={v_gap:.1f}px"
            )
            if iou > best_iou:
                best_index, best_iou, best_confidence = i, iou, float(confidence)

        print("\nBest matching source detection (best geometric match by IoU;")
        print("not confirmed as BoT-SORT's actual internal association source):")
        if best_index is not None:
            print(f"Detection index: {best_index}")
            print(f"Detection confidence: {best_confidence:.2f}")
            print(f"IoU: {best_iou:.2f}")
        else:
            print("No raw YOLO detections available this frame to compare against.")

        track_desc = ", ".join(str(t) for t in track_ids) or "none"
        print(f"\nActive Track IDs: {track_desc}")
        if self._new_track_thresh_label is not None:
            print(f"Runtime new_track_thresh: {self._new_track_thresh_label}")
        print("-------------------------\n", flush=True)

    @staticmethod
    def _iou(box_a, box_b) -> float:
        """Standard bounding-box IoU: intersection_area / (area_a + area_b - intersection_area).

        Diagnostic-only helper; not used by detection or tracking logic.
        """
        ax1, ay1, ax2, ay2, aw, ah = box_a
        bx1, by1, bx2, by2, bw, bh = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        intersection_area = inter_width * inter_height

        area_a = aw * ah
        area_b = bw * bh
        union_area = area_a + area_b - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

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
        print("--- end diagnostic event ---\n", flush=True)

    def print_first_appearance_summary(self) -> None:
        """Print every track ID's first-observed frame number.

        This is a persistent, always-populated record (see `_first_seen_frame`),
        independent of terminal scrollback - intended to be called once at
        program exit so a track's first appearance is never unrecoverable even
        if its live "--- NEW TRACK CREATED ---" print scrolled out of view
        during a long session.
        """
        print("\n--- First appearance summary (all track IDs this session) ---")
        if not self._first_seen_frame:
            print("No track IDs were observed.")
        else:
            for track_id in sorted(self._first_seen_frame):
                print(f"Track ID {track_id}: first observed at frame {self._first_seen_frame[track_id]}")
        print("--- end first appearance summary ---\n", flush=True)

"""Diagnostic-only measurement of geometry/confidence near an existing CONFIRMED person.

Question this module exists to answer (evidence collection only, no rule
implemented yet):

    "When a new CANDIDATE appears or becomes CONFIRMED near an already
    CONFIRMED person, is there a measurable geometry/confidence pattern
    that distinguishes a false object from a real second person?"

This module is strictly downstream and read-only:
    - It consumes `ValidatedTrack` objects already produced by `TrackValidator`
      and raw `Boxes` already produced by `PersonDetector`.
    - It never calls into `PersonDetector`, `PersonTracker`, or `TrackValidator`,
      and never mutates any `TrackedPerson`/`ValidatedTrack`/`Boxes` object.
    - It never affects which tracks exist, when they confirm, or how they are
      drawn as tracks - it only prints extra diagnostic text and (optionally)
      lightweight on-screen hints.

Raw YOLO detections have no persistent ID, so any correspondence drawn between
a candidate's current bbox and a *historical* raw detection is a best-effort
geometric guess (highest IoU, tie-broken by center distance) - never treated,
printed, or implied as certain.
"""

import math
from collections import deque
from dataclasses import dataclass

from track_validator import TrackState, ValidatedTrack
from geometry_utils import iou as _iou, edge_gaps as _edge_gaps, center_distance as _center_distance, as_geom_box as _as_geom_box

RAW_HISTORY_WINDOW = 1.0  # seconds of raw-detection history to keep/print for a new-candidate event


@dataclass
class NearestConfirmed:
    """Diagnostic-only relationship between one track and the nearest other CONFIRMED track."""

    confirmed_id: int
    confirmed_bbox: tuple[int, int, int, int]
    confirmed_confidence: float
    iou: float
    horizontal_gap: float
    vertical_gap: float
    center_distance: float


def nearest_confirmed_relationship(
    subject_bbox, exclude_id: int, validated_tracks: list[ValidatedTrack]
) -> NearestConfirmed | None:
    """Find the nearest other CONFIRMED track (by center distance) to `subject_bbox`.

    Read-only: does not modify `validated_tracks`. Returns None if no other
    CONFIRMED track exists this frame.
    """
    subject_box = _as_geom_box(subject_bbox)
    best = None
    for validated in validated_tracks:
        person = validated.tracked_person
        if validated.state is not TrackState.CONFIRMED or person.track_id == exclude_id:
            continue
        other_box = _as_geom_box(person.bbox)
        dist = _center_distance(subject_box, other_box)
        if best is None or dist < best.center_distance:
            h_gap, v_gap = _edge_gaps(subject_box, other_box)
            best = NearestConfirmed(
                confirmed_id=person.track_id,
                confirmed_bbox=person.bbox,
                confirmed_confidence=person.confidence,
                iou=_iou(subject_box, other_box),
                horizontal_gap=h_gap,
                vertical_gap=v_gap,
                center_distance=dist,
            )
    return best


class ProximityDiagnostics:
    """Event-triggered diagnostic: geometry/confidence at the moment a track becomes a
    CANDIDATE or CONFIRMED near an existing CONFIRMED person.

    Purely observational - see module docstring. Never blocks, delays, or
    otherwise changes CANDIDATE/CONFIRMED state.
    """

    def __init__(self, raw_history_window: float = RAW_HISTORY_WINDOW):
        self._raw_history_window = raw_history_window
        # (timestamp, [(x1,y1,x2,y2,w,h,confidence), ...]) - one entry per observed frame
        self._raw_history: deque[tuple[float, list[tuple[float, ...]]]] = deque()
        self._prev_states: dict[int, TrackState] = {}
        self._start_time: float | None = None

    def observe(self, now: float, boxes, validated_tracks: list[ValidatedTrack], frame_width: int, frame_height: int) -> None:
        """Record this frame's raw detections and report on any CANDIDATE/CONFIRMED
        transition that has another CONFIRMED person to compare against.

        Args:
            now: Current timestamp (same clock as `TrackValidator`).
            boxes: This frame's raw `Boxes`-like YOLO detections - read-only.
            validated_tracks: This frame's `list[ValidatedTrack]` from `TrackValidator.update()` - read-only.
            frame_width, frame_height: For scale/normalization only.
        """
        if self._start_time is None:
            self._start_time = now

        detections = [
            (*_as_geom_box((x1, y1, x2, y2)), float(conf))
            for (x1, y1, x2, y2), conf in zip(boxes.xyxy, boxes.conf)
        ]
        self._raw_history.append((now, detections))
        while self._raw_history and now - self._raw_history[0][0] > self._raw_history_window + 0.5:
            self._raw_history.popleft()

        current_states = {v.tracked_person.track_id: v.state for v in validated_tracks}
        people_by_id = {v.tracked_person.track_id: v.tracked_person for v in validated_tracks}

        for track_id, state in current_states.items():
            prev_state = self._prev_states.get(track_id)
            if prev_state is None and state is TrackState.CANDIDATE:
                self._maybe_report("NEW CANDIDATE", people_by_id[track_id], validated_tracks, now, frame_width, frame_height)
            elif prev_state is TrackState.CANDIDATE and state is TrackState.CONFIRMED:
                self._maybe_report(
                    "CANDIDATE -> CONFIRMED", people_by_id[track_id], validated_tracks, now, frame_width, frame_height
                )

        self._prev_states = current_states

    def _maybe_report(self, event_label, subject, validated_tracks, now, frame_width, frame_height) -> None:
        nearest = nearest_confirmed_relationship(subject.bbox, subject.track_id, validated_tracks)
        if nearest is None:
            return  # no other CONFIRMED person to compare against - nothing to report

        x1, y1, x2, y2 = subject.bbox
        width, height = x2 - x1, y2 - y1
        bbox_scale = height / frame_height if frame_height else 0.0
        frame_diagonal = math.hypot(frame_width, frame_height)
        normalized_center_distance = nearest.center_distance / frame_diagonal if frame_diagonal else 0.0

        subject_box = _as_geom_box(subject.bbox)
        confirmed_box = _as_geom_box(nearest.confirmed_bbox)
        h_overlap_px = max(0.0, -nearest.horizontal_gap)
        v_overlap_px = max(0.0, -nearest.vertical_gap)
        h_overlap_ratio = h_overlap_px / min(subject_box[4], confirmed_box[4]) if min(subject_box[4], confirmed_box[4]) > 0 else 0.0
        v_overlap_ratio = v_overlap_px / min(subject_box[5], confirmed_box[5]) if min(subject_box[5], confirmed_box[5]) > 0 else 0.0

        elapsed = now - self._start_time
        print("\n--- PROXIMITY DIAGNOSTIC ---")
        print(f"Event: {event_label}")
        print(f"Time: t={elapsed:.2f}s")
        print(f"Candidate ID: {subject.track_id}")
        print(f"Candidate confidence: {subject.confidence:.2f}")
        print(f"Candidate bbox: ({x1}, {y1}, {x2}, {y2})")
        print(f"Candidate bbox width: {width}, height: {height}")
        print(f"Candidate bbox scale (height/frame_height): {bbox_scale:.2f}")

        print(f"\nNearest existing CONFIRMED ID: {nearest.confirmed_id}")
        print(f"Confirmed bbox: {nearest.confirmed_bbox}")
        print(f"Confirmed confidence: {nearest.confirmed_confidence:.2f}")

        print(f"\nIoU: {nearest.iou:.2f}")
        print(f"Horizontal edge gap: {nearest.horizontal_gap:.1f}px (negative = overlapping)")
        print(f"Vertical edge gap: {nearest.vertical_gap:.1f}px (negative = overlapping)")
        print(f"Horizontal overlap: {h_overlap_px:.1f}px (ratio {h_overlap_ratio:.2f} of narrower box)")
        print(f"Vertical overlap: {v_overlap_px:.1f}px (ratio {v_overlap_ratio:.2f} of shorter box)")
        print(f"Center distance: {nearest.center_distance:.1f}px")
        print(f"Normalized center distance (/ frame diagonal): {normalized_center_distance:.3f}")

        print(
            "\nBest geometric RAW detection history (diagnostic association only - "
            "matched by geometry to the candidate's CURRENT bbox, not a persistent ID; "
            "NOT confirmed to be the same physical object):"
        )
        for ts, detections in self._raw_history:
            rel_t = ts - now
            if rel_t < -self._raw_history_window:
                continue
            if not detections:
                print(f"t={rel_t:+.2f}s | no RAW detections this frame")
                continue
            best_det, best_iou, best_dist = None, -1.0, None
            for det in detections:
                det_box, conf = det[:6], det[6]
                iou = _iou(subject_box, det_box)
                dist = _center_distance(subject_box, det_box)
                if best_det is None or iou > best_iou or (iou == best_iou and dist < best_dist):
                    best_det, best_iou, best_dist = det, iou, dist
            conf = best_det[6]
            print(
                f"t={rel_t:+.2f}s | conf={conf:.2f} | "
                f"IoU_vs_candidate_now={best_iou:.2f} | center_dist_vs_candidate_now={best_dist:.1f}px"
            )
        print("--------------------------------\n", flush=True)

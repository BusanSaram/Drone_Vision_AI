"""Application-level track validation, downstream of PersonTracker (BoT-SORT).

Pipeline position:
    PersonDetector (YOLO) -> PersonTracker (BoT-SORT) -> TrackValidator -> Confirmed Persons

This module never touches detection or tracking. It only observes the
`TrackedPerson` list PersonTracker already produced each frame and maintains
its own, separate, timestamp-based state per track ID:

    CANDIDATE            - a raw BoT-SORT track ID that has not yet been
                            continuously observed for `confirmation_time`
                            seconds.
    CONFIRMATION_BLOCKED - a CANDIDATE that reached `confirmation_time` but
                            was, at that moment, suspiciously close to an
                            already-CONFIRMED person (see the proximity gate
                            below). It is re-evaluated every frame; once the
                            suspicious proximity clears it returns to
                            CANDIDATE with a FRESH timer (see `update()`).
    CONFIRMED            - a track that has been continuously observed for at
                            least `confirmation_time` seconds AND, at the
                            moment of promotion, was not suspiciously close to
                            an existing CONFIRMED person. CONFIRMED tracks
                            tolerate brief absences up to `lost_grace_time`
                            seconds before expiring, and are NEVER re-examined
                            by the proximity gate once confirmed - only
                            CANDIDATEs are ever blocked.

Proximity gate (experimental, see PROXIMITY_* constants below): this is a
controlled experiment testing whether blocking confirmation of a candidate
that sits suspiciously close to an already-CONFIRMED person can keep a known
flickering false-positive track from ever reaching CONFIRMED, now that we
have evidence that temporal confirmation alone is not sufficient. It is not
assumed to be a final or sufficient solution - a real second person standing
close to someone could show similar geometry, which is exactly what the
webcam test in this experiment is meant to check.
"""

import time
from dataclasses import dataclass, field
from enum import Enum

from person_tracker import TrackedPerson
from geometry_utils import as_geom_box, center_distance, edge_gaps, iou

CONFIRMATION_TIME = 0.5  # seconds; experimental starting value, not tuned
# Experimental starting value: roughly 2x BoT-SORT's own default track_buffer
# (30 frames, ~1s at a typical ~30 FPS) so the application layer's tolerance
# for a brief absence is a bit more generous than what BoT-SORT itself already
# tries to bridge internally - not a final tuned value.
LOST_GRACE_TIME = 2.0  # seconds

# Proximity gate thresholds - all experimental, NOT scientifically tuned.
# Chosen conservatively from the geometry already seen in the proximity
# diagnostics: the known false detection tends to sit beside a confirmed
# person (small/negative horizontal gap) at roughly the same height
# (large vertical overlap), with IoU alone being too small/unreliable to use
# on its own since the boxes often barely touch rather than overlap.
#
# Horizontal gap is normalized by the CONFIRMED person's own bbox width
# (not raw pixels) so the same threshold behaves consistently whether the
# person is near or far from the camera.
PROXIMITY_MAX_HORIZONTAL_GAP_RATIO = 0.5  # candidate's horizontal gap from the confirmed box, as a fraction of the confirmed box's width
# Vertical overlap is normalized by the shorter of the two bbox heights.
PROXIMITY_MIN_VERTICAL_OVERLAP_RATIO = 0.5  # fraction of the shorter box's height that must overlap vertically
# Any IoU at/above this is independently treated as suspicious, regardless
# of the beside/overlap pattern above (catches boxes that overlap heavily
# but happen not to fit the "beside" geometry).
PROXIMITY_MIN_IOU_FOR_BLOCK = 0.10


class TrackState(Enum):
    CANDIDATE = "CANDIDATE"
    CONFIRMATION_BLOCKED = "CONFIRMATION BLOCKED"
    CONFIRMED = "CONFIRMED"


@dataclass
class ValidatedTrack:
    """One currently-visible track paired with its TrackValidator state.

    `tracked_person` is the exact object PersonTracker produced - never
    copied or mutated here. `block_info` is populated only while
    `state is TrackState.CONFIRMATION_BLOCKED`, recomputed fresh every frame,
    and carries the diagnostic detail behind the block (which confirmed ID
    triggered it and the measured geometry) for visualization/logging.
    """

    tracked_person: TrackedPerson
    state: TrackState
    block_info: dict | None = None


@dataclass
class _TrackRecord:
    state: TrackState
    candidate_since: float  # when the current unbroken CANDIDATE streak started; reset on block-clear
    last_seen: float
    missing_notified: bool = False  # for CONFIRMED: already logged "temporarily lost" for this gap?


@dataclass
class _Stats:
    candidates_created: int = 0
    candidates_rejected: int = 0
    confirmed_count: int = 0
    recovered_count: int = 0
    expired_count: int = 0
    blocked_count: int = 0
    block_cleared_count: int = 0
    ever_confirmed: set = field(default_factory=set)
    ever_blocked: set = field(default_factory=set)
    all_ids: set = field(default_factory=set)


def _is_suspicious_proximity(candidate_box, confirmed_box) -> tuple[bool, dict]:
    """Evaluate the experimental proximity-gate rule between one candidate and one
    confirmed box (both already `(x1,y1,x2,y2,w,h)` tuples).

    Suspicious if EITHER:
      - the candidate sits "beside" the confirmed box with a small horizontal
        gap (relative to the confirmed box's own width) AND substantial
        vertical overlap (relative to the shorter box's height), OR
      - the two boxes have meaningful IoU on their own.

    Returns (is_suspicious, measurement_details) - measurements are always
    returned so callers can log/display them regardless of the outcome.
    """
    h_gap, v_gap = edge_gaps(candidate_box, confirmed_box)
    iou_value = iou(candidate_box, confirmed_box)

    confirmed_width = confirmed_box[4]
    shorter_height = min(candidate_box[5], confirmed_box[5])
    horizontal_gap_ratio = h_gap / confirmed_width if confirmed_width > 0 else 0.0
    vertical_overlap_px = max(0.0, -v_gap)
    vertical_overlap_ratio = vertical_overlap_px / shorter_height if shorter_height > 0 else 0.0

    beside_and_aligned = (
        horizontal_gap_ratio <= PROXIMITY_MAX_HORIZONTAL_GAP_RATIO
        and vertical_overlap_ratio >= PROXIMITY_MIN_VERTICAL_OVERLAP_RATIO
    )
    overlapping = iou_value >= PROXIMITY_MIN_IOU_FOR_BLOCK
    suspicious = beside_and_aligned or overlapping

    details = {
        "iou": iou_value,
        "horizontal_gap": h_gap,
        "horizontal_gap_ratio": horizontal_gap_ratio,
        "vertical_overlap_ratio": vertical_overlap_ratio,
    }
    return suspicious, details


class TrackValidator:
    """Promotes raw BoT-SORT track IDs to CONFIRMED only after sustained presence.

    Continuity is defined simply and explicitly: a CANDIDATE must appear in
    every consecutive call to `update()` since it was first observed. Missing
    even one frame drops its candidate state entirely - a later reappearance
    starts a brand-new candidate with a fresh timer (a conservative restart,
    not a resume). This uses wall-clock timestamps throughout, not frame
    counts, since FPS is not guaranteed constant.
    """

    def __init__(
        self,
        confirmation_time: float = CONFIRMATION_TIME,
        lost_grace_time: float = LOST_GRACE_TIME,
        time_fn=time.time,
    ):
        self._confirmation_time = confirmation_time
        self._lost_grace_time = lost_grace_time
        self._time_fn = time_fn
        self._start_time: float | None = None
        self._records: dict[int, _TrackRecord] = {}
        self._stats = _Stats()

    def update(self, tracked_people: list[TrackedPerson], now: float | None = None) -> list[ValidatedTrack]:
        """Advance validator state by one frame and return this frame's validated tracks.

        Args:
            tracked_people: This frame's `TrackedPerson` list from `PersonTracker.update()`.
            now: Timestamp for this frame; defaults to calling `time_fn()` (real time).

        Returns:
            One `ValidatedTrack` per currently-visible track, each carrying its
            current CANDIDATE/CONFIRMED state. Filter with `confirmed_only()`
            for the "usable persons" output.
        """
        now = now if now is not None else self._time_fn()
        if self._start_time is None:
            self._start_time = now

        current_ids = {p.track_id for p in tracked_people}

        # Tracks known from prior frames that are absent this frame.
        for track_id in list(self._records.keys()):
            if track_id in current_ids:
                continue
            record = self._records[track_id]
            if record.state in (TrackState.CANDIDATE, TrackState.CONFIRMATION_BLOCKED):
                duration = now - record.candidate_since
                self._stats.candidates_rejected += 1
                self._log("CANDIDATE LOST BEFORE CONFIRMATION", track_id, now, duration)
                del self._records[track_id]
            else:
                missing_duration = now - record.last_seen
                if not record.missing_notified:
                    record.missing_notified = True
                    self._log("CONFIRMED TRACK TEMPORARILY LOST", track_id, now, missing_duration)
                if missing_duration >= self._lost_grace_time:
                    self._stats.expired_count += 1
                    self._log("CONFIRMED TRACK EXPIRED", track_id, now, missing_duration)
                    del self._records[track_id]

        # Currently-visible CONFIRMED persons' boxes - the only thing the
        # proximity gate is ever compared against. A CONFIRMED person who is
        # temporarily missing this frame (grace period) has no current bbox
        # and simply is not considered this frame.
        confirmed_boxes_by_id = {}
        for person in tracked_people:
            record_for_person = self._records.get(person.track_id)
            if record_for_person is not None and record_for_person.state is TrackState.CONFIRMED:
                confirmed_boxes_by_id[person.track_id] = as_geom_box(person.bbox)

        # Currently-visible tracks.
        validated = []
        for person in tracked_people:
            self._stats.all_ids.add(person.track_id)
            record = self._records.get(person.track_id)
            block_info = None

            if record is None:
                record = _TrackRecord(state=TrackState.CANDIDATE, candidate_since=now, last_seen=now)
                self._records[person.track_id] = record
                self._stats.candidates_created += 1
                self._log("CANDIDATE CREATED", person.track_id, now, 0.0)

            elif record.state is TrackState.CANDIDATE:
                record.last_seen = now
                elapsed = now - record.candidate_since
                if elapsed >= self._confirmation_time:
                    block_info = self._check_proximity_block(person, confirmed_boxes_by_id)
                    if block_info is None:
                        record.state = TrackState.CONFIRMED
                        self._stats.confirmed_count += 1
                        self._stats.ever_confirmed.add(person.track_id)
                        self._log("TRACK CONFIRMED", person.track_id, now, elapsed)
                    else:
                        record.state = TrackState.CONFIRMATION_BLOCKED
                        self._stats.blocked_count += 1
                        self._stats.ever_blocked.add(person.track_id)
                        self._log_block(person.track_id, block_info, now)

            elif record.state is TrackState.CONFIRMATION_BLOCKED:
                record.last_seen = now
                block_info = self._check_proximity_block(person, confirmed_boxes_by_id)
                if block_info is None:
                    # Proximity cleared: restart the confirmation timer
                    # conservatively rather than crediting time spent blocked.
                    record.state = TrackState.CANDIDATE
                    record.candidate_since = now
                    self._stats.block_cleared_count += 1
                    self._log("BLOCK CLEARED", person.track_id, now, 0.0)

            else:  # CONFIRMED, currently visible - never re-evaluated by the proximity gate
                if record.missing_notified:
                    missing_duration = now - record.last_seen
                    record.missing_notified = False
                    self._stats.recovered_count += 1
                    self._log("CONFIRMED TRACK RECOVERED", person.track_id, now, missing_duration)
                record.last_seen = now

            validated.append(ValidatedTrack(tracked_person=person, state=record.state, block_info=block_info))

        return validated

    def _check_proximity_block(self, candidate: TrackedPerson, confirmed_boxes_by_id: dict) -> dict | None:
        """Check `candidate` against the nearest currently-visible CONFIRMED person.

        Returns None if there is no other CONFIRMED person, or the nearest one
        is not suspiciously close; otherwise returns the measurement details
        (plus which confirmed ID triggered it) for logging/display.

        Note: only the single *nearest* (by center distance) CONFIRMED person
        is evaluated, matching the same "nearest confirmed" definition already
        used by `proximity_diagnostics.py` - a deliberate simplification, not
        a check against every CONFIRMED person independently.
        """
        if not confirmed_boxes_by_id:
            return None

        candidate_box = as_geom_box(candidate.bbox)
        nearest_id, nearest_box, nearest_dist = None, None, None
        for confirmed_id, confirmed_box in confirmed_boxes_by_id.items():
            dist = center_distance(candidate_box, confirmed_box)
            if nearest_dist is None or dist < nearest_dist:
                nearest_id, nearest_box, nearest_dist = confirmed_id, confirmed_box, dist

        suspicious, details = _is_suspicious_proximity(candidate_box, nearest_box)
        if not suspicious:
            return None
        return {"confirmed_id": nearest_id, **details}

    def _log(self, event: str, track_id: int, now: float, duration: float) -> None:
        elapsed = now - self._start_time
        print(
            f"[TrackValidator t={elapsed:.2f}s] {event} | Track ID {track_id} | duration={duration:.2f}s",
            flush=True,
        )

    def _log_block(self, track_id: int, block_info: dict, now: float) -> None:
        elapsed = now - self._start_time
        print(
            f"[TrackValidator t={elapsed:.2f}s] CONFIRMATION BLOCKED | Track ID {track_id} | "
            f"near confirmed ID {block_info['confirmed_id']} | "
            f"horizontal_gap={block_info['horizontal_gap']:.1f}px "
            f"(ratio={block_info['horizontal_gap_ratio']:.2f}) | "
            f"vertical_overlap_ratio={block_info['vertical_overlap_ratio']:.2f} | "
            f"IoU={block_info['iou']:.2f} | "
            f"thresholds: max_horizontal_gap_ratio={PROXIMITY_MAX_HORIZONTAL_GAP_RATIO}, "
            f"min_vertical_overlap_ratio={PROXIMITY_MIN_VERTICAL_OVERLAP_RATIO}, "
            f"min_iou={PROXIMITY_MIN_IOU_FOR_BLOCK}",
            flush=True,
        )

    def print_summary(self) -> None:
        """Print end-of-run totals and each track ID's final outcome."""
        print("\n--- TrackValidator summary ---")
        print(f"Total candidate tracks: {self._stats.candidates_created}")
        print(f"Candidates rejected before confirmation: {self._stats.candidates_rejected}")
        print(f"Tracks promoted to confirmed: {self._stats.confirmed_count}")
        print(f"Confirmed tracks recovered: {self._stats.recovered_count}")
        print(f"Confirmed tracks expired: {self._stats.expired_count}")
        print(f"Candidates blocked by proximity at least once: {len(self._stats.ever_blocked)}")
        print(f"Block events (candidate -> blocked): {self._stats.blocked_count}")
        print(f"Block-clear events (blocked -> candidate): {self._stats.block_cleared_count}")

        print("\nPer-track outcome:")
        if not self._stats.all_ids:
            print("No track IDs were observed.")
        else:
            for track_id in sorted(self._stats.all_ids):
                if track_id in self._stats.ever_confirmed:
                    outcome = "confirmed at least once"
                    if track_id in self._stats.ever_blocked:
                        outcome += " (was proximity-blocked earlier)"
                elif track_id in self._records:
                    outcome = f"still {self._records[track_id].state.value} at program exit (never confirmed)"
                else:
                    outcome = "rejected before confirmation (never confirmed)"
                print(f"Track ID {track_id}: {outcome}")
        print("--- end TrackValidator summary ---\n", flush=True)


def confirmed_only(validated_tracks: list[ValidatedTrack]) -> list[TrackedPerson]:
    """Filter to just the TrackedPerson objects currently CONFIRMED.

    This is the clean interface future stages (gesture recognition, target
    selection, etc.) would consume - not called by them yet, since those
    stages are not implemented in this experiment.
    """
    return [v.tracked_person for v in validated_tracks if v.state is TrackState.CONFIRMED]

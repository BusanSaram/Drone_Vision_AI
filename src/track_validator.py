"""Application-level track validation, downstream of PersonTracker (BoT-SORT).

Pipeline position:
    PersonDetector (YOLO) -> PersonTracker (BoT-SORT) -> TrackValidator -> Confirmed Persons

This module never touches detection or tracking. It only observes the
`TrackedPerson` list PersonTracker already produced each frame and maintains
its own, separate, timestamp-based state per track ID:

    CANDIDATE            - a raw BoT-SORT track ID that has not yet been
                            continuously observed for `confirmation_time`
                            seconds.
    CONFIRMED            - a track that has been continuously observed for at
                            least `confirmation_time` seconds. CONFIRMED
                            tracks tolerate brief absences up to
                            `lost_grace_time` seconds before expiring.

Note: an earlier version of this module also gated CONFIRMED promotion on a
proximity check against other CONFIRMED persons (a `CONFIRMATION_BLOCKED`
state), intended to keep a known flickering false-positive track from ever
reaching CONFIRMED. That check was removed after testing with two real
people showed it produced false negatives: a genuine second person standing
close to someone already CONFIRMED could itself get blocked. Confirmation is
now purely temporal, as described above.
"""

import time
from dataclasses import dataclass, field
from enum import Enum

from person_tracker import TrackedPerson

CONFIRMATION_TIME = 0.5  # seconds; experimental starting value, not tuned
# Experimental starting value: roughly 2x BoT-SORT's own default track_buffer
# (30 frames, ~1s at a typical ~30 FPS) so the application layer's tolerance
# for a brief absence is a bit more generous than what BoT-SORT itself already
# tries to bridge internally - not a final tuned value.
LOST_GRACE_TIME = 2.0  # seconds


class TrackState(Enum):
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"


@dataclass
class ValidatedTrack:
    """One currently-visible track paired with its TrackValidator state.

    `tracked_person` is the exact object PersonTracker produced - never
    copied or mutated here.
    """

    tracked_person: TrackedPerson
    state: TrackState


@dataclass
class _TrackRecord:
    state: TrackState
    candidate_since: float  # when the current unbroken CANDIDATE streak started
    last_seen: float
    missing_notified: bool = False  # for CONFIRMED: already logged "temporarily lost" for this gap?


@dataclass
class _Stats:
    candidates_created: int = 0
    candidates_rejected: int = 0
    confirmed_count: int = 0
    recovered_count: int = 0
    expired_count: int = 0
    ever_confirmed: set = field(default_factory=set)
    all_ids: set = field(default_factory=set)


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
        debug: bool = False,
    ):
        self._confirmation_time = confirmation_time
        self._lost_grace_time = lost_grace_time
        self._time_fn = time_fn
        self._debug = debug
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
            if record.state is TrackState.CANDIDATE:
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

        # Currently-visible tracks.
        validated = []
        for person in tracked_people:
            self._stats.all_ids.add(person.track_id)
            record = self._records.get(person.track_id)

            if record is None:
                record = _TrackRecord(state=TrackState.CANDIDATE, candidate_since=now, last_seen=now)
                self._records[person.track_id] = record
                self._stats.candidates_created += 1
                self._log("CANDIDATE CREATED", person.track_id, now, 0.0)

            elif record.state is TrackState.CANDIDATE:
                record.last_seen = now
                elapsed = now - record.candidate_since
                if elapsed >= self._confirmation_time:
                    record.state = TrackState.CONFIRMED
                    self._stats.confirmed_count += 1
                    self._stats.ever_confirmed.add(person.track_id)
                    self._log("TRACK CONFIRMED", person.track_id, now, elapsed)

            else:  # CONFIRMED, currently visible
                if record.missing_notified:
                    missing_duration = now - record.last_seen
                    record.missing_notified = False
                    self._stats.recovered_count += 1
                    self._log("CONFIRMED TRACK RECOVERED", person.track_id, now, missing_duration)
                record.last_seen = now

            validated.append(ValidatedTrack(tracked_person=person, state=record.state))

        return validated

    def _log(self, event: str, track_id: int, now: float, duration: float) -> None:
        """Per-event debug log (CANDIDATE CREATED, TRACK CONFIRMED, etc.).

        Silent unless `debug=True` was passed to `__init__` - state transitions
        themselves are unaffected either way, only this print.
        """
        if not self._debug:
            return
        elapsed = now - self._start_time
        print(
            f"[TrackValidator t={elapsed:.2f}s] {event} | Track ID {track_id} | duration={duration:.2f}s",
            flush=True,
        )

    def print_summary(self) -> None:
        """Print end-of-run tracking totals.

        Always prints the minimal counts used in the normal run's exit
        summary. The rest of the (previously always-on) detail - candidate
        counts and each track ID's final outcome - is kept for debugging but
        only printed when `debug=True` was passed to `__init__`.
        """
        print("--- Tracking Summary ---")
        print(f"Total unique track IDs: {len(self._stats.all_ids)}")
        print(f"Tracks promoted to CONFIRMED: {self._stats.confirmed_count}")
        print(f"Confirmed tracks recovered: {self._stats.recovered_count}")
        print(f"Confirmed tracks expired: {self._stats.expired_count}")

        if not self._debug:
            return

        print(f"Total candidate tracks: {self._stats.candidates_created}")
        print(f"Candidates rejected before confirmation: {self._stats.candidates_rejected}")

        print("\nPer-track outcome:")
        if not self._stats.all_ids:
            print("No track IDs were observed.")
        else:
            for track_id in sorted(self._stats.all_ids):
                if track_id in self._stats.ever_confirmed:
                    outcome = "confirmed at least once"
                elif track_id in self._records:
                    outcome = f"still {self._records[track_id].state.value} at program exit (never confirmed)"
                else:
                    outcome = "rejected before confirmation (never confirmed)"
                print(f"Track ID {track_id}: {outcome}")


def confirmed_only(validated_tracks: list[ValidatedTrack]) -> list[TrackedPerson]:
    """Filter to just the TrackedPerson objects currently CONFIRMED.

    This is the clean interface future stages (gesture recognition, target
    selection, etc.) would consume - not called by them yet, since those
    stages are not implemented in this experiment.
    """
    return [v.tracked_person for v in validated_tracks if v.state is TrackState.CONFIRMED]

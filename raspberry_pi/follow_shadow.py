"""Shadow Mode: run the forward-only FollowController on live IMX500 tracking results and only LOG it.

Nothing in this module can move a vehicle. It has no MAVLink, serial, GPIO or
network code (a unit test scans the imports); its only outputs are a JSON Lines
file, an optional text line drawn on the streamed frame and console lines.

Used by yolo_pose_imx_tracking_stream.py only when --follow-shadow is given:
    NEW AI result   -> on_new_result(pipeline, now): adapter -> controller.update()
                       -> one "new" record (input + full-body metrics + decision)
    REUSED frame    -> on_tick(now): controller.command_at(now), i.e. what a command
                       publisher would send now; logged only when it overrides the
                       last decision (communication timeout / stale result)
    each frame      -> draw(image) with --shadow-overlay
It only READS the pipeline (target_status, last_result); TargetSelector, gestures,
tracking and streaming are untouched. Any exception inside Shadow Mode disables
Shadow Mode (one console warning) and never reaches the camera loop.

Records go through a bounded queue to a writer thread, so file I/O never blocks
the camera loop: when the queue is full or the size limit is reached, records
are dropped and counted. The "new" records hold every FollowInput field, so
src/diagnostics/follow_shadow_replay.py can re-run the exact same decisions.
"""

import hashlib
import json
import os
import queue
import threading
import time
from collections import Counter
from dataclasses import asdict

import cv2

from follow_controller import FollowController, FollowParams, body_metrics
from follow_record import input_from_record, input_to_record
from imx500_follow_adapter import follow_input_from_pipeline

SCHEMA_VERSION = 1
SUMMARY_RESERVE_BYTES = 16 * 1024  # kept free under the size limit for the error / summary records
OVERLAY_HEIGHT = 22


def _round(value, digits):
    return None if value is None else round(float(value), digits)


def body_record(metrics) -> dict | None:
    if metrics is None:
        return None
    return {k: (_round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}


def command_record(cmd, controller) -> dict:
    return {
        "state": cmd.state.value, "stage": cmd.stage, "reason": cmd.reason,
        "fwd": round(cmd.forward_mps, 4), "yaw": round(cmd.yaw_rate_dps, 3), "hold": cmd.hold, "land": cmd.land,
        "target_id": cmd.target_id, "h_norm": _round(cmd.box_height, 4), "cx_err": _round(cmd.center_error, 4),
        "diag": list(cmd.diagnostics), "ind": list(cmd.indicators),
        "streak": controller.valid_streak, "baseline_h": _round(controller.baseline_height, 4),
    }


class JsonlWriter:
    """Background JSON Lines writer: write() never blocks; drops (and counts) instead."""

    _STOP = object()

    def __init__(self, path: str, max_bytes: int, queue_size: int = 2000, opener=open):
        self.path = path
        self.max_bytes = max_bytes
        self.lines = 0
        self.bytes = 0
        self.dropped_queue_full = 0
        self.dropped_size_limit = 0
        self.dropped_after_error = 0
        self.dropped_unserializable = 0
        self.error: str | None = None
        self._opener = opener
        self._closed = False
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, name="shadow-jsonl", daemon=True)
        self._thread.start()

    def write(self, record: dict) -> bool:
        if self._closed or self.error is not None:
            self.dropped_after_error += 1
            return False
        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            self.dropped_queue_full += 1
            return False

    def close(self, timeout: float = 3.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(self._STOP, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)

    def _run(self) -> None:
        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            f = self._opener(self.path, "a", encoding="utf-8")
        except Exception as e:  # noqa: BLE001 - any I/O problem must stay inside this thread
            self.error = f"open failed: {e!r}"
            return
        try:
            while True:
                record = self._queue.get()
                if record is self._STOP:
                    break
                if self.error is not None:
                    continue
                try:
                    line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                except (TypeError, ValueError):
                    self.dropped_unserializable += 1  # one bad record (e.g. NaN) does not stop the log
                    continue
                try:
                    size = len(line.encode("utf-8"))
                    limit = self.max_bytes if record.get("type") in ("summary", "error") \
                        else self.max_bytes - SUMMARY_RESERVE_BYTES
                    if self.bytes + size > limit:
                        self.dropped_size_limit += 1
                        continue
                    f.write(line)
                    f.flush()
                    self.bytes += size
                    self.lines += 1
                except Exception as e:  # noqa: BLE001
                    self.error = f"write failed: {e!r}"
        finally:
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass


class ShadowFollower:
    def __init__(self, writer, frame_width: int, frame_height: int, params: FollowParams = FollowParams(),
                 metadata: dict | None = None):
        self.writer = writer
        self.width = frame_width
        self.height = frame_height
        self.params = params
        self.controller = FollowController(params)
        self.enabled = True
        self.error: str | None = None
        self.last_command = None
        self.new_results = 0
        self.duplicates_skipped = 0
        self.tick_records = 0
        self.states: Counter = Counter()
        self.diagnostics: Counter = Counter()
        self.max_forward = 0.0
        self.max_abs_yaw = 0.0
        self.process_ms: list[float] = []
        self._last_result_t = None
        self._last_tick_key = None
        self.writer.write({"type": "meta", "schema": SCHEMA_VERSION, "wall_time": time.time(),
                           "frame": [frame_width, frame_height], "params": asdict(params), **(metadata or {})})

    def on_new_result(self, pipeline, now: float):
        """Call once after pipeline.update_new(); returns the hypothetical FollowCommand (or None)."""
        if not self.enabled:
            return None
        try:
            t0 = time.perf_counter()
            result = pipeline.last_result
            if result is None:
                return None
            if self._last_result_t is not None and result.timestamp <= self._last_result_t:
                self.duplicates_skipped += 1  # the same NEW result is never evaluated twice
                return None
            self._last_result_t = result.timestamp
            # Round-trip through the log format so the controller sees exactly what is logged.
            record_in = input_to_record(follow_input_from_pipeline(pipeline, now, self.width, self.height))
            inp = input_from_record(record_in)
            cmd = self.controller.update(inp)
            body = None if inp.keypoints is None or inp.bbox is None else \
                body_metrics(inp.keypoints, inp.bbox, self.width, self.height, self.params)
            self._count(cmd)
            self.writer.write({"type": "new", "t": now, "det_count": len(result.detections),
                               "input": record_in, "body": body_record(body),
                               "ctl": command_record(cmd, self.controller)})
            self.process_ms.append((time.perf_counter() - t0) * 1000)
            return cmd
        except Exception as e:  # noqa: BLE001 - Shadow Mode must never break the camera loop
            self._fail("on_new_result", e)
            return None

    def on_tick(self, now: float):
        """Call on frames without a NEW result: what a publisher would send now (logged only on override)."""
        if not self.enabled:
            return None
        try:
            cmd = self.controller.command_at(now)
            if cmd is self.controller.last_command:
                self._last_tick_key = None
                return cmd
            key = (cmd.state, cmd.reason)
            if key != self._last_tick_key:
                self._last_tick_key = key
                self.tick_records += 1
                self.states[cmd.state.value] += 1
                self.writer.write({"type": "tick", "t": now, "ctl": command_record(cmd, self.controller)})
            self.last_command = cmd
            return cmd
        except Exception as e:  # noqa: BLE001
            self._fail("on_tick", e)
            return None

    def overlay_text(self) -> str:
        cmd = self.last_command
        if cmd is None:
            return "SHADOW (no AI result yet)"
        h = "-" if cmd.box_height is None else f"{cmd.box_height:.0%}"
        motion = f"FWD {cmd.forward_mps:.2f}m/s" if cmd.forward_mps > 0 else "HOLD"
        yaw = "-" if cmd.yaw_rate_dps == 0 else f"{'R' if cmd.yaw_rate_dps > 0 else 'L'} {abs(cmd.yaw_rate_dps):.0f}dps"
        occ = " OCC?" if "OCCLUSION_SUSPECTED" in cmd.diagnostics else ""
        return f"SHADOW id={cmd.target_id} {cmd.state.value} h={h} {motion} yaw={yaw}{occ} | {cmd.reason}"

    def draw(self, image) -> None:
        """One text line in a dark band at the bottom of the frame (the top status band is not touched)."""
        if not self.enabled:
            return
        try:
            height, width = image.shape[:2]
            cv2.rectangle(image, (0, height - OVERLAY_HEIGHT), (width - 1, height - 1), (0, 0, 0), -1)
            cv2.putText(image, self.overlay_text()[:95], (6, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 255, 255), 1, cv2.LINE_AA)
        except Exception as e:  # noqa: BLE001
            self._fail("draw", e)

    def summary(self) -> dict:
        ms = sorted(self.process_ms)
        return {
            "type": "summary", "enabled": self.enabled, "error": self.error, "new_results": self.new_results,
            "duplicates_skipped": self.duplicates_skipped, "tick_records": self.tick_records,
            "states": dict(self.states), "diagnostics": dict(self.diagnostics),
            "max_forward_mps": self.max_forward, "max_abs_yaw_dps": self.max_abs_yaw,
            "process_ms_mean": round(sum(ms) / len(ms), 3) if ms else None,
            "process_ms_max": round(ms[-1], 3) if ms else None,
            "log_lines": self.writer.lines, "log_bytes": self.writer.bytes,
            "dropped_queue_full": self.writer.dropped_queue_full,
            "dropped_size_limit": self.writer.dropped_size_limit, "writer_error": self.writer.error,
        }

    def close(self) -> list[str]:
        summary = self.summary()
        self.writer.write(summary)
        self.writer.close()
        summary.update(log_lines=self.writer.lines, log_bytes=self.writer.bytes, writer_error=self.writer.error)
        return [f"[SHADOW] log: {self.writer.path}",
                f"[SHADOW] {json.dumps({k: v for k, v in summary.items() if k != 'type'})}"]

    def _count(self, cmd) -> None:
        self.new_results += 1
        self.last_command = cmd
        self.states[cmd.state.value] += 1
        for label in cmd.diagnostics:
            self.diagnostics[label] += 1
        self.max_forward = max(self.max_forward, cmd.forward_mps)
        self.max_abs_yaw = max(self.max_abs_yaw, abs(cmd.yaw_rate_dps))

    def _fail(self, where: str, error: Exception) -> None:
        self.enabled = False
        self.error = f"{where}: {error!r}"
        print(f"[SHADOW] disabled after error in {where}: {error!r} (camera / tracking continue)", flush=True)
        self.writer.write({"type": "error", "where": where, "error": repr(error)})


def default_log_path(directory: str) -> str:
    return os.path.join(directory, time.strftime("follow_shadow_%Y%m%d_%H%M%S.jsonl"))


def file_sha256(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def create_shadow(log_path: str, max_mb: float, frame_size: tuple[int, int], metadata: dict) -> ShadowFollower:
    writer = JsonlWriter(log_path, max_bytes=int(max_mb * 1024 * 1024))
    return ShadowFollower(writer, frame_size[0], frame_size[1], metadata=metadata)

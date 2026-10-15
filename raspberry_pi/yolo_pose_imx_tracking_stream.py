"""Live tracking view: IMX500 YOLOv8n-Pose -> reused Windows tracking/gesture pipeline -> RTP to Windows.

Pipeline:
    IMX500 (pose inference on the camera - the only inference backend)
        -> modlib AiCamera: frame.image (640x480 BGR) + Poses + new_detection
        -> NEW results only: imx500_pose_backend.poses_to_detections
                             -> ImxTrackingPipeline (BoT-SORT, TrackValidator, association,
                                pose recognition, stabilizer, TargetSelector - Windows modules, unchanged)
        -> every frame:      pipeline.check_stall (no NEW result for 2.0 s -> tracks expire, TARGET_LOST)
                             imx500_tracking_overlay (last pipeline result + status / STALE / target warning)
        -> rtp_h264_sender -> Windows GStreamer (unchanged receiver)

Nothing here moves a drone. `ImxTrackingPipeline.target_status()` is what a future
following controller would have to check; this script only displays and logs it.

Measurements logged every --log-interval: FPS / DPS, NEW / REUSED frames, tracking
time per NEW result, whole-loop time per frame and frames over the 1/frame-rate
budget, detection score histogram, primary track states, comparison-tracker
stats, CPU / RSS / MemAvailable / temperature. The summary checks that the
primary tracker was updated exactly once per NEW result.

--follow-shadow (off by default) additionally runs the forward-only follow
controller in SHADOW MODE (follow_shadow.py): its hypothetical decisions are
written to a JSONL log and, with --shadow-overlay, one line at the bottom of the
stream. It never sends any command. Without the flag the shadow code is not
even imported.

The existing yolo_pose_imx_stream.py (detection-only view) is left unchanged.

Run on the Pi (inside the yolo_pose_imx folder), after starting the Windows receiver:
    <venv>/bin/python yolo_pose_imx_tracking_stream.py --duration 60
"""

import argparse
import os
import signal
import tempfile
import time

import numpy as np
from modlib.devices import AiCamera

from imx500_pose_backend import poses_to_detections
from imx500_pose_overlay import is_stale
from imx500_tracking_overlay import draw_tracking_status, draw_tracks, status_lines, warning_text
from imx500_tracking_pipeline import (
    COMPARISON_NEW_TRACK_THRESHOLDS, NEW_TRACK_THRESH, ImxTrackingPipeline, buffer_seconds, tracker_config_with_buffer,
)
from person_tracker import DEFAULT_TRACKER_CONFIG
from rtp_h264_sender import RtpH264Sender
from system_monitor import CpuMonitor, format_sample
from yolo_pose_modlib_test import DEFAULT_MODEL, YoloPoseImx

DEFAULT_HOST = "172.20.10.8"
DEFAULT_PORT = 5000
IMAGE_SIZE = (640, 480)
SCORE_BINS = (0.25, 0.6, 0.7, 0.8, 1.01)  # log-only histogram edges for detection scores


def _stop_on_signal(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


def memory_mb() -> tuple[float | None, float | None]:
    """(this process RSS, system MemAvailable) in MB, from /proc (None if unavailable)."""
    def field(path, key):
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith(key):
                        return int(line.split()[1]) / 1024
        except OSError:
            return None
    return field("/proc/self/status", "VmRSS:"), field("/proc/meminfo", "MemAvailable:")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to packerOut.zip")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Windows receiver IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Seconds after the first inference result; 0 = until Ctrl+C")
    parser.add_argument("--frame-rate", type=int, default=16, help="Camera frame rate")
    parser.add_argument("--log-interval", type=float, default=2.0, help="Seconds between console lines")
    parser.add_argument("--new-track-thresh", type=float, default=NEW_TRACK_THRESH,
                        help="Primary BoT-SORT new_track_thresh (original Windows value 0.8)")
    parser.add_argument("--track-buffer", type=int, default=None,
                        help="Primary BoT-SORT track_buffer in updates (default: botsort.yaml's 30)")
    parser.add_argument("--compare-thresholds", type=float, nargs="*", default=list(COMPARISON_NEW_TRACK_THRESHOLDS),
                        help="Log-only comparison trackers' new_track_thresh values (no GMC)")
    parser.add_argument("--compare-track-buffers", type=int, nargs="*", default=[],
                        help="Also run comparison trackers with these track_buffer values (for every threshold)")
    # Shadow Mode (off by default): log what the follow controller WOULD command - never sends anything.
    parser.add_argument("--follow-shadow", action="store_true",
                        help="Run the follow controller in shadow mode (JSONL log only, no commands)")
    parser.add_argument("--shadow-log", default=None,
                        help="Shadow JSONL path (default: shadow_logs/follow_shadow_<time>.jsonl next to this script)")
    parser.add_argument("--shadow-log-max-mb", type=float, default=50.0, help="Stop logging past this size")
    parser.add_argument("--shadow-overlay", action="store_true",
                        help="With --follow-shadow: draw one shadow-status line at the bottom of the stream")
    return parser.parse_args()


def build_pipeline(args, config_dir: str) -> ImxTrackingPipeline:
    primary_config = (DEFAULT_TRACKER_CONFIG if args.track_buffer is None
                      else tracker_config_with_buffer(args.track_buffer, config_dir))
    comparisons = [(thr, DEFAULT_TRACKER_CONFIG) for thr in args.compare_thresholds]
    for buf in args.compare_track_buffers:
        cfg = tracker_config_with_buffer(buf, config_dir)
        comparisons += [(thr, cfg) for thr in [args.new_track_thresh, *args.compare_thresholds]]
    return ImxTrackingPipeline(new_track_thresh=args.new_track_thresh, tracker_config=primary_config,
                               comparisons=tuple(comparisons), comparison_gmc=False)


def new_interval() -> dict:
    return {"new": 0, "reused": 0, "max_age": 0.0, "track_ms": [], "loop_ms": [], "over_budget": 0}


def ms_stats(values) -> str:
    return f"{np.mean(values):5.1f}/{np.max(values):5.1f}ms" if values else "-"


def main():
    args = get_args()
    signal.signal(signal.SIGTERM, _stop_on_signal)
    signal.signal(signal.SIGHUP, _stop_on_signal)

    config_dir = tempfile.mkdtemp(prefix="botsort_cfg_")
    pipeline = build_pipeline(args, config_dir)
    variants = [pipeline.primary, *pipeline.comparisons]
    for variant in variants:
        role = "primary" if variant is pipeline.primary else "compare"
        print(f"[TRACK] {role}: {variant.label} (buffer ~{buffer_seconds(variant.track_buffer, 8.0):.2f}s at 8 results/s)",
              flush=True)

    shadow = None
    if args.follow_shadow:  # imported only when enabled: the default run loads none of the shadow code
        from follow_shadow import create_shadow, default_log_path, file_sha256
        log_path = args.shadow_log or default_log_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "shadow_logs"))
        shadow = create_shadow(log_path, args.shadow_log_max_mb, IMAGE_SIZE, metadata={
            "model": args.model, "model_sha256": file_sha256(args.model), "frame_rate": args.frame_rate,
            "new_track_thresh": args.new_track_thresh, "track_buffer": args.track_buffer})
        print(f"[SHADOW] follow controller in SHADOW MODE (no commands sent) -> {log_path}", flush=True)

    model = YoloPoseImx(args.model)
    device = AiCamera(headless=False, frame_rate=args.frame_rate, image_size=IMAGE_SIZE)
    device.deploy(model, overwrite=False)

    width, height = IMAGE_SIZE
    frame_budget_ms = 1000.0 / args.frame_rate
    sender = RtpH264Sender(width, height, args.frame_rate, args.host, args.port)
    monitor = CpuMonitor()
    counts = {"new": 0, "reused": 0, "stale": 0, "over_budget": 0, "stalls": 0}
    all_track_ms, all_loop_ms = [], []
    score_hist = np.zeros(len(SCORE_BINS) - 1, dtype=int)
    score_values: set[float] = set()
    interval = new_interval()
    last_frame = None
    start = None
    peak_rss = 0.0

    try:
        with device as stream, sender:
            print(f"[TRACK] camera started, sending RTP to {args.host}:{args.port}; waiting for the first inference result",
                  flush=True)
            camera_start = time.monotonic()
            for frame in stream:
                now = time.monotonic()
                fresh = bool(frame.new_detection)
                if start is None and fresh:
                    start = now
                    next_log = start + args.log_interval
                    print(f"[TRACK] first inference result {now - camera_start:.1f}s after camera start - test clock started",
                          flush=True)
                if start is not None and args.duration > 0 and now - start >= args.duration:
                    break
                last_frame = frame

                clean = frame.image.copy()  # BGR (480, 640, 3); BoT-SORT GMC input, never drawn on
                if clean.shape != (height, width, 3):
                    raise RuntimeError(f"unexpected frame.image shape {clean.shape}, expected {(height, width, 3)}")

                if fresh:
                    detections = poses_to_detections(frame.detections, frame.roi, width, height)
                    t0 = time.perf_counter()
                    result = pipeline.update_new(detections, clean, now)
                    track_ms = (time.perf_counter() - t0) * 1000
                    interval["track_ms"].append(track_ms)
                    all_track_ms.append(track_ms)
                    score_hist += np.histogram(detections.conf, bins=SCORE_BINS)[0]
                    score_values.update(round(float(c), 3) for c in detections.conf)
                    event = result.selection
                    if shadow is not None:
                        shadow.on_new_result(pipeline, now)  # read-only; once per NEW result
                else:
                    event = pipeline.check_stall(now)  # REUSED frame: no tracking update
                    if event is not None:
                        counts["stalls"] += 1
                        print(f"[TRACK] no NEW result for {pipeline.no_result_timeout:.1f}s - all tracks expired", flush=True)
                    if shadow is not None:
                        shadow.on_tick(now)  # what a command publisher would send now (no measurement)
                if event is not None and event.event is not None:
                    ids = ", ".join(str(i) for i in event.event_track_ids)
                    print(f"[TRACK] Selection: {event.event.value} (track {ids}) -> {event.state.value}", flush=True)

                result_age = pipeline.result_age(now)
                target = pipeline.target_status(now)
                image = clean.copy()
                draw_tracks(image, pipeline.last_result, pipeline.stabilizers.hold_time)
                elapsed = None if start is None else now - start
                draw_tracking_status(image, status_lines(frame.fps, frame.dps, pipeline.last_result, target, elapsed),
                                     warning_text(result_age, target))
                if shadow is not None and args.shadow_overlay:
                    shadow.draw(image)
                sender.send(image)

                loop_ms = (time.monotonic() - now) * 1000
                interval["loop_ms"].append(loop_ms)
                all_loop_ms.append(loop_ms)
                over = loop_ms > frame_budget_ms
                interval["over_budget"] += over
                counts["over_budget"] += over
                kind = "new" if fresh else "reused"
                counts[kind] += 1
                interval[kind] += 1
                interval["max_age"] = max(interval["max_age"], result_age or 0.0)
                counts["stale"] += bool(result_age is not None and is_stale(result_age))

                if start is not None and now >= next_log:
                    next_log = now + args.log_interval
                    usage = monitor.sample({"self": os.getpid(), "ffmpeg": sender.pid})
                    rss, avail = memory_mb()
                    peak_rss = max(peak_rss, rss or 0.0)
                    tracks = ",".join(f"{v.tracked_person.track_id}:{v.state.value[:4]}"
                                      for v in (pipeline.last_result.validated_tracks if pipeline.last_result else []))
                    print(f"[{now - start:6.1f}s] FPS={frame.fps:4.1f} DPS={frame.dps:4.1f} "
                          f"new={interval['new']} reused={interval['reused']} max_age={interval['max_age']:.2f}s "
                          f"track(mean/max)={ms_stats(interval['track_ms'])} loop={ms_stats(interval['loop_ms'])} "
                          f"over_budget={interval['over_budget']} "
                          f"det={len(pipeline.last_result.detections) if pipeline.last_result else 0} "
                          f"tracks=[{tracks}] state={target.state.value} target={target.target_id} usable={target.usable} "
                          f"sent={sender.sent} dropped={sender.dropped} {format_sample(usage)} "
                          f"rss={rss:.0f}MB avail={avail:.0f}MB", flush=True)
                    for variant in variants:
                        print(f"           {variant.label:24s} {variant.stats.summary()}", flush=True)
                    interval = new_interval()
    except KeyboardInterrupt as e:
        print(f"[TRACK] stopping ({e or 'Ctrl+C'})", flush=True)
    finally:
        if shadow is not None:
            for line in shadow.close():  # flushes the JSONL writer
                print(line, flush=True)

    print("[TRACK] --- Summary ---")
    print(f"[TRACK] frames: new={counts['new']} reused={counts['reused']} stale(>0.5s)={counts['stale']}, "
          f"sent={sender.sent} dropped={sender.dropped} ffmpeg restarts={sender.restarts}")
    expected = counts["new"] + counts["stalls"]  # one validator update per NEW result + one per stall expiry
    print(f"[TRACK] primary validator updates={pipeline.primary.stats.updates}, NEW results={counts['new']}, "
          f"stall expiries={counts['stalls']} -> {'OK' if pipeline.primary.stats.updates == expected else 'MISMATCH'}")
    print(f"[TRACK] tracking time per NEW result: {ms_stats(all_track_ms)} (mean/max), "
          f"p95={np.percentile(all_track_ms, 95):.1f}ms" if all_track_ms else "[TRACK] no NEW results")
    print(f"[TRACK] loop time per frame: {ms_stats(all_loop_ms)}, frames over {frame_budget_ms:.1f}ms budget: "
          f"{counts['over_budget']}")
    labels = [f"[{lo:.2f},{hi:.2f})" for lo, hi in zip(SCORE_BINS[:-1], SCORE_BINS[1:])]
    print(f"[TRACK] detection scores (>=0.25): " + " ".join(f"{l}:{n}" for l, n in zip(labels, score_hist)))
    print(f"[TRACK] distinct detection score values: {sorted(score_values)[:30]}")
    print(f"[TRACK] peak RSS: {peak_rss:.0f}MB")
    for variant in variants:
        role = "primary" if variant is pipeline.primary else "compare"
        print(f"[TRACK] {role} {variant.label:24s} {variant.stats.summary()}")
    if last_frame is not None:
        print(f"[TRACK] last reported FPS={last_frame.fps:.1f} DPS={last_frame.dps:.1f}")


if __name__ == "__main__":
    main()

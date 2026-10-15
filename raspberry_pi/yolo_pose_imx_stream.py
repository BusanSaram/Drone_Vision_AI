"""Live view: YOLOv8n-Pose on the IMX500 -> annotated video -> Windows over RTP.

Pipeline:
    IMX500 (pose inference on the camera)
        -> modlib AiCamera: frame.image (640x480 BGR) + frame.detections (Poses)
        -> imx500_pose_overlay: boxes, scores, 17 keypoints, COCO skeleton, FPS/DPS/people, STALE warning
        -> rtp_h264_sender: ffmpeg libx264 -> RTP -> Windows GStreamer (unchanged receiver)

Windows only decodes and displays - no YOLO runs there.

NEW vs REUSED: the camera delivers ~16 FPS but the network produces ~8 results/s,
so modlib repeats the last result on frames without a new output tensor
(`frame.new_detection == False`). That alternation is normal and is only counted
in the console log, not shown on the video. The video instead tracks the age of
the newest result (monotonic clock, reset on every NEW frame) and shows STALE
only when no new result has arrived for more than STALE_AFTER_S seconds.

Model, packaging and deploy are the same as yolo_pose_modlib_test.py (imported
from there, unchanged).

Run on the Pi (inside /home/pi/yolo_pose_imx), after starting the Windows receiver:
    .venv/bin/python yolo_pose_imx_stream.py --duration 60
"""

import argparse
import os
import signal
import time

from modlib.devices import AiCamera

from imx500_pose_overlay import STALE_AFTER_S, draw_poses, draw_status, is_stale, to_pixel_poses
from rtp_h264_sender import RtpH264Sender
from system_monitor import CpuMonitor, format_sample
from yolo_pose_modlib_test import DEFAULT_MODEL, YoloPoseImx

DEFAULT_HOST = "172.20.10.8"  # Windows PC, same target as the HigherHRNet streaming
DEFAULT_PORT = 5000
IMAGE_SIZE = (640, 480)  # modlib default VGA stream: full sensor field of view, 4:3


def _stop_on_signal(signum, _frame):
    # SIGTERM / SIGHUP (e.g. the SSH session drops) take the same clean-shutdown
    # path as Ctrl+C: the `with AiCamera` block stops the camera exactly once.
    raise KeyboardInterrupt(f"signal {signum}")


def new_interval() -> dict:
    """Counters for one console log interval."""
    return {"new": 0, "reused": 0, "people_new": 0, "stale": 0, "max_age": 0.0}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to packerOut.zip")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Windows receiver IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds after camera start; 0 = until Ctrl+C")
    parser.add_argument("--frame-rate", type=int, default=16, help="Camera frame rate (same as the IMX500 test)")
    parser.add_argument("--log-interval", type=float, default=2.0, help="Seconds between console lines")
    return parser.parse_args()


def main():
    args = get_args()
    signal.signal(signal.SIGTERM, _stop_on_signal)
    signal.signal(signal.SIGHUP, _stop_on_signal)

    model = YoloPoseImx(args.model)
    device = AiCamera(headless=False, frame_rate=args.frame_rate, image_size=IMAGE_SIZE)
    device.deploy(model, overwrite=False)  # reuses pack/network.rpk from the first test

    width, height = IMAGE_SIZE
    sender = RtpH264Sender(width, height, args.frame_rate, args.host, args.port)
    monitor = CpuMonitor()
    totals = {"new": 0, "reused": 0, "stale": 0, "max_age": 0.0}
    interval = new_interval()
    last_frame = None
    last_result_time = None  # time.monotonic() of the newest NEW inference result

    try:
        with device as stream, sender:
            print(f"[STREAM] camera started, sending RTP to {args.host}:{args.port}; "
                  f"waiting for the first inference result (firmware may still be loading)", flush=True)
            camera_start = time.monotonic()
            start = None  # test clock: starts at the first NEW inference result, not during firmware loading
            for frame in stream:
                now = time.monotonic()
                if start is None and frame.new_detection:
                    start = now
                    next_log = start + args.log_interval
                    print(f"[STREAM] first inference result {now - camera_start:.1f}s after camera start - "
                          f"test clock started ({'until Ctrl+C' if args.duration <= 0 else f'{args.duration:.0f}s'})",
                          flush=True)
                if start is not None and args.duration > 0 and now - start >= args.duration:
                    break
                last_frame = frame

                image = frame.image.copy()  # BGR, (480, 640, 3), row 0 = top of the scene
                if image.shape != (height, width, 3):
                    raise RuntimeError(f"unexpected frame.image shape {image.shape}, expected {(height, width, 3)}")

                poses = frame.detections
                poses.compensate_for_roi(frame.roi)  # no-op for the full-FOV ROI; same step as modlib's Annotator
                pixel_poses = to_pixel_poses(poses.bbox, poses.keypoints, poses.keypoint_scores,
                                             poses.confidence, width, height)
                fresh = bool(frame.new_detection)
                if fresh or last_result_time is None:  # modlib's first frame always carries a result
                    last_result_time = now
                result_age = now - last_result_time
                stale = is_stale(result_age)

                draw_poses(image, pixel_poses)
                draw_status(image, frame.fps, frame.dps, len(pixel_poses), result_age)
                sender.send(image)

                kind = "new" if fresh else "reused"
                totals[kind] += 1
                totals["stale"] += stale
                totals["max_age"] = max(totals["max_age"], result_age)
                interval[kind] += 1
                interval["stale"] += stale
                interval["max_age"] = max(interval["max_age"], result_age)
                if fresh:
                    interval["people_new"] += len(pixel_poses)

                if start is not None and now >= next_log:
                    next_log = now + args.log_interval
                    avg_people = interval["people_new"] / interval["new"] if interval["new"] else 0.0
                    usage = monitor.sample({"self": os.getpid(), "ffmpeg": sender.pid})
                    print(
                        f"[{now - start:6.1f}s] FPS={frame.fps:4.1f} DPS={frame.dps:4.1f} "
                        f"frames new={interval['new']} reused={interval['reused']} "
                        f"max_age={interval['max_age']:.2f}s stale={interval['stale']} "
                        f"avg_people(new)={avg_people:.2f} sent={sender.sent} dropped={sender.dropped} "
                        f"restarts={sender.restarts} {format_sample(usage)}",
                        flush=True,
                    )
                    interval = new_interval()
    except KeyboardInterrupt as e:
        print(f"[STREAM] stopping ({e or 'Ctrl+C'})", flush=True)

    print("[STREAM] --- Summary ---")
    print(f"[STREAM] frames: new={totals['new']} reused={totals['reused']}, "
          f"sent={sender.sent} dropped={sender.dropped} ffmpeg restarts={sender.restarts}")
    print(f"[STREAM] result age: max={totals['max_age']:.2f}s, "
          f"frames shown STALE (age > {STALE_AFTER_S}s): {totals['stale']}")
    if last_frame is not None:
        print(f"[STREAM] last reported FPS={last_frame.fps:.1f} DPS={last_frame.dps:.1f}")


if __name__ == "__main__":
    main()

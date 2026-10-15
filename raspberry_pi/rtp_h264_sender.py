"""Send annotated BGR frames to Windows as H.264 over RTP, using an ffmpeg subprocess.

Same wire format as the earlier HigherHRNet streaming (picamera2 FfmpegOutput
"-f rtp rtp://172.20.10.8:5000"), so the existing Windows GStreamer receiver
works unchanged:
    udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000"
    ! rtpjitterbuffer latency=100 drop-on-latency=true ! rtph264depay ! h264parse
    ! avdec_h264 ! videoconvert ! d3d11videosink sync=false

picamera2's encoder cannot be reused because modlib drives the camera through
libcamera directly, not through picamera2. Encoding is libx264 on the Pi CPU
(video only - pose inference stays on the IMX500).

RTP is UDP: a missing or restarted receiver does not affect the sender. If
ffmpeg itself exits (e.g. the network interface goes away), `send()` drops
frames and restarts it at most once per `restart_interval` seconds, so the
camera loop keeps running.
"""

import subprocess
import time

import numpy as np

DEFAULT_BITRATE = "4M"  # same 4 Mbit/s as the earlier H264Encoder(4000000)
RTP_PAYLOAD_TYPE = 96  # matches payload=96 in the Windows receiver caps


def build_ffmpeg_command(width: int, height: int, fps: int, host: str, port: int,
                         bitrate: str = DEFAULT_BITRATE) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        # input: raw BGR frames on stdin
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        # low-latency H.264: no B-frames, one keyframe per second so a (re)started receiver recovers fast
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-bf", "0", "-g", str(fps), "-b:v", bitrate, "-pix_fmt", "yuv420p",
        "-f", "rtp", "-payload_type", str(RTP_PAYLOAD_TYPE), f"rtp://{host}:{port}",
    ]


class RtpH264Sender:
    def __init__(self, width: int, height: int, fps: int, host: str, port: int,
                 restart_interval: float = 2.0, popen=subprocess.Popen, clock=time.monotonic):
        self.width = width
        self.height = height
        self.command = build_ffmpeg_command(width, height, fps, host, port)
        self.restart_interval = restart_interval
        self._popen = popen
        self._clock = clock
        self._proc = None
        self._last_start = None
        self.sent = 0
        self.dropped = 0
        self.restarts = 0

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self.alive else None

    def start(self) -> None:
        # stdout gets the SDP description ffmpeg prints for RTP - not needed (receiver caps are fixed).
        self._proc = self._popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        self._last_start = self._clock()

    def send(self, frame: np.ndarray) -> bool:
        """Write one BGR frame. Returns False if it was dropped (ffmpeg down or pipe broken)."""
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 frame of shape {(self.height, self.width, 3)}, got {frame.dtype} {frame.shape}")

        if not self.alive:
            if self._last_start is not None and self._clock() - self._last_start < self.restart_interval:
                self.dropped += 1
                return False
            self._stop_process()
            print("[STREAM] ffmpeg not running - restarting", flush=True)
            self.start()
            self.restarts += 1

        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            print(f"[STREAM] ffmpeg pipe error ({type(e).__name__}) - frame dropped", flush=True)
            self._stop_process()
            self.dropped += 1
            return False
        self.sent += 1
        return True

    def close(self) -> None:
        self._stop_process()

    def _stop_process(self) -> None:
        """Close stdin so ffmpeg flushes and exits; escalate to terminate/kill if it hangs."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

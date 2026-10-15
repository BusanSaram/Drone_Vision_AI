"""Unit tests for the RTP/H.264 sender (raspberry_pi/rtp_h264_sender.py).

A fake Popen stands in for ffmpeg - nothing is encoded or sent.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "raspberry_pi"))

import rtp_h264_sender as rs  # noqa: E402

W, H, FPS = 640, 480, 16


class FakeStdin:
    def __init__(self, broken=False):
        self.broken = broken
        self.written = 0
        self.closed = False

    def write(self, data):
        if self.broken:
            raise BrokenPipeError()
        self.written += len(data)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeProc:
    def __init__(self, broken=False):
        self.stdin = FakeStdin(broken)
        self.returncode = None
        self.pid = 1234

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


class FakePopen:
    def __init__(self, broken_first=False):
        self.procs = []
        self.broken_first = broken_first

    def __call__(self, cmd, **kwargs):
        proc = FakeProc(broken=self.broken_first and not self.procs)
        self.procs.append(proc)
        return proc


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


class CommandTest(unittest.TestCase):
    def test_matches_existing_windows_receiver(self):
        cmd = rs.build_ffmpeg_command(W, H, FPS, "172.20.10.8", 5000)
        self.assertEqual(cmd[-1], "rtp://172.20.10.8:5000")
        self.assertEqual(cmd[cmd.index("-payload_type") + 1], "96")
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "libx264")
        self.assertEqual(cmd[cmd.index("-s") + 1], "640x480")
        self.assertEqual(cmd[cmd.index("-pix_fmt") + 1], "bgr24")  # input: OpenCV BGR
        self.assertIn("zerolatency", cmd)
        self.assertEqual(cmd[cmd.index("-bf") + 1], "0")
        self.assertEqual(cmd[cmd.index("-g") + 1], str(FPS))


class SenderTest(unittest.TestCase):
    def test_sends_whole_frames(self):
        popen = FakePopen()
        with rs.RtpH264Sender(W, H, FPS, "h", 5000, popen=popen) as sender:
            self.assertTrue(sender.send(frame()))
            self.assertTrue(sender.send(frame()))
            self.assertEqual(popen.procs[0].stdin.written, 2 * W * H * 3)
            self.assertEqual(sender.pid, 1234)
        self.assertTrue(popen.procs[0].stdin.closed)  # clean shutdown closes ffmpeg's stdin
        self.assertEqual(sender.sent, 2)

    def test_rejects_wrong_frame_shape(self):
        with rs.RtpH264Sender(W, H, FPS, "h", 5000, popen=FakePopen()) as sender:
            with self.assertRaises(ValueError):
                sender.send(np.zeros((640, 480, 3), dtype=np.uint8))  # transposed

    def test_broken_pipe_drops_frame_then_restarts_after_interval(self):
        popen, clock = FakePopen(broken_first=True), FakeClock()
        sender = rs.RtpH264Sender(W, H, FPS, "h", 5000, restart_interval=2.0, popen=popen, clock=clock)
        sender.start()

        self.assertFalse(sender.send(frame()))  # pipe breaks
        clock.t = 1.0
        self.assertFalse(sender.send(frame()))  # within restart interval: dropped, no restart
        self.assertEqual(len(popen.procs), 1)

        clock.t = 2.5
        self.assertTrue(sender.send(frame()))  # restarted ffmpeg takes the frame
        self.assertEqual(len(popen.procs), 2)
        self.assertEqual((sender.sent, sender.dropped, sender.restarts), (1, 2, 1))
        sender.close()

    def test_ffmpeg_exit_is_detected(self):
        popen, clock = FakePopen(), FakeClock()
        sender = rs.RtpH264Sender(W, H, FPS, "h", 5000, restart_interval=2.0, popen=popen, clock=clock)
        sender.start()
        popen.procs[0].returncode = 1  # ffmpeg died on its own
        self.assertIsNone(sender.pid)
        clock.t = 3.0
        self.assertTrue(sender.send(frame()))
        self.assertEqual(sender.restarts, 1)
        sender.close()

    def test_close_is_idempotent(self):
        sender = rs.RtpH264Sender(W, H, FPS, "h", 5000, popen=FakePopen())
        sender.start()
        sender.close()
        sender.close()
        self.assertFalse(sender.alive)


if __name__ == "__main__":
    unittest.main()

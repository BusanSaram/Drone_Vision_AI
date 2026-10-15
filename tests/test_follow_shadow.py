"""Unit tests for Shadow Mode (raspberry_pi/follow_shadow.py), the record format and the offline replay.

A fake pipeline stands in for ImxTrackingPipeline; no modlib, no camera, nothing is sent anywhere.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import ast
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "diagnostics"))
sys.path.insert(0, str(ROOT / "raspberry_pi"))

import follow_shadow as fs  # noqa: E402
import follow_shadow_replay as replay  # noqa: E402
from follow_controller import FollowParams, FollowState  # noqa: E402
from follow_record import input_from_record, input_to_record  # noqa: E402
from target_selector import SelectionState  # noqa: E402
from test_follow_controller import H, W, person  # noqa: E402

DT = 0.125


class FakePipeline:
    """Just the attributes the shadow code reads: target_status(now) and last_result."""

    def __init__(self):
        self.last_result = None
        self.status = SimpleNamespace(state=SelectionState.SEARCHING, target_id=None, usable=False)

    def new_result(self, t, h=0.35, cx=0.5, pose="standing", target=3, following=True, visible=True, others=()):
        bbox, kp = person(h, cx, pose)
        boxes, kps, confs = [bbox], [kp], [0.816]
        for ob in others:
            obox, okp = person(*ob)
            boxes.append(obox), kps.append(okp), confs.append(0.73)
        detections = SimpleNamespace(xyxy=np.array(boxes), conf=np.array(confs), keypoints=np.stack(kps))
        detections.__len__ = None
        tracked = [SimpleNamespace(track_id=target)] if visible else []
        self.last_result = SimpleNamespace(timestamp=t, tracked_people=tracked,
                                           track_to_detection={target: 0} if visible else {},
                                           detections=_Dets(detections))
        self.status = SimpleNamespace(state=SelectionState.FOLLOWING if following else SelectionState.SEARCHING,
                                      target_id=target if following else None, usable=following and visible)


class _Dets:
    def __init__(self, d):
        self.xyxy, self.conf, self.keypoints = d.xyxy, d.conf, d.keypoints

    def __len__(self):
        return len(self.xyxy)


FakePipeline.target_status = lambda self, now: self.status


class MemoryWriter:
    """Stand-in for JsonlWriter that keeps records in memory (JSON round-tripped, like the real file)."""

    def __init__(self):
        self.records, self.lines, self.bytes = [], 0, 0
        self.path, self.error = "<memory>", None
        self.dropped_queue_full = self.dropped_size_limit = 0

    def write(self, record):
        self.records.append(json.loads(json.dumps(record, allow_nan=False)))
        self.lines += 1
        return True

    def close(self):
        pass


def shadow_with_memory():
    writer = MemoryWriter()
    return fs.ShadowFollower(writer, W, H, metadata={"test": True}), writer


class ShadowFollowerTest(unittest.TestCase):
    def run_session(self, n=16, **kw):
        shadow, writer = shadow_with_memory()
        pipe = FakePipeline()
        cmds = []
        for i in range(n):
            pipe.new_result(i * DT, **kw)
            cmds.append(shadow.on_new_result(pipe, i * DT))
        return shadow, writer, pipe, cmds

    def test_each_new_result_evaluated_once(self):
        shadow, writer, pipe, cmds = self.run_session(8)
        self.assertIsNone(shadow.on_new_result(pipe, 7 * DT + 0.01))  # same result again
        self.assertIsNone(shadow.on_new_result(pipe, 7 * DT + 0.02))
        self.assertEqual(shadow.new_results, 8)
        self.assertEqual(shadow.duplicates_skipped, 2)
        self.assertEqual(sum(r["type"] == "new" for r in writer.records), 8)
        self.assertEqual(writer.records[0]["type"], "meta")

    def test_records_hold_inputs_body_metrics_and_decision(self):
        shadow, writer, _, cmds = self.run_session(12, h=0.35)
        rec = writer.records[-1]
        self.assertEqual(rec["ctl"]["state"], "ACTIVE")
        self.assertGreater(rec["ctl"]["fwd"], 0)
        self.assertEqual(len(rec["input"]["kp"]), 17)
        self.assertEqual(rec["input"]["meas_id"], 3)
        for key in ("lower_conf", "upper_conf", "leg_ratio", "thigh_ratio", "shoulder_ratio", "span", "tilt_deg"):
            self.assertIsNotNone(rec["body"][key], key)
        self.assertEqual(rec["body"]["reason"], "ok")

    def test_replay_reproduces_logged_decisions_exactly(self):
        shadow, writer = shadow_with_memory()
        pipe = FakePipeline()
        script = [dict(h=0.35)] * 10 + [dict(h=0.35, pose="crouch")] + [dict(h=0.36)] * 6 + \
                 [dict(h=0.36, visible=False)] * 2 + [dict(h=0.37, others=[(0.4, 0.55)])] + [dict(h=0.37)] * 8 + \
                 [dict(h=0.37, following=False)] * 2
        for i, kw in enumerate(script):
            pipe.new_result(i * DT + 0.0004 * i, **kw)  # irregular timestamps
            shadow.on_new_result(pipe, i * DT + 0.0004 * i)
        records = [r for r in writer.records if r["type"] != "meta"]
        meta = writer.records[0]
        pairs = replay.replay(records, replay.params_from(meta, []))
        self.assertEqual(len(pairs), len(script))
        self.assertEqual([r for r, c in pairs if not replay.same_decision(r["ctl"], c)], [])
        states = {r["ctl"]["state"] for r in records}
        self.assertTrue({"ACTIVE", "HOLD_INVALID", "HOLD_TARGET_LOST", "INACTIVE"} <= states)
        self.assertIn("OCCLUSION_SUSPECTED", {d for r in records for d in r["ctl"]["diag"]})

    def test_replay_with_override_changes_decisions(self):
        _, writer, _, _ = self.run_session(12, h=0.35)
        meta, records = writer.records[0], writer.records[1:]
        pairs = replay.replay(records, replay.params_from(meta, ["resume_valid_count=20"]))
        self.assertTrue(all(c.forward_mps == 0 for _, c in pairs))
        self.assertEqual(replay.params_from(meta, ["resume_valid_count=20"]).resume_valid_count, 20)

    def test_inference_stop_holds_via_tick(self):
        shadow, writer, pipe, cmds = self.run_session(12, h=0.35)
        self.assertGreater(cmds[-1].forward_mps, 0)
        last = 11 * DT
        self.assertGreater(shadow.on_tick(last + 0.06).forward_mps, 0)  # normal REUSED frame: nothing logged
        self.assertFalse(any(r["type"] == "tick" for r in writer.records))
        # In shadow mode update() runs on NEW results only, so "no update for 0.3 s" and "result older than
        # 0.3 s" coincide; the timeout is checked first. (A stale result alone is covered by test_11b.)
        timeout = shadow.on_tick(last + 0.31)
        self.assertIs(timeout.state, FollowState.HOLD_TIMEOUT)
        self.assertEqual((timeout.forward_mps, timeout.yaw_rate_dps), (0.0, 0.0))
        shadow.on_tick(last + 0.6)
        shadow.on_tick(last + 0.7)  # same override: not logged again
        ticks = [r for r in writer.records if r["type"] == "tick"]
        self.assertEqual([t["ctl"]["state"] for t in ticks], ["HOLD_TIMEOUT"])
        self.assertTrue(all(t["ctl"]["fwd"] == 0 and t["ctl"]["yaw"] == 0 for t in ticks))
        pipe.new_result(last + 1.0)  # results resume: revalidation first
        self.assertIn("revalidating", shadow.on_new_result(pipe, last + 1.0).reason)

    def test_errors_disable_shadow_only(self):
        shadow, writer = shadow_with_memory()
        broken = SimpleNamespace(last_result=SimpleNamespace(timestamp=1.0),
                                 target_status=lambda now: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertIsNone(shadow.on_new_result(broken, 1.0))  # no exception escapes
        self.assertFalse(shadow.enabled)
        self.assertIn("boom", shadow.error)
        self.assertIsNone(shadow.on_tick(1.1))
        self.assertEqual(writer.records[-1]["type"], "error")
        image = np.zeros((H, W, 3), np.uint8)
        shadow.draw(image)  # disabled: draws nothing
        self.assertFalse(image.any())

    def test_never_backward_through_shadow(self):
        import random
        rng = random.Random(3)
        shadow, writer = shadow_with_memory()
        pipe = FakePipeline()
        t = 0.0
        for _ in range(600):
            t += rng.choice([DT, DT, DT, 0.3, 0.6])
            pipe.new_result(t, h=rng.uniform(0.15, 0.95), cx=rng.uniform(0.15, 0.85),
                            pose=rng.choice(["standing"] * 5 + ["crouch", "sitting", "sideways"]),
                            following=rng.random() > 0.05, visible=rng.random() > 0.05,
                            others=[(0.4, rng.uniform(0.1, 0.9))] if rng.random() < 0.2 else [])
            shadow.on_new_result(pipe, t)
            shadow.on_tick(t + rng.choice([0.06, 0.2, 0.4]))
        fwd = [r["ctl"]["fwd"] for r in writer.records if "ctl" in r]
        self.assertTrue(fwd)
        self.assertGreaterEqual(min(fwd), 0.0)
        self.assertLessEqual(max(fwd), FollowParams().max_forward_mps)

    def test_overlay_draws_only_bottom_band(self):
        shadow, _, _, _ = self.run_session(12, h=0.35, cx=0.7)
        text = shadow.overlay_text()
        self.assertIn("ACTIVE", text)
        self.assertIn("FWD", text)
        self.assertIn("yaw=R", text)
        image = np.full((H, W, 3), 128, np.uint8)
        shadow.draw(image)
        self.assertTrue((image[: H - fs.OVERLAY_HEIGHT] == 128).all())  # rest of the frame untouched
        self.assertFalse((image[H - fs.OVERLAY_HEIGHT:] == 128).all())

    def test_overlay_marks_suspected_occlusion(self):
        shadow, writer = shadow_with_memory()
        pipe = FakePipeline()
        for i in range(12):
            pipe.new_result(i * DT)
            shadow.on_new_result(pipe, i * DT)
        pipe.new_result(12 * DT, others=[(0.35, 0.55)])
        cmd = shadow.on_new_result(pipe, 12 * DT)
        self.assertEqual(cmd.diagnostics, ("OCCLUSION_SUSPECTED",))
        self.assertIn("OCC?", shadow.overlay_text())
        self.assertIn("HOLD", shadow.overlay_text())


class RecordTest(unittest.TestCase):
    def test_round_trip_is_stable(self):
        pipe = FakePipeline()
        pipe.new_result(1.2345678, others=[(0.3, 0.8)])
        from imx500_follow_adapter import follow_input_from_pipeline
        rec = input_to_record(follow_input_from_pipeline(pipe, 1.2345678, W, H))
        again = input_to_record(input_from_record(json.loads(json.dumps(rec))))
        self.assertEqual(rec, again)
        self.assertEqual(rec["now"], 1.2345678)  # timestamps keep full precision
        self.assertEqual(len(rec["others"]), 1)


class JsonlWriterTest(unittest.TestCase):
    def test_writes_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "sub" / "log.jsonl")
            w = fs.JsonlWriter(path, max_bytes=1_000_000)
            for i in range(50):
                self.assertTrue(w.write({"type": "new", "i": i}))
            w.close()
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(l)["i"] for l in lines], list(range(50)))
            self.assertEqual(w.lines, 50)

    def test_write_never_blocks_when_disk_is_slow(self):
        gate = threading.Event()

        class SlowFile:
            def write(self, s):
                gate.wait(5)

            def flush(self):
                pass

            def close(self):
                pass

        w = fs.JsonlWriter("unused", max_bytes=10_000_000, queue_size=10, opener=lambda *a, **k: SlowFile())
        t0 = time.perf_counter()
        results = [w.write({"type": "new", "i": i}) for i in range(200)]
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.5)  # the camera loop is never held up
        self.assertGreater(w.dropped_queue_full, 0)
        self.assertIn(False, results)
        gate.set()
        w.close()

    def test_size_limit_stops_logging_but_keeps_summary(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "log.jsonl")
            w = fs.JsonlWriter(path, max_bytes=fs.SUMMARY_RESERVE_BYTES + 2000)
            for i in range(500):
                w.write({"type": "new", "pad": "x" * 50, "i": i})
            w.write({"type": "summary", "ok": True})
            w.close()
            self.assertGreater(w.dropped_size_limit, 0)
            self.assertLessEqual(Path(path).stat().st_size, fs.SUMMARY_RESERVE_BYTES + 2000)
            self.assertEqual(json.loads(Path(path).read_text().splitlines()[-1])["type"], "summary")

    def test_unwritable_path_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            blocker = Path(d) / "file"
            blocker.write_text("x")
            w = fs.JsonlWriter(str(blocker / "log.jsonl"), max_bytes=1000)  # parent is a file
            time.sleep(0.2)
            self.assertIsNotNone(w.error)
            self.assertFalse(w.write({"type": "new"}))
            w.close()

    def test_unserializable_record_is_dropped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "log.jsonl")
            w = fs.JsonlWriter(path, max_bytes=100_000)
            w.write({"type": "new", "bad": float("nan")})
            w.write({"type": "new", "good": 1})
            w.close()
            self.assertEqual(w.dropped_unserializable, 1)
            self.assertIsNone(w.error)
            self.assertEqual(len(Path(path).read_text().splitlines()), 1)


FORBIDDEN_MODULES = {"pymavlink", "mavsdk", "dronekit", "serial", "socket", "socketserver", "RPi", "gpiozero",
                     "smbus", "smbus2", "spidev", "pigpio", "lgpio", "can", "usb", "requests", "urllib", "http",
                     "subprocess"}
SHADOW_FILES = [ROOT / "raspberry_pi" / "follow_shadow.py", ROOT / "src" / "follow_controller.py",
                ROOT / "src" / "follow_record.py", ROOT / "raspberry_pi" / "imx500_follow_adapter.py",
                ROOT / "src" / "diagnostics" / "follow_shadow_replay.py"]
STREAM = ROOT / "raspberry_pi" / "yolo_pose_imx_tracking_stream.py"


def imported_modules(path: Path, top_level_only=False) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if top_level_only else ast.walk(tree)
    names = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


class NoFlightCommandPathTest(unittest.TestCase):
    def test_shadow_code_has_no_flight_or_network_io(self):
        for path in SHADOW_FILES:
            bad = imported_modules(path) & FORBIDDEN_MODULES
            self.assertEqual(bad, set(), f"{path.name} imports {bad}")

    def test_stream_app_adds_no_flight_io(self):
        # the app already used subprocess (ffmpeg via rtp_h264_sender) - only flight / serial / GPIO are checked here
        bad = imported_modules(STREAM) & (FORBIDDEN_MODULES - {"subprocess", "socket"})
        self.assertEqual(bad, set())

    def test_shadow_disabled_by_default_and_imported_lazily(self):
        top = imported_modules(STREAM, top_level_only=True)
        self.assertNotIn("follow_shadow", top)
        self.assertNotIn("follow_controller", top)
        source = STREAM.read_text(encoding="utf-8")
        tree = ast.parse(source)
        flags = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
                name = getattr(node.args[0], "value", None)
                kw = {k.arg: getattr(k.value, "value", None) for k in node.keywords}
                flags[name] = kw
        self.assertEqual(flags["--follow-shadow"].get("action"), "store_true")  # default False
        self.assertEqual(flags["--shadow-overlay"].get("action"), "store_true")
        self.assertIn("if args.follow_shadow:", source)
        self.assertEqual(source.count("shadow.on_new_result("), 1)  # exactly one call site, in the NEW branch


if __name__ == "__main__":
    unittest.main()

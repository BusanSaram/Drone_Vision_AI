"""Unit tests for src/follow_controller.py (forward-only person following) and the IMX500 adapter.

Synthetic standing / crouching / sitting / sideways stick figures, AI results every
0.125 s (the IMX500's ~8 results/s). Nothing here talks to a flight controller.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "raspberry_pi"))

from coco_keypoints import (  # noqa: E402
    LEFT_ANKLE, LEFT_EAR, LEFT_EYE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, NOSE, RIGHT_ANKLE, RIGHT_EAR, RIGHT_EYE,
    RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER,
)
from coco_pose_recognizer import BodyPose  # noqa: E402
from follow_controller import (  # noqa: E402
    FollowCommand, FollowController, FollowInput, FollowParams, FollowState, validate_full_body,
)
from imx500_follow_adapter import follow_input_from_pipeline  # noqa: E402
from target_selector import SelectionEvent, SelectionState, TargetSelector  # noqa: E402

W, H = 640, 480
DT = 0.125
P = FollowParams()


def person(h=0.35, cx=0.5, pose="standing", conf=0.9, top=None):
    """(bbox, keypoints) for a person whose box is h * H tall, centered at cx * W."""
    hp = h * H
    x = cx * W
    y0 = (H - hp) / 2 if top is None else top
    # vertical positions as fractions of box height
    rows = {"standing": dict(shoulder=0.20, hip=0.50, knee=0.74, ankle=0.96, sw=0.12),
            "crouch": dict(shoulder=0.20, hip=0.55, knee=0.62, ankle=0.90, sw=0.12),
            "sitting": dict(shoulder=0.20, hip=0.50, knee=0.52, ankle=0.95, sw=0.12),
            "sideways": dict(shoulder=0.20, hip=0.50, knee=0.74, ankle=0.96, sw=0.03)}[pose]
    kp = np.zeros((17, 3))
    pts = {NOSE: (0, 0.06), LEFT_EYE: (0.02, 0.05), RIGHT_EYE: (-0.02, 0.05), LEFT_EAR: (0.04, 0.055),
           RIGHT_EAR: (-0.04, 0.055), LEFT_SHOULDER: (rows["sw"], rows["shoulder"]),
           RIGHT_SHOULDER: (-rows["sw"], rows["shoulder"]), LEFT_HIP: (0.08, rows["hip"]), RIGHT_HIP: (-0.08, rows["hip"]),
           LEFT_KNEE: (0.07, rows["knee"]), RIGHT_KNEE: (-0.07, rows["knee"]),
           LEFT_ANKLE: (0.07, rows["ankle"]), RIGHT_ANKLE: (-0.07, rows["ankle"])}
    for i, (dx, fy) in pts.items():
        kp[i] = (x + dx * hp, y0 + fy * hp, conf)
    bbox = (x - 0.15 * hp, y0, x + 0.15 * hp, y0 + hp)
    return bbox, kp


class Sim:
    """Feeds the controller one AI result every DT seconds (input built 0.02 s after the result)."""

    def __init__(self, params=P):
        self.ctl = FollowController(params)
        self.t = 0.0
        self.commands = []

    def step(self, h=0.35, cx=0.5, pose="standing", target=3, meas_id=None, state=SelectionState.FOLLOWING,
             observed=True, usable=True, conf=0.816, age=0.02, keypoints=None, bbox=None):
        bb, kp = person(h, cx, pose)
        inp = FollowInput(now=self.t + age, result_timestamp=self.t, selection_state=state, target_id=target,
                          usable=usable, target_observed=observed,
                          measurement_track_id=target if meas_id is None else meas_id,
                          bbox=bbox or bb, confidence=conf, keypoints=kp if keypoints is None else keypoints,
                          frame_width=W, frame_height=H)
        cmd = self.ctl.update(inp)
        self.commands.append(cmd)
        self.t += DT
        return cmd

    def run(self, n, **kw):
        return [self.step(**kw) for _ in range(n)][-1]


class ForwardOnlyTest(unittest.TestCase):
    def test_01_distant_standing_person_moves_forward(self):
        sim = Sim()
        cmd = sim.run(12, h=0.35)
        self.assertIs(cmd.state, FollowState.ACTIVE)
        self.assertGreater(cmd.forward_mps, 0)
        self.assertLessEqual(cmd.forward_mps, P.max_forward_mps)
        self.assertFalse(cmd.hold)

    def test_02_desired_height_holds(self):
        sim = Sim()
        cmd = sim.run(12, h=0.60)  # target already within the [0.55, 0.70] band
        self.assertIs(cmd.state, FollowState.HOLD_DISTANCE)
        self.assertEqual(cmd.reason, "in distance band")
        self.assertEqual(cmd.forward_mps, 0.0)
        self.assertTrue(cmd.hold)

    def test_02b_gradual_approach_stops_at_band(self):
        sim = Sim()
        for h in np.linspace(0.40, 0.62, 40):  # +0.55 % per result: well inside the rate limit
            cmd = sim.step(h=float(h))
        self.assertIs(cmd.state, FollowState.HOLD_DISTANCE)
        self.assertEqual(cmd.forward_mps, 0.0)

    def test_03_too_close_holds_never_backward(self):
        sim = Sim()
        cmd = sim.run(8, h=0.80)
        self.assertIs(cmd.state, FollowState.HOLD_DISTANCE)
        self.assertIn("too close", cmd.reason)
        self.assertEqual(cmd.forward_mps, 0.0)
        self.assertTrue(all(c.forward_mps >= 0 for c in sim.commands))

    def test_03b_box_touching_frame_edge_is_invalid(self):
        sim = Sim()
        cmd = sim.run(6, h=0.99)
        self.assertIs(cmd.state, FollowState.HOLD_INVALID)
        self.assertEqual(cmd.forward_mps, 0.0)


class SizeJumpTest(unittest.TestCase):
    def check_jump(self, new_h):
        sim = Sim()
        self.assertGreater(sim.run(12, h=0.40).forward_mps, 0)
        cmd = sim.step(h=new_h)
        self.assertIs(cmd.state, FollowState.HOLD_INVALID)
        self.assertIn("not explained", cmd.reason)
        self.assertEqual((cmd.forward_mps, cmd.yaw_rate_dps), (0.0, 0.0))
        # the new size stays rejected even when stable for a long time
        stable = sim.run(40, h=new_h)
        self.assertIs(stable.state, FollowState.HOLD_INVALID)
        self.assertEqual(stable.forward_mps, 0.0)
        # back to the old size: accepted, but only after revalidation
        back = [sim.step(h=0.40) for _ in range(P.resume_valid_count)]
        self.assertTrue(all("revalidating" in c.reason for c in back[:-1]))
        self.assertIs(back[-1].state, FollowState.ACTIVE)
        return sim

    def test_04_sudden_shrink_holds(self):
        self.check_jump(0.28)

    def test_05_sudden_expand_holds(self):
        self.check_jump(0.55)

    def test_05b_operator_rebaseline_accepts_new_size_after_revalidation(self):
        sim = Sim()
        sim.run(12, h=0.40)
        sim.run(5, h=0.28)
        sim.ctl.rebaseline()
        cmds = sim.run(P.resume_valid_count, h=0.28)
        self.assertIs(cmds.state, FollowState.ACTIVE)


class FullBodyTest(unittest.TestCase):
    def test_06_missing_hips_holds(self):
        bbox, kp = person(0.35)
        kp[[LEFT_HIP, RIGHT_HIP], 2] = 0.2
        self.assertEqual(validate_full_body(kp, bbox, W, H), (False, "hips not visible"))
        sim = Sim()
        sim.run(12, h=0.35)
        cmd = sim.step(keypoints=kp)
        self.assertIs(cmd.state, FollowState.HOLD_INVALID)
        self.assertEqual(cmd.forward_mps, 0.0)

    def test_06b_one_occluded_leg_is_still_valid(self):
        bbox, kp = person(0.35)
        kp[[RIGHT_KNEE, RIGHT_ANKLE], 2] = 0.1
        self.assertEqual(validate_full_body(kp, bbox, W, H), (True, "ok"))

    def test_06c_missing_head_or_feet_holds(self):
        bbox, kp = person(0.35)
        kp[[NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR], 2] = 0.1
        self.assertEqual(validate_full_body(kp, bbox, W, H)[1], "head not visible")
        bbox, kp = person(0.35)
        kp[[LEFT_ANKLE, RIGHT_ANKLE], 2] = 0.1
        self.assertEqual(validate_full_body(kp, bbox, W, H)[1], "legs not visible")

    def test_07_crouch_and_sit_hold(self):
        for pose, reason in (("crouch", "legs folded"), ("sitting", "thighs not vertical")):
            bbox, kp = person(0.35, pose=pose)
            ok, why = validate_full_body(kp, bbox, W, H)
            self.assertFalse(ok)
            self.assertIn(reason, why)
            sim = Sim()
            sim.run(12, h=0.35)
            cmd = sim.step(pose=pose)
            self.assertIs(cmd.state, FollowState.HOLD_INVALID, pose)
            self.assertEqual(cmd.forward_mps, 0.0)

    def test_08_sideways_person_no_forward(self):
        bbox, kp = person(0.35, pose="sideways")
        self.assertEqual(validate_full_body(kp, bbox, W, H), (False, "sideways to camera"))
        sim = Sim()
        sim.run(12, h=0.35)
        cmd = sim.run(5, pose="sideways")
        self.assertIs(cmd.state, FollowState.HOLD_INVALID)
        self.assertEqual((cmd.forward_mps, cmd.yaw_rate_dps), (0.0, 0.0))

    def test_keypoint_count_alone_is_not_enough(self):
        bbox, kp = person(0.35)
        kp[:, 2] = 0.9
        kp[LEFT_SHOULDER, 1], kp[LEFT_HIP, 1] = kp[LEFT_HIP, 1], kp[LEFT_SHOULDER, 1]  # all 17 seen, nonsense layout
        kp[RIGHT_SHOULDER, 1], kp[RIGHT_HIP, 1] = kp[RIGHT_HIP, 1], kp[RIGHT_SHOULDER, 1]
        self.assertFalse(validate_full_body(kp, bbox, W, H)[0])


class TargetTest(unittest.TestCase):
    def test_09_target_disappears_holds(self):
        sim = Sim()
        sim.run(12, h=0.35)
        cmd = sim.step(observed=False, usable=False)
        self.assertIs(cmd.state, FollowState.HOLD_TARGET_LOST)
        self.assertEqual((cmd.forward_mps, cmd.yaw_rate_dps), (0.0, 0.0))
        self.assertEqual(cmd.target_id, 3)  # selection is kept

    def test_10_other_track_never_followed(self):
        sim = Sim()
        sim.run(12, h=0.35)
        cmd = sim.step(meas_id=7)  # measurement from another person while target 3 is selected
        self.assertIs(cmd.state, FollowState.HOLD_INVALID)
        self.assertEqual(cmd.forward_mps, 0.0)
        # only a new TargetSelector selection changes the target - and it starts from scratch
        cmds = [sim.step(target=7) for _ in range(P.resume_valid_count)]
        self.assertTrue(all(c.target_id == 7 for c in cmds))
        self.assertTrue(all(c.forward_mps == 0.0 for c in cmds))
        self.assertIn("revalidating 1/", cmds[0].reason)

    def test_11_stale_result_holds(self):
        sim = Sim()
        sim.run(12, h=0.35)
        cmd = sim.step(age=0.5)
        self.assertIs(cmd.state, FollowState.HOLD_INVALID)
        self.assertEqual(cmd.reason, "AI result stale")
        self.assertEqual(cmd.forward_mps, 0.0)

    def test_11b_publisher_holds_when_results_stop(self):
        sim = Sim()
        active = sim.run(12, h=0.35)
        self.assertGreater(active.forward_mps, 0)
        last_result = sim.t - DT
        self.assertGreater(sim.ctl.command_at(last_result + 0.1).forward_mps, 0)
        cmd = sim.ctl.command_at(last_result + 0.31)  # still updated recently enough, but result too old
        self.assertEqual((cmd.forward_mps, cmd.yaw_rate_dps), (0.0, 0.0))

    def test_12_resume_only_after_validation(self):
        sim = Sim()
        sim.run(12, h=0.35)
        sim.step(pose="crouch")
        cmds = [sim.step(h=0.35) for _ in range(P.resume_valid_count)]
        self.assertTrue(all(c.forward_mps == 0.0 for c in cmds[:-1]))
        self.assertEqual([c.reason for c in cmds[:-1]], [f"revalidating {i}/{P.resume_valid_count}"
                                                         for i in range(1, P.resume_valid_count)])
        self.assertIs(cmds[-1].state, FollowState.ACTIVE)

    def test_13_t_pose_release_is_preserved(self):
        selector = TargetSelector()
        selector.update({3: BodyPose.ONE_ARM_UP}, True)
        sim = Sim()
        self.assertGreater(sim.run(12, state=selector.state, target=selector.target_id).forward_mps, 0)
        result = selector.update({3: BodyPose.T_POSE}, True)
        self.assertIs(result.event, SelectionEvent.TARGET_RELEASED)
        cmd = sim.step(state=result.state, target=result.target_id)
        self.assertIs(cmd.state, FollowState.INACTIVE)
        self.assertEqual((cmd.forward_mps, cmd.yaw_rate_dps), (0.0, 0.0))

    def test_14_double_biceps_is_a_landing_request(self):
        selector = TargetSelector()
        selector.update({3: BodyPose.ONE_ARM_UP}, True)
        sim = Sim()
        sim.run(12, state=selector.state, target=selector.target_id)
        result = selector.update({3: BodyPose.DOUBLE_BICEPS}, True)
        self.assertIs(result.event, SelectionEvent.LAND_REQUESTED)
        for _ in range(5):  # stays a landing request, target still in view
            cmd = sim.step(state=result.state, target=result.target_id)
            self.assertIs(cmd.state, FollowState.LAND_REQUESTED)
            self.assertTrue(cmd.land)
            self.assertEqual((cmd.forward_mps, cmd.yaw_rate_dps), (0.0, 0.0))

    def test_15_communication_timeout_holds_and_requires_revalidation(self):
        sim = Sim()
        sim.run(12, h=0.35)
        cmd = sim.ctl.command_at(sim.t + 0.4)
        self.assertIs(cmd.state, FollowState.HOLD_TIMEOUT)
        self.assertEqual((cmd.forward_mps, cmd.yaw_rate_dps), (0.0, 0.0))
        sim.t += 0.4
        self.assertIn("revalidating", sim.step(h=0.35).reason)

    def test_16_forward_velocity_never_negative(self):
        with self.assertRaises(ValueError):
            FollowCommand(FollowState.ACTIVE, "x", -0.01, 0.0, False, False, 3)
        rng = random.Random(0)
        sim = Sim()
        for _ in range(3000):
            cmd = sim.step(h=rng.uniform(0.1, 1.0), cx=rng.uniform(0.1, 0.9),
                           pose=rng.choice(["standing"] * 6 + ["crouch", "sitting", "sideways"]),
                           observed=rng.random() > 0.05, usable=rng.random() > 0.05,
                           conf=rng.choice([0.5, 0.621, 0.73, 0.816, 0.879]), age=rng.choice([0.02] * 9 + [0.6]),
                           state=rng.choice([SelectionState.FOLLOWING] * 18 + [SelectionState.SEARCHING,
                                                                               SelectionState.LAND_REQUESTED]),
                           target=rng.choice([3] * 19 + [5]))
            self.assertGreaterEqual(cmd.forward_mps, 0.0)
            self.assertLessEqual(cmd.forward_mps, P.max_forward_mps)
            self.assertLessEqual(abs(cmd.yaw_rate_dps), P.max_yaw_rate_dps)
            for t in (0.05, 0.2, 0.5):
                self.assertGreaterEqual(sim.ctl.command_at(sim.t + t).forward_mps, 0.0)


class YawAndLimitsTest(unittest.TestCase):
    def test_yaw_sign_and_deadband(self):
        right = Sim().run(20, h=0.6, cx=0.75)  # in band: forward 0, yaw toward the person
        left = Sim().run(20, h=0.6, cx=0.25)
        centered = Sim().run(20, h=0.6, cx=0.52)  # error 0.04 < deadband
        self.assertGreater(right.yaw_rate_dps, 0)  # positive = yaw right
        self.assertLess(left.yaw_rate_dps, 0)
        self.assertEqual(centered.yaw_rate_dps, 0.0)
        self.assertLessEqual(abs(Sim().run(40, h=0.6, cx=0.95).yaw_rate_dps), P.max_yaw_rate_dps)

    def test_yaw_rate_and_forward_accel_limited(self):
        sim = Sim()
        sim.run(P.resume_valid_count + 2, h=0.30, cx=0.9)
        a, b = sim.commands[-2], sim.commands[-1]
        self.assertLessEqual(b.forward_mps - a.forward_mps, P.max_forward_accel * DT + 1e-9)
        self.assertLessEqual(abs(b.yaw_rate_dps - a.yaw_rate_dps), P.max_yaw_accel_dps2 * DT + 1e-9)
        self.assertGreater(b.forward_mps, a.forward_mps)  # still ramping up, not jumping to the target speed

    def test_invalid_data_zeroes_yaw_immediately(self):
        sim = Sim()
        self.assertGreater(sim.run(20, h=0.35, cx=0.8).yaw_rate_dps, 0)
        self.assertEqual(sim.step(h=0.35, cx=0.8, pose="crouch").yaw_rate_dps, 0.0)

    def test_hysteresis(self):
        sim = Sim()
        sim.run(8, h=0.535)  # below 0.55 but inside the 0.03 hysteresis: do not start
        self.assertIs(sim.commands[-1].state, FollowState.HOLD_DISTANCE)
        self.assertEqual(sim.commands[-1].forward_mps, 0.0)

    def test_repeated_result_is_not_a_new_measurement(self):
        sim = Sim()
        bbox, kp = person(0.35)
        inp = FollowInput(0.02, 0.0, SelectionState.FOLLOWING, 3, True, True, 3, bbox, 0.816, kp, W, H)
        for i in range(6):  # same result_timestamp six times (REUSED frames)
            cmd = sim.ctl.update(FollowInput(**{**inp.__dict__, "now": 0.02 + i * 0.01}))
        self.assertEqual(cmd.reason, "revalidating 1/4")

    def test_low_confidence_holds(self):
        cmd = Sim().run(8, h=0.35, conf=0.5)
        self.assertIs(cmd.state, FollowState.HOLD_INVALID)
        self.assertEqual(cmd.reason, "detection confidence low")


ALLOWED_DIAGNOSTICS = {"OCCLUSION_SUSPECTED"}


class OcclusionTest(unittest.TestCase):
    """Occlusion is only ever SUSPECTED (never reported as confirmed); every suspicion holds."""

    def assert_held(self, cmd, stage=None):
        self.assertEqual((cmd.forward_mps, cmd.yaw_rate_dps), (0.0, 0.0))
        self.assertTrue(cmd.hold)
        self.assertEqual(cmd.target_id, 3)  # selection kept
        if stage:
            self.assertEqual(cmd.stage, stage)

    def test_other_person_overlapping_target_holds(self):
        sim = Sim()
        self.assertGreater(sim.run(12, h=0.35).forward_mps, 0)
        bbox, kp = person(0.35)
        other = (bbox[0] + 0.5 * (bbox[2] - bbox[0]), bbox[1], bbox[2] + 60, bbox[3])  # covers right half
        inp = FollowInput(sim.t + 0.02, sim.t, SelectionState.FOLLOWING, 3, True, True, 3, bbox, 0.816, kp, W, H,
                          other_boxes=(other,))
        cmd = sim.ctl.update(inp)
        self.assertEqual(cmd.diagnostics, ("OCCLUSION_SUSPECTED",))
        self.assert_held(cmd, "occlusion")
        self.assertTrue(cmd.indicators[0].startswith("other_overlap="))

    def test_far_away_other_person_does_not_hold(self):
        sim = Sim()
        sim.run(12, h=0.35, cx=0.3)
        bbox, kp = person(0.35, cx=0.3)
        other, _ = person(0.35, cx=0.8)
        inp = FollowInput(sim.t + 0.02, sim.t, SelectionState.FOLLOWING, 3, True, True, 3, bbox, 0.816, kp, W, H,
                          other_boxes=(other,))
        self.assertIs(sim.ctl.update(inp).state, FollowState.ACTIVE)

    def test_lower_body_confidence_drop_holds(self):
        sim = Sim()
        sim.run(12, h=0.35)
        bbox, kp = person(0.35)
        kp[[LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE], 2] = 0.52  # still above 0.5, but 0.9 -> 0.52
        self.assertTrue(validate_full_body(kp, bbox, W, H)[0])  # the full-body check alone would pass
        cmd = sim.step(keypoints=kp)
        self.assertEqual(cmd.diagnostics, ("OCCLUSION_SUSPECTED",))
        self.assert_held(cmd, "occlusion")

    def test_hidden_legs_inside_frame_are_suspected(self):
        sim = Sim()
        sim.run(12, h=0.35)
        bbox, kp = person(0.35)
        kp[[LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE], 2] = 0.05
        cmd = sim.step(keypoints=kp)
        self.assertEqual(cmd.diagnostics, ("OCCLUSION_SUSPECTED",))
        self.assertIn("legs not visible", cmd.indicators)
        self.assert_held(cmd, "full_body")

    def test_leaving_the_frame_is_not_labelled_occlusion(self):
        sim = Sim()
        bbox, kp = person(0.6, top=230)  # feet below the image: box clipped at the bottom
        bbox = (bbox[0], bbox[1], bbox[2], H - 1)
        kp[[LEFT_ANKLE, RIGHT_ANKLE], 2] = 0.05
        cmd = sim.step(bbox=bbox, keypoints=kp)
        self.assertEqual(cmd.diagnostics, ())
        self.assertIn("box clipped by image edge", cmd.indicators)
        self.assertEqual(cmd.forward_mps, 0.0)

    def test_shrink_with_fixed_top_edge_is_suspected(self):
        sim = Sim()
        sim.run(12, h=0.40)
        ref, _ = person(0.40)
        bbox, kp = person(0.28, top=ref[1])  # same top, bottom moved up by 30 % of the box
        cmd = sim.step(bbox=bbox, keypoints=kp)
        self.assertIn("not explained", cmd.reason)
        self.assertEqual(cmd.diagnostics, ("OCCLUSION_SUSPECTED",))
        self.assertIn("bottom edge rose, top stable", cmd.indicators)
        self.assert_held(cmd, "measurement")

    def test_uniform_shrink_is_not_labelled_occlusion(self):
        sim = Sim()
        sim.run(12, h=0.40)
        cmd = sim.step(h=0.28)  # centered: both edges move -> size change, no occlusion label
        self.assertIn("not explained", cmd.reason)
        self.assertEqual(cmd.diagnostics, ())
        self.assertEqual(cmd.forward_mps, 0.0)

    def test_labels_are_only_suspicions(self):
        rng = random.Random(1)
        sim = Sim()
        for _ in range(1500):
            bbox, kp = person(rng.uniform(0.2, 0.9), rng.uniform(0.2, 0.8),
                              rng.choice(["standing"] * 5 + ["crouch", "sideways"]))
            kp[:, 2] = np.clip(kp[:, 2] - rng.choice([0, 0, 0, 0.3, 0.6]) * (np.arange(17) > 10), 0, 1)
            others = () if rng.random() < 0.7 else (person(0.4, rng.uniform(0.2, 0.8))[0],)
            inp = FollowInput(sim.t + 0.02, sim.t, SelectionState.FOLLOWING, 3, True, True, 3, bbox, 0.816, kp, W, H,
                              other_boxes=others)
            sim.t += DT
            cmd = sim.ctl.update(inp)
            self.assertTrue(set(cmd.diagnostics) <= ALLOWED_DIAGNOSTICS)
            if cmd.diagnostics:
                self.assertEqual((cmd.forward_mps, cmd.yaw_rate_dps), (0.0, 0.0))
            self.assertGreaterEqual(cmd.forward_mps, 0.0)


class BodyMetricsTest(unittest.TestCase):
    def test_reason_matches_validator_and_values(self):
        from follow_controller import body_metrics
        for pose in ("standing", "crouch", "sitting", "sideways"):
            bbox, kp = person(0.4, pose=pose)
            m = body_metrics(kp, bbox, W, H)
            self.assertEqual((m["reason"] == "ok", m["reason"]), validate_full_body(kp, bbox, W, H))
        bbox, kp = person(0.4)
        m = body_metrics(kp, bbox, W, H)
        self.assertAlmostEqual(m["leg_ratio"], (0.96 - 0.50) / 0.30, places=6)
        self.assertAlmostEqual(m["thigh_ratio"], (0.74 - 0.50) / 0.30, places=6)
        self.assertEqual((m["head"], m["shoulders"], m["hips"], m["knees"], m["ankles"]), (5, 2, 2, 2, 2))
        self.assertEqual(m["failures"], [])

    def test_all_failures_listed(self):
        from follow_controller import body_metrics
        bbox, kp = person(0.4)
        kp[[LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE], 2] = 0.1
        m = body_metrics(kp, bbox, W, H)
        self.assertEqual(m["failures"], ["hips not visible", "legs not visible"])
        self.assertIsNone(m["leg_ratio"])

    def test_stage_is_reported(self):
        sim = Sim()
        stages = [sim.step(h=0.35).stage for _ in range(5)]
        self.assertEqual(stages, ["revalidation"] * 3 + ["control"] * 2)
        self.assertEqual(sim.step(observed=False).stage, "data")
        self.assertEqual(sim.step(state=SelectionState.SEARCHING).stage, "selection")
        self.assertEqual(sim.ctl.command_at(sim.t + 1).stage, "timeout")


class AdapterTest(unittest.TestCase):
    def fake_pipeline(self, target_id=3, tracked_ids=(3,), assoc=True, usable=True):
        bbox, kp = person(0.4)
        detections = SimpleNamespace(xyxy=np.array([bbox]), conf=np.array([0.816]), keypoints=kp[None])
        result = SimpleNamespace(timestamp=10.0, tracked_people=[SimpleNamespace(track_id=t) for t in tracked_ids],
                                 track_to_detection={target_id: 0} if assoc else {}, detections=detections)
        status = SimpleNamespace(state=SelectionState.FOLLOWING, target_id=target_id, usable=usable)
        return SimpleNamespace(target_status=lambda now: status, last_result=result), bbox

    def test_uses_raw_detection_of_the_selected_track(self):
        pipeline, bbox = self.fake_pipeline()
        inp = follow_input_from_pipeline(pipeline, 10.05, W, H)
        self.assertEqual(inp.result_timestamp, 10.0)
        self.assertEqual((inp.target_id, inp.measurement_track_id, inp.target_observed), (3, 3, True))
        np.testing.assert_allclose(inp.bbox, bbox)
        self.assertAlmostEqual(inp.confidence, 0.816)

    def test_no_association_means_no_measurement(self):
        pipeline, _ = self.fake_pipeline(assoc=False)
        inp = follow_input_from_pipeline(pipeline, 10.05, W, H)
        self.assertIsNone(inp.bbox)
        self.assertIs(FollowController().update(inp).state, FollowState.HOLD_INVALID)

    def test_other_detections_are_passed_for_occlusion_check(self):
        bbox, kp = person(0.4)
        other, okp = person(0.4, cx=0.8)
        detections = SimpleNamespace(xyxy=np.array([other, bbox]), conf=np.array([0.73, 0.816]),
                                     keypoints=np.stack([okp, kp]))
        result = SimpleNamespace(timestamp=10.0, tracked_people=[SimpleNamespace(track_id=3)],
                                 track_to_detection={3: 1}, detections=detections)
        status = SimpleNamespace(state=SelectionState.FOLLOWING, target_id=3, usable=True)
        inp = follow_input_from_pipeline(SimpleNamespace(target_status=lambda now: status, last_result=result),
                                         10.05, W, H)
        np.testing.assert_allclose(inp.bbox, bbox)  # index 1 = the target's own detection
        self.assertEqual(len(inp.other_boxes), 1)
        np.testing.assert_allclose(inp.other_boxes[0], other)

    def test_target_missing_from_result(self):
        pipeline, _ = self.fake_pipeline(tracked_ids=(7,), usable=False)
        inp = follow_input_from_pipeline(pipeline, 10.05, W, H)
        self.assertFalse(inp.target_observed)
        self.assertIs(FollowController().update(inp).state, FollowState.HOLD_TARGET_LOST)


if __name__ == "__main__":
    unittest.main()

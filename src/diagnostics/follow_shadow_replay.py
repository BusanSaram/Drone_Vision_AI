"""Replay and analyze a follow-controller Shadow Mode log (offline; Windows or Pi; no camera).

    python src/diagnostics/follow_shadow_replay.py LOG.jsonl
    python src/diagnostics/follow_shadow_replay.py LOG.jsonl --set min_leg_to_torso_ratio=1.1 --set resume_valid_count=3
    python src/diagnostics/follow_shadow_replay.py LOG.jsonl --segment 0:30:standing --segment 30:60:crouch

1. Rebuilds every "new" record's FollowInput and runs a fresh FollowController over them in order.
   With the logged parameters it must reproduce the logged decisions exactly (reported as
   "replay mismatches"); with --set overrides it shows what other parameters WOULD have decided.
2. Summarizes decisions (state / stage / reason / OCCLUSION_SUSPECTED), measured distributions
   (h_norm, confidence, keypoint confidences, posture ratios, box-height change rate) and lists
   FORWARD decisions taken in conditions worth a manual look (review candidates).
Segments are seconds since the first AI result: label scenario time ranges from your test notes.
It never sends anything anywhere.
"""

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/

from follow_controller import FollowController, FollowParams  # noqa: E402
from follow_record import input_from_record as _input_from_input_record  # noqa: E402

METRICS = ("conf", "h_norm", "upper_conf", "lower_conf", "span", "tilt_deg", "leg_ratio", "thigh_ratio",
           "shoulder_ratio")


def load(path) -> tuple[dict, list[dict]]:
    meta, records = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "meta":
                meta = rec
            else:
                records.append(rec)
    return meta, records


def input_from_record(rec: dict):
    """FollowInput of a logged 'new' record."""
    return _input_from_input_record(rec["input"])


def params_from(meta: dict, overrides: list[str]) -> FollowParams:
    names = {f.name: f.type for f in fields(FollowParams)}
    values = {k: v for k, v in meta.get("params", {}).items() if k in names}
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in names:
            raise SystemExit(f"unknown FollowParams field: {key}")
        values[key] = type(getattr(FollowParams(), key))(value)
    return replace(FollowParams(), **values)


def replay(records: list[dict], params: FollowParams) -> list[tuple[dict, object]]:
    """Feed the 'new' records (in order) to a fresh controller; returns (record, replayed command) pairs."""
    controller = FollowController(params)
    return [(rec, controller.update(input_from_record(rec))) for rec in records if rec.get("type") == "new"]


def same_decision(logged: dict, cmd) -> bool:
    return (logged["state"] == cmd.state.value and logged["reason"] == cmd.reason
            and math.isclose(logged["fwd"], round(cmd.forward_mps, 4), abs_tol=1e-9)
            and math.isclose(logged["yaw"], round(cmd.yaw_rate_dps, 3), abs_tol=1e-9)
            and logged.get("diag", []) == list(cmd.diagnostics))


def review_flags(rec: dict, prev_h: float | None, dt: float | None) -> list[str]:
    """Conditions worth a manual look when the controller decided FORWARD."""
    flags = []
    body, i = rec.get("body") or {}, rec["input"]
    if rec.get("det_count", 0) > 1:
        flags.append("other people in frame")
    if i["conf"] is not None and i["conf"] < 0.7:
        flags.append(f"conf {i['conf']}")
    if body.get("lower_conf") is not None and body["lower_conf"] < 0.6:
        flags.append(f"lower_conf {body['lower_conf']}")
    if prev_h and dt and rec["ctl"]["h_norm"] is not None:
        rate = abs(rec["ctl"]["h_norm"] / prev_h - 1) / dt
        if rate > 0.3:
            flags.append(f"h change {rate:.2f}/s")
    return flags


def describe(values) -> str:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return "n=0"
    p = np.percentile(v, [0, 5, 50, 95, 100])
    return f"n={v.size} min={p[0]:.3f} p5={p[1]:.3f} med={p[2]:.3f} p95={p[3]:.3f} max={p[4]:.3f}"


def summarize(pairs, t0: float, label: str = "all", start=None, end=None) -> list[str]:
    sel = [(r, c) for r, c in pairs if (start is None or r["t"] - t0 >= start) and (end is None or r["t"] - t0 < end)]
    out = [f"=== {label} ({len(sel)} AI results"
           + ("" if start is None else f", t={start:g}..{end:g}s") + ") ==="]
    if not sel:
        return out
    states = Counter(c.state.value for _, c in sel)
    reasons = Counter(f"{c.stage}: {c.reason}" for _, c in sel if c.state.value != "ACTIVE")
    diags = Counter(d for _, c in sel for d in c.diagnostics)
    out.append("states: " + ", ".join(f"{k}={v}" for k, v in states.most_common()))
    out.append("diagnostics: " + (", ".join(f"{k}={v}" for k, v in diags.items()) or "-"))
    out.append("hold/other reasons (top 8):")
    out += [f"  {n:5d}  {k}" for k, n in reasons.most_common(8)]
    fwd = [c.forward_mps for _, c in sel]
    out.append(f"forward: max={max(fwd):.3f} m/s, results with forward>0: {sum(f > 0 for f in fwd)}; "
               f"min forward={min(fwd):.3f} (must be >= 0)")
    measured = [(r, c) for r, c in sel if r["input"]["bbox"] is not None]
    out.append("measured values (selected target):")
    for m in METRICS:
        if m == "conf":
            vals = [r["input"]["conf"] for r, _ in measured]
        elif m == "h_norm":
            vals = [c.box_height for _, c in measured]
        else:
            vals = [(r.get("body") or {}).get(m) for r, _ in measured]
        out.append(f"  {m:14s} {describe(vals)}")
    rates, prev = [], None
    for r, c in measured:
        if prev is not None and c.box_height and prev[1] and r["input"]["result_t"] > prev[0]:
            rates.append(abs(c.box_height / prev[1] - 1) / (r["input"]["result_t"] - prev[0]))
        prev = (r["input"]["result_t"], c.box_height)
    out.append(f"  {'h_rate_per_s':14s} {describe(rates)}")
    review, prev = [], None
    for r, c in measured:
        dt = None if prev is None else r["input"]["result_t"] - prev[0]
        if c.forward_mps > 0:
            flags = review_flags(r, None if prev is None else prev[1], dt)
            if flags:
                review.append(f"  t={r['t'] - t0:7.2f}s fwd={c.forward_mps:.2f} h={c.box_height:.3f}: {', '.join(flags)}")
        prev = (r["input"]["result_t"], c.box_height)
    out.append(f"FORWARD review candidates: {len(review)}")
    out += review[:20]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override a FollowParams field")
    ap.add_argument("--segment", action="append", default=[], metavar="START:END:LABEL", help="seconds since first result")
    args = ap.parse_args(argv)

    meta, records = load(args.log)
    params = params_from(meta, args.set)
    pairs = replay(records, params)
    if not pairs:
        print("no 'new' records in log")
        return 1
    t0 = pairs[0][0]["t"]
    mismatches = [r for r, c in pairs if not same_decision(r["ctl"], c)]
    print(f"log: {args.log}  records: {len(records)}  AI results: {len(pairs)}  overrides: {args.set or '-'}")
    print(f"replay mismatches vs logged decisions: {len(mismatches)}"
          + ("  (expected when --set is used)" if args.set else ""))
    ticks = Counter(r["ctl"]["state"] for r in records if r.get("type") == "tick")
    print("tick overrides (publisher view): " + (", ".join(f"{k}={v}" for k, v in ticks.items()) or "-"))
    errors = [r for r in records if r.get("type") == "error"]
    if errors:
        print(f"SHADOW ERRORS: {errors}")
    for line in summarize(pairs, t0):
        print(line)
    for seg in args.segment:
        start, end, label = seg.split(":", 2)
        for line in summarize(pairs, t0, label, float(start), float(end)):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

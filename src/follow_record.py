"""FollowInput <-> JSON record conversion shared by Shadow Mode (Pi) and offline replay (any machine).

Shadow Mode feeds the controller the input AFTER this round trip, so the logged
input is exactly what the controller saw and a replay reproduces every decision.
Pixel values are kept to 0.1 px and confidences to 3 decimals (keeps the log small);
timestamps are kept at full float precision.
"""

import numpy as np

from follow_controller import FollowInput
from target_selector import SelectionState


def _round(value, digits):
    return None if value is None else round(float(value), digits)


def input_to_record(inp: FollowInput) -> dict:
    return {
        "now": float(inp.now),
        "result_t": None if inp.result_timestamp is None else float(inp.result_timestamp),
        "sel_state": inp.selection_state.value, "target_id": inp.target_id, "usable": bool(inp.usable),
        "observed": bool(inp.target_observed), "meas_id": inp.measurement_track_id,
        "bbox": None if inp.bbox is None else [_round(v, 1) for v in inp.bbox],
        "conf": _round(inp.confidence, 3),
        "kp": None if inp.keypoints is None else [[_round(x, 1), _round(y, 1), _round(c, 3)]
                                                  for x, y, c in np.asarray(inp.keypoints)],
        "frame": [int(inp.frame_width), int(inp.frame_height)],
        "others": [[_round(v, 1) for v in box] for box in inp.other_boxes],
    }


def input_from_record(rec: dict) -> FollowInput:
    return FollowInput(
        now=rec["now"], result_timestamp=rec["result_t"], selection_state=SelectionState(rec["sel_state"]),
        target_id=rec["target_id"], usable=rec["usable"], target_observed=rec["observed"],
        measurement_track_id=rec["meas_id"], bbox=None if rec["bbox"] is None else tuple(rec["bbox"]),
        confidence=rec["conf"], keypoints=None if rec["kp"] is None else np.asarray(rec["kp"], dtype=np.float64),
        frame_width=rec["frame"][0], frame_height=rec["frame"][1],
        other_boxes=tuple(tuple(b) for b in rec.get("others", [])),
    )

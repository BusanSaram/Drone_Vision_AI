"""Build a FollowInput from ImxTrackingPipeline state (not wired into any running script yet).

Uses only attributes the pipeline already exposes:
    pipeline.target_status(now)  -> state, target_id, usable
    pipeline.last_result         -> timestamp, tracked_people, track_to_detection, detections
The measurement is the target's RAW detection (box, score, 17 keypoints) found
through `track_to_detection` - not TargetStatus.bbox, which is BoT-SORT's
Kalman-filtered box and would hide the size jumps the controller must see.
If the target's track has no associated detection this result, no measurement
is passed (the controller then holds). The boxes of all OTHER detections of the
same result are passed too, for the controller's occlusion indicator.
"""

from follow_controller import FollowInput


def follow_input_from_pipeline(pipeline, now: float, frame_width: int, frame_height: int) -> FollowInput:
    status = pipeline.target_status(now)
    result = pipeline.last_result
    observed, measurement_id, bbox, confidence, keypoints, others = False, None, None, None, None, ()
    if result is not None and status.target_id is not None:
        observed = any(p.track_id == status.target_id for p in result.tracked_people)
        index = result.track_to_detection.get(status.target_id)
        if observed and index is not None:
            measurement_id = status.target_id
            bbox = tuple(float(v) for v in result.detections.xyxy[index])
            confidence = float(result.detections.conf[index])
            keypoints = result.detections.keypoints[index]
            others = tuple(tuple(float(v) for v in box)
                           for i, box in enumerate(result.detections.xyxy) if i != index)
    return FollowInput(
        now=now,
        result_timestamp=None if result is None else result.timestamp,
        selection_state=status.state,
        target_id=status.target_id,
        usable=status.usable,
        target_observed=observed,
        measurement_track_id=measurement_id,
        bbox=bbox,
        confidence=confidence,
        keypoints=keypoints,
        frame_width=frame_width,
        frame_height=frame_height,
        other_boxes=others,
    )

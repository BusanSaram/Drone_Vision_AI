"""IMX500 pose backend: modlib `Poses` -> the existing pipeline's `PoseDetections`.

This is the only new piece between the IMX500 and the reused Windows modules
(PersonTracker, TrackValidator, keypoint association, pose recognition,
stabilizer, TargetSelector). Inference itself stays on the IMX500 - nothing
here runs a model.

Conversion (same coordinate rules as the live overlay, imx500_pose_overlay.py):
    - `Poses.compensate_for_roi(roi)` first (no-op for the full-FOV ROI we use)
    - normalized 0..1 coordinates * frame.image width/height -> pixels, clipped
      to the image; x right, y down, origin top-left, no flip/rotation
    - keypoints become (N, 17, 3) (x, y, score), the layout coco_keypoints expects
    - class id 0 (person) for every row

MIN_DETECTION_SCORE: the Windows pipeline got its detections from Ultralytics
`predict()` with the default conf=0.25, so BoT-SORT never saw anything lower.
The IMX500 export runs NMS inside the model with a 0.001 floor and always
returns 300 slots, so the same 0.25 floor is applied here to give the reused
tracker the same kind of input. This matches the original input - it is not a
tuned value.
"""

import numpy as np

from imx500_pose_overlay import to_pixel_poses
from pose_detections import PoseDetections

MIN_DETECTION_SCORE = 0.25  # = Ultralytics predict() default conf used by YoloPoseDetector on Windows


def poses_to_detections(poses, roi, width: int, height: int,
                        min_score: float = MIN_DETECTION_SCORE) -> PoseDetections:
    """Convert one modlib `Poses` result to pixel-space `PoseDetections` for a width x height frame.

    `roi` is `frame.roi`; pass None to skip ROI compensation.
    """
    if roi is not None:
        poses.compensate_for_roi(roi)
    pixel = to_pixel_poses(poses.bbox, poses.keypoints, poses.keypoint_scores, poses.confidence,
                           width, height, min_score=min_score)
    return PoseDetections.from_arrays(pixel.boxes, pixel.confidence, np.zeros(len(pixel), dtype=np.float32),
                                      pixel.keypoints)

import numpy as np


def convert_imx500_output(boxes, scores, keypoints):
    """Convert HigherHRNet outputs into arrays for PoseDetections."""

    if scores is None or len(scores) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 17, 3), dtype=np.float32),
        )

    boxes_yxyx = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes_xyxy = boxes_yxyx[:, [1, 0, 3, 2]]

    confidence = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.zeros(len(confidence), dtype=np.float32)
    poses = np.asarray(keypoints, dtype=np.float32).reshape(-1, 17, 3)

    return boxes_xyxy, confidence, classes, poses

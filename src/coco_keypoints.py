"""COCO 17-keypoint body model: names, skeleton, and reliability checks.

Backend-independent - pure numpy, no Ultralytics import. A person's keypoints
are a (17, 3) array of (x, y, confidence), with x/y in frame pixels. This is
the format produced by `YoloPoseDetector` today and the format a future
IMX500 pose backend would also have to produce.
"""

import numpy as np

KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
NUM_KEYPOINTS = len(KEYPOINT_NAMES)  # 17

NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# Standard COCO skeleton (same connections Ultralytics draws), 0-indexed.
COCO_SKELETON = (
    # legs
    (LEFT_ANKLE, LEFT_KNEE), (LEFT_KNEE, LEFT_HIP),
    (RIGHT_ANKLE, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_HIP),
    # torso
    (LEFT_HIP, RIGHT_HIP), (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    # arms
    (LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST), (RIGHT_ELBOW, RIGHT_WRIST),
    # head
    (LEFT_EYE, RIGHT_EYE), (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
    (LEFT_EAR, LEFT_SHOULDER), (RIGHT_EAR, RIGHT_SHOULDER),
)

# Keypoints below this confidence are treated as unreliable (not drawn, and
# not to be used by later stages). Experimental starting value, not tuned;
# Ultralytics' own plotting uses a looser 0.25.
KEYPOINT_CONF_THRESHOLD = 0.5


def validate_keypoints(keypoints) -> np.ndarray:
    """Return `keypoints` as a float array, raising ValueError unless its shape is (17, 3)."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape != (NUM_KEYPOINTS, 3):
        raise ValueError(f"expected keypoints of shape ({NUM_KEYPOINTS}, 3), got {keypoints.shape}")
    return keypoints


def reliable_mask(keypoints, min_conf: float = KEYPOINT_CONF_THRESHOLD) -> np.ndarray:
    """Boolean (17,) mask: True where the keypoint's confidence >= min_conf and x/y are finite."""
    keypoints = validate_keypoints(keypoints)
    finite = np.isfinite(keypoints).all(axis=1)
    return finite & (keypoints[:, 2] >= min_conf)


def skeleton_segments(keypoints, min_conf: float = KEYPOINT_CONF_THRESHOLD) -> list[tuple[int, int]]:
    """Skeleton connections whose two endpoints are both reliable."""
    reliable = reliable_mask(keypoints, min_conf)
    return [(a, b) for a, b in COCO_SKELETON if reliable[a] and reliable[b]]

"""Backend-independent container for one frame of person detections with keypoints.

Pipeline position:
    pose backend (YOLOv8n-Pose today, IMX500 later) -> PoseDetections -> PersonTracker (BoT-SORT)

`PoseDetections` keeps each person's bbox, confidence, class and 17 COCO
keypoints in parallel arrays, so row i of every array always describes the
same detection. It also exposes the small `Boxes`-like interface the
Ultralytics BOTSORT tracker reads (`conf`, `cls`, `xyxy`, `xywh`, `len()`,
boolean-mask indexing), so it can be passed to `PersonTracker.update()`
directly - no Ultralytics object has to cross this boundary.

A future IMX500 backend only needs to build one of these with
`PoseDetections.from_arrays()`.
"""

from dataclasses import dataclass

import numpy as np

from coco_keypoints import NUM_KEYPOINTS


@dataclass(frozen=True, eq=False)
class PoseDetections:
    xyxy: np.ndarray  # (N, 4) float32, x1 y1 x2 y2 in frame pixels
    conf: np.ndarray  # (N,) float32 detection confidence
    cls: np.ndarray  # (N,) float32 class id (always person here)
    keypoints: np.ndarray  # (N, 17, 3) float32, (x, y, confidence) in frame pixels

    def __post_init__(self):
        n = len(self.xyxy)
        if self.xyxy.shape != (n, 4):
            raise ValueError(f"xyxy must have shape (N, 4), got {self.xyxy.shape}")
        if self.conf.shape != (n,) or self.cls.shape != (n,):
            raise ValueError(f"conf/cls must have shape ({n},), got {self.conf.shape} and {self.cls.shape}")
        if self.keypoints.shape != (n, NUM_KEYPOINTS, 3):
            raise ValueError(f"keypoints must have shape ({n}, {NUM_KEYPOINTS}, 3), got {self.keypoints.shape}")

    @classmethod
    def from_arrays(cls, xyxy, conf, classes, keypoints) -> "PoseDetections":
        """Build from array-likes, converting to float32 numpy.

        `keypoints` must include a per-keypoint confidence column, shape
        (N, 17, 3). `None` is only accepted when there are no detections.
        """
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        if len(xyxy) == 0:
            keypoints = np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32)
        elif keypoints is None:
            raise ValueError("keypoints are required when there are detections")
        return cls(
            xyxy=xyxy,
            conf=np.asarray(conf, dtype=np.float32).reshape(-1),
            cls=np.asarray(classes, dtype=np.float32).reshape(-1),
            keypoints=np.asarray(keypoints, dtype=np.float32),  # shape checked in __post_init__
        )

    @classmethod
    def empty(cls) -> "PoseDetections":
        return cls.from_arrays(np.zeros((0, 4)), [], [], None)

    @property
    def xywh(self) -> np.ndarray:
        """(N, 4) center-x, center-y, width, height - the format BoT-SORT reads."""
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, index) -> "PoseDetections":
        """Row selection (e.g. a boolean mask) applied to all arrays together."""
        return PoseDetections(
            xyxy=self.xyxy[index].reshape(-1, 4),
            conf=self.conf[index].reshape(-1),
            cls=self.cls[index].reshape(-1),
            keypoints=self.keypoints[index].reshape(-1, NUM_KEYPOINTS, 3),
        )

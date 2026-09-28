"""YOLOv8n-Pose person detector (Ultralytics, local GPU/CPU).

The only module in the pose pipeline that knows about Ultralytics' pose
model. It returns a backend-independent `PoseDetections`, so this is the
piece to swap out for IMX500 on-camera pose output later.

Inference settings deliberately mirror `PersonDetector` (same `classes`,
default confidence/NMS thresholds, default image size) so the pose pipeline
can be compared against the detection-only baseline.
"""

from ultralytics import YOLO

from person_detector import PERSON_CLASS_ID, get_device
from pose_detections import PoseDetections

POSE_MODEL_PATH = "yolov8n-pose.pt"


class YoloPoseDetector:
    """Runs YOLOv8n-Pose on individual frames."""

    def __init__(self, model_path: str = POSE_MODEL_PATH, device: str | None = None):
        self.device = device or get_device()
        self.model = YOLO(model_path)

    def detect(self, frame) -> PoseDetections:
        """Run pose detection on one BGR frame; returns bboxes + 17 keypoints per person."""
        result = self.model.predict(
            frame, classes=[PERSON_CLASS_ID], device=self.device, verbose=False,
        )[0]
        boxes = result.boxes.cpu().numpy()
        keypoints = None if result.keypoints is None else result.keypoints.data.cpu().numpy()
        return PoseDetections.from_arrays(boxes.xyxy, boxes.conf, boxes.cls, keypoints)

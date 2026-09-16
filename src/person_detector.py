"""YOLO-based person detector.

Owns model loading, device selection, and running inference on a single
frame. This is the piece that would be swapped out if the detection
backend changes later (e.g. to an IMX500-based Raspberry Pi AI Camera) -
downstream code only depends on the `Boxes`-like object returned by
`detect()`, not on how it was produced.
"""

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

PERSON_CLASS_ID = 0  # COCO class index for "person"
MODEL_PATH = "yolov8n.pt"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class PersonDetector:
    """Runs YOLO person detection on individual frames."""

    def __init__(self, model_path: str = MODEL_PATH, device: str | None = None):
        self.device = device or get_device()
        self.model = YOLO(model_path)

    def detect(self, frame) -> Boxes:
        """Run person detection on one BGR frame.

        Returns an Ultralytics `Boxes` object (xyxy, xywh, conf, cls)
        detached to CPU/numpy, which is the format `PersonTracker` expects.
        """
        results = self.model.predict(
            frame, classes=[PERSON_CLASS_ID], device=self.device, verbose=False,
        )[0]
        return results.boxes.cpu().numpy()

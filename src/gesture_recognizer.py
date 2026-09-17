"""MediaPipe Tasks-based hand gesture recognition.

Wraps MediaPipe's `GestureRecognizer` task (current Tasks API, not the
deprecated `mp.solutions.hands` Solutions API) so gesture recognition stays
its own independently testable stage - mirroring how `person_detector.py`
wraps YOLO for the detection stage. This module never touches the YOLO/
BoT-SORT pipeline and does not depend on it.

Model: the stock MediaPipe "gesture_recognizer.task" bundle (canned gesture
classifier only, no custom gesture head configured). Out of the box, its
canned classifier recognizes these categories: "None", "Closed_Fist",
"Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou".
This directly covers the three gestures this project currently cares about:
    - V-sign      -> "Victory"   (renamed to "V_Sign" at the app level, see
                                   `_APP_GESTURE_NAMES` below)
    - Open Palm   -> "Open_Palm"
    - Thumbs Down -> "Thumb_Down"

V_Sign landmark fallback: webcam testing found the canned classifier
recognizes "Victory" reliably when the palm faces the camera, but often
fails to when the back of the hand faces the camera instead. Investigating
with the raw 21-point landmarks confirmed MediaPipe's hand landmark model
itself keeps detecting all 21 landmarks fine in that orientation (it doesn't
depend on which side of the hand faces the camera) - only the canned
classifier's own prediction becomes unreliable. Since the landmarks stay
usable, `_is_v_sign_pose()` below adds a small structural fallback (index +
middle extended, ring + pinky folded) that only runs when the canned
classifier's top prediction is not already "V_Sign", so it never touches
Open_Palm/Thumb_Down/etc. recognition. It is deliberately not meant to be
rotation-invariant in general - it only targets the specific palm/back-of-
hand flip.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    GestureRecognizer as _MPGestureRecognizer,
    GestureRecognizerOptions,
    RunningMode,
)

# Resolved relative to this file (not the caller's cwd) so this module works
# the same way whether it's run from the repo root or from src/ - unlike
# ultralytics' YOLO(), MediaPipe's BaseOptions has no auto-download fallback
# and hard-fails if the relative path doesn't resolve.
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "gesture_recognizer.task")

# Experimental starting values, not tuned - same spirit as the detection
# confidence thresholds elsewhere in this project.
NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Canned model category name -> application-level gesture name. Only
# "Victory" is renamed; every other canned category is passed through as-is.
_APP_GESTURE_NAMES = {"Victory": "V_Sign"}

CANNED = "CANNED"
LANDMARK_FALLBACK = "LANDMARK_FALLBACK"

# Landmark indices (standard 21-point MediaPipe hand model).
_WRIST = 0
_INDEX_PIP, _INDEX_TIP = 6, 8
_MIDDLE_PIP, _MIDDLE_TIP = 10, 12
_RING_PIP, _RING_TIP = 14, 16
_PINKY_PIP, _PINKY_TIP = 18, 20

# A finger is "extended" if its fingertip sits meaningfully farther from the
# wrist than its own PIP joint does, and "folded" if it sits meaningfully
# closer - both measured as a ratio of (tip-to-wrist) / (pip-to-wrist)
# distance. This only compares each finger's own joints to the wrist, so it
# does not depend on which absolute image direction the fingers point in,
# which is what makes it survive the palm/back-of-hand flip (unlike a raw
# "fingertip Y < PIP Y" image-coordinate check). Experimental starting
# values, not tuned; the gap between them is a dead zone where a finger's
# state counts as unclear rather than forcing a guess.
V_SIGN_EXTENDED_RATIO = 1.1
V_SIGN_FOLDED_RATIO = 0.9

# Fixed confidence reported for a fallback match: it's a structural yes/no
# rule, not a graded model probability, so no finer-grained score is claimed.
V_SIGN_FALLBACK_CONFIDENCE = 1.0


def _finger_state(landmarks: list[tuple[float, float]], pip_idx: int, tip_idx: int) -> str:
    """"extended", "folded", or "unclear" for one finger, relative to the wrist."""
    wrist = landmarks[_WRIST]
    pip = landmarks[pip_idx]
    tip = landmarks[tip_idx]
    pip_dist = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])
    tip_dist = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
    if pip_dist <= 0:
        return "unclear"
    ratio = tip_dist / pip_dist
    if ratio >= V_SIGN_EXTENDED_RATIO:
        return "extended"
    if ratio <= V_SIGN_FOLDED_RATIO:
        return "folded"
    return "unclear"


def _is_v_sign_pose(landmarks: list[tuple[float, float]]) -> bool:
    """Structural V-sign check: index + middle extended, ring + pinky folded.

    Any finger landing in the "unclear" dead zone fails the match rather than
    guessing - callers should keep whatever the canned classifier already
    said in that case, not force a V_Sign result.
    """
    return (
        _finger_state(landmarks, _INDEX_PIP, _INDEX_TIP) == "extended"
        and _finger_state(landmarks, _MIDDLE_PIP, _MIDDLE_TIP) == "extended"
        and _finger_state(landmarks, _RING_PIP, _RING_TIP) == "folded"
        and _finger_state(landmarks, _PINKY_PIP, _PINKY_TIP) == "folded"
    )


@dataclass
class HandGesture:
    """One detected hand, its top gesture prediction, and its landmarks for drawing.

    `landmarks` are MediaPipe's normalized (x, y) in [0, 1] relative to frame
    width/height - the caller converts to pixel coordinates for drawing.
    `source` is `CANNED` (MediaPipe's own classifier) or `LANDMARK_FALLBACK`
    (the V_Sign structural fallback in this module overrode it).
    """

    handedness: str  # "Left" or "Right", as reported by the model
    gesture_name: str
    confidence: float
    landmarks: list[tuple[float, float]]
    source: str = CANNED


class GestureRecognizer:
    """Runs MediaPipe's Tasks GestureRecognizer on individual video frames.

    Uses `RunningMode.VIDEO`: a synchronous, timestamp-ordered API matching
    the rest of this project's per-frame `detect()`/`update()` style, as
    opposed to `LIVE_STREAM` mode's async callback API.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        num_hands: int = NUM_HANDS,
        min_hand_detection_confidence: float = MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence: float = MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ):
        base_options = BaseOptions(model_asset_path=model_path)
        options = GestureRecognizerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._recognizer = _MPGestureRecognizer.create_from_options(options)

    def recognize(self, frame_rgb, timestamp_ms: int) -> list[HandGesture]:
        """Run gesture recognition on one RGB frame.

        Args:
            frame_rgb: A single frame as RGB (not BGR) - the caller is
                responsible for any colorspace conversion, keeping this
                module free of an OpenCV-specific assumption.
            timestamp_ms: Monotonically increasing timestamp in milliseconds,
                required by `RunningMode.VIDEO`.

        Returns:
            One `HandGesture` per detected hand, each carrying the model's
            top-1 gesture prediction (or the V_Sign landmark fallback's, see
            module docstring). Empty list if no hand is detected.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)

        hands = []
        for gestures, handedness, hand_landmarks in zip(
            result.gestures, result.handedness, result.hand_landmarks
        ):
            top_gesture = gestures[0]
            top_handedness = handedness[0]
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks]

            gesture_name = _APP_GESTURE_NAMES.get(top_gesture.category_name, top_gesture.category_name)
            confidence = top_gesture.score
            source = CANNED

            if gesture_name != "V_Sign" and _is_v_sign_pose(landmarks):
                gesture_name = "V_Sign"
                confidence = V_SIGN_FALLBACK_CONFIDENCE
                source = LANDMARK_FALLBACK

            hands.append(
                HandGesture(
                    handedness=top_handedness.category_name,
                    gesture_name=gesture_name,
                    confidence=confidence,
                    landmarks=landmarks,
                    source=source,
                )
            )
        return hands

    def close(self) -> None:
        self._recognizer.close()

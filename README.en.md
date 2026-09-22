# Drone Vision AI

[한국어](README.md) | [English](README.en.md)

Vision-based computer vision project for a drone (ECE Senior Design / AI Product — a person-following drone).

## AI Pipeline (full target)

```
Camera
  → Person Detection
  → Person Tracking
  → Gesture Recognition
  → Gesture-to-Person Association
  → Target Selection
  → Target-Loss Handling
  → State Management
  → GCS Visualization
```

## Current Status

| Stage | Status |
|---|---|
| Person Detection (YOLOv8n) | ✅ DONE |
| Person Tracking (BoT-SORT) | ✅ BASELINE ESTABLISHED |
| Track Validation (CANDIDATE → CONFIRMED) | ✅ BASELINE ESTABLISHED |
| Gesture Recognition (MediaPipe) | ✅ BASELINE ESTABLISHED (standalone module) |
| Gesture-to-Person Association | 🔜 In design (NEXT) |
| Target Selection | ⬜ Not implemented |
| Target-Loss Handling | ⬜ Not implemented |
| State Management / Drone Control | ⬜ Not implemented |
| GCS Visualization | ⬜ Not implemented |

### Person Detection & Tracking

- **YOLOv8n** detects only the "person" class.
- **BoT-SORT** (ReID disabled) assigns persistent track IDs across frames.
- `TrackValidator` promotes a raw track ID from `CANDIDATE` to `CONFIRMED` purely by time (CONFIRMED after 0.5s of continuous observation, expires after 2.0s missing). A proximity-based promotion gate was tried but removed after real two-person testing showed it could false-negative a genuine second person standing close to someone already CONFIRMED; the logic is now purely time-based.

### Gesture Recognition

- **MediaPipe Tasks `GestureRecognizer`** (VIDEO mode), implemented as a module fully independent of the person-tracking pipeline.
- Target gestures: `V_Sign`, `Open_Palm`, `Thumb_Down`.
- Found that MediaPipe's canned classifier misses the V-sign when the back of the hand faces the camera. Added a landmark-based fallback (index + middle extended, ring + pinky folded, using the 21 hand landmarks) to fix it. Verified on real webcam testing that both palm-facing and back-of-hand V-signs pass.

## Software Structure

```
src/
├── person_detector.py        # YOLOv8n person detection
├── person_tracker.py         # BoT-SORT tracking → persistent track ID
├── track_validator.py        # CANDIDATE/CONFIRMED state management
├── gesture_recognizer.py     # MediaPipe-based hand gesture recognition
├── gesture_test.py           # standalone webcam test for gesture recognition
├── yolo_person_detection.py  # person detection + tracking integration demo
├── webcam_test.py            # minimal webcam connectivity test
├── geometry_utils.py         # shared pure bbox-geometry functions
└── diagnostics/               # read-only dev diagnostics, kept separate from the core pipeline
    ├── tracking_diagnostics.py   # tracking fragmentation diagnostics
    └── proximity_diagnostics.py  # proximity-situation diagnostics (evaluation only)
```

`diagnostics/` holds developer-only tools used when `DEBUG=True`, kept separate from the core pipeline files.

Person detection/tracking and gesture recognition currently run as independent modules and are not yet connected — connecting them is the next step.

## Next Step

**Gesture-to-Person Association** — designing the logic that determines which CONFIRMED person performed a detected gesture. Target selection and drone/motor control are not part of this stage.

## Structure

- `src/` — source code
- `tests/` — test code
- `models/` — local model files (YOLO, MediaPipe gesture bundle, etc.), excluded from Git

## Setup

```bash
pip install -r requirements.txt
```

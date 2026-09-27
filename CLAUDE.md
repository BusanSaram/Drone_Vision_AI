This is an ECE Senior Design / AI Product project for a vision-based person-following drone.

## Current implementation (IMPLEMENTED)

Webcam → YOLOv8n Person Detection → BoT-SORT Person Tracking → Track Validation (CANDIDATE/CONFIRMED) → OpenCV visualization (bounding boxes, track IDs, confidence, validation state, FPS).

This runs on a development PC with a USB/laptop webcam. Nothing downstream of Track Validation exists yet.

## Intended future architecture (PLANNED / NOT IMPLEMENTED)

Camera
→ Person Detection [IMPLEMENTED]
→ Person Tracking [IMPLEMENTED]
→ Track Validation [IMPLEMENTED]
→ MediaPipe Pose [PLANNED / NOT IMPLEMENTED]
→ Body-Pose Commands [PLANNED / NOT IMPLEMENTED]
→ Target Selection [PLANNED / NOT IMPLEMENTED]
→ Person-Following Control [PLANNED / NOT IMPLEMENTED]
→ MAVLink [PLANNED / NOT IMPLEMENTED]
→ ArduPilot [PLANNED / NOT IMPLEMENTED]

Planned body-pose commands (full-body MediaPipe Pose landmarks, not hand gestures):
- One arm raised above the head → Target Select / Follow
- T-pose → Stop Following / Hover
- Double-biceps pose → Land

The earlier MediaPipe hand GestureRecognizer approach (small hand/finger gestures) is obsolete and has been removed. Do not reintroduce it.

Eventual onboard hardware direction (PLANNED / NOT IMPLEMENTED):
- Raspberry Pi 5 (4GB)
- Sony IMX500 AI Camera
- Custom STM32H7 flight controller running ArduPilot, commanded over MAVLink

ESP32, MSP, and Betaflight are not part of the current architecture.

## Architecture rules

Keep the software modular and testable. Detection, tracking, track validation, pose recognition, pose-to-person association, target selection, following control, flight-controller communication, and UI should remain separate responsibilities. Avoid one large script.

## Development rules

- Implement one subsystem at a time, and only the stage the student explicitly requests. Planned stages above are not approved for implementation until requested.
- Prefer simple and reliable solutions.
- Do not add unrequested features.
- Do not silently change requirements.
- Do not tune detection/tracking/validation thresholds or tracker parameters unless explicitly asked.
- Explain significant architecture or dependency decisions.
- Test each phase before proceeding.
- Do not fabricate test results, performance measurements, or evidence.
- Clearly distinguish implemented behavior from planned behavior.
- AI-generated code must remain understandable and reviewable by the student.

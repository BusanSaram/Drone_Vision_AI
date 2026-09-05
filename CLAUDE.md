I want to expand CLAUDE.md before writing any application code.

This is an ECE Senior Design / AI Product project for a vision-based person-following drone.

The AI Product pipeline is:
Camera → Person Detection → Person Tracking → Gesture Recognition → Gesture-to-Person Association → Target Selection → Target-Loss Handling → State Management → GCS Visualization.

Planned technologies are YOLO for person detection, a person tracker for persistent IDs, MediaPipe for hand gestures, and Python for the GCS vision prototype.

Keep the software modular and testable. Detection, tracking, gesture recognition, target association, state logic, and UI should remain separate responsibilities. Avoid one large script.

For now, only Phase 1 is allowed:
Camera → Person Detection → Person Tracking → Bounding Boxes / Tracking IDs / Confidence / FPS.

Do not implement gestures, target selection, drone control, UDP, ESP32, MSP, Betaflight, or motor control yet.

Development rules:

Implement one subsystem at a time.
Prefer simple and reliable solutions.
Do not add unrequested features.
Do not silently change requirements.
Explain significant architecture or dependency decisions.
Test each phase before proceeding.
Do not fabricate test results, performance measurements, or evidence.
Clearly distinguish implemented behavior from planned behavior.
AI-generated code must remain understandable and reviewable by the student.

Update only CLAUDE.md for now. Do not implement application code yet.

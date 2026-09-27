# Drone Vision AI

[English](#english) | [한국어](#한국어)

---

## English

Vision-based computer vision project for a drone (ECE Senior Design / AI Product — a person-following drone).

### Implemented pipeline (current)

```
Webcam
  → YOLOv8n Person Detection
  → BoT-SORT Person Tracking
  → Track Validation (CANDIDATE → CONFIRMED)
  → Visualization (OpenCV window)
```

This is everything the software does today. It runs on a development PC with a webcam.

### Planned pipeline (future target)

```
Camera
  → YOLOv8n Person Detection      [IMPLEMENTED]
  → BoT-SORT Person Tracking      [IMPLEMENTED]
  → Track Validation              [IMPLEMENTED]
  → MediaPipe Pose                [PLANNED / NOT IMPLEMENTED]
  → Body-Pose Recognition         [PLANNED]
  → Pose-to-Person Association    [PLANNED]
  → Target Selection              [PLANNED]
  → Person-Following Control      [PLANNED]
  → MAVLink                       [PLANNED]
  → ArduPilot                     [PLANNED]
```

Planned body-pose commands (full-body pose, not hand gestures — none implemented yet):

| Pose | Planned command |
|---|---|
| One arm raised above the head | Target Select / Follow |
| T-pose | Stop Following / Hover |
| Double-biceps pose | Land |

Planned onboard hardware (not implemented): Raspberry Pi 5 (4GB), Sony IMX500 AI Camera, custom STM32H7 flight controller running ArduPilot.

### Current Status

| Stage | Status |
|---|---|
| Person Detection (YOLOv8n) | ✅ DONE |
| Person Tracking (BoT-SORT) | ✅ BASELINE ESTABLISHED |
| Track Validation (CANDIDATE → CONFIRMED) | ✅ BASELINE ESTABLISHED |
| MediaPipe Pose | ⬜ Not implemented (NEXT) |
| Body-Pose Recognition | ⬜ Not implemented |
| Pose-to-Person Association | ⬜ Not implemented |
| Target Selection | ⬜ Not implemented |
| Person-Following Control | ⬜ Not implemented |
| MAVLink / ArduPilot | ⬜ Not implemented |

#### Person Detection & Tracking

- **YOLOv8n** detects only the "person" class.
- **BoT-SORT** (ReID disabled) assigns persistent track IDs across frames.
- `TrackValidator` promotes a raw track ID from `CANDIDATE` to `CONFIRMED` purely by time (CONFIRMED after 0.5s of continuous observation, expires after 2.0s missing). A proximity-based promotion gate was tried but removed after real two-person testing showed it could false-negative a genuine second person standing close to someone already CONFIRMED; the logic is now purely time-based.

### Software Structure

```
src/
├── person_detector.py        # YOLOv8n person detection
├── person_tracker.py         # BoT-SORT tracking → persistent track ID
├── track_validator.py        # CANDIDATE/CONFIRMED state management
├── yolo_person_detection.py  # person detection + tracking integration demo (entry point)
├── geometry_utils.py         # shared pure bbox-geometry functions
└── diagnostics/               # read-only dev diagnostics, kept separate from the core pipeline
    ├── tracking_diagnostics.py   # tracking fragmentation diagnostics
    └── proximity_diagnostics.py  # proximity-situation diagnostics (evaluation only)
```

`diagnostics/` holds developer-only tools used when `DEBUG=True`, kept separate from the core pipeline files.

### Next Step

**MediaPipe Pose** — setting up full-body pose landmarks as the basis for the planned body-pose commands. Not yet installed, configured, or implemented. Target selection and drone control are not part of this stage.

### Structure

- `src/` — source code
- `tests/` — test code (no automated tests yet)
- `models/` — local model files, excluded from Git

### Setup

```bash
pip install -r requirements.txt
```

---

## 한국어

드론 영상 기반 컴퓨터 비전 프로젝트 (ECE Senior Design / AI Product — 사람을 인식하고 따라가는 person-following 드론).

### 구현된 파이프라인 (현재)

```
Webcam
  → YOLOv8n Person Detection
  → BoT-SORT Person Tracking
  → Track Validation (CANDIDATE → CONFIRMED)
  → Visualization (OpenCV 창)
```

현재 소프트웨어가 실제로 하는 일은 여기까지다. 웹캠이 연결된 개발용 PC에서 동작한다.

### 계획된 파이프라인 (향후 목표)

```
Camera
  → YOLOv8n Person Detection      [구현됨]
  → BoT-SORT Person Tracking      [구현됨]
  → Track Validation              [구현됨]
  → MediaPipe Pose                [계획 / 미구현]
  → Body-Pose Recognition         [계획]
  → Pose-to-Person Association    [계획]
  → Target Selection              [계획]
  → Person-Following Control      [계획]
  → MAVLink                       [계획]
  → ArduPilot                     [계획]
```

계획된 body-pose 명령 (손 제스처가 아닌 전신 자세 — 아직 모두 미구현):

| 자세 | 계획된 명령 |
|---|---|
| 한쪽 팔을 머리 위로 들기 | Target Select / Follow |
| T-pose | Stop Following / Hover |
| 양팔 이두근 포즈 (Double-biceps) | Land |

계획된 온보드 하드웨어 (미구현): Raspberry Pi 5 (4GB), Sony IMX500 AI Camera, ArduPilot을 실행하는 커스텀 STM32H7 비행 컨트롤러.

### 현재 진행 상황 (Current Status)

| 단계 | 상태 |
|---|---|
| Person Detection (YOLOv8n) | ✅ DONE |
| Person Tracking (BoT-SORT) | ✅ BASELINE ESTABLISHED |
| Track Validation (CANDIDATE → CONFIRMED) | ✅ BASELINE ESTABLISHED |
| MediaPipe Pose | ⬜ 미구현 (NEXT) |
| Body-Pose Recognition | ⬜ 미구현 |
| Pose-to-Person Association | ⬜ 미구현 |
| Target Selection | ⬜ 미구현 |
| Person-Following Control | ⬜ 미구현 |
| MAVLink / ArduPilot | ⬜ 미구현 |

#### Person Detection & Tracking

- **YOLOv8n**으로 사람(person) 클래스만 검출.
- **BoT-SORT**(ReID 비활성화)로 프레임 간 persistent track ID 부여.
- `TrackValidator`가 raw track ID를 시간 기반으로 `CANDIDATE` → `CONFIRMED` 상태로 승격한다 (0.5초 연속 관측 시 CONFIRMED, 2.0초 이상 미관측 시 만료). 근접(proximity) 기반 승격 게이팅을 실험했으나 실제 두 사람 테스트에서 정상 사람까지 차단하는 오탐(false negative)이 발생해 제거했고, 현재는 순수 시간 기반 로직만 사용한다.

### Software Structure

```
src/
├── person_detector.py        # YOLOv8n person detection
├── person_tracker.py         # BoT-SORT tracking → persistent track ID
├── track_validator.py        # CANDIDATE/CONFIRMED 상태 관리
├── yolo_person_detection.py  # person detection + tracking 통합 데모 (실행 진입점)
├── geometry_utils.py         # bbox geometry 공용 순수 함수
└── diagnostics/               # 핵심 파이프라인과 분리된 read-only 개발용 진단 도구
    ├── tracking_diagnostics.py   # 추적 fragmentation 진단
    └── proximity_diagnostics.py  # 근접 상황 진단 (평가용)
```

`diagnostics/`는 `DEBUG=True`일 때만 사용되는 개발자용 도구로, 핵심 파이프라인 파일들과 구분해두었다.

### Next Step

**MediaPipe Pose** — 계획된 body-pose 명령의 기반이 될 전신 pose landmark를 준비하는 단계. 아직 설치·설정·구현되지 않았다. Target selection과 드론 제어는 이 단계에 포함되지 않는다.

### Structure

- `src/` — 소스 코드
- `tests/` — 테스트 코드 (아직 자동화 테스트 없음)
- `models/` — 로컬 모델 파일, Git에서 제외됨

### Setup

```bash
pip install -r requirements.txt
```

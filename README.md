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
  → MediaPipe Pose                [STANDALONE TEST ONLY / NOT INTEGRATED]
  → Body-Pose Recognition         [STANDALONE TEST ONLY / NOT INTEGRATED]
  → Pose-to-Person Association    [DEMO ONLY / BEING VERIFIED]
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
| MediaPipe Pose | 🔧 Standalone webcam test (`pose_test.py`), not integrated with tracking |
| Body-Pose Recognition | 🔧 Standalone (`body_pose_recognizer.py` + hold-time stabilizer), shown in `pose_test.py`; visual labels only, not integrated |
| Pose-to-Person Association | 🔧 `pose_track_associator.py` + webcam demo `pose_track_demo.py`; real multi-person behavior not yet verified |
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
├── pose_test.py              # standalone MediaPipe Pose + body-pose command webcam test (not connected to tracking)
├── body_pose_recognizer.py   # single-frame ONE_ARM_UP / T_POSE / DOUBLE_BICEPS classification from pose landmarks
├── pose_command_stabilizer.py # hold-time stabilizer: pose must be held before it becomes an active command
├── pose_track_associator.py  # matches each MediaPipe pose to a tracked person's bbox (track ID)
├── pose_track_demo.py        # webcam demo: detection + tracking + pose, shows each track ID's pose
├── geometry_utils.py         # shared pure bbox-geometry functions
└── diagnostics/               # read-only dev diagnostics, kept separate from the core pipeline
    ├── tracking_diagnostics.py   # tracking fragmentation diagnostics
    └── proximity_diagnostics.py  # proximity-situation diagnostics (evaluation only)
```

`diagnostics/` holds developer-only tools used when `DEBUG=True`, kept separate from the core pipeline files.

### Next Step

**Pose-to-Person Association** — `src/pose_track_demo.py` runs YOLOv8n → BoT-SORT → TrackValidator and MediaPipe Pose (up to 3 people) on the same frame, then `pose_track_associator.py` attaches each pose to at most one CONFIRMED track ID (ambiguous matches are left unassociated). The demo shows each track's raw per-frame pose. The standalone `src/pose_test.py` (one person, 0.7 s hold → TARGET_SELECT / HOVER / LAND labels) is unchanged. Nothing selects a target or triggers drone behavior yet.

### Structure

- `src/` — source code
- `tests/` — synthetic unit tests (`.venv\Scripts\python.exe -m unittest discover -s tests -v`)
- `models/` — local model files, excluded from Git (`pose_test.py` needs `models/pose_landmarker_lite.task`, see Setup)

### Setup

```bash
pip install -r requirements.txt
```

MediaPipe Pose model for `pose_test.py` (not in Git):

```bash
curl -L -o models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
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
  → MediaPipe Pose                [단독 테스트만 / 미통합]
  → Body-Pose Recognition         [단독 테스트만 / 미통합]
  → Pose-to-Person Association    [데모만 / 검증 중]
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
| MediaPipe Pose | 🔧 단독 웹캠 테스트 (`pose_test.py`), tracking과 미연결 |
| Body-Pose Recognition | 🔧 단독 모듈 (`body_pose_recognizer.py` + hold-time stabilizer), `pose_test.py`에서 표시; 시각적 라벨만, 미통합 |
| Pose-to-Person Association | 🔧 `pose_track_associator.py` + 웹캠 데모 `pose_track_demo.py`; 실제 다인원 동작은 미검증 |
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
├── pose_test.py              # MediaPipe Pose + body-pose 명령 단독 웹캠 테스트 (tracking과 미연결)
├── body_pose_recognizer.py   # pose landmark 기반 단일 프레임 ONE_ARM_UP / T_POSE / DOUBLE_BICEPS 분류
├── pose_command_stabilizer.py # hold-time stabilizer: 일정 시간 유지해야 명령이 활성화됨
├── pose_track_associator.py  # 각 MediaPipe pose를 추적 중인 사람의 bbox(track ID)와 매칭
├── pose_track_demo.py        # 웹캠 데모: detection + tracking + pose, track ID별 pose 표시
├── geometry_utils.py         # bbox geometry 공용 순수 함수
└── diagnostics/               # 핵심 파이프라인과 분리된 read-only 개발용 진단 도구
    ├── tracking_diagnostics.py   # 추적 fragmentation 진단
    └── proximity_diagnostics.py  # 근접 상황 진단 (평가용)
```

`diagnostics/`는 `DEBUG=True`일 때만 사용되는 개발자용 도구로, 핵심 파이프라인 파일들과 구분해두었다.

### Next Step

**Pose-to-Person Association** — `src/pose_track_demo.py`가 같은 프레임에서 YOLOv8n → BoT-SORT → TrackValidator와 MediaPipe Pose(최대 3명)를 실행하고, `pose_track_associator.py`가 각 pose를 최대 하나의 CONFIRMED track ID에 연결한다 (애매한 경우 연결하지 않음). 데모는 track별 프레임 단위 raw pose를 표시한다. 단독 `src/pose_test.py`(1명, 0.7초 유지 → TARGET_SELECT / HOVER / LAND 라벨)는 그대로다. 아직 target 선택이나 드론 동작은 없다.

### Structure

- `src/` — 소스 코드
- `tests/` — synthetic 단위 테스트 (`.venv\Scripts\python.exe -m unittest discover -s tests -v`)
- `models/` — 로컬 모델 파일, Git에서 제외됨 (`pose_test.py`는 `models/pose_landmarker_lite.task` 필요, Setup 참고)

### Setup

```bash
pip install -r requirements.txt
```

`pose_test.py`용 MediaPipe Pose 모델 (Git에 포함되지 않음):

```bash
curl -L -o models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

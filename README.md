# Drone Vision AI

드론 영상 기반 컴퓨터 비전 프로젝트 (ECE Senior Design / AI Product — 사람을 인식하고 따라가는 person-following 드론).

## AI Pipeline (전체 목표)

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

## 현재 진행 상황 (Current Status)

| 단계 | 상태 |
|---|---|
| Person Detection (YOLOv8n) | ✅ DONE |
| Person Tracking (BoT-SORT) | ✅ BASELINE ESTABLISHED |
| Track Validation (CANDIDATE → CONFIRMED) | ✅ BASELINE ESTABLISHED |
| Gesture Recognition (MediaPipe) | ✅ BASELINE ESTABLISHED (독립 모듈) |
| Gesture-to-Person Association | 🔜 설계 중 (NEXT) |
| Target Selection | ⬜ 미구현 |
| Target-Loss Handling | ⬜ 미구현 |
| State Management / Drone Control | ⬜ 미구현 |
| GCS Visualization | ⬜ 미구현 |

### Person Detection & Tracking

- **YOLOv8n**으로 사람(person) 클래스만 검출.
- **BoT-SORT**(ReID 비활성화)로 프레임 간 persistent track ID 부여.
- `TrackValidator`가 raw track ID를 시간 기반으로 `CANDIDATE` → `CONFIRMED` 상태로 승격한다 (0.5초 연속 관측 시 CONFIRMED, 2.0초 이상 미관측 시 만료). 근접(proximity) 기반 승격 게이팅을 실험했으나 실제 두 사람 테스트에서 정상 사람까지 차단하는 오탐(false negative)이 발생해 제거했고, 현재는 순수 시간 기반 로직만 사용한다.

### Gesture Recognition

- **MediaPipe Tasks `GestureRecognizer`** (VIDEO 모드)를 사람 추적 파이프라인과 완전히 독립된 모듈로 구현.
- 인식 대상 제스처: `V_Sign`, `Open_Palm`, `Thumb_Down`.
- 손등이 카메라를 향할 때 MediaPipe의 canned classifier가 V-sign을 놓치는 문제를 발견했고, 21개 hand landmark의 구조(검지·중지 펴짐, 약지·새끼 접힘)를 이용한 landmark fallback을 추가해 해결했다. 실제 웹캠 테스트에서 palm-facing/back-of-hand 양쪽 V-sign 모두 통과.

## Software Structure

```
src/
├── person_detector.py        # YOLOv8n person detection
├── person_tracker.py         # BoT-SORT tracking → persistent track ID
├── track_validator.py        # CANDIDATE/CONFIRMED 상태 관리
├── gesture_recognizer.py     # MediaPipe 기반 hand gesture 인식
├── gesture_test.py           # gesture recognition 단독 웹캠 테스트
├── yolo_person_detection.py  # person detection + tracking 통합 데모
├── webcam_test.py            # 웹캠 연결 확인용 최소 테스트
├── geometry_utils.py         # bbox geometry 공용 순수 함수
└── diagnostics/               # 핵심 파이프라인과 분리된 read-only 개발용 진단 도구
    ├── tracking_diagnostics.py   # 추적 fragmentation 진단
    └── proximity_diagnostics.py  # 근접 상황 진단 (평가용)
```

`diagnostics/`는 `DEBUG=True`일 때만 사용되는 개발자용 도구로, 핵심 파이프라인 파일들과 구분해두었다.

Person detection/tracking과 gesture recognition은 각각 독립적으로 동작하는 모듈이며, 아직 서로 연결되어 있지 않다. 두 모듈을 연결하는 것이 다음 단계다.

## Next Step

**Gesture-to-Person Association** — 어떤 CONFIRMED 사람이 어떤 제스처를 취했는지 연결하는 로직을 설계 중이다. Target selection과 drone/motor control은 이 단계에 포함되지 않는다.

## Structure

- `src/` — 소스 코드
- `tests/` — 테스트 코드
- `models/` — 로컬 모델 파일 (YOLO, MediaPipe gesture bundle 등, Git에서 제외됨)

## Setup

```bash
pip install -r requirements.txt
```

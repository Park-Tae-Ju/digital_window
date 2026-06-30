# Jetson Nano 구현체

이 폴더는 실제 Jetson Nano 하드웨어에서 동작하는 디지털 창문 시스템 코드입니다.

---

## 폴더 구조

```
jetson/
├── experiments/
│   ├── main_frame_tactic.py    # 실험 A: 창틀 오버레이
│   └── main4_asynchronous.py   # 실험 B: 비동기 영상 읽기
├── main_integrated.py          # 실험 A + B를 통합한 최종 버전
└── assets/                     # 배경 영상, 창틀 이미지 (별도 준비 필요)
```

---

## 개발 과정 및 실험 배경

이 프로젝트는 하나의 파일로 바로 작성된 것이 아니라, 여러번의 실험을 해보면서 개발했습니다. 
그중 두 가지 실험이 서로 다른 방향으로 어느정도의 개선이 있어서, 둘을 통합하기로 결정했습니다.

### 실험 A: `main_frame_tactic.py` — 창틀 오버레이

배경 영상 위에 투명 배경(Alpha 채널)을 가진 창틀 PNG 이미지를 합성하여, 실제 "창문"처럼 보이도록 만들어본 버전입니다. 
또한 얼굴과의 거리에 따라 Red Zone(중앙 고정 구역)의 크기가 동적으로 변하도록 캘리브레이션 로직을 도입했습니다.
단, PC 버전에서는 다소의 문제가 있을 수 있습니다. 

### 실험 B: `main4_asynchronous.py` — 비동기 영상 읽기

배경 영상을 읽는 작업을 별도의 스레드(`BackgroundVideoReader`)로 분리하여, 메인 루프가 영상 디코딩을 기다리지 않고 항상 최신 프레임을 즉시 가져올 수 있도록 한 버전입니다. 이전 버전들은 메인 루프에서 직접 `cap.read()`를 호출했기 때문에, 디코딩 지연이 발생하면 화면 반응성도 함께 떨어지는 문제가 있었습니다.

### 통합 결정: `main_integrated.py`

두 실험은 서로 다른 문제를 해결했기 때문에 함께 적용해도 충돌하지 않는다고 판단했습니다. 비동기 읽기 구조(실험 B)는 반응성을, 창틀 오버레이(실험 A)는 시각적 완성도를 담당하므로, 이 둘을 결합하여 두 장점을 모두 가진 최종 버전을 만들었습니다.

실험 원본 두 파일은 영상 및 창틀 프레임 등의 경로만 제외하고, 수정 없이 `experiments/` 폴더에 그대로 보존했습니다.

---

## 핵심 기술 결정

### 1. 시그모이드 기반 좌표 보간

**문제**: 얼굴 미감지 시 화면이 원점으로 급격히 복귀하면, 이후 얼굴이 재감지될 때 화면이 갑자기 점프하여 사용자가 어지러움을 느낄 수 있습니다.
**해결**: 거리 차이에 비례하는 동적 보간 계수(`XY_SMOOTHING`, `ZOOM_SMOOTHING`)를 적용하여, 거리가 멀수록 부드럽게, 가까울수록 빠르게 반응하도록 설계했습니다. 해당 과정에서는 시그모이드 함수를 기반으로 좌표를 보간했습니다.

### 2. Red Zone / Green Zone 기반 추적

**문제**: 얼굴이 화면 중앙 근처의 미세한 움직임에도 계속 반응하면, 가만히 있어도 배경이 미세하게 흔들려 불편함을 줍니다.
**해결**: 얼굴이 중앙 일정 범위(Red Zone) 안에 있으면 배경을 정중앙으로 고정하고, 그 범위를 벗어났을 때만(Green Zone) 실제 위치를 따라가도록 했습니다. Red Zone의 크기는 캘리브레이션된 기준 얼굴 크기 대비 비율로 동적으로 조정되어, 사용자가 카메라에 가까이 있든 멀리 있든 일관되게 움직이는 상황을 연출하고자 했습니다.

### 3. 비동기 영상 읽기

**문제**: 메인 루프에서 직접 영상을 디코딩하면, 디코딩 시간만큼 얼굴 추적에 대한 화면 반응이 지연됩니다.
**해결**: 영상 읽기를 별도 스레드로 분리하고, Lock으로 보호된 최신 프레임만 메인 루프가 즉시 가져가도록 구조를 변경했습니다.

### 4. 창틀 오버레이 합성

**문제**: 배경 영상만으로는 "창문"이라는 컨셉이 직관적으로 전달되지 않습니다.
**해결**: 알파 채널을 가진 창틀 PNG 이미지를 배경 위에 합성하여, 실제 창문틀 안에서 풍경이 움직이는 듯한 느낌을 주도록 했습니다.

---

## 실행 환경

- Hardware: Jetson Nano
- 카메라: V4L2 호환 카메라
- GStreamer (NVDEC 하드웨어 가속 디코딩)
- jetson-inference, jetson-utils (facenet-120 모델)

## 의존성 설치

```bash
pip install opencv-python numpy
# jetson-inference, jetson-utils는 NVIDIA 공식 가이드에 따라 별도 설치 필요
```

## 실행

`main_integrated.py` 상단의 `BG_VIDEO_PATH`, `FRAME_IMAGE_PATH`를 본인의 환경에 맞게 수정한 후 실행합니다.

```bash
python3 main_integrated.py
```

종료하려면 `Q` 키를 누르세요.

---

## 알려진 한계

- `BG_VIDEO_PATH`, `FRAME_IMAGE_PATH` 등 일부 설정값이 코드에 하드코딩되어 있어, 다른 환경에서 실행하려면 직접 경로를 수정해야 합니다.
- 캘리브레이션은 매 실행마다 새로 진행해야 하며, 저장/불러오기 기능은 없습니다.
- 단일 얼굴 추적만 지원하며, 여러 명이 동시에 화면 앞에 있을 경우의 동작은 별도로 정의되어 있지 않습니다.

<br>

---

<br>

# Jetson Nano Implementation

This folder contains the implementation of the Digital Window system that runs on actual Jetson Nano hardware.

---

## Folder Structure

```
jetson/
├── experiments/
│   ├── main_frame_tactic.py    # Experiment A: Window frame overlay
│   └── main4_asynchronous.py   # Experiment B: Asynchronous video reading
├── main_integrated.py          # Final version integrating Experiments A + B
└── assets/                     # Background video, frame image (prepare separately)
```

---

## Development Process and Experiment Background

This project was not written as a single finished version from the start; it evolved through several rounds of iterative experimentation. Two experiments achieved meaningful improvements in different directions, so they were integrated into a single version.

### Experiment A: `main_frame_tactic.py` — Window Frame Overlay

This version composites a window frame PNG image with an alpha channel onto the background video, going beyond simply displaying a video to visually resemble an actual "window." It also introduced a calibration mechanism so that the Red Zone (the center-lock region) dynamically resizes based on the viewer's distance from the camera.

### Experiment B: `main4_asynchronous.py` — Asynchronous Video Reading

This version separates background video reading into its own thread (`BackgroundVideoReader`), allowing the main loop to always grab the latest frame immediately instead of waiting on video decoding. In earlier versions, `cap.read()` was called directly in the main loop, so decoding delays directly degraded display responsiveness.

### Integration Decision: `main_integrated.py`

Since the two experiments solved different problems, they were judged not to conflict with each other. The asynchronous reading structure (Experiment B) handles responsiveness, while the frame overlay (Experiment A) handles visual polish. Combining both produced a final version with both strengths.

The two original experiment files are preserved unmodified in the `experiments/` folder.

---

## Key Technical Decisions

### 1. Sigmoid-based Coordinate Interpolation

**Problem**: If the display snaps back to center when the face is lost, the sudden jump when the face is re-detected causes visual discomfort.

**Solution**: A dynamic interpolation factor (`XY_SMOOTHING`, `ZOOM_SMOOTHING`) proportional to the distance difference was applied, so the response is smoother at larger distances and faster at smaller ones.

### 2. Red Zone / Green Zone-based Tracking

**Problem**: If the display reacts to every minor movement near the center of the frame, the background flickers slightly even when the viewer is mostly still, creating discomfort.

**Solution**: When the face stays within a defined central area (Red Zone), the background is locked to center; only when it leaves that area (Green Zone) does the background follow the actual position. The Red Zone size is dynamically scaled relative to the calibrated base face size, keeping the experience consistent regardless of how close or far the viewer is from the camera.

### 3. Asynchronous Video Reading

**Problem**: Decoding video directly in the main loop delays the display's response to face tracking by the decoding time.

**Solution**: Video reading was separated into its own thread, with the main loop immediately grabbing only the latest frame, protected by a lock.

### 4. Window Frame Overlay Compositing

**Problem**: A background video alone does not intuitively convey the "window" concept.

**Solution**: A window frame PNG image with an alpha channel is composited over the background, creating the impression of a landscape moving within an actual window frame.

---

## Runtime Environment

- Hardware: Jetson Nano
- Camera: V4L2-compatible camera
- GStreamer (NVDEC hardware-accelerated decoding)
- jetson-inference, jetson-utils (facenet-120 model)

## Installing Dependencies

```bash
pip install opencv-python numpy
# jetson-inference and jetson-utils must be installed separately per NVIDIA's official guide
```

## Running

Update `BG_VIDEO_PATH` and `FRAME_IMAGE_PATH` at the top of `main_integrated.py` to match your environment, then run:

```bash
python3 main_integrated.py
```

Press `Q` to quit.

---

## Known Limitations

- Some configuration values such as `BG_VIDEO_PATH` and `FRAME_IMAGE_PATH` are hardcoded and must be manually updated to run in a different environment.
- Calibration must be redone on every run; there is no save/load functionality.
- Only single-face tracking is supported; behavior when multiple people are in front of the camera simultaneously is not explicitly defined.

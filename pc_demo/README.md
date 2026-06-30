# PC 데모 버전

Jetson Nano나 별도의 센서 없이, 일반 PC의 웹캠만으로 디지털 창문의 동작을 체험할 수 있는 데모입니다.
프로그램 실행 시, 최초 얼굴 탐색에 대한 시간이 다소 소요되니 기다려주세요.
---

## Jetson 버전과의 차이

이 데모는 [jetson/main_integrated.py](../jetson/main_integrated.py)를 기반으로, 하드웨어 종속적인 부분만 일반 PC에서 동작 가능한 방식으로 교체한 버전입니다. 

| 구분 | Jetson 버전 | PC 데모 버전 |
|---|---|---|
| 얼굴 인식 | jetson_inference (facenet-120, GPU) | MediaPipe Face Detection |
| 영상 디코딩 | GStreamer + NVDEC 하드웨어 가속 | OpenCV 기본 디코딩 |
| 카메라 입력 | V4L2 전용 | OS 무관 |

---

## 요구 환경

- Python 3.8 이상 (3.10 권장)
- 웹캠
- OS: Windows, macOS, Linux 모두 지원

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 실행 전 준비

`assets/` 폴더에 다음 파일을 준비해 주세요.

```
pc_demo/
└── assets/
    ├── demo_background.mp4     # 배경 영상 (필수)
    └── window_frame.png        # 창틀 이미지, 알파 채널 포함 (선택)
```

창틀 이미지가 없어도 실행은 가능하며, 이 경우 오버레이 없이 배경 영상만 표시됩니다.

---

## 실행

```bash
python3 main_demo.py
```

처음 실행 시 카메라 앞에서 약 5초간 가만히 있어 주세요. 얼굴 크기를 기준값으로 캘리브레이션하는 과정입니다.  
종료하려면 `Q` 키를 누르시면 됩니다.
프로그램 실행 시, 최초 얼굴 탐색에 대한 시간이 다소 소요되니 기다려주세요.  

---

## 동작 방식

1. 웹캠으로 얼굴을 인식하고 기준 거리에서의 얼굴 크기를 캘리브레이션합니다.
2. 캘리브레이션 이후, 얼굴이 화면 중앙 근처(Red Zone)에 있으면 배경이 정중앙에 고정됩니다.
3. 얼굴이 중앙에서 벗어나면(Green Zone), 그 방향에 따라 배경 영상의 보이는 영역이 이동합니다.
4. 얼굴이 카메라에 가까워지거나 멀어지면 배경의 확대/축소 비율도 함께 조정됩니다.

---

## 알려진 문제점

- MediaPipe는 jetson_inference 대비 일부 각도나 조명 조건에서 인식률이 다를 수 있습니다.
- 소프트웨어로 디코딩하기 때문에, 고해상도 배경 영상을 사용할 경우 하드웨어 가속 버전(Jetson) 대비 성능이 낮을 수 있습니다.
- 단일 얼굴 추적만 지원합니다.

<br>

---

<br>

# PC Demo Version

A demo that lets you experience the core behavior of Digital Window using just a regular PC's webcam, without Jetson Nano or any dedicated sensors.

---

## Differences from the Jetson Version

This demo is based on [jetson/main_integrated.py](../jetson/main_integrated.py), with only the hardware-dependent parts replaced to run on a regular PC. 

| Aspect | Jetson Version | PC Demo Version |
|---|---|---|
| Face detection | jetson_inference (facenet-120, GPU) | MediaPipe Face Detection |
| Video decoding | GStreamer + NVDEC hardware acceleration | Standard OpenCV decoding |
| Camera input | V4L2-only | OS-independent |

---

## Requirements

- Python 3.8 or higher (3.10 recommended)
- A webcam
- OS: Windows, macOS, and Linux are all supported

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Before Running

Prepare the following files in the `assets/` folder.

```
pc_demo/
└── assets/
    ├── demo_background.mp4     # Background video (required)
    └── window_frame.png        # Window frame image with alpha channel (optional)
```

The demo will still run without a frame image; in that case, only the background video is shown without an overlay.

---

## Running

```bash
python3 main_demo.py
```

On first run, please stay still in front of the camera for about 2 seconds. This is the calibration process that establishes your baseline face size.
Press `Q` to quit.
When running the program, please wait as the initial face detection may take some time.

---

## How It Works

1. The webcam detects your face and calibrates the baseline face size at a reference distance.
2. After calibration, if your face stays near the center of the frame (Red Zone), the background remains locked to center.
3. If your face moves away from center (Green Zone), the visible region of the background shifts accordingly.
4. As you move closer to or farther from the camera, the background's zoom level adjusts as well.

---

## Known Limitations

- MediaPipe may have different detection accuracy compared to jetson_inference under certain angles or lighting conditions.
- Since this uses software decoding, performance may be lower than the hardware-accelerated (Jetson) version when using high-resolution background videos.
- Only single-face tracking is supported.

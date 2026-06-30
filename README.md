# 디지털 창문 (Digital Window)

얼굴 위치를 실시간으로 추적하여 모니터가 실제 창문처럼 보이는 착시 효과를 구현한 프로젝트입니다.

![demo](./assets/demo.gif)

**영상의 경우 현재 유튜브 업로드 중입니다.**

---

## 프로젝트 구성

이 저장소는 두 개의 환경으로 나뉘어 있습니다.

```
digital-window-v2/
├── jetson/         # Jetson Nano에서 동작하는 실제 구현체
│   ├── experiments/    # 개발 과정에서의 실험 버전 (원본 보존)
│   └── main_integrated.py  # 실험을 통합한 최종 버전
└── pc_demo/        # 하드웨어 없이 일반 PC에서 실행 가능한 데모 버전
```

**jetson/** 폴더는 실제 하드웨어(Jetson Nano, 얼굴 인식 카메라)를 사용하는 코드입니다.   
개발 과정에서 두 가지 다른 방향으로 실험을 진행했고, 이를 하나로 통합한 버전이 `main_integrated.py`입니다. 

**pc_demo/** 폴더는 Jetson Nano나 별도의 센서 없이, 일반 PC의 웹캠만으로 동작을 체험할 수 있도록 만든 버전입니다.   
얼굴 인식은 MediaPipe로, 영상 디코딩은 일반 OpenCV 방식으로 대체했습니다.

각 폴더의 상세한 내용은 폴더 내 README를 참고해 주세요.  

- [jetson/README.md](./jetson/README.md) — 실험 과정, 통합 결정 근거, Jetson Nano 실행 방법  
- [pc_demo/README.md](./pc_demo/README.md) — PC 데모 설치 및 실행 방법

---

## 아이디어

사무실에만 앉아있으면 따뜻한 태양빛이 그리워지듯이, 창밖을 보기 힘든 사람들이 모니터를 보면서 바깥에 있는 것 같은 개방감을 받을 수 있게 만들었습니다.  
센서(또는 카메라)로 사람의 얼굴 위치를 추적하고, 그 위치에 따라 배경 영상의 크롭 영역을 실시간으로 조정합니다.   
관찰자의 시점이 바뀔 때 배경이 함께 움직이므로, 마치 창문 너머를 바라보는 듯한 시차(parallax) 효과가 발생합니다.  
자세한 기술적 의사결정(시그모이드 보간, 비동기 영상 읽기, Zone 기반 추적 로직 등)은 jetson/README.md에서 다룹니다.  

주된 아이디어의 동기는 방구석 전자 youtube의 다음 영상을 참조했습니다 [https://www.youtube.com/watch?v=pp2LbGQxkys]  
구현 방법은 많이 다릅니다.

---

## 라이선스

개인 학습 및 포트폴리오 목적으로 공개된 프로젝트입니다.

<br>

---

<br>

# Digital Window

A real-time face-tracking illusion system that makes a monitor appear as a window into another space.

![demo](./assets/demo.gif)

▶ [Demo Video (YouTube)](https://youtube.com/your-link)

---

## Repository Structure

This repository is divided into two environments.

```
digital-window-v2/
├── jetson/         # Actual implementation running on Jetson Nano
│   ├── experiments/    # Experimental versions from development (preserved as-is)
│   └── main_integrated.py  # Final version integrating both experiments
└── pc_demo/        # Hardware-free demo version runnable on a regular PC
```

The **jetson/** folder contains the actual implementation that uses real hardware (Jetson Nano, a face-tracking camera). During development, two different experimental directions were explored, and they were later combined into `main_integrated.py`. The original experiment files are preserved as-is to show the decision-making process.

The **pc_demo/** folder is a version built to let anyone experience the core behavior using just a regular PC's webcam, without Jetson Nano or dedicated sensors. Face detection is replaced with MediaPipe, and video decoding uses standard OpenCV instead of hardware-accelerated pipelines.

See each folder's README for details.

- [jetson/README.md](./jetson/README.md) — Experiment process, integration rationale, and how to run on Jetson Nano
- [pc_demo/README.md](./pc_demo/README.md) — PC demo installation and usage

---

## Core Idea

A sensor (or camera) tracks the viewer's face position, and the background video's crop region is adjusted in real time accordingly. As the viewer's perspective shifts, the background moves with it, creating a parallax effect that makes the display feel like a window into another space.

Detailed technical decisions (sigmoid-based interpolation, asynchronous video reading, zone-based tracking logic, etc.) are covered in jetson/README.md.

---

## License

This project is published for personal learning and portfolio purposes. Plz take care CCL

"""
Digital Window - PC 데모 버전 (PC Demo Version)

jetson/main_integrated.py를 기반으로 하드웨어 종속성을 제거한 버전입니다.
Based on jetson/main_integrated.py with hardware dependencies removed.

원본(Jetson) 대비 교체된 부분 / Changes from the original (Jetson) version:

  1. 얼굴 인식 / Face detection
     jetson_inference.detectNet (facenet-120, GPU 추론)
     → MediaPipe Face Detection (CPU/GPU 겸용, 일반 PC에서 동작)

  2. 영상 디코딩 / Video decoding
     GStreamer + NVDEC 하드웨어 가속 파이프라인
     → cv2.VideoCapture(file_path) (일반 소프트웨어 디코딩)

  3. 카메라 입력 / Camera input
     cv2.VideoCapture(0, cv2.CAP_V4L2) (Linux/Jetson 전용 V4L2)
     → cv2.VideoCapture(0) (OS 무관)

Zone 판정 로직, 스무딩 공식, 창틀 오버레이 합성 방식은 원본과 동일하게 유지했습니다.
Zone detection logic, smoothing formulas, and frame overlay compositing remain
identical to the original.
"""

import time
import cv2
import numpy as np
import threading
import os
import urllib.request

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("오류: mediapipe가 설치되지 않았습니다. 'pip install mediapipe'로 설치해주세요.")
    print("Error: mediapipe is not installed. Please run 'pip install mediapipe'.")
    raise SystemExit(1)

# MediaPipe 0.10.x Task API 기준으로 작성되었습니다.
# Written against the MediaPipe 0.10.x Task API.
#
# 구버전 API(mp.solutions.face_detection)는 0.10대에서 일부 환경에 따라
# 정상 동작하지 않을 수 있어, 공식 권장 방식인 Task API로 작성했습니다.
# The legacy API (mp.solutions.face_detection) may not work reliably in some
# environments on 0.10.x, so this uses the officially recommended Task API.

FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
FACE_DETECTOR_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'assets', 'blaze_face_short_range.tflite'
)


def ensure_face_detector_model():
    """얼굴 인식 모델 파일이 없으면 자동으로 다운로드합니다. (최초 1회만)
    Downloads the face detection model file automatically if missing. (Once only)
    """
    if os.path.exists(FACE_DETECTOR_MODEL_PATH):
        return

    os.makedirs(os.path.dirname(FACE_DETECTOR_MODEL_PATH), exist_ok=True)
    print(">>> 얼굴 인식 모델 다운로드 중 (최초 1회) / Downloading face detection model (first run only)...")
    print(f">>> {FACE_DETECTOR_MODEL_URL}")
    try:
        urllib.request.urlretrieve(FACE_DETECTOR_MODEL_URL, FACE_DETECTOR_MODEL_PATH)
        print(">>> 모델 다운로드 완료 / Model download complete.")
    except Exception as e:
        print(f"!!! 모델 다운로드 실패 / Model download failed: {e}")
        print(f">>> 수동으로 다운로드 후 다음 경로에 저장해주세요 / Please download manually and save to:")
        print(f">>> {FACE_DETECTOR_MODEL_PATH}")
        raise SystemExit(1)

# ============================================================================
# [설정] 파라미터 튜닝 / Configuration parameters
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_VIDEO_PATH = os.path.join(BASE_DIR, 'assets', 'demo_background.mp4')
FRAME_IMAGE_PATH = os.path.join(BASE_DIR, 'assets', 'window_frame.png')

WINDOW_W = 960
WINDOW_H = 540
WINDOW_NAME = 'Digital Window (PC Demo)'
DEBUG_WINDOW_NAME = 'Camera Debug'

BASE_RED_W = 320
BASE_RED_H = 240

XY_SMOOTHING = 0.03
ZOOM_SMOOTHING = 0.015

TARGET_FRAME_TIME = 0.016  # 60FPS 기준 / 60FPS basis
# TARGET_FRAME_TIME = 0.033  # 30FPS로 낮추려면 주석 해제 / Uncomment for 30FPS

shared_data = {
    'target_cx': WINDOW_W // 2,
    'target_cy': WINDOW_H // 2,
    'target_zoom': 1.0,
    'running': True,
    'debug_frame': None,
    'face_detected': False
}


# ----------------------------------------------------------------------------
# [클래스] 배경 영상 비동기 로더 (PC 버전: 일반 VideoCapture 사용)
# Async background video loader (PC version: plain VideoCapture)
# ----------------------------------------------------------------------------
class BackgroundVideoReader:
    def __init__(self, path, width, height):
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(path)

        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.stopped = False

        if self.cap.isOpened():
            self.ret, frame = self.cap.read()
            if self.ret:
                self.frame = cv2.resize(frame, (width, height))
        else:
            print(f">>> [Error] 배경 영상 열기 실패 / Failed to open background video: {path}")

        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()

            if not ret:
                # 영상 루프백 / Video loopback
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (self.width, self.height))

            with self.lock:
                self.frame = frame
                self.ret = ret

            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.t.join()
        self.cap.release()


# ----------------------------------------------------------------------------
# [스레드] Face Tracking (PC 버전: MediaPipe Task API 사용)
# Face Tracking thread (PC version: MediaPipe Task API)
# ----------------------------------------------------------------------------
def face_tracking_thread():
    print(">>> [Thread] Face Tracking Started (MediaPipe Task API)")

    ensure_face_detector_model()

    base_options = mp_python.BaseOptions(model_asset_path=FACE_DETECTOR_MODEL_PATH)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,  # 실시간 영상 프레임 처리 모드 / video frame processing mode
        min_detection_confidence=0.5,
    )
    detector = mp_vision.FaceDetector.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(">>> [Error] 카메라를 열 수 없습니다. / Could not open camera.")
        shared_data['running'] = False
        return

    # 웜업 / Warm-up
    for _ in range(10):
        cap.read()
        time.sleep(0.03)

    calibration_samples = []
    CALIBRATION_FRAMES = 60
    base_face_width = 0
    is_calibrated = False

    cam_w, cam_h = 640, 480

    # VIDEO 모드는 단조 증가하는 타임스탬프(ms)가 필요합니다.
    # VIDEO mode requires a monotonically increasing timestamp (ms).
    start_time = time.time()

    while shared_data['running']:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)  # 웹캠 좌우 반전 보정 / Mirror correction for webcam
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = detector.detect_for_video(mp_image, timestamp_ms)
        detections = result.detections if result.detections else []

        # --- [1] 캘리브레이션 / Calibration ---
        if not is_calibrated:
            if len(detections) > 0:
                det = detections[0]
                bbox = det.bounding_box  # origin_x, origin_y, width, height (픽셀 단위 / pixel units)

                calibration_samples.append(bbox.width)
                cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y),
                              (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (255, 255, 0), 2)
                progress = len(calibration_samples) / CALIBRATION_FRAMES * 100
                cv2.putText(frame, f"Calibrating... {int(progress)}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 255, 0), 2)

                if len(calibration_samples) >= CALIBRATION_FRAMES:
                    base_face_width = sum(calibration_samples) / len(calibration_samples)
                    is_calibrated = True
                    print(f">>> [완료] 기준 얼굴 크기 / Base face width: {base_face_width:.2f}")
            else:
                cv2.putText(frame, "Face Not Found!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            shared_data['debug_frame'] = frame.copy()
            time.sleep(0.01)
            continue

        # --- [2] 트래킹 / Tracking ---
        if len(detections) > 0:
            shared_data['face_detected'] = True
            det = detections[0]
            bbox = det.bounding_box

            cx = int(bbox.origin_x + bbox.width / 2)
            cy = int(bbox.origin_y + bbox.height / 2)
            w = bbox.width

            ratio = max(0.5, min(w / base_face_width, 2.0))

            curr_red_w = int(BASE_RED_W * ratio)
            curr_red_h = int(BASE_RED_H * ratio)

            red_x1, red_x2 = (cam_w - curr_red_w) // 2, (cam_w + curr_red_w) // 2
            red_y1, red_y2 = (cam_h - curr_red_h) // 2, (cam_h + curr_red_h) // 2

            in_red_zone = (red_x1 < cx < red_x2) and (red_y1 < cy < red_y2)

            if in_red_zone:
                shared_data['target_cx'] = 320
                shared_data['target_cy'] = 240
                debug_color = (0, 0, 255)
            else:
                shared_data['target_cx'] = cx
                shared_data['target_cy'] = cy
                debug_color = (0, 255, 0)

            target_z = max(0.8, min(1.0 - (ratio - 1.0) * 0.4, 1.3))
            shared_data['target_zoom'] = target_z

            cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y),
                          (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), debug_color, 2)
            cv2.rectangle(frame, (red_x1, red_y1), (red_x2, red_y2), (0, 0, 255), 1)
        else:
            shared_data['face_detected'] = False

        shared_data['debug_frame'] = frame.copy()
        time.sleep(0.015)

    cap.release()
    detector.close()


# ----------------------------------------------------------------------------
# [메인] 렌더링 루프 / Main rendering loop
# ----------------------------------------------------------------------------
def main():
    print(">>> System Initializing (PC Demo)...")

    if not os.path.exists(BG_VIDEO_PATH):
        print(f"!!! 오류: 배경 영상을 찾을 수 없습니다 / Error: background video not found: {BG_VIDEO_PATH}")
        print(">>> assets/demo_background.mp4 위치에 영상을 넣어주세요. / Place a video at assets/demo_background.mp4")
        return

    # 1. 창틀 이미지 로드 및 전처리 / Load and preprocess window frame image
    frame_overlay = None
    alpha_mask = None
    bg_mask = None

    if os.path.exists(FRAME_IMAGE_PATH):
        print(f">>> 창틀 이미지 로드 중 / Loading frame image: {FRAME_IMAGE_PATH}")
        img_raw = cv2.imread(FRAME_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

        if img_raw is not None:
            img_raw = cv2.resize(img_raw, (WINDOW_W, WINDOW_H))

            if img_raw.shape[2] == 4:
                frame_overlay = img_raw[:, :, :3]
                alpha = img_raw[:, :, 3] / 255.0
                alpha_mask = np.dstack([alpha] * 3)
                bg_mask = 1.0 - alpha_mask
                print(">>> 창틀 오버레이 준비 완료 / Frame overlay ready")
            else:
                print("!!! 경고: 이미지가 투명도(Alpha)를 포함하지 않습니다. / Warning: image has no alpha channel.")
        else:
            print("!!! 오류: 이미지 파일을 읽을 수 없습니다. / Error: could not read image file.")
    else:
        print("!!! 안내: 창틀 이미지가 없어 오버레이 없이 진행합니다. / Notice: no frame image, proceeding without overlay.")

    # 2. 배경 영상 비동기 로더 시작 / Start async background video loader
    bg_w_target = int(WINDOW_W * 2.5)
    bg_h_target = int(WINDOW_H * 2.5)

    bg_reader = BackgroundVideoReader(BG_VIDEO_PATH, bg_w_target, bg_h_target)

    # 3. 얼굴 인식 스레드 시작 / Start face tracking thread
    t_face = threading.Thread(target=face_tracking_thread)
    t_face.daemon = True
    t_face.start()

    prev_cx, prev_cy = 320, 240
    prev_zoom = 1.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while shared_data['running']:
            loop_start = time.time()

            ret, bg_frame = bg_reader.read()

            if not ret or bg_frame is None:
                time.sleep(0.005)
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # --- [좌표 스무딩 & 줌 계산] / Coordinate smoothing & zoom calculation ---
            target_cx = shared_data['target_cx']
            target_cy = shared_data['target_cy']
            target_zoom = shared_data['target_zoom']

            smooth_cx = int(target_cx * XY_SMOOTHING + prev_cx * (1 - XY_SMOOTHING))
            smooth_cy = int(target_cy * XY_SMOOTHING + prev_cy * (1 - XY_SMOOTHING))
            smooth_zoom = target_zoom * ZOOM_SMOOTHING + prev_zoom * (1 - ZOOM_SMOOTHING)

            prev_cx, prev_cy = smooth_cx, smooth_cy
            prev_zoom = smooth_zoom

            # --- [뷰포트 크롭 & 리사이즈] / Viewport crop & resize ---
            vp_w = int(WINDOW_W / smooth_zoom)
            vp_h = int(WINDOW_H / smooth_zoom)

            offset_x_ratio = (smooth_cx - 320) / 640.0
            offset_y_ratio = (smooth_cy - 240) / 480.0

            center_x = (bg_w_target // 2) - int(offset_x_ratio * bg_w_target * 0.5)
            center_y = (bg_h_target // 2) - int(offset_y_ratio * bg_h_target * 0.5)

            x1 = center_x - (vp_w // 2)
            y1 = center_y - (vp_h // 2)
            x2 = x1 + vp_w
            y2 = y1 + vp_h

            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > bg_w_target: x2 = bg_w_target; x1 = x2 - vp_w
            if y2 > bg_h_target: y2 = bg_h_target; y1 = y2 - vp_h

            crop = bg_frame[y1:y2, x1:x2]

            if crop.size > 0:
                final_view = cv2.resize(crop, (WINDOW_W, WINDOW_H))
            else:
                final_view = cv2.resize(bg_frame, (WINDOW_W, WINDOW_H))

            # --------------------------------------------------------
            # [오버레이 합성] / Overlay compositing
            # --------------------------------------------------------
            if alpha_mask is not None:
                bg_part = final_view.astype(float) * bg_mask
                fg_part = frame_overlay.astype(float) * alpha_mask
                final_view = cv2.add(bg_part, fg_part).astype(np.uint8)

            cv2.imshow(WINDOW_NAME, final_view)

            if shared_data['debug_frame'] is not None:
                cv2.imshow(DEBUG_WINDOW_NAME, shared_data['debug_frame'])

            elapsed = time.time() - loop_start
            if elapsed < TARGET_FRAME_TIME:
                time.sleep(TARGET_FRAME_TIME - elapsed)

    finally:
        print(">>> Shutting down...")
        shared_data['running'] = False
        bg_reader.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
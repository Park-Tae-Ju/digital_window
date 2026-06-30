"""
Digital Window - Jetson Nano 통합 버전 (Integrated Version)

이 파일은 두 개의 실험 버전을 통합한 결과물입니다.
This file integrates two experimental versions.

  - main4_asynchronous.py : 배경 영상 비동기 읽기 구조 (BackgroundVideoReader)
                             Asynchronous background video reading structure
  - main_frame_tactic.py  : 창틀 PNG 오버레이 합성 기능
                             Window frame PNG overlay compositing

통합 이유 / Reason for integration:
    비동기 읽기 구조는 메인 루프가 영상 디코딩을 기다리지 않아 반응성이 좋고,
    창틀 오버레이는 시각적 완성도를 높여줍니다. 두 장점을 모두 살리기 위해 통합했습니다.

    The asynchronous reading structure keeps the main loop from blocking on video
    decoding, while the frame overlay improves visual polish. This integration
    combines both strengths.

실험 원본은 jetson/experiments/ 폴더에 그대로 보존되어 있습니다.
Original experiment files are preserved in jetson/experiments/.
"""

import time
import cv2
import numpy as np
import threading
import sys
import os

# 1. 디스플레이 환경변수 설정 / Set display environment variable
if os.environ.get("DISPLAY") is None:
    os.environ["DISPLAY"] = ":0"

# [NVIDIA 라이브러리 임포트] / Import NVIDIA libraries
try:
    import jetson_inference
    import jetson_utils
except ImportError:
    print("오류: jetson-inference가 설치되지 않았습니다. / Error: jetson-inference is not installed.")
    sys.exit(0)

# ============================================================================
# [설정] 파라미터 튜닝 / Configuration parameters
# ============================================================================
BG_VIDEO_PATH = ''
FRAME_IMAGE_PATH = ''

WINDOW_W = 960
WINDOW_H = 540
WINDOW_NAME = 'Digital Window (Integrated)'
DEBUG_WINDOW_NAME = 'Camera Debug'

# [기본 Red Zone 크기] (기준 거리일 때) / Base red zone size (at reference distance)
BASE_RED_W = 320
BASE_RED_H = 240

# [스무딩 설정] / Smoothing settings
XY_SMOOTHING = 0.06
ZOOM_SMOOTHING = 0.03

# [루프 속도 제어] / Loop rate control
# main4 기준 60FPS를 기본으로 사용합니다. 오버레이 합성으로 부하가 크면
# TARGET_FRAME_TIME 값을 0.033(약 30FPS)으로 바꾸세요.
# Defaults to 60FPS as in main4. If overlay compositing causes too much load,
# change TARGET_FRAME_TIME to 0.033 (~30FPS).
TARGET_FRAME_TIME = 0.016  # 60FPS 기준 / 60FPS basis
# TARGET_FRAME_TIME = 0.033  # 30FPS로 낮추려면 이 줄의 주석을 해제하세요 / Uncomment to lower to 30FPS

shared_data = {
    'target_cx': WINDOW_W // 2,
    'target_cy': WINDOW_H // 2,
    'target_zoom': 1.0,
    'running': True,
    'debug_frame': None,
    'face_detected': False
}


# ----------------------------------------------------------------------------
# [GStreamer] 파이프라인 생성 함수 / Pipeline creation function
# ----------------------------------------------------------------------------
def get_gst_pipeline(file_path, target_w, target_h):
    return (
        f"filesrc location={file_path} ! "
        "qtdemux ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw(memory:NVMM) ! "
        f"nvvidconv ! video/x-raw, width={target_w}, height={target_h}, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


# ----------------------------------------------------------------------------
# [클래스] 배경 영상 비동기 로더 (main4에서 가져옴) / Async background video loader (from main4)
# ----------------------------------------------------------------------------
class BackgroundVideoReader:
    def __init__(self, path, width, height):
        self.gst_str = get_gst_pipeline(path, width, height)
        self.cap = cv2.VideoCapture(self.gst_str, cv2.CAP_GSTREAMER)

        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.stopped = False

        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
        else:
            print(">>> [Error] 배경 영상 열기 실패 / Failed to open background video")

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
                print(">>> [Info] 영상 루프백 / Video loopback")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

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
# [스레드] Face Tracking (main_frame_tactic의 동적 Zone + 캘리브레이션 진행률 포함)
# Face Tracking thread (includes main_frame_tactic's dynamic zone + calibration progress)
# ----------------------------------------------------------------------------
def face_tracking_thread():
    print(">>> [Thread] Face Tracking Started")
    net = jetson_inference.detectNet("facenet-120", threshold=0.5)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 웜업 / Warm-up
    for _ in range(30):
        cap.read()
        time.sleep(0.03)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    calibration_samples = []
    CALIBRATION_FRAMES = 60
    base_face_width = 0
    is_calibrated = False

    while shared_data['running']:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cuda_img = jetson_utils.cudaFromNumpy(frame_rgb)
        detections = net.Detect(cuda_img)

        cam_w, cam_h = 640, 480

        # --- [1] 캘리브레이션 / Calibration ---
        if not is_calibrated:
            if len(detections) > 0:
                det = detections[0]
                calibration_samples.append(det.Width)
                cv2.rectangle(frame, (int(det.Left), int(det.Top)), (int(det.Right), int(det.Bottom)), (255, 255, 0), 2)
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
            cx, cy, w = int(det.Center[0]), int(det.Center[1]), int(det.Width)

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

            cv2.rectangle(frame, (int(det.Left), int(det.Top)), (int(det.Right), int(det.Bottom)), debug_color, 2)
            cv2.rectangle(frame, (red_x1, red_y1), (red_x2, red_y2), (0, 0, 255), 1)
        else:
            shared_data['face_detected'] = False

        shared_data['debug_frame'] = frame.copy()
        time.sleep(0.015)

    cap.release()


# ----------------------------------------------------------------------------
# [메인] 렌더링 루프 / Main rendering loop
# ----------------------------------------------------------------------------
def main():
    print(">>> System Initializing (Integrated: Async + Overlay)...")

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
        print("!!! 경고: 창틀 이미지 파일이 없습니다. / Warning: frame image file not found.")

    # 2. 배경 영상 비동기 로더 시작 (main4 구조) / Start async background video loader (main4 structure)
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
    # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # 필요시 주석 해제

    try:
        while shared_data['running']:
            loop_start = time.time()

            # [핵심] 비동기 읽기: 대기 없이 즉시 최신 프레임을 줍니다.
            # [Key] Async read: returns the latest frame immediately without waiting.
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

            center_x = (bg_w_target // 2) + int(offset_x_ratio * bg_w_target * 0.5)
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
            # [오버레이 합성] 배경 위에 창틀 얹기 (main_frame_tactic 구조)
            # [Overlay compositing] Place window frame over background (main_frame_tactic structure)
            # --------------------------------------------------------
            if alpha_mask is not None:
                bg_part = final_view.astype(float) * bg_mask
                fg_part = frame_overlay.astype(float) * alpha_mask
                final_view = cv2.add(bg_part, fg_part).astype(np.uint8)

            cv2.imshow(WINDOW_NAME, final_view)

            if shared_data['debug_frame'] is not None:
                cv2.imshow(DEBUG_WINDOW_NAME, shared_data['debug_frame'])

            # 루프 속도 제어 / Loop rate control
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

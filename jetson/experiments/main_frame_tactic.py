import time
import cv2
import numpy as np
import threading
import sys
import os

# 1. 디스플레이 환경변수 설정
if os.environ.get("DISPLAY") is None:
    os.environ["DISPLAY"] = ":0"

# [NVIDIA 라이브러리 임포트]
try:
    import jetson_inference
    import jetson_utils
except ImportError:
    print("오류: jetson-inference가 설치되지 않았습니다.")
    sys.exit(0)

# ============================================================================
# [설정] 파라미터 튜닝
# ============================================================================
BG_VIDEO_PATH = ''
FRAME_IMAGE_PATH = ''

WINDOW_W = 960
WINDOW_H = 540
WINDOW_NAME = 'Digital Window (Overlay Ver)'
DEBUG_WINDOW_NAME = 'Camera Debug'

# [기본 Red Zone 크기] (기준 거리일 때)
BASE_RED_W = 320
BASE_RED_H = 240

# [스무딩 설정]
XY_SMOOTHING = 0.06
ZOOM_SMOOTHING = 0.03

shared_data = {
    'target_cx': WINDOW_W // 2,
    'target_cy': WINDOW_H // 2,
    'target_zoom': 1.0,
    'running': True,
    'debug_frame': None,
    'face_detected': False
}


# ----------------------------------------------------------------------------
# [GStreamer] 배경 영상 디코딩 (GPU)
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
# [스레드] Face Tracking
# ----------------------------------------------------------------------------
def face_tracking_thread():
    print(">>> [Thread] Face Tracking Started")
    net = jetson_inference.detectNet("facenet-120", threshold=0.5)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 웜업
    for _ in range(30):
        cap.read()
        time.sleep(0.03)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    # 캘리브레이션 변수
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

        # [상태 1] 캘리브레이션
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
                    print(f">>> [완료] 기준 얼굴 크기: {base_face_width:.2f}")
            else:
                cv2.putText(frame, "Face Not Found!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            shared_data['debug_frame'] = frame.copy()
            time.sleep(0.01)
            continue

        # [상태 2] 정상 작동
        curr_red_w = BASE_RED_W
        curr_red_h = BASE_RED_H

        if len(detections) > 0:
            shared_data['face_detected'] = True
            det = detections[0]
            cx = int(det.Center[0])
            cy = int(det.Center[1])
            w = int(det.Width)

            ratio = w / base_face_width
            ratio = max(0.5, min(ratio, 2.0))

            curr_red_w = int(BASE_RED_W * ratio)
            curr_red_h = int(BASE_RED_H * ratio)

            red_x1 = (cam_w - curr_red_w) // 2
            red_x2 = (cam_w + curr_red_w) // 2
            red_y1 = (cam_h - curr_red_h) // 2
            red_y2 = (cam_h + curr_red_h) // 2

            in_red_zone = (red_x1 < cx < red_x2) and (red_y1 < cy < red_y2)

            if in_red_zone:
                shared_data['target_cx'] = 320
                shared_data['target_cy'] = 240
                debug_color = (0, 0, 255)
            else:
                shared_data['target_cx'] = cx
                shared_data['target_cy'] = cy
                debug_color = (0, 255, 0)

            zoom_factor = 0.4
            target_z = 1.0 - (ratio - 1.0) * zoom_factor
            target_z = max(0.8, min(target_z, 1.3))
            shared_data['target_zoom'] = target_z

            cv2.rectangle(frame, (int(det.Left), int(det.Top)), (int(det.Right), int(det.Bottom)), debug_color, 2)
            cv2.rectangle(frame, (red_x1, red_y1), (red_x2, red_y2), (0, 0, 255), 1)

        else:
            shared_data['face_detected'] = False

        shared_data['debug_frame'] = frame.copy()
        time.sleep(0.01)

    cap.release()


# ----------------------------------------------------------------------------
# [메인] 렌더링 루프
# ----------------------------------------------------------------------------
def main():
    print(">>> System Initializing (Overlay Mode)...")

    # 1. 창틀 이미지 로드 및 전처리 (최적화)
    frame_overlay = None
    alpha_mask = None
    bg_mask = None

    if os.path.exists(FRAME_IMAGE_PATH):
        print(f">>> 창틀 이미지 로드 중: {FRAME_IMAGE_PATH}")
        img_raw = cv2.imread(FRAME_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

        if img_raw is not None:
            # 창 크기에 맞게 리사이즈
            img_raw = cv2.resize(img_raw, (WINDOW_W, WINDOW_H))

            # 알파 채널 확인 (4채널인지)
            if img_raw.shape[2] == 4:
                # BGR 채널과 Alpha 채널 분리
                frame_overlay = img_raw[:, :, :3]  # 창틀 RGB
                alpha = img_raw[:, :, 3] / 255.0  # 0.0 ~ 1.0 정규화

                # 차원 확장 (H, W) -> (H, W, 3) : 곱셈 연산을 위해
                alpha_mask = np.dstack([alpha] * 3)
                bg_mask = 1.0 - alpha_mask

                print(">>> 창틀 오버레이 준비 완료")
            else:
                print("!!! 경고: 이미지가 투명도(Alpha)를 포함하지 않습니다. 오버레이를 건너뜁니다.")
        else:
            print("!!! 오류: 이미지 파일을 읽을 수 없습니다.")
    else:
        print("!!! 경고: 창틀 이미지 파일이 없습니다. 경로를 확인하세요.")

    # 2. 배경 영상 로드
    bg_w_target = int(WINDOW_W * 2.5)
    bg_h_target = int(WINDOW_H * 2.5)
    gst_str = get_gst_pipeline(BG_VIDEO_PATH, bg_w_target, bg_h_target)
    cap_bg = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    if not cap_bg.isOpened():
        print("Error: 배경 영상 로드 실패")
        return

    # 3. 트래킹 스레드 시작
    t = threading.Thread(target=face_tracking_thread)
    t.daemon = True
    t.start()

    prev_cx, prev_cy = 320, 240
    prev_zoom = 1.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while shared_data['running']:
            loop_start = time.time()

            ret, bg_frame = cap_bg.read()
            if not ret:
                cap_bg.release()
                cap_bg = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

            # --- [스무딩 & 줌 적용] ---
            target_cx = shared_data['target_cx']
            target_cy = shared_data['target_cy']
            target_zoom = shared_data['target_zoom']

            smooth_cx = int(target_cx * XY_SMOOTHING + prev_cx * (1 - XY_SMOOTHING))
            smooth_cy = int(target_cy * XY_SMOOTHING + prev_cy * (1 - XY_SMOOTHING))
            smooth_zoom = target_zoom * ZOOM_SMOOTHING + prev_zoom * (1 - ZOOM_SMOOTHING)

            prev_cx, prev_cy = smooth_cx, smooth_cy
            prev_zoom = smooth_zoom

            # 뷰포트 계산
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
            # [오버레이 합성] 배경 위에 창틀 얹기 (최적화된 연산)
            # --------------------------------------------------------
            if alpha_mask is not None:
                # 수식: Final = (배경 * (1-alpha)) + (창틀 * alpha)
                # 1. float 변환 후 곱셈
                bg_part = final_view.astype(float) * bg_mask
                fg_part = frame_overlay.astype(float) * alpha_mask

                # 2. 합산 및 uint8 변환
                final_view = cv2.add(bg_part, fg_part).astype(np.uint8)

            cv2.imshow(WINDOW_NAME, final_view)
            if shared_data['debug_frame'] is not None:
                cv2.imshow(DEBUG_WINDOW_NAME, shared_data['debug_frame'])

            elapsed = time.time() - loop_start
            if elapsed < 0.033: time.sleep(0.033 - elapsed)

    finally:
        shared_data['running'] = False
        cap_bg.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

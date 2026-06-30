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

WINDOW_W = 960
WINDOW_H = 540
WINDOW_NAME = 'Digital Window (Dynamic Zone)'
DEBUG_WINDOW_NAME = 'Camera Debug'

BASE_RED_W = 320
BASE_RED_H = 240
XY_SMOOTHING = 0.06
ZOOM_SMOOTHING = 0.03

# 데이터 공유를 위한 딕셔너리
shared_data = {
    'target_cx': WINDOW_W // 2,
    'target_cy': WINDOW_H // 2,
    'target_zoom': 1.0,
    'running': True,
    'debug_frame': None,
    'face_detected': False
}


# ----------------------------------------------------------------------------
# [GStreamer] 파이프라인 생성 함수
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
# [클래스] 배경 영상 비동기 로더 (Video Thread)
# ----------------------------------------------------------------------------
class BackgroundVideoReader:
    def __init__(self, path, width, height):
        self.gst_str = get_gst_pipeline(path, width, height)
        self.cap = cv2.VideoCapture(self.gst_str, cv2.CAP_GSTREAMER)

        # 최신 프레임을 담을 변수
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.stopped = False

        # 초기 프레임 확보
        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
        else:
            print(">>> [Error] 배경 영상 열기 실패")

        # 스레드 시작
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()

    def update(self):
        # 무한 루프를 돌며 최신 프레임만 계속 grab 함
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()

            # 영상이 끝나면(루프 재생) 다시 처음부터
            if not ret:
                print(">>> [Info] 영상 루프백")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Thread-Safe하게 프레임 업데이트 (Lock 사용)
            with self.lock:
                self.frame = frame
                self.ret = ret

            # 너무 빠르면 CPU 낭비하므로 약간의 대기 (영상 FPS에 맞춤)
            time.sleep(0.01)

    def read(self):
        # 메인 스레드에서 호출: 가장 최신 프레임을 반환
        with self.lock:
            return self.ret, self.frame if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.t.join()
        self.cap.release()


# ----------------------------------------------------------------------------
# [스레드] Face Tracking (Face Thread)
# ----------------------------------------------------------------------------
def face_tracking_thread():
    print(">>> [Thread] Face Tracking Started")
    # 네트워크 로드는 시간이 걸리므로 메인 루프 전에 완료됨
    net = jetson_inference.detectNet("facenet-120", threshold=0.5)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 웜업
    for _ in range(30): cap.read(); time.sleep(0.03)
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

        # --- [1] 캘리브레이션 ---
        if not is_calibrated:
            if len(detections) > 0:
                det = detections[0]
                calibration_samples.append(det.Width)
                cv2.rectangle(frame, (int(det.Left), int(det.Top)), (int(det.Right), int(det.Bottom)), (255, 255, 0), 2)

                if len(calibration_samples) >= CALIBRATION_FRAMES:
                    base_face_width = sum(calibration_samples) / len(calibration_samples)
                    is_calibrated = True
                    print(f">>> [완료] 캘리브레이션 끝! 기준: {base_face_width:.2f}")

            shared_data['debug_frame'] = frame.copy()
            time.sleep(0.01)
            continue

        # --- [2] 트래킹 ---
        if len(detections) > 0:
            shared_data['face_detected'] = True
            det = detections[0]
            cx, cy, w = int(det.Center[0]), int(det.Center[1]), int(det.Width)

            # 비율 계산
            ratio = max(0.5, min(w / base_face_width, 2.0))

            # Dynamic Red Zone
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

            # Reverse Zoom
            target_z = max(0.8, min(1.0 - (ratio - 1.0) * 0.4, 1.3))
            shared_data['target_zoom'] = target_z

            cv2.rectangle(frame, (int(det.Left), int(det.Top)), (int(det.Right), int(det.Bottom)), debug_color, 2)
            cv2.rectangle(frame, (red_x1, red_y1), (red_x2, red_y2), (0, 0, 255), 1)
        else:
            shared_data['face_detected'] = False

        shared_data['debug_frame'] = frame.copy()
        # 얼굴 인식 루프가 너무 빠르면 GPU를 혼자 다 쓰므로 약간 양보
        time.sleep(0.015)

    cap.release()


# ----------------------------------------------------------------------------
# [메인] 렌더링 루프 (Main Thread)
# ----------------------------------------------------------------------------
def main():
    print(">>> System Initializing (Triple Buffering)...")

    # 1. 배경 영상 비동기 로더 시작
    bg_w_target = int(WINDOW_W * 2.5)
    bg_h_target = int(WINDOW_H * 2.5)

    # 클래스 인스턴스 생성 (내부 스레드 시작)
    bg_reader = BackgroundVideoReader(BG_VIDEO_PATH, bg_w_target, bg_h_target)

    # 2. 얼굴 인식 스레드 시작
    t_face = threading.Thread(target=face_tracking_thread)
    t_face.daemon = True
    t_face.start()

    prev_cx, prev_cy = 320, 240
    prev_zoom = 1.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # 필요시 주석 해제

    try:
        while shared_data['running']:
            loop_start = time.time()

            # [핵심] 이제 read()는 대기하지 않고 즉시 최신 프레임을 줍니다.
            ret, bg_frame = bg_reader.read()

            if not ret or bg_frame is None:
                # 아직 프레임 준비 안됨 -> 이전 프레임 유지 or 대기
                time.sleep(0.005)
                continue

            # --- [좌표 스무딩 & 줌 계산] ---
            # (이 부분은 연산량이 적어 메인 스레드에서 해도 무방합니다)
            target_cx = shared_data['target_cx']
            target_cy = shared_data['target_cy']
            target_zoom = shared_data['target_zoom']

            smooth_cx = int(target_cx * XY_SMOOTHING + prev_cx * (1 - XY_SMOOTHING))
            smooth_cy = int(target_cy * XY_SMOOTHING + prev_cy * (1 - XY_SMOOTHING))
            smooth_zoom = target_zoom * ZOOM_SMOOTHING + prev_zoom * (1 - ZOOM_SMOOTHING)

            prev_cx, prev_cy = smooth_cx, smooth_cy
            prev_zoom = smooth_zoom

            # --- [뷰포트 크롭 & 리사이즈] ---
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

            # 범위 예외처리
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > bg_w_target: x2 = bg_w_target; x1 = x2 - vp_w
            if y2 > bg_h_target: y2 = bg_h_target; y1 = y2 - vp_h

            crop = bg_frame[y1:y2, x1:x2]

            if crop.size > 0:
                # 여기서 시간이 좀 걸리지만, I/O 대기가 없으므로 훨씬 빠름
                final_view = cv2.resize(crop, (WINDOW_W, WINDOW_H))
                cv2.imshow(WINDOW_NAME, final_view)

            # 디버그 창 (필요할 때만 켜세요, 성능 잡아먹습니다)
            if shared_data['debug_frame'] is not None:
                cv2.imshow(DEBUG_WINDOW_NAME, shared_data['debug_frame'])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

            # 루프 속도 제어 (최대 60FPS)
            elapsed = time.time() - loop_start
            if elapsed < 0.016:
                time.sleep(0.016 - elapsed)

    finally:
        print(">>> Shutting down...")
        shared_data['running'] = False
        bg_reader.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

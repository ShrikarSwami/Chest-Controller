import time
import subprocess
import sys

import cv2
import numpy as np
from pynput.keyboard import Controller, Key
import mediapipe as mp

# -----------------------------
# Settings you will tweak
# -----------------------------
GAME_APP_PATH = "/Applications/Crossy Road.app"
MODEL_PATH = "pose_landmarker.task"

CAM_INDEX = 0

# Motion thresholds (start here, then tweak after a quick test)
ENERGY_THRESHOLD = 12.0         # higher = less sensitive
BOTH_BONUS_THRESHOLD = 18.0     # for forward

# Debounce and cooldown
COOLDOWN_SEC = 0.18             # prevents repeated spam moves
SMOOTHING = 0.7                 # 0 to 1, higher = smoother

# Calibration
CALIBRATION_SECONDS = 2.0       # learns baseline motion when you are still

# Pec box geometry (relative to torso width/height)
PEC_GAP = 0.04                  # gap from midline (fraction of torso width)
PEC_HALF_WIDTH = 0.22           # half width of each pec box (fraction of torso width)
PEC_TOP_Y = 0.08                # top of pec box (fraction of torso height below shoulders)
PEC_BOT_Y = 0.48                # bottom of pec box (fraction of torso height below shoulders)

# Landmark based thresholds (normalized coords, small numbers)
SHOULDER_WIDTH_THRESH = 0.010   # 0.006 to 0.020 typical
SIDE_DIFF_THRESH = 0.008        # left vs right difference
BOTH_THRESH = 0.014             # combined change for forward

# -----------------------------
# MediaPipe Tasks setup
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

keyboard = Controller()


def open_game_app(app_path: str) -> None:
    # macOS: open -a "App Name" or open /path/to/App.app
    if sys.platform != "darwin":
        return
    if app_path.endswith(".app"):
        subprocess.Popen(["open", app_path])
    else:
        subprocess.Popen(["open", "-a", app_path])


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def ema(prev: float, cur: float, a: float) -> float:
    return a * prev + (1.0 - a) * cur

def metrics_from_landmarks(lm):
    # normalized coordinates
    ls = lm[11]  # left shoulder
    rs = lm[12]  # right shoulder
    le = lm[13]  # left elbow
    re = lm[14]  # right elbow
    lh = lm[23]  # left hip
    rh = lm[24]  # right hip

    mid_x = (ls.x + rs.x) / 2.0
    sh_y = (ls.y + rs.y) / 2.0
    hip_y = (lh.y + rh.y) / 2.0

    shoulder_w = abs(ls.x - rs.x)

    # Side “pec” proxies: how far each shoulder is from midline (captures chest expansion)
    left_side = abs(mid_x - ls.x)
    right_side = abs(rs.x - mid_x)

    # Optional: shoulder lift (sometimes happens on flex)
    shoulder_lift = (hip_y - sh_y)  # bigger = shoulders higher relative to hips

    # Arm stability: elbows move when you shift posture, can be used as a filter later
    arm_motion = abs(le.y - ls.y) + abs(re.y - rs.y)

    return shoulder_w, left_side, right_side, shoulder_lift, arm_motion

def rect_from_points(w: int, h: int, x0: float, y0: float, x1: float, y1: float):
    # Inputs are normalized 0 to 1 coords. Output is pixel rect (x, y, rw, rh).
    px0 = clamp_int(int(x0 * w), 0, w - 1)
    py0 = clamp_int(int(y0 * h), 0, h - 1)
    px1 = clamp_int(int(x1 * w), 0, w - 1)
    py1 = clamp_int(int(y1 * h), 0, h - 1)

    x = min(px0, px1)
    y = min(py0, py1)
    rw = max(1, abs(px1 - px0))
    rh = max(1, abs(py1 - py0))
    return x, y, rw, rh

def ema(prev: float, cur: float, a: float) -> float:
    return a * prev + (1.0 - a) * cur

def metrics_from_landmarks(lm):
    # normalized coordinates
    ls = lm[11]  # left shoulder
    rs = lm[12]  # right shoulder
    le = lm[13]  # left elbow
    re = lm[14]  # right elbow
    lh = lm[23]  # left hip
    rh = lm[24]  # right hip

    mid_x = (ls.x + rs.x) / 2.0
    sh_y = (ls.y + rs.y) / 2.0
    hip_y = (lh.y + rh.y) / 2.0

    shoulder_w = abs(ls.x - rs.x)

    # Side “pec” proxies: how far each shoulder is from midline (captures chest expansion)
    left_side = abs(mid_x - ls.x)
    right_side = abs(rs.x - mid_x)

    # Optional: shoulder lift (sometimes happens on flex)
    shoulder_lift = (hip_y - sh_y)  # bigger = shoulders higher relative to hips

    # Arm stability: elbows move when you shift posture, can be used as a filter later
    arm_motion = abs(le.y - ls.y) + abs(re.y - rs.y)

    return shoulder_w, left_side, right_side, shoulder_lift, arm_motion

def pec_rects_from_landmarks(landmarks, frame_w: int, frame_h: int):
    # Pose landmark ids:
    # 11 left shoulder, 12 right shoulder, 23 left hip, 24 right hip
    ls = landmarks[11]
    rs = landmarks[12]
    lh = landmarks[23]
    rh = landmarks[24]

    # Midline x as average of shoulders, helps split left and right regions
    mid_x = (ls.x + rs.x) / 2.0

    # Torso geometry based on shoulders and hips
    torso_w = max(0.001, abs(ls.x - rs.x))
    shoulder_y = (ls.y + rs.y) / 2.0
    hip_y = (lh.y + rh.y) / 2.0
    torso_h = max(0.001, abs(hip_y - shoulder_y))

    top_y = shoulder_y + PEC_TOP_Y * torso_h
    bot_y = shoulder_y + PEC_BOT_Y * torso_h
    bot_y = min(bot_y, hip_y - (0.05 * torso_h))

    # Left pec region
    left_center_x = mid_x - (PEC_GAP * torso_w) - (PEC_HALF_WIDTH * torso_w)
    left_x0 = left_center_x - (PEC_HALF_WIDTH * torso_w)
    left_x1 = left_center_x + (PEC_HALF_WIDTH * torso_w)

    # Right pec region
    right_center_x = mid_x + (PEC_GAP * torso_w) + (PEC_HALF_WIDTH * torso_w)
    right_x0 = right_center_x - (PEC_HALF_WIDTH * torso_w)
    right_x1 = right_center_x + (PEC_HALF_WIDTH * torso_w)

    left_rect = rect_from_points(frame_w, frame_h, left_x0, top_y, left_x1, bot_y)
    right_rect = rect_from_points(frame_w, frame_h, right_x0, top_y, right_x1, bot_y)

    return left_rect, right_rect


def roi_energy(prev_gray: np.ndarray, gray: np.ndarray, rect):
    x, y, rw, rh = rect
    a = prev_gray[y:y + rh, x:x + rw]
    b = gray[y:y + rh, x:x + rw]
    if a.size == 0 or b.size == 0:
        return 0.0
    diff = cv2.absdiff(a, b)
    return float(np.mean(diff))


def press_arrow(action: str):
    if action == "LEFT":
        k = Key.left
    elif action == "RIGHT":
        k = Key.right
    elif action == "FORWARD":
        k = Key.up
    else:
        return
    keyboard.press(k)
    keyboard.release(k)


def draw_landmark_dot(frame, landmark, w: int, h: int):
    x = clamp_int(int(landmark.x * w), 0, w - 1)
    y = clamp_int(int(landmark.y * h), 0, h - 1)
    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    open_game_app(GAME_APP_PATH)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    last_move_time = 0.0
    prev_gray = None

    smooth_left = 0.0
    smooth_right = 0.0
    # Baseline for landmark metrics
    baseline = None
    calibrating = False
    calib_samples = []

    # Smoothed deltas
    d_shoulder = 0.0
    d_left = 0.0
    d_right = 0.0

    last_action = "NONE"

    # Baseline calibration
    baseline_left_vals = []
    baseline_right_vals = []
    calib_start = time.time()

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror view feels more natural
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            action = "NONE"

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_gray = gray.copy()

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                left_rect, right_rect = pec_rects_from_landmarks(lm, w, h)

                # Draw boxes
                lx, ly, lrw, lrh = left_rect
                rx, ry, rrw, rrh = right_rect
                cv2.rectangle(frame, (lx, ly), (lx + lrw, ly + lrh), (0, 255, 0), 2)
                cv2.rectangle(frame, (rx, ry), (rx + rrw, ry + rrh), (0, 255, 0), 2)

                # Optional landmark debug dots
                draw_landmark_dot(frame, lm[11], w, h)
                draw_landmark_dot(frame, lm[12], w, h)
                draw_landmark_dot(frame, lm[23], w, h)
                draw_landmark_dot(frame, lm[24], w, h)

                # Motion energy inside each box
                e_left = roi_energy(prev_gray, gray, left_rect)
                e_right = roi_energy(prev_gray, gray, right_rect)

                # Smooth it
                smooth_left = SMOOTHING * smooth_left + (1.0 - SMOOTHING) * e_left
                smooth_right = SMOOTHING * smooth_right + (1.0 - SMOOTHING) * e_right

                # Calibration baseline, assume you are mostly still for first few seconds
                if time.time() - calib_start < CALIBRATION_SECONDS:
                    baseline_left_vals.append(smooth_left)
                    baseline_right_vals.append(smooth_right)
                    action = "CALIBRATING"
                else:
                    base_left = float(np.mean(baseline_left_vals)) if baseline_left_vals else 0.0
                    base_right = float(np.mean(baseline_right_vals)) if baseline_right_vals else 0.0

                    left_delta = smooth_left - base_left
                    right_delta = smooth_right - base_right

                    left_active = left_delta > ENERGY_THRESHOLD
                    right_active = right_delta > ENERGY_THRESHOLD
                    both_strong = (left_delta + right_delta) > BOTH_BONUS_THRESHOLD

                    if left_active and right_active and both_strong:
                        action = "FORWARD"
                    elif left_active and not right_active:
                        action = "LEFT"
                    elif right_active and not left_active:
                        action = "RIGHT"
                    else:
                        action = "NONE"

                    # Cooldown so it moves one step per flex
                    now = time.time()
                    if action in ["LEFT", "RIGHT", "FORWARD"] and (now - last_move_time) > COOLDOWN_SEC:
                        press_arrow(action)
                        last_move_time = now

                # Debug text inside boxes
                cv2.putText(frame, f"L {smooth_left:.1f}", (lx + 8, ly + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"R {smooth_right:.1f}", (rx + 8, ry + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Top right action label
            label = f"MOVE: {action}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(frame, label, (w - tw - 20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow("Chest Test", frame)
            prev_gray = gray

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

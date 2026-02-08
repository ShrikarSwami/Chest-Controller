"""
Setup (macOS, Python 3.10):
1) python -m venv .venv
2) source .venv/bin/activate
3) pip install opencv-python mediapipe pynput numpy
4) curl -L -o pose_landmarker.task \
   https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

Tuning:
- SHOULDER_WIDTH_THRESH, SIDE_DIFF_THRESH, BOTH_THRESH, SMOOTHING, COOLDOWN_SEC
- BOX_WIDTH_RATIO, BOX_HEIGHT_RATIO, BOX_Y_OFFSET
"""

from __future__ import annotations

import time
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from pynput.keyboard import Controller, Key
import mediapipe as mp


# -----------------------------
# Constants
# -----------------------------
MODEL_PATH = "pose_landmarker.task"
CALIBRATION_SECONDS = 2.0

SHOULDER_WIDTH_THRESH = 0.08
SIDE_DIFF_THRESH = 0.015
BOTH_THRESH = 0.02
SMOOTHING = 0.2  # EMA alpha
COOLDOWN_SEC = 0.7

BOX_WIDTH_RATIO = 0.38
BOX_HEIGHT_RATIO = 0.45
BOX_Y_OFFSET = 0.35

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Baseline:
	left_proxy: float
	right_proxy: float


# -----------------------------
# Helper functions
# -----------------------------
def clamp(value: float, min_value: float, max_value: float) -> float:
	return max(min_value, min(max_value, value))


def norm_to_pixel(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
	return int(clamp(x, 0.0, 1.0) * width), int(clamp(y, 0.0, 1.0) * height)


def ema(prev: Optional[float], new: float, alpha: float) -> float:
	if prev is None:
		return new
	return alpha * new + (1.0 - alpha) * prev


def compute_proxies(landmarks) -> Optional[Tuple[float, float, float, float, float, float]]:
	try:
		ls = landmarks[11]
		rs = landmarks[12]
		le = landmarks[13]
		re = landmarks[14]
		lh = landmarks[23]
		rh = landmarks[24]
	except IndexError:
		return None

	shoulder_w = abs(ls.x - rs.x)
	if shoulder_w < 1e-6:
		return None

	mid_x = (ls.x + rs.x) / 2.0
	shoulder_mid_y = (ls.y + rs.y) / 2.0
	hip_mid_y = (lh.y + rh.y) / 2.0
	torso_h = abs(shoulder_mid_y - hip_mid_y)

	# Chest activation proxies using torso geometry + elbow relation.
	left_proxy = abs(mid_x - ls.x) + 0.35 * abs(ls.x - le.x) + 0.25 * abs(ls.x - lh.x)
	right_proxy = abs(rs.x - mid_x) + 0.35 * abs(re.x - rs.x) + 0.25 * abs(rh.x - rs.x)

	return shoulder_w, torso_h, mid_x, left_proxy, right_proxy, shoulder_mid_y


def compute_boxes(
	ls, rs, mid_x: float, shoulder_mid_y: float, shoulder_w: float, torso_h: float
):
	box_w = shoulder_w * BOX_WIDTH_RATIO
	box_h = torso_h * BOX_HEIGHT_RATIO
	center_y = shoulder_mid_y + torso_h * BOX_Y_OFFSET

	left_center_x = (ls.x + mid_x) / 2.0
	right_center_x = (rs.x + mid_x) / 2.0

	left_box = (left_center_x, center_y, box_w, box_h)
	right_box = (right_center_x, center_y, box_w, box_h)
	return left_box, right_box


def draw_box(frame, box, label: str, value: float) -> None:
	h, w, _ = frame.shape
	cx, cy, bw, bh = box
	x1, y1 = norm_to_pixel(cx - bw / 2.0, cy - bh / 2.0, w, h)
	x2, y2 = norm_to_pixel(cx + bw / 2.0, cy + bh / 2.0, w, h)
	cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

	text = f"{label}: {value:+.3f}"
	tx, ty = x1 + 5, max(20, y1 - 8)
	cv2.putText(frame, text, (tx, ty), FONT, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)


def launch_crossy_road() -> None:
	subprocess.run(["open", "-a", "Crossy Road"], check=False)


def main() -> None:
	launch_crossy_road()

	base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
	options = mp.tasks.vision.PoseLandmarkerOptions(
		base_options=base_options,
		running_mode=mp.tasks.vision.RunningMode.IMAGE,
		num_poses=1,
	)
	landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("Unable to open webcam.")

	keyboard = Controller()

	baseline: Optional[Baseline] = None
	calibration_start: Optional[float] = None
	calib_left: list[float] = []
	calib_right: list[float] = []

	d_left_ema: Optional[float] = None
	d_right_ema: Optional[float] = None
	last_trigger_time = 0.0

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
		result = landmarker.detect(mp_image)

		move_label = "NONE"
		pose_found = bool(result.pose_landmarks)

		if pose_found:
			landmarks = result.pose_landmarks[0]
			proxy_data = compute_proxies(landmarks)

			if proxy_data is not None:
				shoulder_w, torso_h, mid_x, left_proxy, right_proxy, shoulder_mid_y = proxy_data

				if shoulder_w >= SHOULDER_WIDTH_THRESH:
					ls = landmarks[11]
					rs = landmarks[12]
					left_box, right_box = compute_boxes(
						ls, rs, mid_x, shoulder_mid_y, shoulder_w, torso_h
					)

					if calibration_start is not None:
						calib_left.append(left_proxy)
						calib_right.append(right_proxy)

						if (time.time() - calibration_start) >= CALIBRATION_SECONDS:
							if calib_left and calib_right:
								baseline = Baseline(
									left_proxy=float(np.mean(calib_left)),
									right_proxy=float(np.mean(calib_right)),
								)
							calibration_start = None
							calib_left.clear()
							calib_right.clear()

						move_label = "CALIBRATING"
					else:
						if baseline is None:
							move_label = "PRESS C TO CALIBRATE"
						else:
							d_left = left_proxy - baseline.left_proxy
							d_right = right_proxy - baseline.right_proxy
							d_left_ema = ema(d_left_ema, d_left, SMOOTHING)
							d_right_ema = ema(d_right_ema, d_right, SMOOTHING)

							left_active = d_left_ema > BOTH_THRESH
							right_active = d_right_ema > BOTH_THRESH
							side_diff = d_left_ema - d_right_ema
							now = time.time()

							if (now - last_trigger_time) >= COOLDOWN_SEC:
								if left_active and right_active and (d_left_ema + d_right_ema) > (
									BOTH_THRESH * 2.0
								):
									keyboard.press(Key.up)
									keyboard.release(Key.up)
									last_trigger_time = now
									move_label = "FORWARD"
								elif left_active and side_diff > SIDE_DIFF_THRESH:
									keyboard.press(Key.left)
									keyboard.release(Key.left)
									last_trigger_time = now
									move_label = "LEFT"
								elif right_active and (-side_diff) > SIDE_DIFF_THRESH:
									keyboard.press(Key.right)
									keyboard.release(Key.right)
									last_trigger_time = now
									move_label = "RIGHT"

					# Draw boxes and debug values
					if d_left_ema is None:
						d_left_ema = 0.0
					if d_right_ema is None:
						d_right_ema = 0.0
					draw_box(frame, left_box, "dL", d_left_ema)
					draw_box(frame, right_box, "dR", d_right_ema)

		if not pose_found and calibration_start is None and baseline is not None:
			move_label = "NONE"

		# Overlay move label
		label_text = f"MOVE: {move_label}"
		h, w, _ = frame.shape
		text_size, _ = cv2.getTextSize(label_text, FONT, TEXT_SCALE, TEXT_THICKNESS)
		text_x = max(10, w - text_size[0] - 10)
		text_y = 30
		cv2.putText(frame, label_text, (text_x, text_y), FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

		cv2.imshow("Chest Controller", frame)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
		if key == ord("c"):
			calibration_start = time.time()
			calib_left.clear()
			calib_right.clear()
			d_left_ema = None
			d_right_ema = None

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

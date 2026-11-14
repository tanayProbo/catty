from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np

from config import (
    CV_BLUR_KERNEL,
    CV_CAMERA_INDEX,
    CV_CROUCH_Y_THRESHOLD,
    CV_FRAME_HEIGHT,
    CV_FRAME_WIDTH,
    CV_JUMP_Y_THRESHOLD,
    CV_LEFT_X_THRESHOLD,
    CV_RIGHT_X_THRESHOLD,
)

PostureType = Literal["STAND", "JUMP", "CROUCH"]
HandGestureType = Literal[
    "BOTH_HANDS_UP",
    "BOTH_HANDS_DOWN",
    "ONE_HAND_LEFT",
    "ONE_HAND_RIGHT",
    "HANDS_NEUTRAL",
    "OPEN_PALM",
    "CLOSED_FIST",
    "NEUTRAL_HAND",
]


@dataclass(slots=True)
class MotionSample:
    motion_score: float
    frame_rgb: np.ndarray
    is_simulation: bool
    status_label: str
    sim_motion_active: bool
    human_lane: int = 0
    human_posture: PostureType = "STAND"
    body_x: float = 0.5
    body_y: float = 0.5
    hand_gesture: HandGestureType = "HANDS_NEUTRAL"
    finger_count: int = 0
    hand_detected: bool = False


def _fast_mask_stats(mask: np.ndarray) -> tuple[int, float, float, float, float]:
    row_counts = np.count_nonzero(mask, axis=1)
    total = int(row_counts.sum())
    if total < 24:
        return total, 0.5, 0.5, 0.5, 0.5

    col_counts = np.count_nonzero(mask, axis=0)
    row_cdf = np.cumsum(row_counts)
    col_cdf = np.cumsum(col_counts)

    h = float(mask.shape[0])
    w = float(mask.shape[1])

    med_x = float(np.searchsorted(col_cdf, total * 0.50)) / w
    top_y = float(np.searchsorted(row_cdf, total * 0.16)) / h
    med_y = float(np.searchsorted(row_cdf, total * 0.50)) / h
    bot_y = float(np.searchsorted(row_cdf, total * 0.84)) / h
    return total, med_x, top_y, med_y, bot_y


class MotionTracker:
    def __init__(
        self,
        camera_index: int = CV_CAMERA_INDEX,
        width: int = CV_FRAME_WIDTH,
        height: int = CV_FRAME_HEIGHT,
        force_simulation: bool = False,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._force_simulation = force_simulation

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_simulation = force_simulation
        self._status_label = "Starting"

        # Pre-allocated image buffers for the vision loop (zero per-frame heap churn)
        self._prev_gray: Optional[np.ndarray] = None
        self._bg_gray: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._gray_buf = np.empty((height, width), dtype=np.uint8)
        self._blur_buf = np.empty((height, width), dtype=np.uint8)
        self._delta_buf = np.empty((height, width), dtype=np.uint8)
        self._mask_buf = np.empty((height, width), dtype=np.uint8)
        self._fg_curr_buf = np.empty((height, width), dtype=np.uint8)
        self._fg_prev_buf = np.empty((height, width), dtype=np.uint8)
        self._fg_mask_buf = np.empty((height, width), dtype=np.uint8)
        self._motion_mask_buf = np.empty((height, width), dtype=np.uint8)
        self._fg_motion_mask = np.empty((height, width), dtype=np.uint8)
        self._arriving_mask = np.empty((height, width), dtype=np.uint8)
        self._departing_mask = np.empty((height, width), dtype=np.uint8)
        self._ycrcb_buf = np.empty((height, width, 3), dtype=np.uint8)
        self._hsv_buf = np.empty((height, width, 3), dtype=np.uint8)
        self._skin_mask = np.empty((height, width), dtype=np.uint8)
        self._skin_hsv_mask = np.empty((height, width), dtype=np.uint8)
        self._combined_hand_mask = np.empty((height, width), dtype=np.uint8)
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._small_gray = np.empty((height // 2, width // 2), dtype=np.uint8)
        self._eq_small_gray = np.empty((height // 2, width // 2), dtype=np.uint8)
        self._latest_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        self._motion_score: float = 0.0
        self._sim_motion_trigger: bool = False
        self._sim_phase: float = 0.0
        self._consecutive_failures: int = 0

        # Persistent face anchor & adaptive skin calibration
        self._face_cascade = self._load_cascade("haarcascade_frontalface_default.xml")
        self._face_box_smooth: Optional[tuple[float, float, float, float]] = None
        self._face_last_seen: float = 0.0
        self._face_cr: Optional[float] = None
        self._face_cb: Optional[float] = None

        # Smoothed hand kinematics & persistence
        self._smoothed_x: float = 0.50
        self._smoothed_y: float = 0.50
        self._prev_raw_x: float = 0.50
        self._prev_raw_y: float = 0.50
        self._neutral_y: float = 0.50
        self._neutral_initialized: bool = False
        self._human_lane: int = 0
        self._human_posture: PostureType = "STAND"
        self._hand_gesture: HandGestureType = "HANDS_NEUTRAL"
        self._finger_count: int = 0
        self._hand_detected: bool = False
        self._detected_hands: list[tuple[float, float]] = []
        self._last_valid_hands: list[tuple[float, float]] = []
        self._last_valid_hands_time: float = 0.0
        self._last_hand_contours: list[np.ndarray] = []
        self._jump_hold_until: float = 0.0
        self._crouch_hold_until: float = 0.0

        self._sim_lane_override: Optional[int] = None
        self._sim_posture_override: Optional[PostureType] = None

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _load_cascade(self, filename: str) -> Optional[cv2.CascadeClassifier]:
        try:
            cascade_dir = getattr(cv2, "data", None)
            if cascade_dir is not None:
                xml_path = os.path.join(cascade_dir.haarcascades, filename)
                if os.path.exists(xml_path):
                    classifier = cv2.CascadeClassifier(xml_path)
                    if not classifier.empty():
                        return classifier
        except Exception:
            pass
        return None

    def start(self) -> None:
        if self._running:
            return
        if not self._force_simulation:
            self._open_camera()
        else:
            self._is_simulation = True
            self._status_label = "Hands Sim Mode"

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CVMotionTracker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._release_camera()

    def _open_camera(self) -> None:
        try:
            cap = cv2.VideoCapture(self.camera_index)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    self._cap = cap
                    self._is_simulation = False
                    self._status_label = "Hands Cam Live"
                    self._consecutive_failures = 0
                    self._prev_gray = None
                    self._bg_gray = None
                    self._frame_count = 0
                    return
            if cap is not None:
                cap.release()
        except Exception:
            pass

        self._cap = None
        self._is_simulation = True
        self._status_label = "Hands Sim Mode"
        self._prev_gray = None
        self._bg_gray = None
        self._frame_count = 0

    def _release_camera(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def set_simulation_mode(self, enabled: bool) -> None:
        with self._lock:
            self._force_simulation = enabled
            self._prev_gray = None
            self._bg_gray = None
            self._frame_count = 0
            self._motion_score = 0.0
            self._last_valid_hands = []
            if enabled:
                self._release_camera()
                self._is_simulation = True
                self._status_label = "Hands Sim Mode"
            else:
                self._open_camera()

    def set_simulated_motion(self, active: bool) -> None:
        with self._lock:
            self._sim_motion_trigger = active

    def set_pose_overrides(
        self,
        lane: Optional[int],
        posture: Optional[PostureType],
    ) -> None:
        with self._lock:
            was_posture_ovr = self._sim_posture_override is not None
            self._sim_lane_override = lane
            self._sim_posture_override = posture
            if lane is not None:
                self._human_lane = lane
                self._smoothed_x = 0.20 if lane < 0 else (0.80 if lane > 0 else 0.50)
                if lane < 0:
                    self._hand_gesture = "ONE_HAND_LEFT"
                    self._finger_count = 1
                elif lane > 0:
                    self._hand_gesture = "ONE_HAND_RIGHT"
                    self._finger_count = 1
            if posture is not None:
                self._human_posture = posture
                if posture == "JUMP":
                    self._smoothed_y = 0.18
                    self._hand_gesture = "BOTH_HANDS_UP"
                    self._finger_count = 2
                elif posture == "CROUCH":
                    self._smoothed_y = 0.80
                    self._hand_gesture = "BOTH_HANDS_DOWN"
                    self._finger_count = 2
                else:
                    self._hand_gesture = "HANDS_NEUTRAL"
                    self._finger_count = 1
            elif was_posture_ovr:
                self._crouch_hold_until = 0.0
                self._jump_hold_until = 0.0

    def calibrate_neutral(self) -> None:
        with self._lock:
            self._neutral_y = max(0.24, min(0.76, self._smoothed_y))
            self._neutral_initialized = True
            self._jump_hold_until = 0.0
            self._crouch_hold_until = 0.0
            self._bg_gray = None
            self._frame_count = 0
            self._last_valid_hands = []

    def reset_reference(self) -> None:
        with self._lock:
            self._prev_gray = None
            self._motion_score = 0.0

    def _update_face_and_skin_calibration(
        self, frame_bgr: np.ndarray, gray: np.ndarray, now: float
    ) -> Optional[tuple[int, int, int, int]]:
        """
        Detects the user's face, persists the smoothed anchor box across brief dropouts (1.2s),
        and adaptively samples the face center (Cr, Cb) to tune skin detection to room lighting.
        """
        detected_box: Optional[tuple[float, float, float, float]] = None
        if self._face_cascade is not None and (self._frame_count <= 2 or self._frame_count % 5 == 0):
            try:
                cv2.resize(
                    gray,
                    (self.width // 2, self.height // 2),
                    dst=self._small_gray,
                    interpolation=cv2.INTER_LINEAR,
                )
                cv2.equalizeHist(self._small_gray, self._eq_small_gray)
                faces = self._face_cascade.detectMultiScale(
                    self._eq_small_gray,
                    scaleFactor=1.15,
                    minNeighbors=3,
                    minSize=(14, 14),
                )
                if len(faces) > 0:
                    fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                    detected_box = (float(fx * 2), float(fy * 2), float(fw * 2), float(fh * 2))
            except Exception:
                pass

        if detected_box is not None:
            if self._face_box_smooth is None:
                self._face_box_smooth = detected_box
            else:
                ax, ay, aw, ah = self._face_box_smooth
                bx, by, bw, bh = detected_box
                self._face_box_smooth = (
                    ax * 0.65 + bx * 0.35,
                    ay * 0.65 + by * 0.35,
                    aw * 0.65 + bw * 0.35,
                    ah * 0.65 + bh * 0.35,
                )
            self._face_last_seen = now

            # Sample center patch of face in YCrCb to adapt skin tone to current lighting
            fx, fy, fw, fh = detected_box
            x0 = max(0, int(fx + fw * 0.28))
            x1 = min(self.width, int(fx + fw * 0.72))
            y0 = max(0, int(fy + fh * 0.26))
            y1 = min(self.height, int(fy + fh * 0.66))
            if x1 > x0 + 4 and y1 > y0 + 4:
                patch = self._ycrcb_buf[y0:y1, x0:x1]
                med_cr = float(np.median(patch[:, :, 1]))
                med_cb = float(np.median(patch[:, :, 2]))
                if 125.0 <= med_cr <= 185.0 and 70.0 <= med_cb <= 140.0:
                    if self._face_cr is None or self._face_cb is None:
                        self._face_cr = med_cr
                        self._face_cb = med_cb
                    else:
                        self._face_cr = self._face_cr * 0.80 + med_cr * 0.20
                        self._face_cb = self._face_cb * 0.80 + med_cb * 0.20

        if self._face_box_smooth is not None and (now - self._face_last_seen) <= 1.2:
            sx, sy, sw, sh = self._face_box_smooth
            return int(sx), int(sy), int(sw), int(sh)
        return None

    def _detect_body_pose(
        self,
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        delta: Optional[np.ndarray],
        now: float,
    ) -> list[tuple[float, float]]:
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb, dst=self._ycrcb_buf)
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV, dst=self._hsv_buf)

        # Foreground motion detection (dynamic body tracking)
        if delta is not None:
            cv2.threshold(delta, 10, 255, cv2.THRESH_BINARY, dst=self._combined_hand_mask)
        else:
            cv2.threshold(self._fg_curr_buf, 14, 255, cv2.THRESH_BINARY, dst=self._combined_hand_mask)

        # Include skin regions when present
        cv2.inRange(self._ycrcb_buf, (15, 128, 70), (255, 185, 135), dst=self._skin_mask)
        cv2.inRange(self._hsv_buf, (0, 20, 30), (32, 255, 255), dst=self._skin_hsv_mask)
        cv2.bitwise_or(self._skin_mask, self._skin_hsv_mask, dst=self._skin_mask)
        cv2.bitwise_or(self._combined_hand_mask, self._skin_mask, dst=self._combined_hand_mask)

        cv2.morphologyEx(self._combined_hand_mask, cv2.MORPH_OPEN, self._morph_kernel, dst=self._combined_hand_mask)
        cv2.morphologyEx(self._combined_hand_mask, cv2.MORPH_CLOSE, self._morph_kernel, dst=self._combined_hand_mask)

        face_box = self._update_face_and_skin_calibration(frame_bgr, gray, now)

        contours, _ = cv2.findContours(self._combined_hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area >= 160.0:
                valid_contours.append((c, area))

        valid_contours.sort(key=lambda item: item[1], reverse=True)

        points: list[tuple[float, float]] = []

        if face_box is not None:
            fx, fy, fw, fh = face_box
            head_cx = (fx + fw * 0.5) / float(self.width)
            head_cy = (fy + fh * 0.45) / float(self.height)
            points.append((max(0.02, min(0.98, head_cx)), max(0.02, min(0.98, head_cy))))

        for c, area in valid_contours[:2]:
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx = float(m["m10"] / m["m00"]) / float(self.width)
                cy = float(m["m01"] / m["m00"]) / float(self.height)
                if not any(abs(cx - px) < 0.10 and abs(cy - py) < 0.10 for px, py in points):
                    points.append((max(0.02, min(0.98, cx)), max(0.02, min(0.98, cy))))

        if not points and delta is not None:
            arr_cnt = int(np.count_nonzero(self._arriving_mask))
            if arr_cnt >= 200:
                arr_pts = np.argwhere(self._arriving_mask > 0)
                if len(arr_pts) > 0:
                    cy = float(np.mean(arr_pts[:, 0])) / float(self.height)
                    cx = float(np.mean(arr_pts[:, 1])) / float(self.width)
                    points.append((max(0.02, min(0.98, cx)), max(0.02, min(0.98, cy))))

        points.sort(key=lambda pt: pt[0])
        self._last_hand_contours = []
        self._last_valid_hands = points
        self._last_valid_hands_time = now if points else 0.0
        self._detected_hands = points
        return points

    def _detect_human_pose(
        self,
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        delta: Optional[np.ndarray],
        is_sim: bool,
        lane_override: Optional[int],
        posture_override: Optional[PostureType],
    ) -> tuple[float, float, int, PostureType]:
        """
        Box-Based Hand-Only Control Logic:
        - TOP JUMP BOX (y < CV_JUMP_Y_THRESHOLD = 0.28)       -> JUMP
        - BOTTOM CROUCH BOX (y > CV_CROUCH_Y_THRESHOLD = 0.72) -> CROUCH
        - LEFT BOX (x < CV_LEFT_X_THRESHOLD = 0.32)            -> LANE LEFT (-1)
        - RIGHT BOX (x > CV_RIGHT_X_THRESHOLD = 0.68)          -> LANE RIGHT (+1)
        - ENLARGED CENTER STAND BOX (36% x 44%)                -> CENTER / STAND
        """
        now = time.perf_counter()
        jump_enter = CV_JUMP_Y_THRESHOLD
        jump_exit = jump_enter + 0.04
        crouch_enter = CV_CROUCH_Y_THRESHOLD
        crouch_exit = crouch_enter - 0.04

        left_enter = CV_LEFT_X_THRESHOLD
        left_exit = left_enter + 0.04
        right_enter = CV_RIGHT_X_THRESHOLD
        right_exit = right_enter - 0.04

        points: list[tuple[float, float]] = []
        if not is_sim:
            points = self._detect_body_pose(frame_bgr, gray, delta, now)

        num_points = len(points)
        target_x = self._smoothed_x
        target_y = self._smoothed_y
        detected = num_points > 0
        box_jump_hit = False
        box_crouch_hit = False
        box_lane: Optional[int] = None
        in_stand_box = False

        if num_points >= 1:
            any_top = any(py < jump_enter for _, py in points)
            any_bot = any(py > crouch_enter for _, py in points)
            any_left = any(px < left_enter for px, _ in points)
            any_right = any(px > right_enter for px, _ in points)

            mean_x = sum(px for px, _ in points) / float(num_points)
            mean_y = sum(py for _, py in points) / float(num_points)
            target_x = mean_x
            target_y = mean_y

            if any_top and not any_bot:
                box_jump_hit = True
                target_y = min(py for _, py in points)
            elif any_bot and not any_top:
                box_crouch_hit = True
                target_y = max(py for _, py in points)
            elif not any_top and not any_bot:
                target_y = self._neutral_y

            if any_left and not any_right:
                box_lane = -1
                target_x = min(px for px, _ in points)
            elif any_right and not any_left:
                box_lane = 1
                target_x = max(px for px, _ in points)
            elif not any_left and not any_right:
                box_lane = 0
                in_stand_box = not any_top and not any_bot

        # Direct peripheral box pixel density triggers (higher threshold to prevent false-triggers from ambient light)
        if not is_sim and lane_override is None and posture_override is None:
            top_row = int(self.height * jump_enter)
            bot_row = int(self.height * crouch_enter)
            left_col = int(self.width * left_enter)
            right_col = int(self.width * right_enter)

            # Measure pixel density only inside the designated peripheral fist zones
            top_px = int(np.count_nonzero(self._combined_hand_mask[:top_row, :]))
            bot_px = int(np.count_nonzero(self._combined_hand_mask[bot_row:, :]))
            left_px = int(np.count_nonzero(self._combined_hand_mask[top_row:bot_row, :left_col]))
            right_px = int(np.count_nonzero(self._combined_hand_mask[top_row:bot_row, right_col:]))

            if top_px >= 180 and top_px >= bot_px * 1.5:
                box_jump_hit = True
                target_y = min(target_y, jump_enter - 0.06)
                in_stand_box = False
                detected = True
            elif bot_px >= 180 and bot_px >= top_px * 1.5:
                box_crouch_hit = True
                target_y = max(target_y, crouch_enter + 0.06)
                in_stand_box = False
                detected = True

            if left_px >= 140 and left_px >= right_px * 1.5:
                box_lane = -1
                in_stand_box = False
                detected = True
            elif right_px >= 140 and right_px >= left_px * 1.5:
                box_lane = 1
                in_stand_box = False
                detected = True

        if lane_override is not None:
            target_x = 0.20 if lane_override < 0 else (0.80 if lane_override > 0 else 0.50)
            detected = True
        elif is_sim:
            target_x = 0.50

        if posture_override is not None:
            if posture_override == "JUMP":
                target_y = 0.16
                box_jump_hit = True
                num_hands = 2
                self._detected_hands = [(0.34, 0.16), (0.66, 0.16)]
            elif posture_override == "CROUCH":
                target_y = 0.84
                box_crouch_hit = True
                num_hands = 2
                self._detected_hands = [(0.34, 0.84), (0.66, 0.84)]
            else:
                target_y = self._neutral_y
                num_hands = 1
                self._detected_hands = [(target_x, self._neutral_y)]
            detected = True
        elif is_sim:
            target_y = self._neutral_y
            num_hands = 1
            self._detected_hands = [(target_x, self._neutral_y)]

        vx = 0.0
        vy = 0.0
        if detected or is_sim:
            vx = target_x - self._prev_raw_x
            vy = target_y - self._prev_raw_y
            self._prev_raw_x = target_x
            self._prev_raw_y = target_y
            # Smooth with heavy dampening (0.60 previous + 0.40 current) to eliminate twitchy sensitivity
            self._smoothed_x = self._smoothed_x * 0.60 + target_x * 0.40
            self._smoothed_y = self._smoothed_y * 0.60 + target_y * 0.40

        # Horizontal Lane Selection: Requires deliberate hand positioning into the side boxes
        if lane_override is not None:
            lane = lane_override
        elif box_lane is not None:
            lane = box_lane
        else:
            prev_lane = self._human_lane
            # Deliberate threshold crossings with generous deadzone
            if self._smoothed_x < left_enter - 0.02 or vx < -0.08:
                lane = -1
            elif self._smoothed_x > right_enter + 0.02 or vx > 0.08:
                lane = 1
            elif prev_lane == -1 and self._smoothed_x < left_exit:
                lane = -1
            elif prev_lane == 1 and self._smoothed_x > right_exit:
                lane = 1
            else:
                lane = 0

        # Vertical Posture Selection: Requires clear deliberate fist placement in Top or Bottom box
        if posture_override is not None:
            posture: PostureType = posture_override
            if posture == "JUMP":
                self._jump_hold_until = now + 0.35
                self._crouch_hold_until = 0.0
            elif posture == "CROUCH":
                self._crouch_hold_until = now + 0.35
                self._jump_hold_until = 0.0
        elif box_jump_hit or self._smoothed_y < jump_enter:
            posture = "JUMP"
            self._jump_hold_until = now + 0.28
            self._crouch_hold_until = 0.0
        elif box_crouch_hit or self._smoothed_y > crouch_enter:
            posture = "CROUCH"
            self._crouch_hold_until = now + 0.28
            self._jump_hold_until = 0.0
        elif in_stand_box:
            # Explicit hand placement inside enlarged center STAND box responds immediately
            posture = "STAND"
            self._jump_hold_until = 0.0
            self._crouch_hold_until = 0.0
        else:
            prev_posture = self._human_posture
            if vy < -0.09 and self._smoothed_y < self._neutral_y - 0.06:
                posture = "JUMP"
                self._jump_hold_until = now + 0.25
                self._crouch_hold_until = 0.0
            elif vy > 0.09 and self._smoothed_y > self._neutral_y + 0.06:
                posture = "CROUCH"
                self._crouch_hold_until = now + 0.25
                self._jump_hold_until = 0.0
            elif prev_posture == "JUMP" and now < self._jump_hold_until and self._smoothed_y < crouch_enter:
                posture = "JUMP"
            elif prev_posture == "CROUCH" and now < self._crouch_hold_until and self._smoothed_y > jump_enter:
                posture = "CROUCH"
            else:
                posture = "STAND"

        if posture == "JUMP":
            gesture: HandGestureType = "BOTH_HANDS_UP"
        elif posture == "CROUCH":
            gesture = "BOTH_HANDS_DOWN"
        elif lane < 0:
            gesture = "ONE_HAND_LEFT"
        elif lane > 0:
            gesture = "ONE_HAND_RIGHT"
        else:
            gesture = "HANDS_NEUTRAL"

        self._hand_detected = detected or is_sim
        self._hand_gesture = gesture
        self._finger_count = num_points
        return self._smoothed_x, self._smoothed_y, lane, posture

    def _tint_roi(
        self, frame_bgr: np.ndarray, y0: int, y1: int, x0: int, x1: int, bgr: tuple[int, int, int], alpha: float = 0.25
    ) -> None:
        if y1 <= y0 or x1 <= x0:
            return
        roi = frame_bgr[y0:y1, x0:x1]
        overlay = np.full_like(roi, bgr)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    def _draw_corner_brackets(
        self,
        frame_bgr: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        thickness: int = 2,
        length: int = 12,
    ) -> None:
        lx = min(length, max(4, (x1 - x0) // 4))
        ly = min(length, max(4, (y1 - y0) // 4))
        # Top-Left
        cv2.line(frame_bgr, (x0, y0), (x0 + lx, y0), color, thickness)
        cv2.line(frame_bgr, (x0, y0), (x0, y0 + ly), color, thickness)
        # Top-Right
        cv2.line(frame_bgr, (x1, y0), (x1 - lx, y0), color, thickness)
        cv2.line(frame_bgr, (x1, y0), (x1, y0 + ly), color, thickness)
        # Bottom-Left
        cv2.line(frame_bgr, (x0, y1), (x0 + lx, y1), color, thickness)
        cv2.line(frame_bgr, (x0, y1), (x0, y1 - ly), color, thickness)
        # Bottom-Right
        cv2.line(frame_bgr, (x1, y1), (x1 - lx, y1), color, thickness)
        cv2.line(frame_bgr, (x1, y1), (x1, y1 - ly), color, thickness)

    def _draw_cyber_reticle(
        self,
        frame_bgr: np.ndarray,
        px: int,
        py: int,
        color: tuple[int, int, int],
        phase: float,
        label: str,
    ) -> None:
        # Outer rotated diamond
        d = 16
        pts = np.array([[px, py - d], [px + d, py], [px, py + d], [px - d, py]], dtype=np.int32)
        cv2.polylines(frame_bgr, [pts], True, color, 1, cv2.LINE_AA)
        # Dual concentric rings
        cv2.circle(frame_bgr, (px, py), 12, color, 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)
        # Animated rotating crosshair ticks
        for k in range(4):
            a = phase + k * (np.pi * 0.5)
            ca, sa = float(np.cos(a)), float(np.sin(a))
            p_in = (int(px + ca * 13), int(py + sa * 13))
            p_out = (int(px + ca * 20), int(py + sa * 20))
            cv2.line(frame_bgr, p_in, p_out, color, 2, cv2.LINE_AA)
        # Glowing core dot & tag
        cv2.circle(frame_bgr, (px, py), 3, (255, 255, 255), -1)
        if label:
            cv2.putText(frame_bgr, label, (px - 6, max(12, py - 21)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)

    def _draw_pose_hud_on_frame(
        self,
        frame_bgr: np.ndarray,
        bx: float,
        by: float,
        lane: int,
        posture: PostureType,
    ) -> None:
        w, h = self.width, self.height
        jump_y = int(h * CV_JUMP_Y_THRESHOLD)
        crouch_y = int(h * CV_CROUCH_Y_THRESHOLD)
        left_x = int(w * CV_LEFT_X_THRESHOLD)
        right_x = int(w * CV_RIGHT_X_THRESHOLD)

        # Highlight active zone with sleek translucent cyber neon tint
        in_center_stand = posture == "STAND" and lane == 0
        if posture == "JUMP":
            self._tint_roi(frame_bgr, 0, jump_y, 0, w, (0, 225, 255), 0.25)
        elif posture == "CROUCH":
            self._tint_roi(frame_bgr, crouch_y, h, 0, w, (255, 200, 35), 0.25)
        if lane < 0:
            self._tint_roi(frame_bgr, jump_y, crouch_y, 0, left_x, (245, 210, 56), 0.22)
        elif lane > 0:
            self._tint_roi(frame_bgr, jump_y, crouch_y, right_x, w, (245, 210, 56), 0.22)
        if in_center_stand:
            self._tint_roi(frame_bgr, jump_y, crouch_y, left_x, right_x, (210, 175, 55), 0.14)

        # Grid divider lines for the 5 Target Boxes (with enlarged center STAND box)
        cv2.line(frame_bgr, (0, jump_y), (w, jump_y), (0, 225, 255) if posture == "JUMP" else (55, 95, 110), 2 if posture == "JUMP" else 1)
        cv2.line(frame_bgr, (0, crouch_y), (w, crouch_y), (255, 200, 35) if posture == "CROUCH" else (95, 85, 55), 2 if posture == "CROUCH" else 1)
        cv2.line(frame_bgr, (left_x, jump_y), (left_x, crouch_y), (245, 210, 56) if lane < 0 else (65, 90, 95), 2 if lane < 0 else 1)
        cv2.line(frame_bgr, (right_x, jump_y), (right_x, crouch_y), (245, 210, 56) if lane > 0 else (65, 90, 95), 2 if lane > 0 else 1)

        # Cyber corner brackets on each of the 5 boxes
        self._draw_corner_brackets(frame_bgr, 3, 3, w - 4, jump_y - 3, (0, 235, 255) if posture == "JUMP" else (65, 110, 130), 2 if posture == "JUMP" else 1, 14)
        self._draw_corner_brackets(frame_bgr, 3, crouch_y + 3, w - 4, h - 4, (255, 210, 45) if posture == "CROUCH" else (110, 95, 60), 2 if posture == "CROUCH" else 1, 14)
        self._draw_corner_brackets(frame_bgr, 3, jump_y + 3, left_x - 3, crouch_y - 3, (245, 220, 75) if lane < 0 else (70, 95, 105), 2 if lane < 0 else 1, 10)
        self._draw_corner_brackets(frame_bgr, right_x + 3, jump_y + 3, w - 4, crouch_y - 3, (245, 220, 75) if lane > 0 else (70, 95, 105), 2 if lane > 0 else 1, 10)
        self._draw_corner_brackets(frame_bgr, left_x + 3, jump_y + 3, right_x - 3, crouch_y - 3, (125, 240, 220) if in_center_stand else (75, 115, 115), 2 if in_center_stand else 1, 16)

        # Clean Box Labels
        cv2.putText(frame_bgr, "UP: JUMP", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 235, 255), 1)
        cv2.putText(frame_bgr, "DOWN: CROUCH", (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 210, 60), 1)
        cv2.putText(frame_bgr, "L", (10, (jump_y + crouch_y) // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 215, 75), 1)
        cv2.putText(frame_bgr, "R", (w - 22, (jump_y + crouch_y) // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 215, 75), 1)
        cv2.putText(frame_bgr, "CENTER: STAND", ((left_x + right_x) // 2 - 38, jump_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (145, 240, 220), 1)

        ret_color = (125, 240, 220)
        if posture == "JUMP":
            ret_color = (80, 235, 255)
        elif posture == "CROUCH":
            ret_color = (255, 210, 60)
        elif lane != 0:
            ret_color = (245, 215, 75)

        phase = (time.perf_counter() * 3.2) % (np.pi * 2.0)
        if len(self._detected_hands) >= 2:
            p1 = (int(self._detected_hands[0][0] * w), int(self._detected_hands[0][1] * h))
            p2 = (int(self._detected_hands[1][0] * w), int(self._detected_hands[1][1] * h))
            cv2.line(frame_bgr, p1, p2, ret_color, 1, cv2.LINE_AA)
            self._draw_cyber_reticle(frame_bgr, p1[0], p1[1], ret_color, phase, "TRACK")
            self._draw_cyber_reticle(frame_bgr, p2[0], p2[1], ret_color, -phase, "TRACK")
        elif len(self._detected_hands) == 1:
            p0 = (int(self._detected_hands[0][0] * w), int(self._detected_hands[0][1] * h))
            self._draw_cyber_reticle(frame_bgr, p0[0], p0[1], ret_color, phase, "TRACK")
        else:
            px, py = int(bx * w), int(by * h)
            self._draw_cyber_reticle(frame_bgr, px, py, ret_color, phase, "")

        cv2.putText(frame_bgr, f"POSE: {posture}", (w - 110, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.34, ret_color, 1)

    def _compute_motion_and_pose(
        self,
        frame_bgr: np.ndarray,
        is_sim: bool,
        lane_override: Optional[int],
        posture_override: Optional[PostureType],
    ) -> tuple[float, np.ndarray, float, float, int, PostureType]:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            resized = cv2.resize(frame_bgr, (self.width, self.height))
        else:
            resized = frame_bgr.copy()

        if not is_sim:
            cv2.flip(resized, 1, dst=resized)

        self._frame_count += 1
        cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY, dst=self._gray_buf)
        cv2.GaussianBlur(self._gray_buf, CV_BLUR_KERNEL, 0, dst=self._blur_buf)

        if self._bg_gray is None or self._bg_gray.shape != self._blur_buf.shape:
            med_val = int(np.median(self._blur_buf))
            self._bg_gray = np.full_like(self._blur_buf, med_val)

        cv2.absdiff(self._blur_buf, self._bg_gray, dst=self._fg_curr_buf)

        score = 0.0
        delta: Optional[np.ndarray] = None
        if self._prev_gray is not None and self._prev_gray.shape == self._blur_buf.shape:
            cv2.absdiff(self._prev_gray, self._blur_buf, dst=self._delta_buf)
            delta = self._delta_buf
            cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY, dst=self._mask_buf)
            changed_pct = (float(np.count_nonzero(self._mask_buf)) / float(self._mask_buf.size)) * 100.0
            mean_delta = float(delta.mean())
            raw_score = min(100.0, (changed_pct * 1.4) + (mean_delta * 0.6))
            score = raw_score if is_sim else (self._motion_score * 0.25 + raw_score * 0.75)

            cv2.absdiff(self._prev_gray, self._bg_gray, dst=self._fg_prev_buf)
            cv2.threshold(self._delta_buf, 17, 255, cv2.THRESH_BINARY, dst=self._motion_mask_buf)
            cv2.subtract(self._fg_curr_buf, self._fg_prev_buf, dst=self._arriving_mask)
            cv2.threshold(self._arriving_mask, 3, 255, cv2.THRESH_BINARY, dst=self._arriving_mask)
            cv2.bitwise_and(self._arriving_mask, self._motion_mask_buf, dst=self._arriving_mask)

            cv2.subtract(self._fg_prev_buf, self._fg_curr_buf, dst=self._departing_mask)
            cv2.threshold(self._departing_mask, 3, 255, cv2.THRESH_BINARY, dst=self._departing_mask)
            cv2.bitwise_and(self._departing_mask, self._motion_mask_buf, dst=self._departing_mask)

            if self._frame_count <= 18 or self._frame_count % 3 == 0:
                alpha = 0.12 if self._frame_count <= 18 else 0.03
                cv2.addWeighted(self._blur_buf, alpha, self._bg_gray, 1.0 - alpha, 0.0, dst=self._bg_gray)
        else:
            self._arriving_mask.fill(0)
            self._departing_mask.fill(0)

        if self._prev_gray is None:
            self._prev_gray = self._blur_buf.copy()
        else:
            np.copyto(self._prev_gray, self._blur_buf)

        bx, by, lane, posture = self._detect_human_pose(
            resized, self._gray_buf, delta, is_sim, lane_override, posture_override
        )
        self._draw_pose_hud_on_frame(resized, bx, by, lane, posture)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return round(score, 2), rgb, bx, by, lane, posture

    def _generate_sim_frame(
        self,
        motion_active: bool,
        lane_override: Optional[int] = None,
        posture_override: Optional[PostureType] = None,
    ) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), (10, 14, 12), dtype=np.uint8)
        frame[:, ::32] = (22, 34, 28)
        frame[::32, :] = (22, 34, 28)

        cx = self.width // 2
        if lane_override is not None:
            cx = int(self.width * (0.20 if lane_override < 0 else (0.80 if lane_override > 0 else 0.50)))

        cy = self.height // 2
        if posture_override == "JUMP":
            cy = int(self.height * 0.20)
        elif posture_override == "CROUCH":
            cy = int(self.height * 0.80)

        if motion_active:
            self._sim_phase += 0.45
            cx += int(np.sin(self._sim_phase) * 42)
            cy += int(np.cos(self._sim_phase * 1.3) * 18)
            cv2.circle(frame, (cx - 48, cy), 22, (230, 230, 230), -1)
            cv2.circle(frame, (cx + 48, cy), 22, (230, 230, 230), -1)
        elif posture_override in ("JUMP", "CROUCH"):
            lh_x = max(32, cx - 54)
            rh_x = min(self.width - 32, cx + 54)
            cv2.circle(frame, (lh_x, cy), 22, (95, 165, 125), -1)
            cv2.circle(frame, (rh_x, cy), 22, (95, 165, 125), -1)
        else:
            cv2.circle(frame, (cx, cy), 24, (90, 145, 110), -1)

        cv2.putText(
            frame,
            "2 Hands Up/Down | 1 Hand L/R",
            (10, self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (140, 225, 175),
            1,
        )
        return frame

    def _compute_motion(self, frame_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        score, rgb, bx, by, lane, posture = self._compute_motion_and_pose(
            frame_bgr, True, self._sim_lane_override, self._sim_posture_override
        )
        self._smoothed_x = bx
        self._smoothed_y = by
        self._human_lane = lane
        self._human_posture = posture
        return score, rgb

    def _capture_loop(self) -> None:
        target_period = 1.0 / 30.0
        while self._running:
            loop_start = time.perf_counter()
            frame_bgr: Optional[np.ndarray] = None

            with self._lock:
                sim_mode = self._is_simulation
                sim_motion = self._sim_motion_trigger
                lane_ovr = self._sim_lane_override
                posture_ovr = self._sim_posture_override
                cap = self._cap

            if not sim_mode and cap is not None:
                try:
                    ret, raw = cap.read()
                    if ret and raw is not None and raw.size > 0:
                        frame_bgr = raw
                        self._consecutive_failures = 0
                    else:
                        self._consecutive_failures += 1
                except Exception:
                    self._consecutive_failures += 1

                if self._consecutive_failures >= 15:
                    with self._lock:
                        self._release_camera()
                        self._is_simulation = True
                        self._status_label = "CAMERA LOST (SIM)"
                        self._prev_gray = None
                        self._bg_gray = None
                        self._frame_count = 0
                    sim_mode = True
                    frame_bgr = self._generate_sim_frame(sim_motion, lane_ovr, posture_ovr)
            else:
                frame_bgr = self._generate_sim_frame(sim_motion, lane_ovr, posture_ovr)

            if frame_bgr is not None:
                score, rgb, bx, by, lane, posture = self._compute_motion_and_pose(
                    frame_bgr, sim_mode, lane_ovr, posture_ovr
                )
                with self._lock:
                    self._motion_score = score
                    self._latest_rgb = rgb
                    self._smoothed_x = bx
                    self._smoothed_y = by
                    self._human_lane = lane
                    self._human_posture = posture

            elapsed = time.perf_counter() - loop_start
            if elapsed < target_period:
                time.sleep(max(0.002, target_period - elapsed))

    def get_sample(self) -> MotionSample:
        with self._lock:
            return MotionSample(
                motion_score=self._motion_score,
                frame_rgb=self._latest_rgb,
                is_simulation=self._is_simulation,
                status_label=self._status_label,
                sim_motion_active=self._sim_motion_trigger,
                human_lane=self._human_lane,
                human_posture=self._human_posture,
                body_x=self._smoothed_x,
                body_y=self._smoothed_y,
                hand_gesture=self._hand_gesture,
                finger_count=self._finger_count,
                hand_detected=self._hand_detected,
            )

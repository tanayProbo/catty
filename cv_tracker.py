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

    def _detect_hands_only(
        self,
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        delta: Optional[np.ndarray],
        now: float,
    ) -> list[tuple[float, float]]:
        """
        Foreground-Gated Dual-Hand Detector:
        - Converts frame to YCrCb & HSV once.
        - Uses persistent face anchor (1.2s) + adaptive face skin calibration (Cr/Cb +-20).
        - Gates the skin mask strictly with (fg_mask | motion_mask) to eliminate 100% of static
          background walls/furniture, while unioning strong arriving motion for dim light & tests.
        - Masks out the face region and extracts up to 2 hand centroids sorted left-to-right.
        """
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb, dst=self._ycrcb_buf)
        face_box = self._update_face_and_skin_calibration(frame_bgr, gray, now)

        if self._face_cr is not None and self._face_cb is not None:
            cr_lo = max(124, int(self._face_cr - 20))
            cr_hi = min(195, int(self._face_cr + 20))
            cb_lo = max(65, int(self._face_cb - 20))
            cb_hi = min(145, int(self._face_cb + 20))
            cv2.inRange(self._ycrcb_buf, (20, cr_lo, cb_lo), (255, cr_hi, cb_hi), dst=self._skin_mask)
        else:
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV, dst=self._hsv_buf)
            cv2.inRange(self._ycrcb_buf, (20, 132, 77), (255, 179, 132), dst=self._skin_mask)
            cv2.inRange(self._hsv_buf, (0, 26, 40), (28, 250, 255), dst=self._skin_hsv_mask)
            cv2.bitwise_and(self._skin_mask, self._skin_hsv_mask, dst=self._skin_mask)

        # Foreground mask (absdiff vs learned background >= 14) + Motion mask (delta >= 12)
        cv2.threshold(self._fg_curr_buf, 14, 255, cv2.THRESH_BINARY, dst=self._fg_mask_buf)
        if delta is not None:
            cv2.threshold(delta, 12, 255, cv2.THRESH_BINARY, dst=self._motion_mask_buf)
            cv2.bitwise_or(self._fg_mask_buf, self._motion_mask_buf, dst=self._fg_motion_mask)
        else:
            np.copyto(self._fg_motion_mask, self._fg_mask_buf)

        # Gate skin mask with foreground/motion so static walls & furniture are ignored
        cv2.bitwise_and(self._skin_mask, self._fg_motion_mask, dst=self._combined_hand_mask)

        # Union strong arriving motion so hands in dim light or synthetic frames always trigger
        if delta is not None:
            arr_count = int(np.count_nonzero(self._arriving_mask))
            if arr_count >= 25:
                cv2.bitwise_or(self._combined_hand_mask, self._arriving_mask, dst=self._combined_hand_mask)

        # Erase the persistent face box so the head never registers as a hand
        if face_box is not None:
            fx, fy, fw, fh = face_box
            pad_x = max(0, fx - int(fw * 0.22))
            pad_y = max(0, fy - int(fh * 0.25))
            end_x = min(self.width, fx + fw + int(fw * 0.22))
            end_y = min(self.height, fy + int(fh * 1.25))
            self._combined_hand_mask[pad_y:end_y, pad_x:end_x] = 0

        cv2.morphologyEx(self._combined_hand_mask, cv2.MORPH_OPEN, self._morph_kernel, dst=self._combined_hand_mask)
        cv2.morphologyEx(self._combined_hand_mask, cv2.MORPH_CLOSE, self._morph_kernel, dst=self._combined_hand_mask)

        contours, _ = cv2.findContours(self._combined_hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= 120.0]
        valid_contours.sort(key=cv2.contourArea, reverse=True)

        hands: list[tuple[float, float]] = []
        drawn_contours: list[np.ndarray] = []

        if len(valid_contours) >= 2:
            c1, c2 = valid_contours[0], valid_contours[1]
            a1 = float(cv2.contourArea(c1))
            a2 = float(cv2.contourArea(c2))
            if a2 >= max(120.0, a1 * 0.14):
                for c in (c1, c2):
                    m = cv2.moments(c)
                    if m["m00"] > 0:
                        hx = float(m["m10"] / m["m00"]) / float(self.width)
                        hy = float(m["m01"] / m["m00"]) / float(self.height)
                        hands.append((max(0.02, min(0.98, hx)), max(0.02, min(0.98, hy))))
                        drawn_contours.append(c)
                if len(hands) == 2 and abs(hands[0][0] - hands[1][0]) < 0.09:
                    hands = [((hands[0][0] + hands[1][0]) * 0.5, min(hands[0][1], hands[1][1]))]
                    drawn_contours = [c1]

        if not hands and len(valid_contours) >= 1:
            c1 = valid_contours[0]
            bx, by, bw, bh = cv2.boundingRect(c1)
            drawn_contours = [c1]
            # Split wide connected two-hand blobs across their horizontal midpoint
            if bw >= int(self.width * 0.34):
                mid_x = bx + bw // 2
                left_slice = self._combined_hand_mask[by : by + bh, bx:mid_x]
                right_slice = self._combined_hand_mask[by : by + bh, mid_x : bx + bw]
                lc, lx, _, lmed_y, _ = _fast_mask_stats(left_slice)
                rc, rx, _, rmed_y, _ = _fast_mask_stats(right_slice)
                if lc >= 40 and rc >= 40:
                    lh_x = (float(bx) + lx * float(bw * 0.5)) / float(self.width)
                    lh_y = (float(by) + lmed_y * float(bh)) / float(self.height)
                    rh_x = (float(mid_x) + rx * float(bw * 0.5)) / float(self.width)
                    rh_y = (float(by) + rmed_y * float(bh)) / float(self.height)
                    hands = [(max(0.02, min(0.98, lh_x)), max(0.02, min(0.98, lh_y))),
                             (max(0.02, min(0.98, rh_x)), max(0.02, min(0.98, rh_y)))]

            if not hands:
                m = cv2.moments(c1)
                if m["m00"] > 0:
                    hx = float(m["m10"] / m["m00"]) / float(self.width)
                    hy = float(m["m01"] / m["m00"]) / float(self.height)
                    hands = [(max(0.02, min(0.98, hx)), max(0.02, min(0.98, hy)))]

        hands.sort(key=lambda pt: pt[0])
        if hands:
            self._last_valid_hands = hands
            self._last_valid_hands_time = now
            self._last_hand_contours = drawn_contours
            self._detected_hands = hands
            return hands

        # Short pose persistence (0.40s) when hands are held stationary
        if self._last_valid_hands and (now - self._last_valid_hands_time) <= 0.40:
            self._detected_hands = list(self._last_valid_hands)
            self._last_hand_contours = []
            return self._detected_hands

        self._last_hand_contours = []
        self._detected_hands = []
        return []

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
        Hands-Only Gesture Control Logic:
        - BOTH HANDS RAISED (or raised contour in Top Jump Zone) -> JUMP
        - BOTH HANDS LOWERED (or lowered contour in Bottom Duck Zone) -> CROUCH
        - ANY ONE HAND MOVED LEFT OR RIGHT (middle band) -> STEER LEFT (-1) OR RIGHT (+1)
        """
        now = time.perf_counter()
        jump_enter = max(0.18, min(CV_JUMP_Y_THRESHOLD, self._neutral_y - 0.08))
        jump_exit = jump_enter + 0.035
        crouch_enter = min(0.82, max(CV_CROUCH_Y_THRESHOLD - 0.02, self._neutral_y + 0.08))
        crouch_exit = crouch_enter - 0.035

        left_enter = CV_LEFT_X_THRESHOLD + 0.02
        left_exit = CV_LEFT_X_THRESHOLD + 0.055
        right_enter = CV_RIGHT_X_THRESHOLD - 0.02
        right_exit = CV_RIGHT_X_THRESHOLD - 0.055

        hands: list[tuple[float, float]] = []
        if not is_sim:
            hands = self._detect_hands_only(frame_bgr, gray, delta, now)

        num_hands = len(hands)
        target_x = self._smoothed_x
        target_y = self._smoothed_y
        detected = num_hands > 0
        both_hands_jump = False
        both_hands_crouch = False
        zone_lane: Optional[int] = None

        if num_hands >= 2:
            (lx, ly), (rx, ry) = hands[0], hands[1]
            mean_x = (lx + rx) * 0.5
            mean_y = (ly + ry) * 0.5
            target_x = mean_x
            target_y = mean_y

            # Both hands raised -> JUMP
            if (
                (ly < jump_enter and ry < jump_enter + 0.08)
                or (ry < jump_enter and ly < jump_enter + 0.08)
                or mean_y < jump_enter
            ):
                both_hands_jump = True
            # Both hands lowered -> CROUCH
            elif (
                (ly > crouch_enter and ry > crouch_enter - 0.08)
                or (ry > crouch_enter and ly > crouch_enter - 0.08)
                or mean_y > crouch_enter
            ):
                both_hands_crouch = True

            if lx < left_enter and rx <= right_enter:
                zone_lane = -1
                target_x = lx
            elif rx > right_enter and lx >= left_enter:
                zone_lane = 1
                target_x = rx
            else:
                zone_lane = 0

        elif num_hands == 1:
            hx, hy = hands[0]
            target_x = hx
            if hx < left_enter:
                zone_lane = -1
            elif hx > right_enter:
                zone_lane = 1
            else:
                zone_lane = 0

            # Allow single merged/clear contour in Top/Bottom zones to trigger Jump/Crouch,
            # while clamping middle-band steering strictly to neutral Y so L/R never false-triggers!
            if hy < jump_enter:
                target_y = hy
                both_hands_jump = True
            elif hy > crouch_enter:
                target_y = hy
                both_hands_crouch = True
            else:
                target_y = self._neutral_y

        # Check vertical zone-energy on real webcam frames when skin+fg pixels are active
        if not is_sim and lane_override is None and posture_override is None:
            top_row = int(self.height * jump_enter)
            bot_row = int(self.height * crouch_enter)
            top_px = int(np.count_nonzero(self._combined_hand_mask[:top_row, :]))
            mid_px = int(np.count_nonzero(self._combined_hand_mask[top_row:bot_row, :]))
            bot_px = int(np.count_nonzero(self._combined_hand_mask[bot_row:, :]))
            if top_px >= 220 and top_px > bot_px * 2 and top_px >= mid_px * 0.65:
                both_hands_jump = True
                target_y = min(target_y, jump_enter - 0.05)
                detected = True
            elif bot_px >= 220 and bot_px > top_px * 2 and bot_px >= mid_px * 0.65:
                both_hands_crouch = True
                target_y = max(target_y, crouch_enter + 0.05)
                detected = True

        if lane_override is not None:
            target_x = 0.20 if lane_override < 0 else (0.80 if lane_override > 0 else 0.50)
            detected = True
        elif is_sim:
            target_x = 0.50

        if posture_override is not None:
            if posture_override == "JUMP":
                target_y = 0.18
                both_hands_jump = True
                num_hands = 2
                self._detected_hands = [(0.34, 0.18), (0.66, 0.18)]
            elif posture_override == "CROUCH":
                target_y = 0.80
                both_hands_crouch = True
                num_hands = 2
                self._detected_hands = [(0.34, 0.80), (0.66, 0.80)]
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
            self._smoothed_x = self._smoothed_x * 0.30 + target_x * 0.70
            self._smoothed_y = self._smoothed_y * 0.30 + target_y * 0.70

        # Horizontal Lane Selection
        if lane_override is not None:
            lane = lane_override
        elif zone_lane is not None:
            lane = zone_lane
        else:
            prev_lane = self._human_lane
            if self._smoothed_x < left_enter or (vx < -0.042 and self._smoothed_x < 0.45):
                lane = -1
            elif self._smoothed_x > right_enter or (vx > 0.042 and self._smoothed_x > 0.55):
                lane = 1
            elif prev_lane == -1 and self._smoothed_x < left_exit:
                lane = -1
            elif prev_lane == 1 and self._smoothed_x > right_exit:
                lane = 1
            else:
                lane = 0

        # Vertical Posture Selection
        if posture_override is not None:
            posture: PostureType = posture_override
            if posture == "JUMP":
                self._jump_hold_until = now + 0.38
                self._crouch_hold_until = 0.0
            elif posture == "CROUCH":
                self._crouch_hold_until = now + 0.40
                self._jump_hold_until = 0.0
        else:
            prev_posture = self._human_posture
            if both_hands_jump or self._smoothed_y < jump_enter or (
                vy < -0.040 and self._smoothed_y < self._neutral_y - 0.03
            ):
                posture = "JUMP"
                self._jump_hold_until = now + 0.38
                self._crouch_hold_until = 0.0
            elif both_hands_crouch or self._smoothed_y > crouch_enter or (
                vy > 0.040 and self._smoothed_y > self._neutral_y + 0.03
            ):
                posture = "CROUCH"
                self._crouch_hold_until = now + 0.40
                self._jump_hold_until = 0.0
            elif (prev_posture == "JUMP" and self._smoothed_y < jump_exit) or (
                now < self._jump_hold_until and self._smoothed_y < crouch_enter
            ):
                posture = "JUMP"
            elif (prev_posture == "CROUCH" and self._smoothed_y > crouch_exit) or (
                now < self._crouch_hold_until and self._smoothed_y > jump_enter
            ):
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
        self._finger_count = num_hands
        return self._smoothed_x, self._smoothed_y, lane, posture

    def _tint_roi(
        self, frame_bgr: np.ndarray, y0: int, y1: int, x0: int, x1: int, bgr: tuple[int, int, int], alpha: float = 0.25
    ) -> None:
        if y1 <= y0 or x1 <= x0:
            return
        roi = frame_bgr[y0:y1, x0:x1]
        overlay = np.full_like(roi, bgr)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    def _draw_pose_hud_on_frame(
        self,
        frame_bgr: np.ndarray,
        bx: float,
        by: float,
        lane: int,
        posture: PostureType,
    ) -> None:
        w, h = self.width, self.height
        jump_thresh = max(0.18, min(CV_JUMP_Y_THRESHOLD, self._neutral_y - 0.08))
        crouch_thresh = min(0.82, max(CV_CROUCH_Y_THRESHOLD - 0.02, self._neutral_y + 0.08))

        jump_y = int(h * jump_thresh)
        crouch_y = int(h * crouch_thresh)
        neutral_y = int(h * self._neutral_y)
        left_x = int(w * (CV_LEFT_X_THRESHOLD + 0.02))
        right_x = int(w * (CV_RIGHT_X_THRESHOLD - 0.02))

        # Highlight active zone with translucent neon tint
        if posture == "JUMP":
            self._tint_roi(frame_bgr, 0, jump_y, 0, w, (0, 215, 255), 0.26)
        elif posture == "CROUCH":
            self._tint_roi(frame_bgr, crouch_y, h, 0, w, (255, 195, 40), 0.26)
        if lane < 0:
            self._tint_roi(frame_bgr, jump_y, crouch_y, 0, left_x, (80, 240, 140), 0.22)
        elif lane > 0:
            self._tint_roi(frame_bgr, jump_y, crouch_y, right_x, w, (80, 240, 140), 0.22)

        if self._last_hand_contours:
            cv2.drawContours(frame_bgr, self._last_hand_contours, -1, (46, 230, 125), 2)

        # Draw subtle face exclusion box when face anchor is active
        if self._face_box_smooth is not None and (time.perf_counter() - self._face_last_seen) <= 1.2:
            fx, fy, fw, fh = (int(v) for v in self._face_box_smooth)
            cv2.rectangle(frame_bgr, (fx, fy), (fx + fw, fy + fh), (75, 75, 95), 1)

        cv2.line(frame_bgr, (0, neutral_y), (w, neutral_y), (60, 80, 70), 1)
        cv2.line(frame_bgr, (0, jump_y), (w, jump_y), (0, 220, 255), 2 if posture == "JUMP" else 1)
        cv2.line(frame_bgr, (0, crouch_y), (w, crouch_y), (255, 200, 0), 2 if posture == "CROUCH" else 1)
        cv2.line(frame_bgr, (left_x, 0), (left_x, h), (70, 210, 120) if lane < 0 else (70, 90, 80), 2 if lane < 0 else 1)
        cv2.line(frame_bgr, (right_x, 0), (right_x, h), (70, 210, 120) if lane > 0 else (70, 90, 80), 2 if lane > 0 else 1)

        cv2.putText(frame_bgr, "2 HANDS UP: JUMP", (6, max(12, jump_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 225, 255), 1)
        cv2.putText(frame_bgr, "2 HANDS DOWN: CROUCH", (6, min(h - 6, crouch_y + 13)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 210, 60), 1)

        ret_color = (60, 235, 130)
        if posture == "JUMP":
            ret_color = (80, 225, 255)
        elif posture == "CROUCH":
            ret_color = (255, 210, 60)

        if len(self._detected_hands) >= 2:
            p1 = (int(self._detected_hands[0][0] * w), int(self._detected_hands[0][1] * h))
            p2 = (int(self._detected_hands[1][0] * w), int(self._detected_hands[1][1] * h))
            cv2.line(frame_bgr, p1, p2, ret_color, 2)
            for idx, pt in enumerate((p1, p2)):
                cv2.circle(frame_bgr, pt, 12, ret_color, 2)
                cv2.circle(frame_bgr, pt, 3, ret_color, -1)
                lbl = "L" if idx == 0 else "R"
                cv2.putText(frame_bgr, lbl, (pt[0] - 4, pt[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ret_color, 1)
        else:
            px, py = int(bx * w), int(by * h)
            cv2.circle(frame_bgr, (px, py), 14, ret_color, 2)
            cv2.circle(frame_bgr, (px, py), 3, ret_color, -1)

        gest_str = self._hand_gesture.replace("_", " ")
        cv2.putText(frame_bgr, gest_str, (w - 142, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.34, ret_color, 1)

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

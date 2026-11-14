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


def _fast_mask_stats(mask: np.ndarray) -> tuple[int, float, float, float, float]:
    row_counts = np.count_nonzero(mask, axis=1)
    total = int(row_counts.sum())
    if total < 30:
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
        self._status_label = "STARTING"

        # Pre-allocated image buffers for the vision loop
        self._prev_gray: Optional[np.ndarray] = None
        self._gray_buf = np.empty((height, width), dtype=np.uint8)
        self._blur_buf = np.empty((height, width), dtype=np.uint8)
        self._delta_buf = np.empty((height, width), dtype=np.uint8)
        self._mask_buf = np.empty((height, width), dtype=np.uint8)
        self._small_gray = np.empty((height // 2, width // 2), dtype=np.uint8)
        self._latest_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        self._motion_score: float = 0.0
        self._sim_motion_trigger: bool = False
        self._sim_phase: float = 0.0
        self._consecutive_failures: int = 0

        self._face_cascade = self._load_cascade("haarcascade_frontalface_default.xml")
        self._smoothed_x: float = 0.50
        self._smoothed_y: float = 0.50
        self._prev_raw_x: float = 0.50
        self._prev_raw_y: float = 0.50
        self._neutral_y: float = 0.50
        self._human_lane: int = 0
        self._human_posture: PostureType = "STAND"
        self._jump_hold_until: float = 0.0
        self._crouch_hold_until: float = 0.0
        self._crouch_suppress_until: float = 0.0
        self._jump_suppress_until: float = 0.0

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
            self._status_label = "SIMULATION"

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
                    self._status_label = "WEBCAM LIVE"
                    self._consecutive_failures = 0
                    self._prev_gray = None
                    return
            if cap is not None:
                cap.release()
        except Exception:
            pass

        self._cap = None
        self._is_simulation = True
        self._status_label = "NO CAMERA (SIM)"
        self._prev_gray = None

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
            self._motion_score = 0.0
            if enabled:
                self._release_camera()
                self._is_simulation = True
                self._status_label = "SIMULATION"
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
            self._sim_lane_override = lane
            self._sim_posture_override = posture
            if posture is None:
                self._crouch_hold_until = 0.0
                self._jump_suppress_until = 0.0

    def calibrate_neutral(self) -> None:
        with self._lock:
            self._neutral_y = max(0.32, min(0.68, self._smoothed_y))
            self._jump_hold_until = 0.0
            self._crouch_hold_until = 0.0

    def reset_reference(self) -> None:
        with self._lock:
            self._prev_gray = None
            self._motion_score = 0.0

    def _detect_human_pose(
        self,
        gray: np.ndarray,
        delta: Optional[np.ndarray],
        is_sim: bool,
        lane_override: Optional[int],
        posture_override: Optional[PostureType],
    ) -> tuple[float, float, int, PostureType]:
        now = time.perf_counter()
        target_x = self._smoothed_x
        target_y = self._smoothed_y
        detected = False
        upward_impulse_jump = False
        downward_impulse_crouch = False

        if not is_sim:
            face_found = False
            if self._face_cascade is not None:
                try:
                    cv2.resize(
                        gray,
                        (self.width // 2, self.height // 2),
                        dst=self._small_gray,
                        interpolation=cv2.INTER_LINEAR,
                    )
                    faces = self._face_cascade.detectMultiScale(
                        self._small_gray,
                        scaleFactor=1.18,
                        minNeighbors=3,
                        minSize=(14, 14),
                    )
                    if len(faces) > 0:
                        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                        target_x = float((fx + fw * 0.5) * 2.0) / float(self.width)
                        target_y = float((fy + fh * 0.44) * 2.0) / float(self.height)
                        detected = True
                        face_found = True
                except Exception:
                    pass

            if delta is not None:
                cv2.threshold(delta, 16, 255, cv2.THRESH_BINARY, dst=self._mask_buf)
                count, motion_x, top_crown_y, median_y, bottom_y = _fast_mask_stats(self._mask_buf)
                if count >= 30:
                    upper_cut = int(self.height * 0.38)
                    lower_cut = int(self.height * 0.62)
                    upper_active_pct = (
                        float(np.count_nonzero(self._mask_buf[:upper_cut, :]))
                        / float(max(1, upper_cut * self.width))
                    ) * 100.0
                    lower_active_pct = (
                        float(np.count_nonzero(self._mask_buf[lower_cut:, :]))
                        / float(max(1, (self.height - lower_cut) * self.width))
                    ) * 100.0

                    if not face_found:
                        target_x = motion_x
                        if lower_active_pct > upper_active_pct * 1.45 and top_crown_y > self._neutral_y - 0.02:
                            target_y = max(median_y, bottom_y - 0.06)
                        else:
                            target_y = min(self._smoothed_y, top_crown_y + 0.05)
                        detected = True
                    else:
                        target_x = target_x * 0.72 + motion_x * 0.28
                        if top_crown_y + 0.04 < target_y:
                            target_y = top_crown_y + 0.04

                    vy = target_y - self._prev_raw_y
                    if (vy < -0.026 and top_crown_y < self._neutral_y - 0.03) or (
                        upper_active_pct > 3.6 and top_crown_y < self._neutral_y - 0.04
                    ):
                        upward_impulse_jump = True
                    elif now >= self._crouch_suppress_until and (
                        (vy > 0.026 and median_y > self._neutral_y + 0.04)
                        or (lower_active_pct > 4.2 and top_crown_y > self._neutral_y + 0.03)
                    ):
                        downward_impulse_crouch = True

        if lane_override is not None:
            target_x = 0.20 if lane_override < 0 else (0.80 if lane_override > 0 else 0.50)
            detected = True
        elif is_sim:
            target_x = 0.50

        if posture_override is not None:
            if posture_override == "JUMP":
                target_y = 0.18
                upward_impulse_jump = True
            elif posture_override == "CROUCH":
                target_y = 0.80
                downward_impulse_crouch = True
            else:
                target_y = self._neutral_y
            detected = True
        elif is_sim:
            target_y = self._neutral_y

        vx = 0.0
        if detected or is_sim:
            vx = target_x - self._prev_raw_x
            self._prev_raw_x = target_x
            self._prev_raw_y = target_y
            self._smoothed_x = self._smoothed_x * 0.34 + target_x * 0.66
            if target_y < self._smoothed_y:
                self._smoothed_y = self._smoothed_y * 0.16 + target_y * 0.84
            else:
                self._smoothed_y = self._smoothed_y * 0.28 + target_y * 0.72

        if lane_override is not None:
            lane = lane_override
        else:
            prev_lane = self._human_lane
            left_enter = CV_LEFT_X_THRESHOLD + 0.02
            left_exit = CV_LEFT_X_THRESHOLD + 0.055
            right_enter = CV_RIGHT_X_THRESHOLD - 0.02
            right_exit = CV_RIGHT_X_THRESHOLD - 0.055

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

        jump_thresh = max(0.18, min(CV_JUMP_Y_THRESHOLD, self._neutral_y - 0.068))
        crouch_thresh = min(0.80, max(CV_CROUCH_Y_THRESHOLD - 0.03, self._neutral_y + 0.078))

        if posture_override is not None:
            posture: PostureType = posture_override
            if posture == "JUMP":
                self._jump_hold_until = now + 0.36
                self._crouch_suppress_until = now + 0.62
            elif posture == "CROUCH":
                self._crouch_hold_until = now + 0.40
                self._jump_suppress_until = now + 0.55
        elif upward_impulse_jump or self._smoothed_y < jump_thresh:
            posture = "JUMP"
            self._jump_hold_until = now + 0.38
            self._crouch_hold_until = 0.0
            self._crouch_suppress_until = now + 0.65
        elif now < self._jump_hold_until:
            posture = "JUMP"
        elif now >= self._crouch_suppress_until and (downward_impulse_crouch or self._smoothed_y > crouch_thresh):
            posture = "CROUCH"
            self._crouch_hold_until = now + 0.42
            self._jump_suppress_until = now + 0.58
        elif now < self._crouch_hold_until:
            posture = "CROUCH"
        else:
            posture = "STAND"
            if not is_sim and detected and abs(self._smoothed_y - self._neutral_y) < 0.055:
                self._neutral_y = self._neutral_y * 0.97 + self._smoothed_y * 0.03

        return self._smoothed_x, self._smoothed_y, lane, posture

    def _draw_pose_hud_on_frame(
        self,
        frame_bgr: np.ndarray,
        bx: float,
        by: float,
        lane: int,
        posture: PostureType,
    ) -> None:
        w, h = self.width, self.height
        jump_thresh = max(0.18, min(CV_JUMP_Y_THRESHOLD, self._neutral_y - 0.068))
        crouch_thresh = min(0.80, max(CV_CROUCH_Y_THRESHOLD - 0.03, self._neutral_y + 0.078))

        jump_y = int(h * jump_thresh)
        crouch_y = int(h * crouch_thresh)
        left_x = int(w * (CV_LEFT_X_THRESHOLD + 0.02))
        right_x = int(w * (CV_RIGHT_X_THRESHOLD - 0.02))

        cv2.line(frame_bgr, (0, jump_y), (w, jump_y), (0, 220, 255), 2 if posture == "JUMP" else 1)
        cv2.line(frame_bgr, (0, crouch_y), (w, crouch_y), (255, 200, 0), 2 if posture == "CROUCH" else 1)
        cv2.line(frame_bgr, (left_x, 0), (left_x, h), (70, 90, 80), 1)
        cv2.line(frame_bgr, (right_x, 0), (right_x, h), (70, 90, 80), 1)

        cv2.putText(frame_bgr, "JUMP", (8, max(12, jump_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 255), 1)
        cv2.putText(frame_bgr, "CROUCH", (8, min(h - 6, crouch_y + 13)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 210, 0), 1)

        px, py = int(bx * w), int(by * h)
        ret_color = (0, 255, 102)
        if posture == "JUMP":
            ret_color = (0, 230, 255)
        elif posture == "CROUCH":
            ret_color = (255, 210, 0)

        cv2.circle(frame_bgr, (px, py), 16, ret_color, 2)
        cv2.line(frame_bgr, (px - 22, py), (px + 22, py), ret_color, 1)
        cv2.line(frame_bgr, (px, py - 22), (px, py + 22), ret_color, 1)

        lane_str = "LEFT" if lane < 0 else ("RIGHT" if lane > 0 else "CENTER")
        tag = f"{lane_str} / {posture}"
        cv2.putText(frame_bgr, tag, (w - 132, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ret_color, 1)

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

        cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY, dst=self._gray_buf)
        cv2.GaussianBlur(self._gray_buf, CV_BLUR_KERNEL, 0, dst=self._blur_buf)

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

        if self._prev_gray is None:
            self._prev_gray = self._blur_buf.copy()
        else:
            np.copyto(self._prev_gray, self._blur_buf)

        bx, by, lane, posture = self._detect_human_pose(
            self._gray_buf, delta, is_sim, lane_override, posture_override
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

        cy = self.height // 2 + 12
        if posture_override == "JUMP":
            cy = int(self.height * 0.22)
        elif posture_override == "CROUCH":
            cy = int(self.height * 0.78)

        if motion_active:
            self._sim_phase += 0.45
            cx += int(np.sin(self._sim_phase) * 42)
            cy += int(np.cos(self._sim_phase * 1.3) * 18)
            cv2.circle(frame, (cx, cy - 24), 20, (230, 230, 230), -1)
            cv2.rectangle(frame, (cx - 28, cy - 2), (cx + 28, cy + 48), (210, 210, 210), -1)
        else:
            cv2.circle(frame, (cx, cy - 22), 18, (90, 145, 110), -1)
            body_h = 26 if posture_override == "CROUCH" else 46
            cv2.rectangle(frame, (cx - 24, cy - 2), (cx + 24, cy + body_h), (80, 125, 95), -1)

        cv2.putText(
            frame,
            "W/S: JUMP/ROLL  A/D: LANE",
            (10, self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (120, 240, 160),
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
            )

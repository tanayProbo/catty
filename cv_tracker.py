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


@dataclass
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
        self._prev_gray: Optional[np.ndarray] = None
        self._latest_rgb: np.ndarray = self._blank_frame()
        self._motion_score: float = 0.0
        self._sim_motion_trigger: bool = False
        self._sim_phase: float = 0.0
        self._consecutive_failures: int = 0

        self._face_cascade = self._load_face_cascade()
        self._smoothed_x: float = 0.5
        self._smoothed_y: float = 0.50
        self._prev_raw_y: float = 0.50
        self._neutral_y: float = 0.50
        self._human_lane: int = 0
        self._human_posture: PostureType = "STAND"
        self._jump_hold_until: float = 0.0
        self._crouch_suppress_until: float = 0.0

        self._sim_lane_override: Optional[int] = None
        self._sim_posture_override: Optional[PostureType] = None

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _load_face_cascade(self) -> Optional[cv2.CascadeClassifier]:
        try:
            cascade_dir = getattr(cv2, "data", None)
            if cascade_dir is not None:
                xml_path = os.path.join(cascade_dir.haarcascades, "haarcascade_frontalface_default.xml")
                if os.path.exists(xml_path):
                    classifier = cv2.CascadeClassifier(xml_path)
                    if not classifier.empty():
                        return classifier
        except Exception:
            pass
        return None

    def _blank_frame(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

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

    def calibrate_neutral(self) -> None:
        with self._lock:
            self._neutral_y = max(0.32, min(0.68, self._smoothed_y))
            self._jump_hold_until = 0.0

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

        if not is_sim:
            face_found = False
            if self._face_cascade is not None:
                try:
                    faces = self._face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.15,
                        minNeighbors=3,
                        minSize=(24, 24),
                    )
                    if len(faces) > 0:
                        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                        target_x = float(fx + fw * 0.5) / float(self.width)
                        target_y = float(fy + fh * 0.45) / float(self.height)
                        detected = True
                        face_found = True
                except Exception:
                    pass

            if delta is not None:
                _, mask = cv2.threshold(delta, 18, 255, cv2.THRESH_BINARY)
                ys, xs = np.nonzero(mask)
                if len(ys) > 35:
                    motion_x = float(np.mean(xs)) / float(self.width)
                    top_crown_y = float(np.percentile(ys, 18)) / float(self.height)
                    bottom_y = float(np.percentile(ys, 85)) / float(self.height)

                    upper_zone_cutoff = int(self.height * 0.38)
                    upper_active_pct = (
                        float(np.count_nonzero(mask[:upper_zone_cutoff, :]))
                        / float(max(1, upper_zone_cutoff * self.width))
                    ) * 100.0

                    if not face_found:
                        target_x = motion_x
                        if bottom_y > 0.72 and top_crown_y > 0.48:
                            target_y = float(np.mean(ys)) / float(self.height)
                        else:
                            target_y = min(self._smoothed_y, top_crown_y + 0.06)
                        detected = True
                    else:
                        target_y = min(target_y, top_crown_y + 0.05)

                    vy = target_y - self._prev_raw_y
                    if (vy < -0.032 and top_crown_y < self._neutral_y - 0.04) or (
                        upper_active_pct > 4.2 and top_crown_y < self._neutral_y - 0.05
                    ):
                        upward_impulse_jump = True

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
            else:
                target_y = self._neutral_y
            detected = True
        elif is_sim:
            target_y = self._neutral_y

        if detected or is_sim:
            self._prev_raw_y = target_y
            self._smoothed_x = self._smoothed_x * 0.50 + target_x * 0.50
            if target_y < self._smoothed_y:
                self._smoothed_y = self._smoothed_y * 0.18 + target_y * 0.82
            else:
                self._smoothed_y = self._smoothed_y * 0.45 + target_y * 0.55

        if lane_override is not None:
            lane = lane_override
        elif self._smoothed_x < CV_LEFT_X_THRESHOLD:
            lane = -1
        elif self._smoothed_x > CV_RIGHT_X_THRESHOLD:
            lane = 1
        else:
            lane = 0

        jump_thresh = max(0.18, min(CV_JUMP_Y_THRESHOLD, self._neutral_y - 0.075))
        crouch_thresh = min(0.82, max(CV_CROUCH_Y_THRESHOLD, self._neutral_y + 0.105))

        if posture_override is not None:
            posture: PostureType = posture_override
            if posture == "JUMP":
                self._jump_hold_until = now + 0.35
                self._crouch_suppress_until = now + 0.60
        elif upward_impulse_jump or self._smoothed_y < jump_thresh:
            posture = "JUMP"
            self._jump_hold_until = now + 0.38
            self._crouch_suppress_until = now + 0.68
        elif now < self._jump_hold_until:
            posture = "JUMP"
        elif self._smoothed_y > crouch_thresh and now >= self._crouch_suppress_until:
            posture = "CROUCH"
        else:
            posture = "STAND"
            if not is_sim and detected and abs(self._smoothed_y - self._neutral_y) < 0.06:
                self._neutral_y = self._neutral_y * 0.98 + self._smoothed_y * 0.02

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
        jump_thresh = max(0.18, min(CV_JUMP_Y_THRESHOLD, self._neutral_y - 0.075))
        crouch_thresh = min(0.82, max(CV_CROUCH_Y_THRESHOLD, self._neutral_y + 0.105))

        jump_y = int(h * jump_thresh)
        crouch_y = int(h * crouch_thresh)
        left_x = int(w * CV_LEFT_X_THRESHOLD)
        right_x = int(w * CV_RIGHT_X_THRESHOLD)

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
        resized = cv2.resize(frame_bgr, (self.width, self.height))
        if not is_sim:
            resized = cv2.flip(resized, 1)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, CV_BLUR_KERNEL, 0)

        score = 0.0
        delta: Optional[np.ndarray] = None
        if self._prev_gray is not None and self._prev_gray.shape == blurred.shape:
            delta = cv2.absdiff(self._prev_gray, blurred)
            _, thresh = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)
            changed_pct = (float(np.count_nonzero(thresh)) / float(thresh.size)) * 100.0
            mean_delta = float(np.mean(delta))
            score = min(100.0, (changed_pct * 1.4) + (mean_delta * 0.6))

        self._prev_gray = blurred

        bx, by, lane, posture = self._detect_human_pose(
            gray, delta, is_sim, lane_override, posture_override
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
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (10, 14, 12)

        for x in range(0, self.width, 32):
            cv2.line(frame, (x, 0), (x, self.height), (22, 34, 28), 1)
        for y in range(0, self.height, 32):
            cv2.line(frame, (0, y), (self.width, y), (22, 34, 28), 1)

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
            sleep_time = max(0.005, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)

    def get_sample(self) -> MotionSample:
        with self._lock:
            return MotionSample(
                motion_score=self._motion_score,
                frame_rgb=self._latest_rgb.copy(),
                is_simulation=self._is_simulation,
                status_label=self._status_label,
                sim_motion_active=self._sim_motion_trigger,
                human_lane=self._human_lane,
                human_posture=self._human_posture,
                body_x=self._smoothed_x,
                body_y=self._smoothed_y,
            )

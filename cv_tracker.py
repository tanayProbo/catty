"""Real-time OpenCV motion tracker with automatic mock simulation fallback."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import (
    CV_BLUR_KERNEL,
    CV_CAMERA_INDEX,
    CV_FRAME_HEIGHT,
    CV_FRAME_WIDTH,
)


@dataclass
class MotionSample:
    motion_score: float
    frame_rgb: np.ndarray
    is_simulation: bool
    status_label: str
    sim_motion_active: bool


class MotionTracker:
    """Thread-safe OpenCV frame-differencing motion detector with simulation fallback."""

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
        self._status_label = "INITIALIZING"
        self._prev_gray: Optional[np.ndarray] = None
        self._latest_rgb: np.ndarray = self._blank_frame()
        self._motion_score: float = 0.0
        self._sim_motion_trigger: bool = False
        self._sim_phase: float = 0.0
        self._consecutive_failures: int = 0

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _blank_frame(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def start(self) -> None:
        if self._running:
            return
        if not self._force_simulation:
            self._open_camera()
        else:
            self._is_simulation = True
            self._status_label = "SIMULATION MODE"

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
                    self._status_label = "CAM_00 LIVE"
                    self._consecutive_failures = 0
                    self._prev_gray = None
                    return
            if cap is not None:
                cap.release()
        except Exception:
            pass

        self._cap = None
        self._is_simulation = True
        self._status_label = "SIMULATION (NO CAM)"
        self._prev_gray = None

    def _release_camera(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def set_simulation_mode(self, enabled: bool) -> None:
        """Toggle between live webcam and mock simulation mode."""
        with self._lock:
            self._force_simulation = enabled
            self._prev_gray = None
            self._motion_score = 0.0
            if enabled:
                self._release_camera()
                self._is_simulation = True
                self._status_label = "SIMULATION MODE"
            else:
                self._open_camera()

    def set_simulated_motion(self, active: bool) -> None:
        """Used in simulation mode (e.g. holding SPACE) to inject real frame delta."""
        with self._lock:
            self._sim_motion_trigger = active

    def reset_reference(self) -> None:
        """Reset previous frame baseline before a new freeze window."""
        with self._lock:
            self._prev_gray = None
            self._motion_score = 0.0

    def _compute_motion(self, frame_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        """Run OpenCV absdiff motion detection and return (score, rgb_frame)."""
        resized = cv2.resize(frame_bgr, (self.width, self.height))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, CV_BLUR_KERNEL, 0)

        score = 0.0
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            delta = cv2.absdiff(self._prev_gray, gray)
            # Combine mean intensity delta and percentage of pixels above noise floor
            _, thresh = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)
            changed_pct = (float(np.count_nonzero(thresh)) / float(thresh.size)) * 100.0
            mean_delta = float(np.mean(delta))
            score = min(100.0, (changed_pct * 1.4) + (mean_delta * 0.6))

        self._prev_gray = gray
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return round(score, 2), rgb

    def _generate_sim_frame(self, motion_active: bool) -> np.ndarray:
        """Generate a synthetic CCTV frame for simulation fallback."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (10, 14, 12)

        # Subtle grid lines
        for x in range(0, self.width, 32):
            cv2.line(frame, (x, 0), (x, self.height), (22, 34, 28), 1)
        for y in range(0, self.height, 32):
            cv2.line(frame, (0, y), (self.width, y), (22, 34, 28), 1)

        # Static subject silhouette in center
        cx, cy = self.width // 2, self.height // 2 + 12
        if motion_active:
            self._sim_phase += 0.45
            dx = int(np.sin(self._sim_phase) * 42)
            dy = int(np.cos(self._sim_phase * 1.3) * 18)
            cx += dx
            cy += dy
            # Draw high-contrast moving intruder silhouette
            cv2.circle(frame, (cx, cy - 28), 22, (230, 230, 230), -1)
            cv2.rectangle(frame, (cx - 30, cy - 4), (cx + 30, cy + 54), (210, 210, 210), -1)
            cv2.putText(
                frame,
                "SIM_MOTION!",
                (12, self.height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (60, 60, 255),
                1,
            )
        else:
            cv2.circle(frame, (cx, cy - 28), 22, (90, 115, 100), -1)
            cv2.rectangle(frame, (cx - 28, cy - 4), (cx + 28, cy + 52), (80, 105, 90), -1)
            cv2.putText(
                frame,
                "HOLD [SPACE] TO SIMULATE MOTION",
                (12, self.height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (100, 220, 140),
                1,
            )

        return frame

    def _capture_loop(self) -> None:
        while self._running:
            loop_start = time.perf_counter()
            frame_bgr: Optional[np.ndarray] = None

            with self._lock:
                sim_mode = self._is_simulation
                sim_motion = self._sim_motion_trigger
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
                    # Graceful mid-session disconnect fallback
                    with self._lock:
                        self._release_camera()
                        self._is_simulation = True
                        self._status_label = "SIMULATION (CAM LOST)"
                        self._prev_gray = None
                    frame_bgr = self._generate_sim_frame(sim_motion)
            else:
                frame_bgr = self._generate_sim_frame(sim_motion)

            if frame_bgr is not None:
                score, rgb = self._compute_motion(frame_bgr)
                with self._lock:
                    self._motion_score = score
                    self._latest_rgb = rgb

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.005, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)

    def get_sample(self) -> MotionSample:
        """Return the latest thread-safe CV sample."""
        with self._lock:
            return MotionSample(
                motion_score=self._motion_score,
                frame_rgb=self._latest_rgb.copy(),
                is_simulation=self._is_simulation,
                status_label=self._status_label,
                sim_motion_active=self._sim_motion_trigger,
            )

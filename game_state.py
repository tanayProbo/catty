"""Core game state machine and entity animation state for LASER_PAW_PATROL."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from backend.models import RunCreate, TelemetryEventIn
from config import (
    CAT_WALK_SPEED_MPS,
    CLEAR_DURATION_SEC,
    FREEZE_CLEAR_BONUS_MPS,
    HEIST_GOAL_METERS,
    SWEEP_INTERVAL_MAX,
    SWEEP_INTERVAL_MIN,
    WARNING_DURATION_SEC,
    GameConfig,
)


class Phase(str, Enum):
    MENU = "MENU"
    CREEPING = "CREEPING"
    WARNING = "WARNING"
    FREEZE = "FREEZE"
    CLEAR = "CLEAR"
    BUSTED = "BUSTED"
    VICTORY = "VICTORY"


@dataclass
class GameState:
    config: GameConfig = field(default_factory=GameConfig)
    phase: Phase = Phase.MENU

    # Heist progression
    score: int = 0
    distance_meters: float = 0.0
    goal_meters: float = HEIST_GOAL_METERS
    streak: int = 0
    max_streak: int = 0
    elapsed_sec: float = 0.0
    motion_peak: float = 0.0
    current_motion: float = 0.0

    # Timers & animations
    phase_timer: float = 0.0
    next_sweep_timer: float = 3.5
    walk_cycle: float = 0.0
    laser_sweep_offset: float = 0.0
    flash_timer: float = 0.0
    step_sound_timer: float = 0.0

    # Busted chase animation (normalized 0..1+ screen progress)
    cat_chase_offset: float = 0.0
    dog_chase_x: float = -0.25

    # Telemetry & persistence
    telemetry: list[TelemetryEventIn] = field(default_factory=list)
    run_submitted: bool = False
    show_leaderboard: bool = False

    # Optional callbacks wired by main loop
    on_sound: Optional[Callable[[str], None]] = None
    on_reset_cv: Optional[Callable[[], None]] = None
    on_submit_run: Optional[Callable[[RunCreate], None]] = None

    def _emit_sound(self, name: str) -> None:
        if self.on_sound is not None:
            self.on_sound(name)

    def start_heist(self) -> None:
        """Reset run metrics and start creeping toward the golden fish bowl."""
        self.phase = Phase.CREEPING
        self.score = 0
        self.distance_meters = 0.0
        self.streak = 0
        self.max_streak = 0
        self.elapsed_sec = 0.0
        self.motion_peak = 0.0
        self.current_motion = 0.0
        self.phase_timer = 0.0
        self.next_sweep_timer = random.uniform(SWEEP_INTERVAL_MIN, SWEEP_INTERVAL_MAX)
        self.walk_cycle = 0.0
        self.laser_sweep_offset = 0.0
        self.flash_timer = 0.0
        self.cat_chase_offset = 0.0
        self.dog_chase_x = -0.22
        self.telemetry = []
        self.run_submitted = False
        self.show_leaderboard = False

    def return_to_menu(self) -> None:
        self.phase = Phase.MENU
        self.flash_timer = 0.0

    def update(self, dt: float, motion_score: float, is_simulation: bool) -> None:
        """Advance the state machine by `dt` seconds."""
        dt = min(dt, 0.1)  # Clamp huge frame spikes
        self.current_motion = motion_score
        if motion_score > self.motion_peak:
            self.motion_peak = motion_score

        if self.flash_timer > 0.0:
            self.flash_timer = max(0.0, self.flash_timer - dt)

        if self.phase == Phase.MENU:
            self.walk_cycle += dt * 3.0
            return

        if self.phase in (Phase.BUSTED, Phase.VICTORY):
            if self.phase == Phase.BUSTED:
                self.walk_cycle += dt * 18.0
                self.cat_chase_offset += dt * 0.65
                self.dog_chase_x += dt * 0.80
            else:
                self.walk_cycle += dt * 5.0
            return

        self.elapsed_sec += dt
        self.laser_sweep_offset = (self.laser_sweep_offset + dt * 1.4) % 1.0

        if self.phase == Phase.CREEPING:
            self._advance_cat(dt, CAT_WALK_SPEED_MPS)
            self.next_sweep_timer -= dt
            if self.distance_meters >= self.goal_meters:
                self._trigger_victory(is_simulation)
            elif self.next_sweep_timer <= 0.0:
                self._enter_warning()

        elif self.phase == Phase.WARNING:
            self.phase_timer -= dt
            if self.phase_timer <= 0.0:
                self._enter_freeze()

        elif self.phase == Phase.FREEZE:
            self.phase_timer -= dt
            # Tiny 0.15s grace window at freeze start so residual frame-lag doesn't false-trigger
            grace_elapsed = (self.config.freeze_duration - self.phase_timer) >= 0.15
            if grace_elapsed and motion_score > self.config.motion_threshold:
                self._trigger_busted(motion_score, is_simulation)
            elif self.phase_timer <= 0.0:
                self._enter_clear()

        elif self.phase == Phase.CLEAR:
            self._advance_cat(dt, FREEZE_CLEAR_BONUS_MPS)
            self.phase_timer -= dt
            if self.distance_meters >= self.goal_meters:
                self._trigger_victory(is_simulation)
            elif self.phase_timer <= 0.0:
                self.phase = Phase.CREEPING
                self.next_sweep_timer = random.uniform(SWEEP_INTERVAL_MIN, SWEEP_INTERVAL_MAX)

    def _advance_cat(self, dt: float, speed_mps: float) -> None:
        delta_m = speed_mps * dt
        self.distance_meters = min(self.goal_meters, self.distance_meters + delta_m)
        self.score += int(round(delta_m * 25.0))
        self.walk_cycle += dt * (speed_mps * 4.5)
        self.step_sound_timer -= dt
        if self.step_sound_timer <= 0.0:
            self._emit_sound("step")
            self.step_sound_timer = 0.28

    def _enter_warning(self) -> None:
        self.phase = Phase.WARNING
        self.phase_timer = WARNING_DURATION_SEC
        self._emit_sound("klaxon")
        self.telemetry.append(
            TelemetryEventIn(
                event_type="LASER_SWEEP",
                motion_delta=self.current_motion,
                heist_distance=self.distance_meters,
                elapsed_sec=self.elapsed_sec,
            )
        )

    def _enter_freeze(self) -> None:
        self.phase = Phase.FREEZE
        self.phase_timer = self.config.freeze_duration
        if self.on_reset_cv is not None:
            self.on_reset_cv()

    def _enter_clear(self) -> None:
        self.phase = Phase.CLEAR
        self.phase_timer = CLEAR_DURATION_SEC
        self.streak += 1
        if self.streak > self.max_streak:
            self.max_streak = self.streak
        self.score += 250 + (self.streak * 50)
        self._emit_sound("clear")
        self.telemetry.append(
            TelemetryEventIn(
                event_type="FREEZE_PASS",
                motion_delta=self.current_motion,
                heist_distance=self.distance_meters,
                elapsed_sec=self.elapsed_sec,
            )
        )

    def _trigger_busted(self, motion_score: float, is_simulation: bool) -> None:
        self.phase = Phase.BUSTED
        self.flash_timer = 1.2
        self.cat_chase_offset = 0.0
        self.dog_chase_x = max(-0.20, self.progress_ratio - 0.35)
        self._emit_sound("alarm")
        self.telemetry.append(
            TelemetryEventIn(
                event_type="MOTION_DETECTED",
                motion_delta=motion_score,
                heist_distance=self.distance_meters,
                elapsed_sec=self.elapsed_sec,
            )
        )
        self._submit_run("BUSTED", is_simulation)

    def _trigger_victory(self, is_simulation: bool) -> None:
        self.phase = Phase.VICTORY
        self.distance_meters = self.goal_meters
        self.score += 1000
        self._emit_sound("victory")
        self.telemetry.append(
            TelemetryEventIn(
                event_type="HEIST_COMPLETE",
                motion_delta=self.current_motion,
                heist_distance=self.distance_meters,
                elapsed_sec=self.elapsed_sec,
            )
        )
        self._submit_run("VICTORY", is_simulation)

    def _submit_run(self, outcome: str, is_simulation: bool) -> None:
        if self.run_submitted:
            return
        self.run_submitted = True
        if self.on_submit_run is not None:
            payload = RunCreate(
                callsign=self.config.callsign,
                score=self.score,
                distance_meters=self.distance_meters,
                max_streak=self.max_streak,
                motion_peak=self.motion_peak,
                cv_mode="simulation" if is_simulation else "webcam",
                outcome=outcome,  # type: ignore[arg-type]
                duration_sec=self.elapsed_sec,
                telemetry=list(self.telemetry),
            )
            self.on_submit_run(payload)

    @property
    def progress_ratio(self) -> float:
        if self.goal_meters <= 0:
            return 0.0
        return max(0.0, min(1.0, self.distance_meters / self.goal_meters))

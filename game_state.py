"""Core game state machine, 3-lane POV obstacle physics, and cat bounce/jump/roll state."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Optional

from backend.models import RunCreate, TelemetryEventIn
from config import (
    CAT_BOUNCE_FREQ,
    CAT_JUMP_DURATION_SEC,
    CAT_ROLL_DURATION_SEC,
    CAT_ROLL_SPEED_BOOST_MPS,
    CAT_WALK_SPEED_MPS,
    CLEAR_DURATION_SEC,
    FREEZE_CLEAR_BONUS_MPS,
    HEIST_GOAL_METERS,
    OBSTACLE_SPAWN_MAX_SEC,
    OBSTACLE_SPAWN_MIN_SEC,
    SWEEP_INTERVAL_MAX,
    SWEEP_INTERVAL_MIN,
    WARNING_DURATION_SEC,
    GameConfig,
)

ObstacleKind = Literal["HIGH_GATE", "LOW_BEAM", "SIDE_BLOCK"]


class Phase(str, Enum):
    MENU = "MENU"
    CREEPING = "CREEPING"
    WARNING = "WARNING"
    FREEZE = "FREEZE"
    CLEAR = "CLEAR"
    BUSTED = "BUSTED"
    VICTORY = "VICTORY"


@dataclass
class CorridorObstacle:
    """Obstacle in the 3D Cat POV corridor and 2D PiP side-view."""

    kind: ObstacleKind              # "HIGH_GATE" (crouch-roll under), "LOW_BEAM" (jump over), "SIDE_BLOCK" (move left/right)
    distance_m: float               # Absolute distance along the heist track
    lane: int = 0                   # -1 = LEFT, 0 = CENTER, +1 = RIGHT (used by SIDE_BLOCK; 0 = all lanes for gates/beams)
    cleared: bool = False


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

    # Cat POV & 2D Bouncing / Jump / Crouch-Roll Physics
    target_lane: int = 0            # -1 (LEFT), 0 (CENTER), +1 (RIGHT)
    cat_lane_pos: float = 0.0       # Smoothly interpolated -1.0 .. +1.0
    cat_bounce_height: float = 0.0  # Continuous rhythmic bounce 0.0 .. 1.0
    cat_jump_timer: float = 0.0     # Remaining jump airtime
    cat_jump_height: float = 0.0    # Parabolic jump height 0.0 .. 1.0
    cat_roll_timer: float = 0.0     # Remaining crouch-roll time
    cat_roll_angle: float = 0.0     # 0 .. 360 deg barrel roll spin
    is_crouching: bool = False      # Whether human is currently crouching
    action_banner: str = ""         # e.g. "ROLLED UNDER GATE! +150"
    action_banner_timer: float = 0.0
    busted_reason: str = "MOTION DETECTED ON LASER SWEEP!"

    # Corridor obstacles
    obstacles: list[CorridorObstacle] = field(default_factory=list)
    next_obstacle_dist: float = 9.0

    # Timers & animations
    phase_timer: float = 0.0
    next_sweep_timer: float = 6.5
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
        """Reset run metrics and start bouncing down the corridor in Cat POV."""
        self.phase = Phase.CREEPING
        self.score = 0
        self.distance_meters = 0.0
        self.streak = 0
        self.max_streak = 0
        self.elapsed_sec = 0.0
        self.motion_peak = 0.0
        self.current_motion = 0.0

        self.target_lane = 0
        self.cat_lane_pos = 0.0
        self.cat_bounce_height = 0.0
        self.cat_jump_timer = 0.0
        self.cat_jump_height = 0.0
        self.cat_roll_timer = 0.0
        self.cat_roll_angle = 0.0
        self.is_crouching = False
        self.action_banner = "JUMP [UP] // CROUCH-ROLL [DOWN] // MOVE [LEFT/RIGHT]"
        self.action_banner_timer = 2.8
        self.busted_reason = "MOTION DETECTED ON LASER SWEEP!"

        self.obstacles = []
        self.next_obstacle_dist = 8.5
        self._spawn_obstacles_ahead()

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

    def trigger_jump(self) -> None:
        """Trigger a high cat jump when the human jumps (or presses UP/W)."""
        if self.phase not in (Phase.CREEPING, Phase.WARNING, Phase.CLEAR):
            return
        if self.cat_jump_timer <= 0.05:
            self.cat_jump_timer = CAT_JUMP_DURATION_SEC
            self.cat_roll_timer = 0.0
            self.is_crouching = False
            self._emit_sound("jump")

    def trigger_crouch_roll(self) -> None:
        """Trigger a fast cat tuck-and-roll under obstacles when the human crouches (or presses DOWN/S)."""
        if self.phase not in (Phase.CREEPING, Phase.WARNING, Phase.CLEAR):
            return
        self.is_crouching = True
        if self.cat_roll_timer <= 0.10:
            self.cat_roll_timer = CAT_ROLL_DURATION_SEC
            self.cat_jump_timer = 0.0
            self.cat_jump_height = 0.0
            self._emit_sound("roll")

    def set_target_lane(self, lane: int) -> None:
        """Set target corridor lane (-1 = LEFT, 0 = CENTER, +1 = RIGHT)."""
        self.target_lane = max(-1, min(1, lane))

    def shift_lane(self, delta: int) -> None:
        """Step one lane left (-1) or right (+1)."""
        self.target_lane = max(-1, min(1, self.target_lane + delta))

    def _spawn_obstacles_ahead(self) -> None:
        """Keep a stream of HIGH_GATE (roll under), LOW_BEAM (jump), and SIDE_BLOCK obstacles ahead."""
        while self.next_obstacle_dist < self.goal_meters - 4.0 and (
            self.next_obstacle_dist - self.distance_meters
        ) < 42.0:
            # Emphasize HIGH_GATE so the player gets plenty of crouch-roll action, mixed with jumps & side dodges
            r = random.random()
            if r < 0.44:
                kind: ObstacleKind = "HIGH_GATE"
                lane = 0
            elif r < 0.76:
                kind = "LOW_BEAM"
                lane = 0
            else:
                kind = "SIDE_BLOCK"
                lane = random.choice([-1, 0, 1])

            self.obstacles.append(
                CorridorObstacle(
                    kind=kind,
                    distance_m=round(self.next_obstacle_dist, 2),
                    lane=lane,
                )
            )
            self.next_obstacle_dist += random.uniform(
                OBSTACLE_SPAWN_MIN_SEC * CAT_WALK_SPEED_MPS,
                OBSTACLE_SPAWN_MAX_SEC * CAT_WALK_SPEED_MPS,
            )

    def update(
        self,
        dt: float,
        motion_score: float,
        is_simulation: bool,
        human_lane: Optional[int] = None,
        human_posture: Optional[str] = None,
    ) -> None:
        """Advance the state machine, Cat POV physics, and obstacle collisions by `dt` seconds."""
        dt = min(dt, 0.1)
        self.current_motion = motion_score
        if motion_score > self.motion_peak:
            self.motion_peak = motion_score

        if self.flash_timer > 0.0:
            self.flash_timer = max(0.0, self.flash_timer - dt)
        if self.action_banner_timer > 0.0:
            self.action_banner_timer = max(0.0, self.action_banner_timer - dt)

        # Apply human pose input when provided
        if human_lane is not None and self.phase in (Phase.CREEPING, Phase.WARNING, Phase.CLEAR):
            self.set_target_lane(human_lane)
        if human_posture is not None and self.phase in (Phase.CREEPING, Phase.WARNING, Phase.CLEAR):
            if human_posture == "JUMP":
                self.trigger_jump()
            elif human_posture == "CROUCH":
                self.trigger_crouch_roll()
            else:
                self.is_crouching = False

        if self.phase == Phase.MENU:
            self.walk_cycle += dt * 4.5
            self.cat_bounce_height = abs(math.sin(self.walk_cycle))
            return

        if self.phase in (Phase.BUSTED, Phase.VICTORY):
            if self.phase == Phase.BUSTED:
                self.walk_cycle += dt * 18.0
                self.cat_bounce_height = abs(math.sin(self.walk_cycle))
                self.cat_chase_offset += dt * 0.65
                self.dog_chase_x += dt * 0.80
            else:
                self.walk_cycle += dt * 6.0
                self.cat_bounce_height = abs(math.sin(self.walk_cycle))
            return

        self.elapsed_sec += dt
        self.laser_sweep_offset = (self.laser_sweep_offset + dt * 1.4) % 1.0

        # Smoothly interpolate horizontal lane position (-1.0 .. +1.0)
        lane_diff = float(self.target_lane) - self.cat_lane_pos
        self.cat_lane_pos += lane_diff * min(1.0, dt * 11.0)

        # Update Jump & Crouch-Roll timers
        if self.cat_jump_timer > 0.0:
            self.cat_jump_timer = max(0.0, self.cat_jump_timer - dt)
            prog = 1.0 - (self.cat_jump_timer / CAT_JUMP_DURATION_SEC)
            self.cat_jump_height = math.sin(prog * math.pi)
        else:
            self.cat_jump_height = 0.0

        if self.cat_roll_timer > 0.0:
            self.cat_roll_timer = max(0.0, self.cat_roll_timer - dt)
            prog = 1.0 - (self.cat_roll_timer / CAT_ROLL_DURATION_SEC)
            self.cat_roll_angle = (prog * 360.0) % 360.0
        else:
            self.cat_roll_angle = 0.0

        if self.phase == Phase.CREEPING:
            speed = CAT_ROLL_SPEED_BOOST_MPS if self.is_rolling else CAT_WALK_SPEED_MPS
            self._advance_cat(dt, speed)
            self._spawn_obstacles_ahead()
            if human_lane is not None or human_posture is not None:
                self._check_obstacles(is_simulation)
            if self.phase == Phase.BUSTED:
                return

            self.next_sweep_timer -= dt
            if self.distance_meters >= self.goal_meters:
                self._trigger_victory(is_simulation)
            elif self.next_sweep_timer <= 0.0:
                self._enter_warning()

        elif self.phase == Phase.WARNING:
            # Allow finishing a jump/roll during warning telegraph
            self.phase_timer -= dt
            if self.phase_timer <= 0.0:
                self._enter_freeze()

        elif self.phase == Phase.FREEZE:
            self.cat_bounce_height = 0.0
            self.phase_timer -= dt
            grace_elapsed = (self.config.freeze_duration - self.phase_timer) >= 0.15
            if grace_elapsed and motion_score > self.config.motion_threshold:
                self.busted_reason = "MOVED DURING RED LASER FREEZE!"
                self._trigger_busted(motion_score, is_simulation)
            elif self.phase_timer <= 0.0:
                self._enter_clear()

        elif self.phase == Phase.CLEAR:
            self._advance_cat(dt, FREEZE_CLEAR_BONUS_MPS)
            self._spawn_obstacles_ahead()
            if human_lane is not None or human_posture is not None:
                self._check_obstacles(is_simulation)
            if self.phase == Phase.BUSTED:
                return

            self.phase_timer -= dt
            if self.distance_meters >= self.goal_meters:
                self._trigger_victory(is_simulation)
            elif self.phase_timer <= 0.0:
                self.phase = Phase.CREEPING
                self.next_sweep_timer = random.uniform(SWEEP_INTERVAL_MIN, SWEEP_INTERVAL_MAX)

    def _check_obstacles(self, is_simulation: bool) -> None:
        """Evaluate obstacle collisions or crouch-roll/jump/lane clears."""
        for obs in self.obstacles:
            if obs.cleared:
                continue
            rel = obs.distance_m - self.distance_meters
            # When the cat passes through the obstacle plane
            if -0.45 <= rel <= 0.55:
                if obs.kind == "HIGH_GATE":
                    if self.is_rolling or self.is_crouching:
                        obs.cleared = True
                        self.score += 150
                        self.streak += 1
                        self.max_streak = max(self.max_streak, self.streak)
                        self.action_banner = "CAT ROLLED UNDER LASER GATE! +150"
                        self.action_banner_timer = 1.4
                        self._emit_sound("dodge")
                    else:
                        obs.cleared = True
                        self.busted_reason = "HIT HIGH LASER! CROUCH TO ROLL UNDER!"
                        self._trigger_busted(self.current_motion, is_simulation)
                        return

                elif obs.kind == "LOW_BEAM":
                    if self.cat_jump_height > 0.25 or self.cat_jump_timer > 0.05:
                        obs.cleared = True
                        self.score += 150
                        self.streak += 1
                        self.max_streak = max(self.max_streak, self.streak)
                        self.action_banner = "JUMPED OVER LASER TRIPWIRE! +150"
                        self.action_banner_timer = 1.4
                        self._emit_sound("dodge")
                    else:
                        obs.cleared = True
                        self.busted_reason = "TRIPPED LOW LASER! JUMP OVER IT!"
                        self._trigger_busted(self.current_motion, is_simulation)
                        return

                elif obs.kind == "SIDE_BLOCK":
                    if (
                        self.target_lane != obs.lane
                        or abs(self.cat_lane_pos - float(obs.lane)) >= 0.40
                        or self.cat_jump_height > 0.45
                    ):
                        obs.cleared = True
                        self.score += 120
                        self.streak += 1
                        self.max_streak = max(self.max_streak, self.streak)
                        self.action_banner = "DODGED SECURITY PYLON! +120"
                        self.action_banner_timer = 1.2
                        self._emit_sound("dodge")
                    else:
                        obs.cleared = True
                        self.busted_reason = "CRASHED INTO LANE PYLON! MOVE LEFT/RIGHT!"
                        self._trigger_busted(self.current_motion, is_simulation)
                        return
            elif rel < -0.55:
                obs.cleared = True

    def _advance_cat(self, dt: float, speed_mps: float) -> None:
        delta_m = speed_mps * dt
        self.distance_meters = min(self.goal_meters, self.distance_meters + delta_m)
        self.score += int(round(delta_m * 25.0))
        self.walk_cycle += dt * CAT_BOUNCE_FREQ * (speed_mps / CAT_WALK_SPEED_MPS)
        self.cat_bounce_height = abs(math.sin(self.walk_cycle))
        self.step_sound_timer -= dt
        if self.step_sound_timer <= 0.0:
            self._emit_sound("step")
            self.step_sound_timer = 0.26

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
    def is_rolling(self) -> bool:
        return self.cat_roll_timer > 0.0 or self.is_crouching

    @property
    def progress_ratio(self) -> float:
        if self.goal_meters <= 0:
            return 0.0
        return max(0.0, min(1.0, self.distance_meters / self.goal_meters))

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Optional

from backend.models import RunCreate, TelemetryEventIn
from config import (
    BOUNCE_ESCALATION_INTERVAL_SEC,
    BOUNCE_TIER_STEP,
    CAT_BOUNCE_FREQ,
    CAT_JUMP_DURATION_SEC,
    CAT_ROLL_DURATION_SEC,
    CAT_WALK_SPEED_MPS,
    CLEAR_DURATION_SEC,
    FREEZE_CLEAR_BONUS_MPS,
    HEIST_GOAL_METERS,
    MAX_BOUNCE_MULTIPLIER,
    MAX_BOUNCE_TIER,
    MAX_CAT_LIVES,
    MAX_LEVEL,
    WARNING_DURATION_SEC,
    GameConfig,
    LevelSpec,
    get_level_spec,
)

ObstacleKind = Literal["HIGH_GATE", "LOW_BEAM", "SIDE_BLOCK"]
CoinTier = Literal["FLOOR", "AIR", "LOW_ROLL"]
MAX_TELEMETRY_EVENTS = 128


class Phase(str, Enum):
    MENU = "MENU"
    CREEPING = "CREEPING"
    WARNING = "WARNING"
    FREEZE = "FREEZE"
    CLEAR = "CLEAR"
    BUSTED = "BUSTED"
    VICTORY = "VICTORY"


@dataclass(slots=True)
class CorridorObstacle:
    kind: ObstacleKind
    distance_m: float
    lane: int = 0
    cleared: bool = False


@dataclass(slots=True)
class CorridorCoin:
    distance_m: float
    lane: int = 0
    tier: CoinTier = "FLOOR"
    collected: bool = False


@dataclass(slots=True)
class ScorePopup:
    text: str
    timer: float = 1.0
    lane: int = 0
    is_gold: bool = True


@dataclass(slots=True)
class BloodParticle:
    rel_x: float
    rel_y: float
    vx: float
    vy: float
    radius: float
    timer: float = 0.85


@dataclass(slots=True)
class BloodStain:
    norm_x: float
    norm_y: float
    radius: float
    alpha: int = 185


@dataclass(slots=True)
class GameState:
    config: GameConfig = field(default_factory=GameConfig)
    phase: Phase = Phase.MENU

    level: int = 1
    level_name: str = "Anubis Foyer"
    score: int = 0
    coins: int = 0
    distance_meters: float = 0.0
    goal_meters: float = HEIST_GOAL_METERS
    total_distance_meters: float = 0.0
    streak: int = 0
    max_streak: int = 0
    elapsed_sec: float = 0.0
    motion_peak: float = 0.0
    current_motion: float = 0.0

    # 9 Lives & Progressive Weakness System
    max_lives: int = MAX_CAT_LIVES
    lives_remaining: int = MAX_CAT_LIVES
    deaths_count: int = 0
    cat_strength: float = 1.0
    blood_flash_timer: float = 0.0
    crash_invuln_timer: float = 0.0
    blood_particles: list[BloodParticle] = field(default_factory=list)
    blood_stains: list[BloodStain] = field(default_factory=list)

    # 10-Second Escalating Bounce System (saturates at MAX_BOUNCE_TIER)
    bounce_tier: int = 0
    bounce_multiplier: float = 1.0
    bounce_saturated: bool = False

    # Hand Gesture Menu Start Hold
    gesture_start_hold: float = 0.0

    target_lane: int = 0
    cat_lane_pos: float = 0.0
    cat_bounce_height: float = 0.0
    cat_jump_timer: float = 0.0
    cat_jump_height: float = 0.0
    jump_grace_timer: float = 0.0
    cat_roll_timer: float = 0.0
    cat_roll_angle: float = 0.0
    cat_crouch_depth: float = 0.0
    crouch_grace_timer: float = 0.0
    is_crouching: bool = False
    last_human_posture: str = "STAND"
    action_banner: str = ""
    action_banner_timer: float = 0.0
    busted_reason: str = "MOTION DETECTED ON LASER SWEEP"

    obstacles: list[CorridorObstacle] = field(default_factory=list)
    coins_list: list[CorridorCoin] = field(default_factory=list)
    popups: list[ScorePopup] = field(default_factory=list)
    next_obstacle_dist: float = 9.0
    coin_spin_angle: float = 0.0

    phase_timer: float = 0.0
    next_sweep_timer: float = 6.5
    walk_cycle: float = 0.0
    laser_sweep_offset: float = 0.0
    flash_timer: float = 0.0
    step_sound_timer: float = 0.0

    cat_chase_offset: float = 0.0
    dog_chase_x: float = -0.25

    telemetry: list[TelemetryEventIn] = field(default_factory=list)
    run_submitted: bool = False
    show_leaderboard: bool = False

    on_sound: Optional[Callable[[str], None]] = None
    on_reset_cv: Optional[Callable[[], None]] = None
    on_submit_run: Optional[Callable[[RunCreate], None]] = None

    @property
    def level_spec(self) -> LevelSpec:
        return get_level_spec(self.level)

    @property
    def combo_multiplier(self) -> int:
        return min(5, 1 + (self.streak // 2))

    @property
    def effective_bounce_multiplier(self) -> float:
        weakness_damp = 0.64 + 0.36 * self.cat_strength
        return round(self.bounce_multiplier * weakness_damp, 3)

    def _emit_sound(self, name: str) -> None:
        if self.on_sound is not None:
            self.on_sound(name)

    def _record_telemetry(self, event_type: str, motion_val: float) -> None:
        if len(self.telemetry) >= MAX_TELEMETRY_EVENTS:
            self.telemetry.pop(0)
        self.telemetry.append(
            TelemetryEventIn(
                event_type=event_type,  # type: ignore[arg-type]
                motion_delta=motion_val,
                heist_distance=self.total_distance_meters or self.distance_meters,
                elapsed_sec=self.elapsed_sec,
            )
        )

    def _add_popup(self, text: str, lane: int = 0, is_gold: bool = True) -> None:
        if len(self.popups) >= 6:
            self.popups.pop(0)
        self.popups.append(ScorePopup(text=text, timer=0.95, lane=lane, is_gold=is_gold))

    def _spawn_blood_splatter(self) -> None:
        self.blood_flash_timer = 1.05
        for _ in range(18):
            angle = random.uniform(-math.pi * 0.92, -math.pi * 0.08)
            spd = random.uniform(75.0, 240.0)
            self.blood_particles.append(
                BloodParticle(
                    rel_x=random.uniform(-22.0, 22.0),
                    rel_y=random.uniform(-28.0, 12.0),
                    vx=math.cos(angle) * spd,
                    vy=math.sin(angle) * spd,
                    radius=random.uniform(3.5, 8.5),
                    timer=random.uniform(0.55, 1.05),
                )
            )
        if len(self.blood_particles) > 54:
            self.blood_particles = self.blood_particles[-54:]

        for _ in range(3):
            self.blood_stains.append(
                BloodStain(
                    norm_x=max(0.12, min(0.88, 0.50 + self.cat_lane_pos * 0.16 + random.uniform(-0.14, 0.14))),
                    norm_y=random.uniform(0.68, 0.93),
                    radius=random.uniform(8.0, 20.0),
                    alpha=random.randint(145, 210),
                )
            )
        if len(self.blood_stains) > 27:
            self.blood_stains = self.blood_stains[-27:]

    def select_starting_level(self, level_num: int) -> None:
        self.config.starting_level = max(1, min(MAX_LEVEL, level_num))
        if self.phase == Phase.MENU:
            self._apply_level_spec(self.config.starting_level)

    def _apply_level_spec(self, level_num: int) -> None:
        self.level = max(1, min(MAX_LEVEL, level_num))
        spec = get_level_spec(self.level)
        self.level_name = spec.name
        self.goal_meters = spec.goal_meters
        self.config.freeze_duration = spec.freeze_duration
        self.distance_meters = 0.0
        self.obstacles.clear()
        self.coins_list.clear()
        self.next_obstacle_dist = 8.0
        self._spawn_obstacles_ahead()
        self.next_sweep_timer = random.uniform(spec.sweep_min_sec, spec.sweep_max_sec)

    def start_heist(self, start_level: Optional[int] = None) -> None:
        lvl = start_level if start_level is not None else self.config.starting_level
        self.phase = Phase.CREEPING
        self.score = 0
        self.coins = 0
        self.total_distance_meters = 0.0
        self.streak = 0
        self.max_streak = 0
        self.elapsed_sec = 0.0
        self.motion_peak = 0.0
        self.current_motion = 0.0

        self.max_lives = MAX_CAT_LIVES
        self.lives_remaining = MAX_CAT_LIVES
        self.deaths_count = 0
        self.cat_strength = 1.0
        self.blood_flash_timer = 0.0
        self.crash_invuln_timer = 0.0
        self.blood_particles.clear()
        self.blood_stains.clear()

        self.bounce_tier = 0
        self.bounce_multiplier = 1.0
        self.bounce_saturated = False
        self.gesture_start_hold = 0.0

        self.target_lane = 0
        self.cat_lane_pos = 0.0
        self.cat_bounce_height = 0.0
        self.cat_jump_timer = 0.0
        self.cat_jump_height = 0.0
        self.jump_grace_timer = 0.0
        self.cat_roll_timer = 0.0
        self.cat_roll_angle = 0.0
        self.cat_crouch_depth = 0.0
        self.crouch_grace_timer = 0.0
        self.is_crouching = False
        self.last_human_posture = "STAND"
        self.popups.clear()
        self.busted_reason = "MOTION DETECTED ON LASER SWEEP"

        self._apply_level_spec(lvl)
        self.action_banner = f"LEVEL {self.level:02d}/{MAX_LEVEL:02d} // {self.level_name} (9 LIVES ACTIVE)"
        self.action_banner_timer = 2.6

        self.phase_timer = 0.0
        self.walk_cycle = 0.0
        self.laser_sweep_offset = 0.0
        self.flash_timer = 0.0
        self.cat_chase_offset = 0.0
        self.dog_chase_x = -0.22
        self.telemetry.clear()
        self.run_submitted = False
        self.show_leaderboard = False

    def return_to_menu(self) -> None:
        self.phase = Phase.MENU
        self.flash_timer = 0.0
        self.gesture_start_hold = 0.0
        self._apply_level_spec(self.config.starting_level)

    def trigger_jump(self, force: bool = False) -> None:
        if self.phase not in (Phase.CREEPING, Phase.WARNING, Phase.CLEAR):
            return
        was_crouching = self.is_crouching or self.cat_roll_timer > 0.0
        if force or was_crouching or self.cat_jump_timer <= 0.14:
            self.cat_jump_timer = CAT_JUMP_DURATION_SEC
            boost = 1.0 + (self.effective_bounce_multiplier - 1.0) * 0.45
            self.cat_jump_height = max(0.38 * boost, self.cat_jump_height)
            self.jump_grace_timer = 0.32
            self.cat_roll_timer = 0.0
            self.cat_crouch_depth = 0.0
            self.crouch_grace_timer = 0.0
            self.is_crouching = False
            self._emit_sound("jump")
        else:
            self.jump_grace_timer = max(self.jump_grace_timer, 0.30)

    def trigger_crouch_roll(self, force: bool = False) -> None:
        if self.phase not in (Phase.CREEPING, Phase.WARNING, Phase.CLEAR):
            return
        was_jumping = self.cat_jump_timer > 0.0 or self.cat_jump_height > 0.0
        self.is_crouching = True
        self.cat_crouch_depth = 1.0
        self.crouch_grace_timer = 0.32
        self.cat_jump_timer = 0.0
        self.cat_jump_height = 0.0
        self.jump_grace_timer = 0.0
        if force or was_jumping or self.cat_roll_timer <= 0.14:
            self.cat_roll_timer = CAT_ROLL_DURATION_SEC
            self._emit_sound("roll")
        else:
            self.cat_roll_timer = max(self.cat_roll_timer, 0.28)

    def set_target_lane(self, lane: int) -> None:
        self.target_lane = max(-1, min(1, lane))

    def shift_lane(self, delta: int) -> None:
        self.target_lane = max(-1, min(1, self.target_lane + delta))

    def _spawn_obstacles_ahead(self) -> None:
        cutoff = self.distance_meters - 2.0
        if len(self.obstacles) > 24:
            self.obstacles = [o for o in self.obstacles if not o.cleared and o.distance_m >= cutoff]
        if len(self.coins_list) > 36:
            self.coins_list = [c for c in self.coins_list if not c.collected and c.distance_m >= cutoff]

        spec = self.level_spec
        while self.next_obstacle_dist < self.goal_meters - 4.0 and (
            self.next_obstacle_dist - self.distance_meters
        ) < 44.0:
            r = random.random()
            if r < 0.42:
                kind: ObstacleKind = "HIGH_GATE"
                lane = 0
                coin_lane = random.choice((-1, 0, 1))
                self.coins_list.append(
                    CorridorCoin(distance_m=round(self.next_obstacle_dist, 2), lane=coin_lane, tier="LOW_ROLL")
                )
            elif r < 0.74:
                kind = "LOW_BEAM"
                lane = 0
                coin_lane = random.choice((-1, 0, 1))
                self.coins_list.append(
                    CorridorCoin(distance_m=round(self.next_obstacle_dist, 2), lane=coin_lane, tier="AIR")
                )
            else:
                kind = "SIDE_BLOCK"
                lane = random.choice((-1, 0, 1))
                open_lanes = tuple(l for l in (-1, 0, 1) if l != lane)
                self.coins_list.append(
                    CorridorCoin(
                        distance_m=round(self.next_obstacle_dist, 2),
                        lane=random.choice(open_lanes),
                        tier="FLOOR",
                    )
                )

            self.obstacles.append(
                CorridorObstacle(
                    kind=kind,
                    distance_m=round(self.next_obstacle_dist, 2),
                    lane=lane,
                )
            )

            gap = random.uniform(spec.obstacle_gap_min, spec.obstacle_gap_max)
            trail_lane = random.choice((-1, 0, 1))
            mid_start = self.next_obstacle_dist + gap * 0.34
            for step in range(2):
                cdist = round(mid_start + step * 1.6, 2)
                if cdist < self.goal_meters - 2.5:
                    self.coins_list.append(CorridorCoin(distance_m=cdist, lane=trail_lane, tier="FLOOR"))

            self.next_obstacle_dist += gap

    def _update_bounce_escalation(self) -> None:
        new_tier = min(MAX_BOUNCE_TIER, int(self.elapsed_sec // BOUNCE_ESCALATION_INTERVAL_SEC))
        if new_tier > self.bounce_tier:
            self.bounce_tier = new_tier
            self.bounce_multiplier = min(
                MAX_BOUNCE_MULTIPLIER,
                round(1.0 + self.bounce_tier * BOUNCE_TIER_STEP, 2),
            )
            self.bounce_saturated = self.bounce_tier >= MAX_BOUNCE_TIER
            if self.bounce_saturated:
                self.action_banner = f"MAX BOUNCE SATURATED ({self.bounce_multiplier:.2f}x HEIGHT)!"
                self._add_popup(f"MAX BOUNCE {self.bounce_multiplier:.1f}x!", lane=self.target_lane, is_gold=True)
            else:
                self.action_banner = f"10S SURVIVED! BOUNCE TIER {self.bounce_tier}/{MAX_BOUNCE_TIER} ({self.bounce_multiplier:.2f}x)"
                self._add_popup(f"BOUNCE UP {self.bounce_multiplier:.1f}x!", lane=self.target_lane, is_gold=True)
            self.action_banner_timer = 1.8

    def _update_blood_effects(self, dt: float) -> None:
        if self.blood_flash_timer > 0.0:
            self.blood_flash_timer = max(0.0, self.blood_flash_timer - dt)
        if self.crash_invuln_timer > 0.0:
            self.crash_invuln_timer = max(0.0, self.crash_invuln_timer - dt)
        if self.blood_particles:
            for bp in self.blood_particles:
                bp.timer -= dt
                bp.rel_x += bp.vx * dt
                bp.rel_y += bp.vy * dt
                bp.vy += 420.0 * dt
            self.blood_particles = [bp for bp in self.blood_particles if bp.timer > 0.0]

    def update(
        self,
        dt: float,
        motion_score: float,
        is_simulation: bool,
        human_lane: Optional[int] = None,
        human_posture: Optional[str] = None,
    ) -> None:
        dt = min(dt, 0.1)
        self.current_motion = motion_score
        if motion_score > self.motion_peak:
            self.motion_peak = motion_score

        if self.flash_timer > 0.0:
            self.flash_timer = max(0.0, self.flash_timer - dt)
        if self.action_banner_timer > 0.0:
            self.action_banner_timer = max(0.0, self.action_banner_timer - dt)
        self._update_blood_effects(dt)

        self.coin_spin_angle = (self.coin_spin_angle + dt * 260.0) % 360.0
        if self.popups:
            for p in self.popups:
                p.timer -= dt
            self.popups = [p for p in self.popups if p.timer > 0.0]

        # Hand-gesture start/retry when on MENU, BUSTED, or VICTORY
        if self.phase in (Phase.MENU, Phase.BUSTED, Phase.VICTORY):
            if human_posture == "JUMP":
                self.gesture_start_hold += dt
                if self.gesture_start_hold >= 0.55:
                    self.gesture_start_hold = 0.0
                    self.start_heist()
                    self.last_human_posture = "JUMP"
                    return
            else:
                self.gesture_start_hold = max(0.0, self.gesture_start_hold - dt * 1.5)

        if human_lane is not None and self.phase in (Phase.CREEPING, Phase.WARNING, Phase.CLEAR):
            self.set_target_lane(human_lane)
        if human_posture is not None and self.phase in (Phase.CREEPING, Phase.WARNING, Phase.CLEAR):
            posture_changed = human_posture != self.last_human_posture
            self.last_human_posture = human_posture
            if human_posture == "JUMP":
                if posture_changed or self.cat_jump_timer <= 0.0:
                    self.trigger_jump(force=True)
                else:
                    boost = 1.0 + (self.effective_bounce_multiplier - 1.0) * 0.45
                    self.cat_jump_timer = max(self.cat_jump_timer, 0.35)
                    self.cat_jump_height = max(0.68 * boost, self.cat_jump_height)
                    self.jump_grace_timer = 0.32
                    self.is_crouching = False
                    self.cat_roll_timer = 0.0
            elif human_posture == "CROUCH":
                if posture_changed or self.cat_roll_timer <= 0.0:
                    self.trigger_crouch_roll(force=True)
                else:
                    self.is_crouching = True
                    self.cat_crouch_depth = 1.0
                    self.cat_roll_timer = max(self.cat_roll_timer, 0.35)
                    self.crouch_grace_timer = 0.32
                    self.cat_jump_timer = 0.0
                    self.cat_jump_height = 0.0
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
        self._update_bounce_escalation()
        self.laser_sweep_offset = (self.laser_sweep_offset + dt * 1.4) % 1.0

        lane_diff = float(self.target_lane) - self.cat_lane_pos
        self.cat_lane_pos += lane_diff * min(1.0, dt * 12.5)

        if self.cat_jump_timer > 0.0:
            self.cat_jump_timer = max(0.0, self.cat_jump_timer - dt)
            prog = 1.0 - (self.cat_jump_timer / CAT_JUMP_DURATION_SEC)
            arc = math.sin((prog ** 0.82) * math.pi)
            boost = 1.0 + (self.effective_bounce_multiplier - 1.0) * 0.45
            self.cat_jump_height = max(0.22 if self.cat_jump_timer > 0.04 else 0.0, arc * boost)
            self.jump_grace_timer = 0.30
        else:
            self.cat_jump_height = 0.0
            if self.jump_grace_timer > 0.0:
                self.jump_grace_timer = max(0.0, self.jump_grace_timer - dt)

        if self.cat_roll_timer > 0.0:
            self.cat_roll_timer = max(0.0, self.cat_roll_timer - dt)
            prog = 1.0 - (self.cat_roll_timer / CAT_ROLL_DURATION_SEC)
            self.cat_roll_angle = (prog * 360.0) % 360.0
            self.crouch_grace_timer = 0.30
            self.cat_crouch_depth = min(1.0, self.cat_crouch_depth + dt * 14.0)
            if self.cat_roll_timer <= 0.0:
                self.is_crouching = False
        else:
            self.is_crouching = False
            self.cat_roll_angle = 0.0
            self.cat_crouch_depth = max(0.0, self.cat_crouch_depth - dt * 10.0)
            if self.crouch_grace_timer > 0.0:
                self.crouch_grace_timer = max(0.0, self.crouch_grace_timer - dt)

        spec = self.level_spec

        if self.phase == Phase.CREEPING:
            base_speed = spec.roll_speed_mps if self.is_rolling else spec.walk_speed_mps
            speed = base_speed * (0.58 + 0.42 * self.cat_strength)
            self._advance_cat(dt, speed)
            self._spawn_obstacles_ahead()
            self._check_coins()
            if human_lane is not None or human_posture is not None:
                self._check_obstacles(is_simulation)
            if self.phase == Phase.BUSTED:
                return

            self.next_sweep_timer -= dt
            if self.distance_meters >= self.goal_meters:
                self._complete_sector_or_heist(is_simulation)
            elif self.next_sweep_timer <= 0.0:
                self._enter_warning()

        elif self.phase == Phase.WARNING:
            self.phase_timer -= dt
            if self.phase_timer <= 0.0:
                self._enter_freeze()

        elif self.phase == Phase.FREEZE:
            self.cat_bounce_height = 0.0
            self.phase_timer -= dt
            grace_elapsed = (self.config.freeze_duration - self.phase_timer) >= 0.15
            if grace_elapsed and motion_score > self.config.motion_threshold:
                self.busted_reason = f"MOVED DURING LEVEL {self.level} LASER FREEZE"
                self._register_crash_or_death(
                    self.busted_reason,
                    motion_score,
                    is_simulation,
                    fatal_if_no_gesture=(human_lane is None and human_posture is None),
                )
            elif self.phase_timer <= 0.0:
                self._enter_clear()

        elif self.phase == Phase.CLEAR:
            base_clear_spd = max(FREEZE_CLEAR_BONUS_MPS, spec.walk_speed_mps + 2.2)
            self._advance_cat(dt, base_clear_spd * (0.58 + 0.42 * self.cat_strength))
            self._spawn_obstacles_ahead()
            self._check_coins()
            if human_lane is not None or human_posture is not None:
                self._check_obstacles(is_simulation)
            if self.phase == Phase.BUSTED:
                return

            self.phase_timer -= dt
            if self.distance_meters >= self.goal_meters:
                self._complete_sector_or_heist(is_simulation)
            elif self.phase_timer <= 0.0:
                self.phase = Phase.CREEPING
                self.next_sweep_timer = random.uniform(spec.sweep_min_sec, spec.sweep_max_sec)

    def _check_coins(self) -> None:
        dist = self.distance_meters
        for coin in self.coins_list:
            if coin.collected:
                continue
            rel = coin.distance_m - dist
            if rel > 0.70:
                continue
            if rel < -0.60:
                coin.collected = True
                continue

            lane_ok = (
                abs(self.cat_lane_pos - float(coin.lane)) <= 0.68
                or self.target_lane == coin.lane
            )
            if not lane_ok:
                continue
            if coin.tier == "AIR" and not (
                self.cat_jump_height > 0.10 or self.cat_jump_timer > 0.0 or self.jump_grace_timer > 0.0
            ):
                continue
            if coin.tier == "LOW_ROLL" and not (
                self.is_rolling or self.is_crouching or self.crouch_grace_timer > 0.0
            ):
                continue

            coin.collected = True
            self.coins += 1
            pts = 50 * self.combo_multiplier
            self.score += pts
            self._add_popup(f"+{pts} COIN", lane=coin.lane, is_gold=True)
            self._emit_sound("coin")

    def _complete_sector_or_heist(self, is_simulation: bool) -> None:
        if self.level < MAX_LEVEL:
            bonus = (400 + self.level * 100) * self.combo_multiplier
            self.score += bonus
            next_lvl = self.level + 1
            self._apply_level_spec(next_lvl)
            self.phase = Phase.CREEPING
            self.action_banner = f"SECTOR CLEAR! +{bonus} -> LEVEL {self.level:02d}: {self.level_name}"
            self.action_banner_timer = 3.0
            self._add_popup(f"LVL {self.level:02d} +{bonus}", lane=0, is_gold=True)
            self._emit_sound("level_up")
        else:
            self._trigger_victory(is_simulation)

    def _check_obstacles(self, is_simulation: bool) -> None:
        dist = self.distance_meters
        for obs in self.obstacles:
            if obs.cleared:
                continue
            rel = obs.distance_m - dist
            if rel > 0.60:
                continue
            if rel < -0.50:
                obs.cleared = True
                continue

            if obs.kind == "HIGH_GATE":
                if self.is_rolling or self.is_crouching or self.crouch_grace_timer > 0.0:
                    obs.cleared = True
                    self.streak += 1
                    self.max_streak = max(self.max_streak, self.streak)
                    pts = 150 * self.combo_multiplier
                    self.score += pts
                    self.action_banner = f"ROLLED UNDER LASER GATE +{pts} (x{self.combo_multiplier})"
                    self.action_banner_timer = 1.4
                    self._add_popup(f"ROLL +{pts}", lane=self.target_lane, is_gold=False)
                    self._emit_sound("dodge")
                elif rel <= 0.12:
                    obs.cleared = True
                    self.busted_reason = "HIT HIGH LASER GATE (FIST / CROUCH TO ROLL)"
                    self._register_crash_or_death(self.busted_reason, self.current_motion, is_simulation)
                    return

            elif obs.kind == "LOW_BEAM":
                if (
                    self.cat_jump_height > 0.12
                    or self.cat_jump_timer > 0.0
                    or self.jump_grace_timer > 0.0
                ):
                    obs.cleared = True
                    self.streak += 1
                    self.max_streak = max(self.max_streak, self.streak)
                    pts = 150 * self.combo_multiplier
                    self.score += pts
                    self.action_banner = f"JUMPED OVER FLOOR LASER +{pts} (x{self.combo_multiplier})"
                    self.action_banner_timer = 1.4
                    self._add_popup(f"JUMP +{pts}", lane=self.target_lane, is_gold=False)
                    self._emit_sound("dodge")
                elif rel <= 0.12:
                    obs.cleared = True
                    self.busted_reason = "TRIPPED SARCOPHAGUS LASER (OPEN PALM / JUMP)"
                    self._register_crash_or_death(self.busted_reason, self.current_motion, is_simulation)
                    return

            elif obs.kind == "SIDE_BLOCK":
                if (
                    self.target_lane != obs.lane
                    or abs(self.cat_lane_pos - float(obs.lane)) >= 0.38
                    or self.cat_jump_height > 0.45
                ):
                    obs.cleared = True
                    self.streak += 1
                    self.max_streak = max(self.max_streak, self.streak)
                    pts = 120 * self.combo_multiplier
                    self.score += pts
                    self.action_banner = f"DODGED SECURITY PYLON +{pts} (x{self.combo_multiplier})"
                    self.action_banner_timer = 1.2
                    self._add_popup(f"DODGE +{pts}", lane=self.target_lane, is_gold=False)
                    self._emit_sound("dodge")
                elif rel <= 0.12:
                    obs.cleared = True
                    self.busted_reason = "HIT PHARAOH PYLON (MOVE HAND LEFT OR RIGHT)"
                    self._register_crash_or_death(self.busted_reason, self.current_motion, is_simulation)
                    return

    def _advance_cat(self, dt: float, speed_mps: float) -> None:
        delta_m = speed_mps * dt
        self.distance_meters = min(self.goal_meters, self.distance_meters + delta_m)
        self.total_distance_meters += delta_m
        self.score += int(round(delta_m * (22.0 + self.level * 3.0) * self.combo_multiplier))
        self.walk_cycle += dt * CAT_BOUNCE_FREQ * (speed_mps / CAT_WALK_SPEED_MPS)
        self.cat_bounce_height = abs(math.sin(self.walk_cycle)) * self.effective_bounce_multiplier
        self.step_sound_timer -= dt
        if self.step_sound_timer <= 0.0:
            self._emit_sound("step")
            self.step_sound_timer = 0.26

    def _enter_warning(self) -> None:
        self.phase = Phase.WARNING
        self.phase_timer = WARNING_DURATION_SEC
        self._emit_sound("klaxon")
        self._record_telemetry("LASER_SWEEP", self.current_motion)

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
        pts = (250 + (self.streak * 50)) * self.combo_multiplier
        self.score += pts
        self._add_popup(f"FREEZE CLEAR +{pts}", lane=self.target_lane, is_gold=True)
        self._emit_sound("clear")
        self._record_telemetry("FREEZE_PASS", self.current_motion)

    def _register_crash_or_death(
        self,
        reason: str,
        motion_score: float,
        is_simulation: bool,
        fatal_if_no_gesture: bool = False,
    ) -> None:
        """
        Handles a cat crash or laser zap:
        - Spawns crimson blood splatter particles & floor/screen blood stains
        - Consumes 1 of the cat's 9 lives and weakens the cat (up to 9 times)
        - When all 9 lives are exhausted (or fatal_if_no_gesture is set), triggers final BUSTED
        """
        self._spawn_blood_splatter()
        self.deaths_count = min(self.max_lives, self.deaths_count + 1)
        self.lives_remaining = max(0, self.max_lives - self.deaths_count)
        self.cat_strength = max(0.20, round(1.0 - self.deaths_count * 0.088, 3))
        self.streak = 0
        self.crash_invuln_timer = 0.85

        if self.lives_remaining <= 0 or fatal_if_no_gesture:
            self._trigger_busted(motion_score, is_simulation)
        else:
            self.flash_timer = 0.65
            self._emit_sound("alarm")
            self._record_telemetry("MOTION_DETECTED", motion_score)
            strength_pct = int(round(self.cat_strength * 100))
            self.action_banner = (
                f"CRASH! -1 LIFE ({self.lives_remaining}/{self.max_lives} LEFT) // "
                f"CAT WEAKENED TO {strength_pct}%"
            )
            self.action_banner_timer = 2.2
            self._add_popup(
                f"BLOOD! {self.lives_remaining}/{self.max_lives} LIVES ({strength_pct}%)",
                lane=self.target_lane,
                is_gold=False,
            )
            if self.phase == Phase.FREEZE:
                self.phase = Phase.CREEPING
                spec = self.level_spec
                self.next_sweep_timer = random.uniform(spec.sweep_min_sec, spec.sweep_max_sec)

    def _trigger_busted(self, motion_score: float, is_simulation: bool) -> None:
        if self.blood_flash_timer <= 0.0:
            self._spawn_blood_splatter()
        self.phase = Phase.BUSTED
        self.flash_timer = 1.2
        self.cat_chase_offset = 0.0
        self.dog_chase_x = max(-0.20, self.progress_ratio - 0.35)
        self._emit_sound("alarm")
        self._record_telemetry("MOTION_DETECTED", motion_score)
        self._submit_run("BUSTED", is_simulation)

    def _trigger_victory(self, is_simulation: bool) -> None:
        self.phase = Phase.VICTORY
        self.distance_meters = self.goal_meters
        self.score += 1500 + self.coins * 100
        self._emit_sound("victory")
        self._record_telemetry("HEIST_COMPLETE", self.current_motion)
        self._submit_run("VICTORY", is_simulation)

    def _submit_run(self, outcome: str, is_simulation: bool) -> None:
        if self.run_submitted:
            return
        self.run_submitted = True
        if self.on_submit_run is not None:
            reported_dist = max(self.distance_meters, self.total_distance_meters)
            payload = RunCreate(
                callsign=self.config.callsign,
                score=self.score,
                distance_meters=reported_dist,
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

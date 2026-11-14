"""Neobrutalist & Cyberpunk Pygame renderer for LASER_PAW_PATROL (Cat 3D POV + 2D Mini-Window Edition)."""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np
import pygame

from backend.models import RunRecord, StatsSummary
from config import (
    BORDER_WIDTH,
    COLOR_AMBER,
    COLOR_BG,
    COLOR_CYAN,
    COLOR_GOLD,
    COLOR_GRAY_DARK,
    COLOR_GRAY_MID,
    COLOR_GREEN,
    COLOR_GREEN_DIM,
    COLOR_RED,
    COLOR_RED_DIM,
    COLOR_SHADOW,
    COLOR_WHITE,
    SHADOW_OFFSET,
)
from cv_tracker import MotionSample
from game_state import CorridorObstacle, GameState, Phase


class UIRenderer:
    """Renders the First-Person Cat POV 3D Corridor, 2D Side-View Mini-Window, and CCTV Pose HUD."""

    def __init__(self) -> None:
        self.font_xs = pygame.font.SysFont("consolas", 13, bold=True)
        self.font_sm = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_md = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 34, bold=True)
        self.font_xl = pygame.font.SysFont("consolas", 50, bold=True)

    def _draw_brutalist_box(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        fill: tuple[int, int, int] = COLOR_BG,
        border: tuple[int, int, int] = COLOR_WHITE,
        shadow: tuple[int, int, int] = COLOR_SHADOW,
        border_w: int = BORDER_WIDTH,
        shadow_off: int = SHADOW_OFFSET,
    ) -> None:
        if shadow_off > 0:
            shadow_rect = rect.move(shadow_off, shadow_off)
            pygame.draw.rect(surface, shadow, shadow_rect)
        pygame.draw.rect(surface, fill, rect)
        pygame.draw.rect(surface, border, rect, border_w)

    def render(
        self,
        surface: pygame.Surface,
        state: GameState,
        cv_sample: MotionSample,
        leaderboard: Sequence[RunRecord],
        stats: StatsSummary,
        api_online: bool,
    ) -> None:
        w, h = surface.get_size()
        surface.fill(COLOR_BG)

        # 1. Main Full-Screen First-Person Cat POV 3D Viewport
        self._draw_cat_pov_viewport(surface, w, h, state)
        self._draw_pov_lasers(surface, w, h, state)
        self._draw_cat_pov_foreground(surface, w, h, state)

        # Red alarm flash overlay when busted
        if state.phase == Phase.BUSTED and state.flash_timer > 0.0:
            pulse = abs(math.sin(time.perf_counter() * 14.0))
            alpha = int(115 * pulse * min(1.0, state.flash_timer))
            if alpha > 0:
                overlay = pygame.Surface((w, h), pygame.SRCALPHA)
                overlay.fill((*COLOR_RED, alpha))
                surface.blit(overlay, (0, 0))

        # 2. Top Brutalist HUD
        self._draw_top_hud(surface, w, state, cv_sample, api_online)

        # 3. Right Side: Live CCTV Human Pose Feed + 2D Environment Mini-Window
        self._draw_cctv_feed(surface, w, state, cv_sample)
        self._draw_2d_pip_window(surface, w, h, state)

        # 4. Dynamic Phase & Action Banners
        self._draw_phase_banner(surface, w, h, state, cv_sample)

        if state.phase == Phase.MENU:
            self._draw_menu_overlay(surface, w, h, state, cv_sample)
        elif state.phase == Phase.BUSTED:
            self._draw_end_overlay(surface, w, h, state, busted=True)
        elif state.phase == Phase.VICTORY:
            self._draw_end_overlay(surface, w, h, state, busted=False)

        if state.show_leaderboard:
            self._draw_leaderboard_modal(surface, w, h, leaderboard, stats)

    # =========================================================================
    # 1. FIRST-PERSON CAT POV 3D CORRIDOR & OBSTACLES
    # =========================================================================

    def _get_pov_camera_offsets(self, w: int, h: int, state: GameState) -> tuple[int, int, float]:
        """Compute vanishing point (vx, vy) and camera height offset from Cat Bounce/Jump/Crouch-Roll."""
        # Center-left of screen so right-side PiP panels don't block the corridor center
        base_vx = int((w - 310) * 0.50)
        base_vy = int(h * 0.43)

        # Horizontal shift when cat moves left/right across lanes
        cam_shift_x = int(-state.cat_lane_pos * (w * 0.09))

        # Vertical camera bounce + jump arc + crouch-roll drop
        bounce_px = int(state.cat_bounce_height * 16.0) if state.phase != Phase.FREEZE else 0
        jump_px = int(state.cat_jump_height * 135.0)
        crouch_px = -52 if state.is_rolling else 0

        # When jumping, camera rises so horizon moves down slightly; when crouching, camera drops close to floor
        cam_height_px = bounce_px + jump_px + crouch_px
        vx = base_vx + cam_shift_x
        vy = base_vy + int(cam_height_px * 0.55)
        return vx, vy, float(cam_height_px)

    def _project_corridor_point(
        self,
        lane_x: float,
        height_y: float,
        depth_m: float,
        vx: int,
        vy: int,
        corridor_w: int,
        corridor_h: int,
        cam_height_px: float,
        cat_lane_pos: float,
    ) -> tuple[int, int, float]:
        """Project a 3D corridor coordinate (lane_x in -1.5..1.5, height_y in 0..1, depth_m > 0) to 2D screen."""
        safe_z = max(0.35, depth_m)
        scale = 2.6 / (safe_z + 2.2)
        rel_x = lane_x - cat_lane_pos * 0.85
        sx = int(vx + rel_x * (corridor_w * 0.34) * scale)
        # height_y = 0.0 is floor, 1.0 is ceiling
        floor_offset = (0.52 - height_y) * corridor_h * 0.72 + cam_height_px * 0.85
        sy = int(vy + floor_offset * scale)
        return sx, sy, scale

    def _draw_cat_pov_viewport(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        corridor_w = max(560, w - 300)
        vx, vy, cam_h = self._get_pov_camera_offsets(w, h, state)

        # Ceiling / Floor backdrop
        pygame.draw.rect(surface, (8, 10, 14), (0, 0, corridor_w, vy))
        pygame.draw.rect(surface, (14, 16, 22), (0, vy, corridor_w, h - vy))

        grid_col = (34, 46, 40)
        if state.phase in (Phase.FREEZE, Phase.BUSTED):
            grid_col = (70, 20, 26)
        elif state.phase == Phase.CLEAR:
            grid_col = (20, 85, 48)
        elif state.is_rolling:
            grid_col = (18, 68, 88)

        # Perspective lane lines on floor & ceiling (-1.5, -0.5, +0.5, +1.5)
        for lane_edge in (-1.5, -0.5, 0.5, 1.5):
            fx0, fy0, _ = self._project_corridor_point(
                lane_edge, 0.0, 0.4, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            fx1, fy1, _ = self._project_corridor_point(
                lane_edge, 0.0, 36.0, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            line_w = 3 if abs(lane_edge) == 1.5 else 1
            col = COLOR_WHITE if abs(lane_edge) == 1.5 else grid_col
            pygame.draw.line(surface, col, (fx0, fy0), (fx1, fy1), line_w)

            # Ceiling edge rails
            if abs(lane_edge) == 1.5:
                cx0, cy0, _ = self._project_corridor_point(
                    lane_edge, 1.0, 0.4, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
                )
                cx1, cy1, _ = self._project_corridor_point(
                    lane_edge, 1.0, 36.0, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
                )
                pygame.draw.line(surface, grid_col, (cx0, cy0), (cx1, cy1), 2)

        # Scrolling depth rings along corridor walls and floor
        scroll_offset = state.distance_meters % 3.0
        for i in range(12, 0, -1):
            z = i * 2.6 - scroll_offset
            if z <= 0.35:
                continue
            lx0, ly0, _ = self._project_corridor_point(
                -1.5, 0.0, z, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            rx0, ry0, _ = self._project_corridor_point(
                1.5, 0.0, z, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            lx1, ly1, _ = self._project_corridor_point(
                -1.5, 1.0, z, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            rx1, ry1, _ = self._project_corridor_point(
                1.5, 1.0, z, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            pygame.draw.line(surface, grid_col, (lx0, ly0), (rx0, ry0), 2)
            pygame.draw.line(surface, grid_col, (lx0, ly0), (lx1, ly1), 1)
            pygame.draw.line(surface, grid_col, (rx0, ry0), (rx1, ry1), 1)
            pygame.draw.line(surface, grid_col, (lx1, ly1), (rx1, ry1), 1)

        # Draw distant Golden Fish Bowl in 3D perspective
        rem_dist = max(1.2, state.goal_meters - state.distance_meters)
        if rem_dist < 45.0:
            bx, by, bscale = self._project_corridor_point(
                0.0, 0.28, rem_dist, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            radius = max(10, int(68 * bscale))
            pygame.draw.circle(surface, COLOR_GOLD, (bx, by), radius)
            pygame.draw.circle(surface, COLOR_WHITE, (bx, by), radius, 2)

        # Draw 3D Obstacles from farthest to nearest
        visible_obs = [
            obs
            for obs in state.obstacles
            if not obs.cleared and -0.4 <= (obs.distance_m - state.distance_meters) <= 34.0
        ]
        visible_obs.sort(key=lambda o: o.distance_m, reverse=True)
        for obs in visible_obs:
            depth = obs.distance_m - state.distance_meters
            self._draw_pov_obstacle(surface, obs, depth, vx, vy, corridor_w, h, cam_h, state)

        # Bottom controls hint bar
        hint = (
            "[W/UP or HUMAN JUMP] JUMP   |   [S/DOWN or HUMAN CROUCH] ROLL UNDER   |   "
            "[A/D or STEP L/R] LANE   |   [C] CALIBRATE CAM   |   [M] SIM MODE"
        )
        hint_surf = self.font_xs.render(hint, True, (175, 175, 185))
        surface.blit(hint_surf, (18, h - 24))

    def _draw_pov_obstacle(
        self,
        surface: pygame.Surface,
        obs: CorridorObstacle,
        depth: float,
        vx: int,
        vy: int,
        corridor_w: int,
        h: int,
        cam_h: float,
        state: GameState,
    ) -> None:
        if obs.kind == "HIGH_GATE":
            # Overhead security gate from ceiling (1.0) down to crouch height (0.36)
            tl_x, tl_y, scale = self._project_corridor_point(
                -1.5, 1.0, depth, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            br_x, br_y, _ = self._project_corridor_point(
                1.5, 0.36, depth, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            gate_rect = pygame.Rect(tl_x, tl_y, max(12, br_x - tl_x), max(10, br_y - tl_y))
            pygame.draw.rect(surface, (45, 12, 22), gate_rect)
            pygame.draw.rect(surface, COLOR_CYAN if state.is_rolling else COLOR_RED, gate_rect, 3)

            # Glowing horizontal laser bars inside the high gate
            for frac in (0.3, 0.6, 0.9):
                ly = int(gate_rect.top + gate_rect.height * frac)
                pygame.draw.line(surface, COLOR_RED, (gate_rect.left, ly), (gate_rect.right, ly), 2)

            if depth < 14.0 and gate_rect.width > 130:
                lbl = self.font_xs.render("CROUCH TO ROLL UNDER v", True, COLOR_CYAN)
                surface.blit(lbl, (gate_rect.centerx - lbl.get_width() // 2, gate_rect.bottom + 4))

        elif obs.kind == "LOW_BEAM":
            # Floor laser barrier from height 0.0 up to 0.34
            tl_x, tl_y, _ = self._project_corridor_point(
                -1.5, 0.34, depth, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            br_x, br_y, _ = self._project_corridor_point(
                1.5, 0.0, depth, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            beam_rect = pygame.Rect(tl_x, tl_y, max(12, br_x - tl_x), max(8, br_y - tl_y))
            pygame.draw.rect(surface, COLOR_RED_DIM, beam_rect)
            pygame.draw.rect(surface, COLOR_AMBER, beam_rect, 3)
            if depth < 14.0 and beam_rect.width > 120:
                lbl = self.font_xs.render("JUMP OVER ^", True, COLOR_GOLD)
                surface.blit(lbl, (beam_rect.centerx - lbl.get_width() // 2, beam_rect.top - 18))

        elif obs.kind == "SIDE_BLOCK":
            # Pylon blocking a single lane (-1, 0, or +1)
            l_min = float(obs.lane) - 0.45
            l_max = float(obs.lane) + 0.45
            tl_x, tl_y, _ = self._project_corridor_point(
                l_min, 0.85, depth, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            br_x, br_y, _ = self._project_corridor_point(
                l_max, 0.0, depth, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            pylon_rect = pygame.Rect(tl_x, tl_y, max(10, br_x - tl_x), max(14, br_y - tl_y))
            pygame.draw.rect(surface, (55, 22, 26), pylon_rect)
            pygame.draw.rect(surface, COLOR_RED, pylon_rect, 3)
            pygame.draw.line(surface, COLOR_AMBER, pylon_rect.topleft, pylon_rect.bottomright, 2)
            pygame.draw.line(surface, COLOR_AMBER, pylon_rect.topright, pylon_rect.bottomleft, 2)

    def _draw_pov_lasers(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        if state.phase not in (Phase.WARNING, Phase.FREEZE, Phase.CLEAR, Phase.BUSTED):
            return
        corridor_w = max(560, w - 300)
        if state.phase == Phase.CLEAR:
            beam_color, glow_color = COLOR_GREEN, COLOR_GREEN_DIM
        elif state.phase == Phase.WARNING:
            beam_color, glow_color = COLOR_AMBER, (90, 60, 0)
        else:
            beam_color, glow_color = COLOR_RED, COLOR_RED_DIM

        num_beams = 5
        for i in range(num_beams):
            offset = ((i / num_beams) + state.laser_sweep_offset * 0.25) % 1.0
            x_top = int(corridor_w * (0.08 + offset * 0.84))
            x_bot = int(corridor_w * (0.04 + ((offset + 0.12) % 1.0) * 0.92))
            pygame.draw.line(surface, glow_color, (x_top, 84), (x_bot, h - 30), 8)
            pygame.draw.line(surface, beam_color, (x_top, 84), (x_bot, h - 30), 2)

    def _draw_cat_pov_foreground(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        """Render the First-Person Cat Ears, Whiskers, Night-Vision HUD, Bouncing Paws, and Roll Spin."""
        corridor_w = max(560, w - 300)
        cx = corridor_w // 2 + int(state.cat_lane_pos * 28)
        bounce_y = int(math.sin(state.walk_cycle) * 14) if state.phase != Phase.FREEZE else 0

        # 1. Top Cat Ears Silhouette & Cyber Visor Frame
        ear_bob = int(abs(math.sin(state.walk_cycle)) * 6)
        left_ear = [(cx - 190, 86), (cx - 145, 38 + ear_bob), (cx - 105, 86)]
        right_ear = [(cx + 105, 86), (cx + 145, 38 + ear_bob), (cx + 190, 86)]
        pygame.draw.polygon(surface, COLOR_BG, left_ear)
        pygame.draw.polygon(surface, COLOR_GREEN, left_ear, 2)
        pygame.draw.polygon(surface, COLOR_BG, right_ear)
        pygame.draw.polygon(surface, COLOR_GREEN, right_ear, 2)

        # 2. Side Cat Whiskers on POV Viewport Edges (tilt when rolling!)
        roll_rad = math.radians(state.cat_roll_angle) if state.is_rolling else 0.0
        whisker_col = COLOR_CYAN if state.is_rolling else COLOR_WHITE
        wy_base = h - 120 - int(state.cat_jump_height * 25)
        for idx, spread in enumerate((-18, 4, 26)):
            ang_off = int(math.sin(roll_rad + idx) * 24) if state.is_rolling else bounce_y // 2
            # Left whiskers
            pygame.draw.line(
                surface,
                whisker_col,
                (18, wy_base + spread),
                (125, wy_base + spread - 12 + ang_off),
                2,
            )
            # Right whiskers
            pygame.draw.line(
                surface,
                whisker_col,
                (corridor_w - 18, wy_base + spread),
                (corridor_w - 125, wy_base + spread - 12 - ang_off),
                2,
            )

        # 3. Crouch-Roll Barrel Spin Ring OR Bouncing First-Person Cat Paws
        if state.is_rolling:
            ring_cx, ring_cy = cx, int(h * 0.56)
            pygame.draw.circle(surface, COLOR_CYAN, (ring_cx, ring_cy), 135, 3)
            for k in range(4):
                a = roll_rad + k * (math.pi * 0.5)
                x1 = ring_cx + int(math.cos(a) * 95)
                y1 = ring_cy + int(math.sin(a) * 95)
                x2 = ring_cx + int(math.cos(a) * 145)
                y2 = ring_cy + int(math.sin(a) * 145)
                pygame.draw.line(surface, COLOR_CYAN, (x1, y1), (x2, y2), 4)
            roll_lbl = self.font_md.render("<< CAT CROUCH-ROLL ACTIVE >>", True, COLOR_CYAN)
            surface.blit(roll_lbl, (ring_cx - roll_lbl.get_width() // 2, h - 78))
        else:
            # Left & Right Bouncing Cat Paws at bottom of POV screen
            left_paw_y = h - 58 + bounce_y - int(state.cat_jump_height * 55)
            right_paw_y = h - 58 - bounce_y - int(state.cat_jump_height * 55)
            self._draw_pov_paw(surface, cx - 110, left_paw_y)
            self._draw_pov_paw(surface, cx + 110, right_paw_y)
            if state.cat_jump_height > 0.12:
                jmp_lbl = self.font_md.render("^^ CAT AIRBORNE // JUMP ACTIVE ^^", True, COLOR_GOLD)
                surface.blit(jmp_lbl, (cx - jmp_lbl.get_width() // 2, h - 108))

    def _draw_pov_paw(self, surface: pygame.Surface, px: int, py: int) -> None:
        paw_rect = pygame.Rect(px - 38, py - 18, 76, 64)
        pygame.draw.ellipse(surface, COLOR_WHITE, paw_rect)
        pygame.draw.ellipse(surface, COLOR_BG, paw_rect, 3)
        # Cute pink/cyber toe beans
        for tx, ty in [(-18, -4), (0, -10), (18, -4)]:
            pygame.draw.circle(surface, COLOR_RED, (px + tx, py + ty), 7)
        pygame.draw.ellipse(surface, COLOR_RED, (px - 14, py + 6, 28, 18))

    # =========================================================================
    # 2. SMALL 2D SIDE-VIEW ENVIRONMENT WINDOW (PiP)
    # =========================================================================

    def _draw_2d_pip_window(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        """Render the synchronized 2D side-scrolling environment inside a small window."""
        box_w = 276
        box_h = 225
        box_x = w - box_w - 18
        box_y = 264

        box_rect = pygame.Rect(box_x, box_y, box_w, box_h)
        self._draw_brutalist_box(
            surface,
            box_rect,
            fill=(10, 12, 18),
            border=COLOR_CYAN if state.is_rolling else COLOR_WHITE,
            shadow=COLOR_GREEN,
            shadow_off=5,
        )

        # Header bar
        hdr_rect = pygame.Rect(box_x, box_y, box_w, 24)
        pygame.draw.rect(surface, COLOR_GRAY_DARK, hdr_rect)
        pygame.draw.line(surface, COLOR_WHITE, (box_x, box_y + 24), (box_x + box_w, box_y + 24), 2)
        hdr_lbl = self.font_xs.render("2D ENV // SIDE-VIEW TRACKER", True, COLOR_GREEN)
        surface.blit(hdr_lbl, (box_x + 8, box_y + 5))

        # Inner 2D viewport subsurface
        inner_rect = pygame.Rect(box_x + 4, box_y + 26, box_w - 8, box_h - 30)
        sub = surface.subsurface(inner_rect)
        sw, sh = inner_rect.width, inner_rect.height
        floor_y = int(sh * 0.78)

        # 2D background grid & floor
        for gx in range(0, sw, 28):
            scrolling_x = (gx - int(state.distance_meters * 18)) % sw
            pygame.draw.line(sub, (25, 32, 38), (scrolling_x, 0), (scrolling_x, floor_y), 1)
        pygame.draw.rect(sub, COLOR_GRAY_DARK, (0, floor_y, sw, sh - floor_y))
        pygame.draw.line(sub, COLOR_WHITE, (0, floor_y), (sw, floor_y), 2)

        # Cat 2D horizontal anchor (left third of mini-window) + progress toward fish bowl
        cat_2d_x = 58
        pixels_per_m = 11.0

        # Draw 2D Obstacles scrolling toward the 2D cat
        for obs in state.obstacles:
            if obs.cleared:
                continue
            rel_m = obs.distance_m - state.distance_meters
            ox = cat_2d_x + int(rel_m * pixels_per_m)
            if -20 <= ox <= sw + 20:
                if obs.kind == "HIGH_GATE":
                    # Hanging laser gate from top down to floor_y - 24 (cat rolls underneath!)
                    g_rect = pygame.Rect(ox - 8, 4, 16, floor_y - 26)
                    pygame.draw.rect(sub, COLOR_RED_DIM, g_rect)
                    pygame.draw.rect(sub, COLOR_CYAN, g_rect, 2)
                elif obs.kind == "LOW_BEAM":
                    # Low floor obstacle (cat jumps over!)
                    b_rect = pygame.Rect(ox - 7, floor_y - 24, 14, 24)
                    pygame.draw.rect(sub, COLOR_RED, b_rect)
                    pygame.draw.rect(sub, COLOR_AMBER, b_rect, 2)
                elif obs.kind == "SIDE_BLOCK":
                    p_rect = pygame.Rect(ox - 6, floor_y - 36, 12, 36)
                    pygame.draw.rect(sub, COLOR_AMBER, p_rect, 2)

        # Golden fish bowl at end of 2D track
        goal_rel_m = state.goal_meters - state.distance_meters
        fish_x = min(sw - 22, cat_2d_x + int(goal_rel_m * pixels_per_m))
        pygame.draw.circle(sub, COLOR_GOLD, (fish_x, floor_y - 16), 11)
        pygame.draw.circle(sub, COLOR_WHITE, (fish_x, floor_y - 16), 11, 2)

        # Draw 2D Cat (bouncing, jumping, or spinning ball when crouch-rolling!)
        bounce_2d = int(state.cat_bounce_height * 9) if state.phase != Phase.FREEZE else 0
        jump_2d = int(state.cat_jump_height * 46)
        cat_2d_y = floor_y - 14 - bounce_2d - jump_2d

        if state.is_rolling:
            # Spinning ball rolling right under the high obstacle!
            ball_y = floor_y - 10
            pygame.draw.circle(sub, COLOR_CYAN, (cat_2d_x, ball_y), 11)
            pygame.draw.circle(sub, COLOR_WHITE, (cat_2d_x, ball_y), 11, 2)
            ang = math.radians(state.cat_roll_angle)
            sx = cat_2d_x + int(math.cos(ang) * 9)
            sy = ball_y + int(math.sin(ang) * 9)
            pygame.draw.line(sub, COLOR_BG, (cat_2d_x, ball_y), (sx, sy), 3)
        else:
            # Mini 2D Chubby Cat bouncing forward
            pygame.draw.ellipse(sub, COLOR_WHITE, (cat_2d_x - 14, cat_2d_y - 8, 26, 17))
            pygame.draw.circle(sub, COLOR_WHITE, (cat_2d_x + 11, cat_2d_y - 6), 8)
            pygame.draw.polygon(
                sub,
                COLOR_WHITE,
                [(cat_2d_x + 7, cat_2d_y - 12), (cat_2d_x + 10, cat_2d_y - 19), (cat_2d_x + 14, cat_2d_y - 12)],
            )
            pygame.draw.rect(sub, COLOR_GREEN, (cat_2d_x + 11, cat_2d_y - 8, 6, 4))

        # Guard dog in 2D window if busted
        if state.phase == Phase.BUSTED:
            pygame.draw.rect(sub, COLOR_RED, (cat_2d_x - 36, floor_y - 22, 24, 16))

        # Status footer inside 2D mini-window
        lane_txt = "LEFT" if state.target_lane < 0 else ("RIGHT" if state.target_lane > 0 else "CENTER")
        state_txt = "ROLLING!" if state.is_rolling else ("JUMPING!" if state.cat_jump_height > 0.1 else "BOUNCING")
        info = self.font_xs.render(f"LANE: {lane_txt} | {state_txt}", True, COLOR_WHITE)
        sub.blit(info, (8, sh - 18))

    # =========================================================================
    # 3. TOP HUD, CCTV HUMAN POSE BOX & MODALS
    # =========================================================================

    def _draw_top_hud(
        self,
        surface: pygame.Surface,
        w: int,
        state: GameState,
        cv_sample: MotionSample,
        api_online: bool,
    ) -> None:
        hud_w = max(460, w - 315)
        hud_rect = pygame.Rect(18, 14, hud_w, 66)
        self._draw_brutalist_box(surface, hud_rect, fill=COLOR_BG, border=COLOR_WHITE, shadow=COLOR_WHITE, shadow_off=5)

        score_lbl = self.font_xs.render("SCORE", True, (170, 170, 180))
        score_val = self.font_md.render(f"{state.score:06d}", True, COLOR_WHITE)
        surface.blit(score_lbl, (32, 22))
        surface.blit(score_val, (32, 40))

        dist_lbl = self.font_xs.render(
            f"CAT POV HEIST: {state.distance_meters:04.1f}M / {state.goal_meters:.0f}M",
            True,
            COLOR_GOLD,
        )
        surface.blit(dist_lbl, (158, 22))
        bar_rect = pygame.Rect(158, 43, max(110, hud_w - 340), 18)
        pygame.draw.rect(surface, COLOR_GRAY_DARK, bar_rect)
        fill_w = int(bar_rect.width * state.progress_ratio)
        if fill_w > 0:
            pygame.draw.rect(surface, COLOR_GREEN, (bar_rect.x, bar_rect.y, fill_w, bar_rect.height))
        pygame.draw.rect(surface, COLOR_WHITE, bar_rect, 2)

        right_x = hud_rect.right - 165
        pose_lbl = self.font_xs.render(
            f"HUMAN: {cv_sample.human_posture} ({'L' if cv_sample.human_lane < 0 else ('R' if cv_sample.human_lane > 0 else 'C')})",
            True,
            COLOR_CYAN if cv_sample.human_posture == "CROUCH" else (COLOR_GOLD if cv_sample.human_posture == "JUMP" else COLOR_GREEN),
        )
        api_text = f"STREAK x{state.streak} | {'API ON' if api_online else 'LOCAL'}"
        api_lbl = self.font_xs.render(api_text, True, COLOR_WHITE)
        surface.blit(pose_lbl, (right_x, 23))
        surface.blit(api_lbl, (right_x, 45))

    def _draw_cctv_feed(
        self,
        surface: pygame.Surface,
        w: int,
        state: GameState,
        cv_sample: MotionSample,
    ) -> None:
        box_w, box_h = 276, 184
        box_x = w - box_w - 18
        box_y = 14
        border_col = COLOR_RED if state.phase == Phase.FREEZE else COLOR_WHITE

        box_rect = pygame.Rect(box_x, box_y, box_w, box_h + 44)
        self._draw_brutalist_box(surface, box_rect, fill=COLOR_BG, border=border_col, shadow=border_col, shadow_off=5)

        try:
            frame_surf = pygame.surfarray.make_surface(np.transpose(cv_sample.frame_rgb, (1, 0, 2)))
            scaled = pygame.transform.scale(frame_surf, (box_w - 8, box_h - 8))
            surface.blit(scaled, (box_x + 4, box_y + 4))
        except Exception:
            pygame.draw.rect(surface, COLOR_GRAY_DARK, (box_x + 4, box_y + 4, box_w - 8, box_h - 8))

        blink = int(time.perf_counter() * 2.5) % 2 == 0
        if blink:
            pygame.draw.circle(surface, COLOR_RED, (box_x + 16, box_y + 16), 5)
        rec_txt = "SIM" if cv_sample.is_simulation else "REC"
        rec_lbl = self.font_xs.render(f"{rec_txt} // {cv_sample.status_label}", True, COLOR_WHITE)
        pygame.draw.rect(surface, COLOR_BG, (box_x + 26, box_y + 8, rec_lbl.get_width() + 8, 16))
        surface.blit(rec_lbl, (box_x + 30, box_y + 9))

        meter_y = box_y + box_h + 2
        meter_rect = pygame.Rect(box_x + 8, meter_y + 16, box_w - 16, 12)
        delta_lbl = self.font_xs.render(
            f"MOTION: {cv_sample.motion_score:04.1f} / THR {state.config.motion_threshold:.0f}",
            True,
            COLOR_RED if cv_sample.motion_score > state.config.motion_threshold else COLOR_GREEN,
        )
        surface.blit(delta_lbl, (box_x + 8, meter_y))
        pygame.draw.rect(surface, COLOR_GRAY_DARK, meter_rect)
        fill_ratio = min(1.0, cv_sample.motion_score / 40.0)
        fill_col = COLOR_RED if cv_sample.motion_score > state.config.motion_threshold else COLOR_GREEN
        pygame.draw.rect(
            surface,
            fill_col,
            (meter_rect.x, meter_rect.y, int(meter_rect.width * fill_ratio), meter_rect.height),
        )
        pygame.draw.rect(surface, COLOR_WHITE, meter_rect, 2)

    def _draw_phase_banner(
        self,
        surface: pygame.Surface,
        w: int,
        h: int,
        state: GameState,
        cv_sample: MotionSample,
    ) -> None:
        cx = (w - 300) // 2
        if state.phase == Phase.WARNING:
            rect = pygame.Rect(cx - 250, 96, 500, 56)
            self._draw_brutalist_box(surface, rect, fill=COLOR_BG, border=COLOR_AMBER, shadow=COLOR_AMBER)
            txt = self.font_md.render("!! LASER SWEEP INCOMING - FREEZE !!", True, COLOR_AMBER)
            surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        elif state.phase == Phase.FREEZE:
            rect = pygame.Rect(cx - 210, 94, 420, 82)
            self._draw_brutalist_box(surface, rect, fill=COLOR_BG, border=COLOR_RED, shadow=COLOR_RED)
            t1 = self.font_lg.render(f"FREEZE! {max(0.0, state.phase_timer):.1f}s", True, COLOR_RED)
            sub = "RELEASE [SPACE]!" if cv_sample.sim_motion_active else "STAY COMPLETELY STILL"
            t2 = self.font_sm.render(sub, True, COLOR_WHITE)
            surface.blit(t1, (rect.centerx - t1.get_width() // 2, rect.y + 10))
            surface.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + 50))

        elif state.phase == Phase.CLEAR:
            rect = pygame.Rect(cx - 210, 96, 420, 54)
            self._draw_brutalist_box(surface, rect, fill=COLOR_BG, border=COLOR_GREEN, shadow=COLOR_GREEN)
            txt = self.font_md.render("ZONE CLEAR // STEALTH BONUS!", True, COLOR_GREEN)
            surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        elif state.action_banner_timer > 0.0 and state.action_banner:
            rect = pygame.Rect(cx - 230, 94, 460, 44)
            self._draw_brutalist_box(surface, rect, fill=COLOR_BG, border=COLOR_CYAN, shadow=COLOR_WHITE, shadow_off=4)
            txt = self.font_sm.render(state.action_banner, True, COLOR_CYAN)
            surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

    def _draw_menu_overlay(
        self,
        surface: pygame.Surface,
        w: int,
        h: int,
        state: GameState,
        cv_sample: MotionSample,
    ) -> None:
        cx = (w - 300) // 2
        box = pygame.Rect(cx - 300, h // 2 - 165, 600, 315)
        self._draw_brutalist_box(surface, box, fill=COLOR_BG, border=COLOR_WHITE, shadow=COLOR_GREEN, shadow_off=8)

        title = self.font_xl.render("LASER_PAW_PATROL", True, COLOR_WHITE)
        sub = self.font_sm.render("CAT 3D POV + 2D MINI-WINDOW // HUMAN POSE EDITION", True, COLOR_GREEN)
        surface.blit(title, (box.centerx - title.get_width() // 2, box.y + 20))
        surface.blit(sub, (box.centerx - sub.get_width() // 2, box.y + 78))

        lines = [
            "1. MAIN SCREEN: BOUNCING CAT FIRST-PERSON POV CORRIDOR.",
            "2. SIDE WINDOW: LIVE 2D SIDE-VIEW ENVIRONMENT TRACKER.",
            "3. HUMAN JUMP (OR [W/UP]) -> CAT JUMPS OVER FLOOR LASERS.",
            "4. HUMAN CROUCH (OR [S/DOWN]) -> CAT ROLLS UNDER HIGH GATES!",
            "5. HUMAN MOVE LEFT/RIGHT (OR [A/D]) -> SHIFT CORRIDOR LANES.",
            "6. RED LASER SWEEP -> FREEZE COMPLETELY STILL FOR 2.6s!",
        ]
        for idx, line in enumerate(lines):
            col = COLOR_CYAN if idx == 3 else (COLOR_GOLD if idx == 2 else COLOR_WHITE)
            lbl = self.font_xs.render(line, True, col)
            surface.blit(lbl, (box.x + 26, box.y + 114 + idx * 23))

        cta = pygame.Rect(box.centerx - 155, box.bottom - 50, 310, 36)
        self._draw_brutalist_box(surface, cta, fill=COLOR_GREEN, border=COLOR_WHITE, shadow=COLOR_WHITE, shadow_off=4)
        cta_txt = self.font_md.render("PRESS [ENTER] TO START", True, COLOR_BG)
        surface.blit(cta_txt, (cta.centerx - cta_txt.get_width() // 2, cta.centery - cta_txt.get_height() // 2))

    def _draw_end_overlay(
        self, surface: pygame.Surface, w: int, h: int, state: GameState, busted: bool
    ) -> None:
        cx = (w - 300) // 2
        accent = COLOR_RED if busted else COLOR_GOLD
        box = pygame.Rect(cx - 265, h // 2 - 130, 530, 245)
        self._draw_brutalist_box(surface, box, fill=COLOR_BG, border=accent, shadow=accent, shadow_off=8)

        headline = "!! BUSTED !!" if busted else "HEIST COMPLETE // FISH SECURED!"
        h_lbl = self.font_lg.render(headline, True, accent)
        surface.blit(h_lbl, (box.centerx - h_lbl.get_width() // 2, box.y + 20))

        reason_line = state.busted_reason if busted else "GOLDEN FISH BOWL CAPTURED IN CAT POV!"
        stats_lines = [
            reason_line,
            f"FINAL SCORE: {state.score}   |   MAX STREAK: x{state.max_streak}",
            f"DISTANCE: {state.distance_meters:.1f}M / {state.goal_meters:.0f}M   |   SAVED TO DB",
        ]
        for idx, line in enumerate(stats_lines):
            col = COLOR_AMBER if idx == 0 else (COLOR_GREEN if idx == 2 else COLOR_WHITE)
            lbl = self.font_sm.render(line, True, col)
            surface.blit(lbl, (box.centerx - lbl.get_width() // 2, box.y + 78 + idx * 30))

        prompt = self.font_sm.render("[ENTER / R] PLAY AGAIN   |   [TAB] LEADERBOARD   |   [ESC] MENU", True, COLOR_WHITE)
        surface.blit(prompt, (box.centerx - prompt.get_width() // 2, box.bottom - 36))

    def _draw_leaderboard_modal(
        self,
        surface: pygame.Surface,
        w: int,
        h: int,
        leaderboard: Sequence[RunRecord],
        stats: StatsSummary,
    ) -> None:
        box = pygame.Rect(w // 2 - 340, h // 2 - 200, 680, 400)
        self._draw_brutalist_box(surface, box, fill=COLOR_BG, border=COLOR_WHITE, shadow=COLOR_GOLD, shadow_off=8)

        title = self.font_lg.render("TOP HEIST LEADERBOARD", True, COLOR_GOLD)
        surface.blit(title, (box.x + 24, box.y + 18))

        summary = self.font_xs.render(
            f"RUNS: {stats.total_runs}  |  WIN RATE: {stats.win_rate:.1f}%  |  BEST: {stats.best_score}",
            True,
            COLOR_GREEN,
        )
        surface.blit(summary, (box.x + 24, box.y + 60))

        header = self.font_xs.render("RK  CALLSIGN       SCORE    DIST     STREAK  MODE   OUTCOME", True, (160, 160, 175))
        surface.blit(header, (box.x + 24, box.y + 90))
        pygame.draw.line(surface, COLOR_WHITE, (box.x + 24, box.y + 110), (box.right - 24, box.y + 110), 2)

        if not leaderboard:
            empty = self.font_sm.render("NO HEIST RUNS RECORDED YET.", True, COLOR_WHITE)
            surface.blit(empty, (box.centerx - empty.get_width() // 2, box.y + 180))
        else:
            for idx, run in enumerate(leaderboard[:8]):
                ry = box.y + 122 + idx * 28
                col = COLOR_GREEN if run.outcome == "VICTORY" else COLOR_RED
                mode_tag = "SIM" if run.cv_mode == "simulation" else "CAM"
                row_txt = (
                    f"{idx + 1:02d}  {run.callsign:<13}  {run.score:06d}   "
                    f"{run.distance_meters:04.1f}M    x{run.max_streak:<5}  {mode_tag:<5}  {run.outcome}"
                )
                lbl = self.font_sm.render(row_txt, True, col)
                surface.blit(lbl, (box.x + 24, ry))

        close_lbl = self.font_xs.render("PRESS [TAB] OR [ESC] TO CLOSE", True, COLOR_WHITE)
        surface.blit(close_lbl, (box.centerx - close_lbl.get_width() // 2, box.bottom - 28))

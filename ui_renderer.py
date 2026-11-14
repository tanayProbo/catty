from __future__ import annotations

import math
import os
import time
from typing import Optional, Sequence

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
    MAX_LEVEL,
    SHADOW_OFFSET,
    get_level_spec,
)
from cv_tracker import MotionSample
from game_state import CorridorCoin, CorridorObstacle, GameState, Phase

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

COLOR_SAND_LIGHT = (218, 198, 148)
COLOR_SAND_MID = (194, 170, 118)
COLOR_SAND_DARK = (156, 132, 86)
COLOR_SAND_DEEP = (118, 96, 58)
COLOR_SKY_TOP = (108, 166, 252)
COLOR_SKY_HORIZON = (212, 230, 255)


class UIRenderer:
    def __init__(self) -> None:
        self.font_xs = pygame.font.SysFont("consolas", 13, bold=True)
        self.font_sm = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_md = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 34, bold=True)
        self.font_xl = pygame.font.SysFont("consolas", 48, bold=True)

        self._text_cache: dict[tuple[int, str, tuple[int, int, int]], pygame.Surface] = {}
        self._scale_cache: dict[tuple[str, int, int], pygame.Surface] = {}
        self._flash_overlay: Optional[pygame.Surface] = None
        self._flash_size: tuple[int, int] = (0, 0)

        self._assets: dict[str, pygame.Surface] = {}
        self._load_assets()

    def _load_assets(self) -> None:
        asset_files = (
            "ruby",
            "ankh",
            "golden_lamp",
            "burglar_cat",
            "chubby_cat_sketch",
            "pillar",
            "temple_archway",
            "pyramid",
            "mummy_sarcophagus",
            "pharaoh_mask",
            "tomb_shrine",
            "floor_tile_1",
            "floor_tile_2",
            "floor_tile_3",
            "sandstone_wall",
            "desert_env",
            "desert_sky",
            "desert_ground",
        )
        for name in asset_files:
            path = os.path.join(ASSETS_DIR, f"{name}.png")
            if os.path.exists(path):
                try:
                    surf = pygame.image.load(path)
                    if pygame.display.get_surface() is not None:
                        surf = surf.convert_alpha()
                    self._assets[name] = surf
                except Exception:
                    pass

    def _sprite(self, name: str, w: int, h: int) -> Optional[pygame.Surface]:
        base = self._assets.get(name)
        if base is None or w <= 2 or h <= 2:
            return None
        w_clamped = max(4, min(1400, w))
        h_clamped = max(4, min(900, h))
        key = (name, w_clamped, h_clamped)
        cached = self._scale_cache.get(key)
        if cached is not None:
            return cached
        if len(self._scale_cache) >= 320:
            self._scale_cache.clear()
        scaled = pygame.transform.smoothscale(base, (w_clamped, h_clamped))
        self._scale_cache[key] = scaled
        return scaled

    def _goal_sprite_name(self, level: int) -> str:
        if level == MAX_LEVEL or level % 3 == 0:
            return "golden_lamp"
        if level % 2 == 0:
            return "ankh"
        return "ruby"

    def _txt(
        self,
        font: pygame.font.Font,
        text: str,
        color: tuple[int, int, int],
    ) -> pygame.Surface:
        key = (id(font), text, color)
        surf = self._text_cache.get(key)
        if surf is not None:
            return surf
        if len(self._text_cache) >= 256:
            self._text_cache.clear()
        surf = font.render(text, True, color)
        self._text_cache[key] = surf
        return surf

    def _draw_panel(
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
            pygame.draw.rect(surface, shadow, rect.move(shadow_off, shadow_off))
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

        # Full-height 3D Egyptian Desert & Temple Corridor on the main viewport
        self._draw_cat_pov_viewport(surface, w, h, state)
        self._draw_pov_lasers(surface, w, h, state)
        self._draw_blood_stains_and_particles(surface, w, h, state)
        self._draw_cat_pov_foreground(surface, w, h, state)
        self._draw_floating_popups(surface, w, h, state)

        if (state.phase == Phase.BUSTED and state.flash_timer > 0.0) or state.blood_flash_timer > 0.0:
            pulse = abs(math.sin(time.perf_counter() * 14.0))
            timer_val = max(state.flash_timer if state.phase == Phase.BUSTED else 0.0, state.blood_flash_timer)
            alpha = int(105 * (0.45 + 0.55 * pulse) * min(1.0, timer_val))
            if alpha > 0:
                if self._flash_overlay is None or self._flash_size != (w, h):
                    self._flash_overlay = pygame.Surface((w, h), pygame.SRCALPHA)
                    self._flash_size = (w, h)
                self._flash_overlay.fill((210, 18, 28, alpha))
                surface.blit(self._flash_overlay, (0, 0))

        # Top HUD + Right Sidebar (CAM 01 + 2D Side View Window + Telemetry Card)
        self._draw_top_hud(surface, w, state, cv_sample, api_online)
        self._draw_cctv_feed(surface, w, state, cv_sample)
        self._draw_2d_pip_window(surface, w, h, state)
        self._draw_sidebar_telemetry_card(surface, w, h, state, cv_sample)
        self._draw_phase_banner(surface, w, h, state, cv_sample)

        if state.phase == Phase.MENU:
            self._draw_menu_overlay(surface, w, h, state, cv_sample)
        elif state.phase == Phase.BUSTED:
            self._draw_end_overlay(surface, w, h, state, busted=True)
        elif state.phase == Phase.VICTORY:
            self._draw_end_overlay(surface, w, h, state, busted=False)

        if state.show_leaderboard:
            self._draw_leaderboard_modal(surface, w, h, leaderboard, stats)

    def _get_pov_camera_offsets(self, w: int, h: int, state: GameState) -> tuple[int, int, float]:
        base_vx = int((w - 310) * 0.50)
        base_vy = int(h * 0.41)

        cam_shift_x = int(-state.cat_lane_pos * (w * 0.09))
        bounce_px = int(state.cat_bounce_height * 18.0) if state.phase != Phase.FREEZE else 0
        jump_px = int(state.cat_jump_height * 135.0)
        crouch_factor = max(state.cat_crouch_depth, 1.0 if state.is_rolling else 0.0)
        crouch_px = -int(crouch_factor * 56.0)

        cam_height_px = bounce_px + jump_px + crouch_px
        vx = base_vx + cam_shift_x
        vy = base_vy + int(cam_height_px * 0.52)
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
        safe_z = 0.35 if depth_m < 0.35 else depth_m
        scale = 2.6 / (safe_z + 2.2)
        rel_x = lane_x - cat_lane_pos * 0.85
        sx = int(vx + rel_x * (corridor_w * 0.34) * scale)
        floor_offset = (0.52 - height_y) * corridor_h * 0.72 + cam_height_px * 0.85
        sy = int(vy + floor_offset * scale)
        return sx, sy, scale

    def _draw_cat_pov_viewport(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        corridor_w = max(560, w - 310)
        vx, vy, cam_h = self._get_pov_camera_offsets(w, h, state)
        vy_clamped = max(90, min(h - 120, vy))
        dist = state.distance_meters

        # 1. DESERT SKY LAYER (0 .. vy_clamped)
        sky_surf = self._sprite("desert_sky", corridor_w, vy_clamped + 18)
        if sky_surf is not None:
            surface.blit(sky_surf, (0, 0))
        else:
            pygame.draw.rect(surface, COLOR_SKY_TOP, (0, 0, corridor_w, vy_clamped // 2))
            pygame.draw.rect(surface, COLOR_SKY_HORIZON, (0, vy_clamped // 2, corridor_w, vy_clamped - vy_clamped // 2))

        # Distant Parallax Pyramids & Temple Silhouettes on the Horizon
        pyr_big = self._sprite("pyramid", 190, 130)
        pyr_sm = self._sprite("pyramid", 125, 86)
        parallax_shift = int(dist * 6.0) % 420
        for idx, px in enumerate(range(-parallax_shift, corridor_w + 220, 260)):
            if idx % 2 == 0 and pyr_big is not None:
                surface.blit(pyr_big, (px, vy_clamped - 118))
            elif pyr_sm is not None:
                surface.blit(pyr_sm, (px, vy_clamped - 78))

        # 2. BEST EGYPTIAN DESERT & SANDSTONE GROUND LAYER (vy_clamped .. h)
        ground_h = max(40, h - vy_clamped)
        ground_tex = self._sprite("desert_ground", corridor_w, ground_h)
        if ground_tex is not None:
            surface.blit(ground_tex, (0, vy_clamped))
        else:
            pygame.draw.rect(surface, COLOR_SAND_MID, (0, vy_clamped, corridor_w, ground_h))

        # Perspective-projected Egyptian Sandstone Roadbed & Hieroglyphic Slabs
        scroll_offset = dist % 2.6
        depth_steps = [0.38] + [i * 2.6 - scroll_offset for i in range(1, 14) if i * 2.6 - scroll_offset > 0.38] + [36.0]
        depth_steps.sort(reverse=True)

        tile_strip_1 = self._sprite("floor_tile_1", 260, 48)
        tile_strip_2 = self._sprite("floor_tile_2", 260, 48)

        for idx in range(len(depth_steps) - 1):
            z_far = depth_steps[idx]
            z_near = depth_steps[idx + 1]
            if z_far <= z_near:
                continue

            fl_x, fl_y, _ = self._project_corridor_point(
                -1.5, 0.0, z_far, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            fr_x, fr_y, _ = self._project_corridor_point(
                1.5, 0.0, z_far, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            nl_x, nl_y, _ = self._project_corridor_point(
                -1.5, 0.0, z_near, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            nr_x, nr_y, _ = self._project_corridor_point(
                1.5, 0.0, z_near, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )

            # Outer desert sand terraces on left & right of the sandstone road
            band_idx = int((dist + z_near) / 2.6)
            outer_col = (206, 184, 134) if band_idx % 2 == 0 else (188, 164, 114)
            if nl_y > fr_y:
                pygame.draw.polygon(
                    surface,
                    outer_col,
                    [(0, fl_y), (fl_x, fl_y), (nl_x, nl_y), (0, nl_y)],
                )
                pygame.draw.polygon(
                    surface,
                    outer_col,
                    [(fr_x, fr_y), (corridor_w, fr_y), (corridor_w, nr_y), (nr_x, nr_y)],
                )

            # Inner 3-lane Egyptian Sandstone Causeway polygon
            slab_col = (176, 148, 92) if band_idx % 2 == 0 else (154, 126, 74)
            if state.phase in (Phase.FREEZE, Phase.BUSTED):
                slab_col = (135, 58, 52) if band_idx % 2 == 0 else (112, 42, 38)
            elif state.phase == Phase.CLEAR:
                slab_col = (78, 148, 96) if band_idx % 2 == 0 else (62, 126, 80)

            quad = [(fl_x, fl_y), (fr_x, fr_y), (nr_x, nr_y), (nl_x, nl_y)]
            pygame.draw.polygon(surface, slab_col, quad)

            # Blit textured hieroglyphic stone strip on near slabs for rich ground detail
            slab_h = nl_y - fl_y
            slab_w = int((nr_x - nl_x + fr_x - fl_x) * 0.48)
            if 10 <= slab_h <= 110 and slab_w > 40:
                strip_src = tile_strip_1 if band_idx % 2 == 0 else tile_strip_2
                if strip_src is not None:
                    scaled_strip = self._sprite(
                        "floor_tile_1" if band_idx % 2 == 0 else "floor_tile_2",
                        slab_w,
                        slab_h,
                    )
                    if scaled_strip is not None:
                        cx_slab = (fl_x + fr_x + nl_x + nr_x) // 4
                        surface.blit(scaled_strip, (cx_slab - slab_w // 2, fl_y))

            # Sandstone mortar joint line across the road
            pygame.draw.line(surface, COLOR_SAND_DEEP, (fl_x, fl_y), (fr_x, fr_y), 2)

        # Horizon separator line
        pygame.draw.line(surface, COLOR_SAND_DARK, (0, vy_clamped), (corridor_w, vy_clamped), 2)

        # 3-Lane dividers & raised golden curb borders
        for lane_edge in (-1.5, -0.5, 0.5, 1.5):
            fx0, fy0, _ = self._project_corridor_point(
                lane_edge, 0.0, 0.4, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            fx1, fy1, _ = self._project_corridor_point(
                lane_edge, 0.0, 36.0, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            if abs(lane_edge) == 1.5:
                pygame.draw.line(surface, COLOR_SHADOW, (fx0 + 3, fy0), (fx1 + 2, fy1), 6)
                pygame.draw.line(surface, COLOR_GOLD, (fx0, fy0), (fx1, fy1), 4)
            else:
                pygame.draw.line(surface, (110, 88, 48), (fx0, fy0), (fx1, fy1), 2)

        # Egyptian Sandstone Pillars & Temple Archways flanking the causeway
        for i in range(11, 0, -1):
            z = i * 2.6 - scroll_offset
            if z <= 0.45:
                continue
            lx0, ly0, _ = self._project_corridor_point(
                -1.62, 0.0, z, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            rx0, ry0, _ = self._project_corridor_point(
                1.62, 0.0, z, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            lx1, ly1, _ = self._project_corridor_point(
                -1.62, 0.92, z, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            rx1, ry1, _ = self._project_corridor_point(
                1.62, 0.92, z, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            pil_h = max(14, int(ly0 - ly1))
            pil_w = max(8, int(pil_h * 0.44))
            pil_surf = self._sprite("pillar", pil_w, pil_h)
            if pil_surf is not None:
                surface.blit(pil_surf, (lx0 - pil_w // 2, ly1))
                surface.blit(pil_surf, (rx0 - pil_w // 2, ry1))

        # Goal Temple Archway & Treasure Relic at the Extraction Point
        rem_dist = max(1.2, state.goal_meters - dist)
        if rem_dist < 45.0:
            bx, by, bscale = self._project_corridor_point(
                0.0, 0.28, rem_dist, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            arch_w = max(28, int(220 * bscale))
            arch_h = max(28, int(195 * bscale))
            arch_surf = self._sprite("temple_archway", arch_w, arch_h)
            if arch_surf is not None:
                surface.blit(arch_surf, (bx - arch_w // 2, by - arch_h // 2))
            goal_name = self._goal_sprite_name(state.level)
            g_size = max(18, int(115 * bscale))
            goal_surf = self._sprite(goal_name, g_size, g_size)
            if goal_surf is not None:
                surface.blit(goal_surf, (bx - g_size // 2, by - g_size // 2))
            else:
                radius = max(10, int(68 * bscale))
                pygame.draw.circle(surface, COLOR_GOLD, (bx, by), radius)

        # Collectibles & Obstacles along the 3D Causeway
        for coin in reversed(state.coins_list):
            if coin.collected:
                continue
            c_depth = coin.distance_m - dist
            if -0.2 <= c_depth <= 32.0:
                self._draw_pov_coin(surface, coin, c_depth, vx, vy, corridor_w, h, cam_h, state)

        for obs in reversed(state.obstacles):
            if obs.cleared:
                continue
            depth = obs.distance_m - dist
            if -0.4 <= depth <= 34.0:
                self._draw_pov_obstacle(surface, obs, depth, vx, vy, corridor_w, h, cam_h, state)

        hint = (
            "2 HANDS UP: JUMP  |  2 HANDS DOWN: CROUCH  |  1 HAND L/R: STEER LANE  |  "
            "1-0: LEVEL  |  C: CALIBRATE  |  TAB: SCORES"
        )
        hint_surf = self._txt(self.font_xs, hint, COLOR_WHITE)
        hint_bg = pygame.Rect(16, h - 30, hint_surf.get_width() + 16, 22)
        pygame.draw.rect(surface, COLOR_BG, hint_bg)
        pygame.draw.rect(surface, COLOR_GOLD, hint_bg, 2)
        surface.blit(hint_surf, (24, h - 26))

    def _draw_pov_coin(
        self,
        surface: pygame.Surface,
        coin: CorridorCoin,
        depth: float,
        vx: int,
        vy: int,
        corridor_w: int,
        h: int,
        cam_h: float,
        state: GameState,
    ) -> None:
        height_y = 0.56 if coin.tier == "AIR" else (0.14 if coin.tier == "LOW_ROLL" else 0.22)
        cx, cy, scale = self._project_corridor_point(
            float(coin.lane), height_y, depth, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
        )
        r = max(6, int(38 * scale))
        sprite_name = "ankh" if int(coin.distance_m * 10) % 2 == 0 else "ruby"
        c_surf = self._sprite(sprite_name, r * 2, r * 2)
        if c_surf is not None:
            surface.blit(c_surf, (cx - r, cy - r))
        else:
            coin_rect = pygame.Rect(cx - r, cy - r, r * 2, r * 2)
            pygame.draw.ellipse(surface, COLOR_GOLD, coin_rect)
            pygame.draw.ellipse(surface, COLOR_WHITE, coin_rect, 2)

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
            tl_x, tl_y, _ = self._project_corridor_point(
                -1.5, 1.0, depth, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            br_x, br_y, _ = self._project_corridor_point(
                1.5, 0.36, depth, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            gate_rect = pygame.Rect(tl_x, tl_y, max(12, br_x - tl_x), max(10, br_y - tl_y))
            pygame.draw.rect(surface, (45, 18, 22), gate_rect)
            shrine = self._sprite("tomb_shrine", min(gate_rect.width, 200), gate_rect.height)
            if shrine is not None:
                surface.blit(shrine, (gate_rect.centerx - shrine.get_width() // 2, gate_rect.y))
            pygame.draw.rect(surface, COLOR_CYAN if state.is_rolling else COLOR_RED, gate_rect, 3)

            for frac in (0.3, 0.6, 0.9):
                ly = int(gate_rect.top + gate_rect.height * frac)
                pygame.draw.line(surface, COLOR_RED, (gate_rect.left, ly), (gate_rect.right, ly), 2)

            if depth < 14.0 and gate_rect.width > 130:
                lbl = self._txt(self.font_xs, "CROUCH TO ROLL UNDER", COLOR_CYAN)
                bg = pygame.Rect(gate_rect.centerx - lbl.get_width() // 2 - 4, gate_rect.bottom + 2, lbl.get_width() + 8, 16)
                pygame.draw.rect(surface, COLOR_BG, bg)
                surface.blit(lbl, (gate_rect.centerx - lbl.get_width() // 2, gate_rect.bottom + 4))

        elif obs.kind == "LOW_BEAM":
            tl_x, tl_y, _ = self._project_corridor_point(
                -1.5, 0.34, depth, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            br_x, br_y, _ = self._project_corridor_point(
                1.5, 0.0, depth, vx, vy, corridor_w, h, cam_h, state.cat_lane_pos
            )
            beam_rect = pygame.Rect(tl_x, tl_y, max(12, br_x - tl_x), max(8, br_y - tl_y))
            pygame.draw.rect(surface, COLOR_RED_DIM, beam_rect)
            pygame.draw.rect(surface, COLOR_AMBER, beam_rect, 3)
            mummy_h = max(18, int(beam_rect.height * 1.55))
            mummy_w = max(14, int(mummy_h * 0.78))
            mummy = self._sprite("mummy_sarcophagus", mummy_w, mummy_h)
            if mummy is not None:
                surface.blit(mummy, (beam_rect.centerx - mummy_w // 2, beam_rect.bottom - mummy_h))
            if depth < 14.0 and beam_rect.width > 120:
                lbl = self._txt(self.font_xs, "JUMP OVER SARCOPHAGUS LASER", COLOR_GOLD)
                bg = pygame.Rect(beam_rect.centerx - lbl.get_width() // 2 - 4, beam_rect.top - 20, lbl.get_width() + 8, 16)
                pygame.draw.rect(surface, COLOR_BG, bg)
                surface.blit(lbl, (beam_rect.centerx - lbl.get_width() // 2, beam_rect.top - 18))

        elif obs.kind == "SIDE_BLOCK":
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
            mask_s = self._sprite("pharaoh_mask", pylon_rect.width, pylon_rect.height)
            if mask_s is not None:
                surface.blit(mask_s, pylon_rect.topleft)
            pygame.draw.rect(surface, COLOR_RED, pylon_rect, 3)

    def _draw_floating_popups(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        if not state.popups:
            return
        corridor_w = max(560, w - 310)
        cx = corridor_w // 2
        for idx, p in enumerate(state.popups):
            px = cx + p.lane * 135
            rise = int((1.0 - p.timer) * 70)
            py = int(h * 0.44) - rise - idx * 22
            col = COLOR_GOLD if p.is_gold else COLOR_CYAN
            lbl = self._txt(self.font_md, p.text, col)
            bg_r = pygame.Rect(px - lbl.get_width() // 2 - 6, py - 2, lbl.get_width() + 12, lbl.get_height() + 4)
            pygame.draw.rect(surface, COLOR_BG, bg_r)
            pygame.draw.rect(surface, col, bg_r, 2)
            surface.blit(lbl, (px - lbl.get_width() // 2, py))

    def _draw_pov_lasers(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        if state.phase not in (Phase.WARNING, Phase.FREEZE, Phase.CLEAR, Phase.BUSTED):
            return
        corridor_w = max(560, w - 310)
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
            pygame.draw.line(surface, glow_color, (x_top, 84), (x_bot, h - 35), 8)
            pygame.draw.line(surface, beam_color, (x_top, 84), (x_bot, h - 35), 2)

    def _draw_blood_stains_and_particles(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        corridor_w = max(560, w - 310)
        # Persistent floor/screen blood stains after crashes
        for stain in state.blood_stains:
            sx = int(stain.norm_x * corridor_w)
            sy = int(stain.norm_y * h)
            r = max(4, int(stain.radius))
            pygame.draw.circle(surface, (145, 14, 22), (sx, sy), r)
            pygame.draw.circle(surface, (205, 24, 34), (sx - 2, sy - 1), max(2, int(r * 0.65)))
            pygame.draw.circle(surface, (125, 10, 18), (sx + int(r * 0.8), sy + int(r * 0.4)), max(2, int(r * 0.38)))

    def _draw_cat_pov_foreground(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        corridor_w = max(560, w - 310)
        cx = corridor_w // 2 + int(state.cat_lane_pos * 85)
        bounce_amp = int(state.cat_bounce_height * 22.0) if state.phase != Phase.FREEZE else 0
        bounce_y = int(math.sin(state.walk_cycle) * 14.0 * state.effective_bounce_multiplier) if state.phase != Phase.FREEZE else 0
        crouch_factor = max(state.cat_crouch_depth, 1.0 if state.is_rolling else 0.0)

        # Cat ears at top of viewport (droop slightly as cat loses lives/strength)
        weak_droop = int(state.deaths_count * 2.5)
        ear_bob = int(state.cat_bounce_height * 8.0) + int(crouch_factor * 24) - int(state.cat_jump_height * 10) + weak_droop
        ear_base_y = 86 + int(crouch_factor * 14)
        ear_col = (
            COLOR_RED
            if state.blood_flash_timer > 0.2
            else (COLOR_CYAN if state.is_rolling else (COLOR_GOLD if state.cat_jump_height > 0.12 else COLOR_GREEN))
        )
        left_ear = [(cx - 190, ear_base_y), (cx - 145, 38 + ear_bob), (cx - 105, ear_base_y)]
        right_ear = [(cx + 105, ear_base_y), (cx + 145, 38 + ear_bob), (cx + 190, ear_base_y)]
        pygame.draw.polygon(surface, COLOR_BG, left_ear)
        pygame.draw.polygon(surface, ear_col, left_ear, 2)
        pygame.draw.polygon(surface, COLOR_BG, right_ear)
        pygame.draw.polygon(surface, ear_col, right_ear, 2)

        # Main Burglar Cat character at bottom of the causeway + POV paws
        jump_lift = int(state.cat_jump_height * 118)
        cat_w = 104 if not state.is_rolling else 94
        cat_h = int(128 * (1.0 - 0.36 * crouch_factor))
        cat_y = h - 42 - cat_h - bounce_amp - jump_lift

        shadow_w = max(32, int(88 * (1.0 - 0.30 * min(1.5, state.cat_jump_height + state.cat_bounce_height * 0.25))))
        pygame.draw.ellipse(surface, (58, 44, 24), (cx - shadow_w // 2, h - 48, shadow_w, 14))

        cat_sprite = self._sprite("burglar_cat", cat_w, cat_h) or self._sprite("chubby_cat_sketch", cat_w, cat_h)
        if state.is_rolling:
            roll_rad = math.radians(state.cat_roll_angle)
            ring_cy = h - 105
            if cat_sprite is not None:
                rot = pygame.transform.rotate(cat_sprite, -state.cat_roll_angle * 0.35)
                r_rect = rot.get_rect(center=(cx, ring_cy))
                surface.blit(rot, r_rect.topleft)
            pygame.draw.circle(surface, COLOR_CYAN, (cx, ring_cy), 64, 3)
            for k in range(4):
                a = roll_rad + k * (math.pi * 0.5)
                x1 = cx + int(math.cos(a) * 46)
                y1 = ring_cy + int(math.sin(a) * 46)
                x2 = cx + int(math.cos(a) * 70)
                y2 = ring_cy + int(math.sin(a) * 70)
                pygame.draw.line(surface, COLOR_CYAN, (x1, y1), (x2, y2), 4)
            roll_lbl = self._txt(self.font_sm, "CROUCH ROLL ACTIVE", COLOR_CYAN)
            bg = pygame.Rect(cx - roll_lbl.get_width() // 2 - 6, h - 195, roll_lbl.get_width() + 12, 22)
            pygame.draw.rect(surface, COLOR_BG, bg)
            pygame.draw.rect(surface, COLOR_CYAN, bg, 2)
            surface.blit(roll_lbl, (cx - roll_lbl.get_width() // 2, h - 192))
        else:
            if cat_sprite is not None:
                # Slight limping tilt when weakened by deaths
                if state.deaths_count > 0:
                    limp_deg = math.sin(state.walk_cycle * 0.5) * min(12.0, state.deaths_count * 1.4)
                    tilted = pygame.transform.rotate(cat_sprite, limp_deg)
                    t_rect = tilted.get_rect(center=(cx, cat_y + cat_h // 2))
                    surface.blit(tilted, t_rect.topleft)
                else:
                    surface.blit(cat_sprite, (cx - cat_w // 2, cat_y))

                # Draw battle scars / blood spots / bandages proportional to deaths_count (1..9)
                if state.deaths_count > 0:
                    scar_count = min(6, state.deaths_count)
                    for s_idx in range(scar_count):
                        sx = cx - 22 + (s_idx % 3) * 18
                        sy = cat_y + 34 + (s_idx // 3) * 26
                        pygame.draw.line(surface, (220, 22, 32), (sx - 6, sy - 4), (sx + 6, sy + 4), 3)
                        pygame.draw.circle(surface, (185, 16, 24), (sx + 2, sy + 5), 3)
                    # White bandage strip across wounded cat
                    if state.deaths_count >= 2:
                        band_rect = pygame.Rect(cx - 24, cat_y + 48, 48, 10)
                        pygame.draw.rect(surface, COLOR_WHITE, band_rect)
                        pygame.draw.rect(surface, COLOR_RED, (cx - 6, cat_y + 48, 12, 10))
                        pygame.draw.rect(surface, COLOR_BG, band_rect, 1)

            left_paw_y = h - 56 + bounce_y - int(state.cat_jump_height * 45)
            right_paw_y = h - 56 - bounce_y - int(state.cat_jump_height * 45)
            self._draw_pov_paw(surface, cx - 135, left_paw_y, wounded=state.deaths_count > 0)
            self._draw_pov_paw(surface, cx + 135, right_paw_y, wounded=state.deaths_count > 0)
            if state.cat_jump_height > 0.12:
                jmp_lbl = self._txt(
                    self.font_sm,
                    f"CAT JUMP ACTIVE ({state.bounce_multiplier:.1f}x)",
                    COLOR_GOLD,
                )
                bg = pygame.Rect(cx - jmp_lbl.get_width() // 2 - 6, cat_y - 28, jmp_lbl.get_width() + 12, 22)
                pygame.draw.rect(surface, COLOR_BG, bg)
                pygame.draw.rect(surface, COLOR_GOLD, bg, 2)
                surface.blit(jmp_lbl, (cx - jmp_lbl.get_width() // 2, cat_y - 25))

        # Active flying crimson blood droplets after a crash
        for bp in state.blood_particles:
            bpx = int(cx + bp.rel_x)
            bpy = int(cat_y + cat_h * 0.45 + bp.rel_y)
            br = max(2, int(bp.radius * max(0.3, bp.timer)))
            pygame.draw.circle(surface, (230, 22, 34), (bpx, bpy), br)
            pygame.draw.circle(surface, (140, 10, 18), (bpx, bpy), br, 1)

    def _draw_pov_paw(self, surface: pygame.Surface, px: int, py: int, wounded: bool = False) -> None:
        paw_rect = pygame.Rect(px - 34, py - 16, 68, 56)
        pygame.draw.ellipse(surface, COLOR_WHITE, paw_rect)
        pygame.draw.ellipse(surface, COLOR_RED if wounded else COLOR_BG, paw_rect, 3)
        for tx, ty in ((-16, -4), (0, -9), (16, -4)):
            pygame.draw.circle(surface, COLOR_RED, (px + tx, py + ty), 6)
        pygame.draw.ellipse(surface, COLOR_RED, (px - 12, py + 5, 24, 15))
        if wounded:
            pygame.draw.circle(surface, (200, 18, 28), (px + 14, py + 12), 4)

    def _draw_2d_pip_window(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        """
        Side Window on the Right Sidebar (Middle Card):
        Renders the 2D Egyptian Museum & Desert Side-Scroller with Sky, Pyramids,
        Pillars, Temple Archways, Tiled Sandstone Ground, Chubby Burglar Cat,
        Escalating Bounce, Blood Splatter, Obstacles, and Extraction Point Treasure.
        """
        box_w = 282
        box_h = 244
        box_x = w - box_w - 16
        box_y = 252

        box_rect = pygame.Rect(box_x, box_y, box_w, box_h)
        self._draw_panel(
            surface,
            box_rect,
            fill=(16, 14, 12),
            border=COLOR_RED if state.blood_flash_timer > 0.1 else (COLOR_CYAN if state.is_rolling else COLOR_GOLD),
            shadow=COLOR_GOLD,
            shadow_off=5,
        )

        hdr_rect = pygame.Rect(box_x, box_y, box_w, 24)
        pygame.draw.rect(surface, COLOR_GRAY_DARK, hdr_rect)
        pygame.draw.line(surface, COLOR_WHITE, (box_x, box_y + 24), (box_x + box_w, box_y + 24), 2)
        b_tag = "MAX" if state.bounce_saturated else f"T{state.bounce_tier}"
        hdr_lbl = self._txt(
            self.font_xs,
            f"2D SIDE VIEW // BOUNCE {state.bounce_multiplier:.1f}x ({b_tag})",
            COLOR_GOLD,
        )
        surface.blit(hdr_lbl, (box_x + 8, box_y + 5))

        inner_rect = pygame.Rect(box_x + 4, box_y + 26, box_w - 8, box_h - 30)
        sub = surface.subsurface(inner_rect)
        sw, sh = inner_rect.width, inner_rect.height
        floor_y = int(sh * 0.74)
        dist = state.distance_meters

        # 1. Sky + Desert Horizon inside the 2D Side Window
        sky_pip = self._sprite("desert_sky", sw, floor_y)
        if sky_pip is not None:
            sub.blit(sky_pip, (0, 0))
        else:
            pygame.draw.rect(sub, COLOR_SKY_TOP, (0, 0, sw, floor_y))

        # Parallax Pyramid Silhouette
        pyr_pip = self._sprite("pyramid", 86, 58)
        if pyr_pip is not None:
            for px_p in range(-(int(dist * 5) % 160), sw + 80, 160):
                sub.blit(pyr_pip, (px_p, floor_y - 56))

        # Scrolling Pillars & Temple Archways
        pil_pip = self._sprite("pillar", 28, 72)
        arch_pip = self._sprite("temple_archway", 58, 66)
        bg_scroll = int(dist * 22.0) % 140
        for idx, bx in enumerate(range(-bg_scroll, sw + 70, 70)):
            if idx % 2 == 0 and pil_pip is not None:
                sub.blit(pil_pip, (bx, floor_y - 72))
            elif idx % 2 == 1 and arch_pip is not None:
                sub.blit(arch_pip, (bx, floor_y - 66))

        # Tiled Egyptian Sandstone Floor across bottom of Side Window
        t1 = self._sprite("floor_tile_1", 46, max(18, sh - floor_y))
        t2 = self._sprite("floor_tile_2", 46, max(18, sh - floor_y))
        pygame.draw.rect(sub, COLOR_SAND_DARK, (0, floor_y, sw, sh - floor_y))
        if t1 is not None and t2 is not None:
            t_scroll = int(dist * 26.0)
            first_i = t_scroll // 46
            off_x = -(t_scroll % 46)
            for idx, tx in enumerate(range(off_x, sw, 46)):
                sub.blit(t1 if (first_i + idx) % 2 == 0 else t2, (tx, floor_y))
        pygame.draw.line(sub, COLOR_GOLD, (0, floor_y), (sw, floor_y), 2)

        cat_2d_x = 54
        pixels_per_m = 12.0

        # Collectible Relics (Ankhs & Rubies)
        for coin in state.coins_list:
            if coin.collected:
                continue
            rel_m = coin.distance_m - dist
            cx_2d = cat_2d_x + int(rel_m * pixels_per_m)
            if 0 <= cx_2d <= sw:
                cy_2d = floor_y - 50 if coin.tier == "AIR" else (floor_y - 14 if coin.tier == "LOW_ROLL" else floor_y - 22)
                c_pip = self._sprite("ankh" if int(coin.distance_m * 10) % 2 == 0 else "ruby", 16, 16)
                if c_pip is not None:
                    sub.blit(c_pip, (cx_2d - 8, cy_2d - 8))
                else:
                    pygame.draw.circle(sub, COLOR_GOLD, (cx_2d, cy_2d), 5)

        # Obstacles (Tomb Shrine, Mummy Sarcophagus, Pharaoh Mask)
        for obs in state.obstacles:
            if obs.cleared:
                continue
            rel_m = obs.distance_m - dist
            ox = cat_2d_x + int(rel_m * pixels_per_m)
            if -20 <= ox <= sw + 20:
                if obs.kind == "HIGH_GATE":
                    g_rect = pygame.Rect(ox - 12, 4, 24, floor_y - 28)
                    shrine_p = self._sprite("tomb_shrine", 28, max(16, floor_y - 28))
                    if shrine_p is not None:
                        sub.blit(shrine_p, (ox - 14, 4))
                    pygame.draw.rect(sub, COLOR_CYAN if state.is_rolling else COLOR_RED, g_rect, 2)
                elif obs.kind == "LOW_BEAM":
                    m_pip = self._sprite("mummy_sarcophagus", 22, 34)
                    if m_pip is not None:
                        sub.blit(m_pip, (ox - 11, floor_y - 34))
                    pygame.draw.line(sub, COLOR_RED, (ox, floor_y - 28), (ox, floor_y), 2)
                elif obs.kind == "SIDE_BLOCK":
                    mask_pip = self._sprite("pharaoh_mask", 24, 26)
                    if mask_pip is not None:
                        sub.blit(mask_pip, (ox - 12, floor_y - 30))
                    else:
                        pygame.draw.rect(sub, COLOR_AMBER, (ox - 7, floor_y - 34, 14, 34), 2)

        # Goal Treasure at far right of 2D Side Window
        goal_rel_m = state.goal_meters - dist
        goal_x = min(sw - 22, cat_2d_x + int(goal_rel_m * pixels_per_m))
        g_pip = self._sprite(self._goal_sprite_name(state.level), 28, 28)
        if g_pip is not None:
            sub.blit(g_pip, (goal_x - 14, floor_y - 30))
        else:
            pygame.draw.circle(sub, COLOR_GOLD, (goal_x, floor_y - 16), 11)

        # Chubby Burglar Cat in 2D Side Window (with 10s escalating bounce & blood droplets)
        bounce_2d = int(state.cat_bounce_height * 12.0) if state.phase != Phase.FREEZE else 0
        jump_2d = int(state.cat_jump_height * 46)
        cat_2d_y = max(18, floor_y - 20 - bounce_2d - jump_2d)

        # Blood stains on 2D floor
        if state.deaths_count > 0:
            for d_i in range(min(5, state.deaths_count)):
                pygame.draw.circle(sub, (185, 18, 26), (cat_2d_x - 8 + d_i * 5, floor_y + 3), 3)

        cat_pip = self._sprite("burglar_cat", 34, 42)
        if state.is_rolling:
            ball_y = floor_y - 12
            pygame.draw.circle(sub, COLOR_CYAN, (cat_2d_x, ball_y), 12)
            pygame.draw.circle(sub, COLOR_WHITE, (cat_2d_x, ball_y), 12, 2)
        elif cat_pip is not None:
            sub.blit(cat_pip, (cat_2d_x - 17, cat_2d_y - 22))
        else:
            pygame.draw.ellipse(sub, COLOR_WHITE, (cat_2d_x - 14, cat_2d_y - 8, 26, 17))

        for bp in state.blood_particles[:8]:
            pygame.draw.circle(
                sub,
                (225, 22, 32),
                (int(cat_2d_x + bp.rel_x * 0.35), int(cat_2d_y + bp.rel_y * 0.35)),
                2,
            )

        lane_txt = "L" if state.target_lane < 0 else ("R" if state.target_lane > 0 else "C")
        state_txt = "ROLL" if state.is_rolling else ("JUMP" if state.cat_jump_height > 0.1 else "CREEP")
        info_bg = pygame.Rect(0, sh - 20, sw, 20)
        pygame.draw.rect(sub, (12, 14, 18), info_bg)
        info = self._txt(
            self.font_xs,
            f"LN:{lane_txt} | {state_txt} | LIVES:{state.lives_remaining}/9 | STR:{int(state.cat_strength * 100)}%",
            COLOR_RED if state.lives_remaining <= 3 else COLOR_GOLD,
        )
        sub.blit(info, (6, sh - 17))

    def _draw_sidebar_telemetry_card(
        self,
        surface: pygame.Surface,
        w: int,
        h: int,
        state: GameState,
        cv_sample: MotionSample,
    ) -> None:
        """
        Bottom Card on the Right Sidebar:
        Displays Hand Gesture status, 9-Lives Vitality & Weakness, 10s Bounce Escalation,
        Target Relic preview, and Laser Sweep countdown.
        """
        box_w = 282
        box_x = w - box_w - 16
        box_y = 508
        box_h = max(150, h - box_y - 12)

        box_rect = pygame.Rect(box_x, box_y, box_w, box_h)
        self._draw_panel(
            surface,
            box_rect,
            fill=COLOR_BG,
            border=COLOR_WHITE,
            shadow=COLOR_RED if state.blood_flash_timer > 0.1 else (COLOR_CYAN if state.is_rolling else COLOR_GREEN),
            shadow_off=5,
        )

        hdr_rect = pygame.Rect(box_x, box_y, box_w, 24)
        pygame.draw.rect(surface, COLOR_GRAY_DARK, hdr_rect)
        pygame.draw.line(surface, COLOR_WHITE, (box_x, box_y + 24), (box_x + box_w, box_y + 24), 2)
        hdr_lbl = self._txt(self.font_xs, "HAND GESTURE & 9-LIVES TELEMETRY", COLOR_GREEN)
        surface.blit(hdr_lbl, (box_x + 8, box_y + 5))

        # Target Relic showcase box
        relic_box = pygame.Rect(box_x + 10, box_y + 32, 58, 58)
        pygame.draw.rect(surface, COLOR_GRAY_DARK, relic_box)
        pygame.draw.rect(surface, COLOR_GOLD, relic_box, 2)
        relic_surf = self._sprite(self._goal_sprite_name(state.level), 44, 44)
        if relic_surf is not None:
            surface.blit(relic_surf, (relic_box.x + 7, relic_box.y + 7))

        # Hand Gesture, Posture & Weakness status next to Relic
        posture_col = (
            COLOR_CYAN
            if cv_sample.human_posture == "CROUCH"
            else (COLOR_GOLD if cv_sample.human_posture == "JUMP" else COLOR_GREEN)
        )
        gest_label = cv_sample.hand_gesture.replace("_", " ")
        lane_name = "LEFT" if state.target_lane < 0 else ("RIGHT" if state.target_lane > 0 else "CENTER")
        g_lbl = self._txt(self.font_xs, f"HANDS: {gest_label} ({cv_sample.finger_count}H)", posture_col)
        p_lbl = self._txt(self.font_xs, f"CMD:  {cv_sample.human_posture} // LN: {lane_name}", COLOR_WHITE)
        w_col = COLOR_RED if state.deaths_count >= 5 else (COLOR_AMBER if state.deaths_count > 0 else COLOR_GREEN)
        w_lbl = self._txt(
            self.font_xs,
            f"POWER: {int(state.cat_strength * 100)}% ({state.deaths_count} CRASHES)",
            w_col,
        )
        surface.blit(g_lbl, (box_x + 76, box_y + 33))
        surface.blit(p_lbl, (box_x + 76, box_y + 52))
        surface.blit(w_lbl, (box_x + 76, box_y + 71))

        # 10-Second Escalating Bounce Meter
        b_y = box_y + 98
        if state.bounce_saturated:
            b_txt = f"BOUNCE HEIGHT: {state.bounce_multiplier:.2f}x [MAX SATURATED]"
            b_ratio = 1.0
            b_col = COLOR_GOLD
        else:
            sec_in_tier = state.elapsed_sec % 10.0
            b_txt = f"BOUNCE: {state.bounce_multiplier:.2f}x (NEXT BOOST IN {max(0.0, 10.0 - sec_in_tier):.1f}S)"
            b_ratio = max(0.0, min(1.0, sec_in_tier / 10.0))
            b_col = COLOR_CYAN
        surface.blit(self._txt(self.font_xs, b_txt, b_col), (box_x + 10, b_y))
        b_bar = pygame.Rect(box_x + 10, b_y + 16, box_w - 20, 10)
        pygame.draw.rect(surface, COLOR_GRAY_DARK, b_bar)
        pygame.draw.rect(surface, b_col, (b_bar.x, b_bar.y, int(b_bar.width * b_ratio), b_bar.height))
        pygame.draw.rect(surface, COLOR_WHITE, b_bar, 1)

        # Next Laser Sweep Countdown Bar
        sweep_y = b_y + 34
        if box_y + box_h >= sweep_y + 34:
            if state.phase == Phase.FREEZE:
                s_txt = f"LASER LOCK ACTIVE: {max(0.0, state.phase_timer):.1f}S (HOLD HAND STILL)"
                s_col = COLOR_RED
                s_ratio = max(0.0, min(1.0, state.phase_timer / max(0.1, state.config.freeze_duration)))
            elif state.phase == Phase.WARNING:
                s_txt = "SWEEP IMMINENT - FREEZE HAND NOW!"
                s_col = COLOR_AMBER
                s_ratio = 1.0
            else:
                s_txt = f"NEXT LASER SWEEP IN: {max(0.0, state.next_sweep_timer):.1f}S"
                s_col = COLOR_GREEN
                s_ratio = max(0.0, min(1.0, 1.0 - (state.next_sweep_timer / 10.0)))

            s_lbl = self._txt(self.font_xs, s_txt, s_col)
            surface.blit(s_lbl, (box_x + 10, sweep_y))
            s_bar = pygame.Rect(box_x + 10, sweep_y + 16, box_w - 20, 11)
            pygame.draw.rect(surface, COLOR_GRAY_DARK, s_bar)
            pygame.draw.rect(
                surface,
                s_col,
                (s_bar.x, s_bar.y, int(s_bar.width * s_ratio), s_bar.height),
            )
            pygame.draw.rect(surface, COLOR_WHITE, s_bar, 1)

        if box_y + box_h >= sweep_y + 52:
            ctrl = self._txt(self.font_xs, "2 HANDS UP/DN: JUMP/CROUCH | 1 HAND: L/R", (185, 195, 205))
            surface.blit(ctrl, (box_x + 10, box_y + box_h - 19))

    def _draw_top_hud(
        self,
        surface: pygame.Surface,
        w: int,
        state: GameState,
        cv_sample: MotionSample,
        api_online: bool,
    ) -> None:
        hud_w = max(460, w - 320)
        hud_rect = pygame.Rect(16, 14, hud_w, 68)
        self._draw_panel(
            surface,
            hud_rect,
            fill=COLOR_BG,
            border=COLOR_RED if state.blood_flash_timer > 0.1 else COLOR_WHITE,
            shadow=COLOR_GOLD,
            shadow_off=5,
        )

        # Column 1: Score & Level
        score_lbl = self._txt(self.font_xs, f"LVL {state.level:02d}/{MAX_LEVEL:02d} // SCORE", COLOR_GOLD)
        score_val = self._txt(self.font_md, f"{state.score:06d}", COLOR_WHITE)
        surface.blit(score_lbl, (26, 20))
        surface.blit(score_val, (26, 39))

        # Column 2: Relics & Combo
        ankh_icon = self._sprite("ankh", 16, 18)
        if ankh_icon is not None:
            surface.blit(ankh_icon, (148, 20))
        coin_lbl = self._txt(self.font_xs, f"RELICS: {state.coins:03d}", COLOR_GOLD)
        combo_lbl = self._txt(
            self.font_xs,
            f"COMBO x{state.combo_multiplier} (STRK {state.streak})",
            COLOR_CYAN if state.combo_multiplier >= 3 else COLOR_GREEN,
        )
        surface.blit(coin_lbl, (168, 21))
        surface.blit(combo_lbl, (148, 44))

        # Column 3: 9 LIVES Heart Bar & Cat Weakness
        lives_x = 292
        lives_col = COLOR_RED if state.lives_remaining <= 3 else (COLOR_AMBER if state.lives_remaining <= 6 else COLOR_GREEN)
        lives_hdr = self._txt(
            self.font_xs,
            f"9 LIVES: {state.lives_remaining}/{state.max_lives} ({int(state.cat_strength * 100)}% STR)",
            lives_col,
        )
        surface.blit(lives_hdr, (lives_x, 21))
        for i in range(state.max_lives):
            pip_rect = pygame.Rect(lives_x + i * 16, 42, 13, 16)
            alive = i < state.lives_remaining
            pygame.draw.rect(surface, COLOR_RED if alive else COLOR_GRAY_DARK, pip_rect)
            pygame.draw.rect(surface, COLOR_WHITE if alive else COLOR_GRAY_MID, pip_rect, 1)

        # Column 4: Sector Progress Bar
        dist_x = 452
        bar_w = max(80, hud_w - 640)
        dist_lbl = self._txt(
            self.font_xs,
            f"{state.level_name.upper()}: {state.distance_meters:04.1f}M/{state.goal_meters:.0f}M",
            COLOR_WHITE,
        )
        surface.blit(dist_lbl, (dist_x, 21))
        bar_rect = pygame.Rect(dist_x, 42, bar_w, 16)
        pygame.draw.rect(surface, COLOR_GRAY_DARK, bar_rect)
        fill_w = int(bar_rect.width * state.progress_ratio)
        if fill_w > 0:
            pygame.draw.rect(surface, COLOR_GOLD, (bar_rect.x, bar_rect.y, fill_w, bar_rect.height))
        pygame.draw.rect(surface, COLOR_WHITE, bar_rect, 2)

        # Column 5: Hand Gesture & Bounce Status
        right_x = hud_rect.right - 185
        pose_lbl = self._txt(
            self.font_xs,
            f"HAND: {cv_sample.human_posture} ({'L' if cv_sample.human_lane < 0 else ('R' if cv_sample.human_lane > 0 else 'C')})",
            COLOR_CYAN if cv_sample.human_posture == "CROUCH" else (COLOR_GOLD if cv_sample.human_posture == "JUMP" else COLOR_GREEN),
        )
        b_tag = "MAX" if state.bounce_saturated else f"T{state.bounce_tier}/5"
        bounce_lbl = self._txt(
            self.font_xs,
            f"BOUNCE: {state.bounce_multiplier:.2f}x ({b_tag})",
            COLOR_GOLD if state.bounce_saturated else COLOR_WHITE,
        )
        surface.blit(pose_lbl, (right_x, 21))
        surface.blit(bounce_lbl, (right_x, 44))

    def _draw_cctv_feed(
        self,
        surface: pygame.Surface,
        w: int,
        state: GameState,
        cv_sample: MotionSample,
    ) -> None:
        box_w, box_h = 282, 184
        box_x = w - box_w - 16
        box_y = 14
        border_col = COLOR_RED if state.phase == Phase.FREEZE else COLOR_WHITE

        box_rect = pygame.Rect(box_x, box_y, box_w, box_h + 44)
        self._draw_panel(surface, box_rect, fill=COLOR_BG, border=border_col, shadow=border_col, shadow_off=5)

        try:
            frame_surf = pygame.surfarray.make_surface(np.transpose(cv_sample.frame_rgb, (1, 0, 2)))
            scaled = pygame.transform.scale(frame_surf, (box_w - 8, box_h - 8))
            surface.blit(scaled, (box_x + 4, box_y + 4))
        except Exception:
            pygame.draw.rect(surface, COLOR_GRAY_DARK, (box_x + 4, box_y + 4, box_w - 8, box_h - 8))

        cat_badge = self._sprite("burglar_cat", 42, 52)
        if cat_badge is not None:
            badge_bg = pygame.Rect(box_x + box_w - 52, box_y + box_h - 60, 46, 54)
            pygame.draw.rect(surface, COLOR_BG, badge_bg)
            pygame.draw.rect(surface, COLOR_GOLD, badge_bg, 2)
            surface.blit(cat_badge, (badge_bg.x + 2, badge_bg.y + 1))

        blink = int(time.perf_counter() * 2.5) % 2 == 0
        rec_lbl = self._txt(self.font_xs, "CAM 01 - LIVE FEED", COLOR_WHITE)
        pygame.draw.rect(surface, COLOR_BG, (box_x + 8, box_y + 8, rec_lbl.get_width() + 26, 18))
        pygame.draw.rect(surface, COLOR_WHITE, (box_x + 8, box_y + 8, rec_lbl.get_width() + 26, 18), 1)
        if blink:
            pygame.draw.circle(surface, COLOR_RED, (box_x + 17, box_y + 17), 5)
        surface.blit(rec_lbl, (box_x + 26, box_y + 10))

        meter_y = box_y + box_h + 2
        meter_rect = pygame.Rect(box_x + 8, meter_y + 16, box_w - 16, 12)
        delta_lbl = self._txt(
            self.font_xs,
            f"MOTION: {cv_sample.motion_score:04.1f} / THR {state.config.motion_threshold:.0f}",
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
        cx = (w - 310) // 2
        if state.phase == Phase.WARNING:
            rect = pygame.Rect(cx - 250, 96, 500, 56)
            self._draw_panel(surface, rect, fill=COLOR_BG, border=COLOR_AMBER, shadow=COLOR_AMBER)
            txt = self._txt(self.font_md, "LASER SWEEP INCOMING - PREPARE TO FREEZE", COLOR_AMBER)
            surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        elif state.phase == Phase.FREEZE:
            rect = pygame.Rect(cx - 210, 94, 420, 82)
            self._draw_panel(surface, rect, fill=COLOR_BG, border=COLOR_RED, shadow=COLOR_RED)
            t1 = self._txt(self.font_lg, f"FREEZE {max(0.0, state.phase_timer):.1f}s", COLOR_RED)
            sub = "RELEASE SPACE" if cv_sample.sim_motion_active else "HOLD STILL"
            t2 = self._txt(self.font_sm, sub, COLOR_WHITE)
            surface.blit(t1, (rect.centerx - t1.get_width() // 2, rect.y + 10))
            surface.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + 50))

        elif state.phase == Phase.CLEAR:
            rect = pygame.Rect(cx - 210, 96, 420, 54)
            self._draw_panel(surface, rect, fill=COLOR_BG, border=COLOR_GREEN, shadow=COLOR_GREEN)
            txt = self._txt(self.font_md, "SWEEP CLEAR - SPEED BONUS", COLOR_GREEN)
            surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        elif state.action_banner_timer > 0.0 and state.action_banner:
            rect = pygame.Rect(cx - 240, 94, 480, 44)
            self._draw_panel(surface, rect, fill=COLOR_BG, border=COLOR_CYAN, shadow=COLOR_WHITE, shadow_off=4)
            txt = self._txt(self.font_sm, state.action_banner, COLOR_CYAN)
            surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

    def _draw_menu_overlay(
        self,
        surface: pygame.Surface,
        w: int,
        h: int,
        state: GameState,
        cv_sample: MotionSample,
    ) -> None:
        cx = (w - 310) // 2
        box = pygame.Rect(cx - 310, h // 2 - 190, 620, 365)
        self._draw_panel(surface, box, fill=COLOR_BG, border=COLOR_WHITE, shadow=COLOR_GOLD, shadow_off=8)

        cat_menu = self._sprite("burglar_cat", 62, 76)
        if cat_menu is not None:
            surface.blit(cat_menu, (box.x + 22, box.y + 16))
        lamp_menu = self._sprite("golden_lamp", 64, 58)
        if lamp_menu is not None:
            surface.blit(lamp_menu, (box.right - 86, box.y + 22))

        title = self._txt(self.font_xl, "LASER PAW PATROL", COLOR_GOLD)
        sub = self._txt(self.font_sm, "EGYPTIAN MUSEUM HEIST // 10-LEVEL STEALTH RUNNER", COLOR_GREEN)
        surface.blit(title, (box.centerx - title.get_width() // 2, box.y + 18))
        surface.blit(sub, (box.centerx - sub.get_width() // 2, box.y + 74))

        sel_spec = get_level_spec(state.config.starting_level)
        lvl_hdr = self._txt(
            self.font_sm,
            f"SELECTED START: LEVEL {sel_spec.number:02d}/{MAX_LEVEL:02d} - {sel_spec.name} ({sel_spec.goal_meters:.0f}M)",
            COLOR_GOLD,
        )
        surface.blit(lvl_hdr, (box.centerx - lvl_hdr.get_width() // 2, box.y + 106))

        pip_w = 46
        total_pips_w = MAX_LEVEL * (pip_w + 6) - 6
        start_px = box.centerx - total_pips_w // 2
        for i in range(1, MAX_LEVEL + 1):
            p_rect = pygame.Rect(start_px + (i - 1) * (pip_w + 6), box.y + 132, pip_w, 24)
            active = i == state.config.starting_level
            pygame.draw.rect(surface, COLOR_GOLD if active else COLOR_GRAY_DARK, p_rect)
            pygame.draw.rect(surface, COLOR_WHITE, p_rect, 2)
            n_lbl = self._txt(self.font_xs, f"L{i:02d}", COLOR_BG if active else COLOR_WHITE)
            surface.blit(n_lbl, (p_rect.centerx - n_lbl.get_width() // 2, p_rect.centery - n_lbl.get_height() // 2))

        lines = (
            "RAISE BOTH HANDS UP: JUMP OVER LASERS & SARCOPHAGI",
            "LOWER BOTH HANDS DOWN: CROUCH ROLL UNDER LASER GATES",
            "MOVE ANY 1 HAND LEFT OR RIGHT: STEER TEMPLE LANES",
            "EVERY 10S: CAT BOUNCES HIGHER (SATURATES AT 2.1x MAX!)",
            "9 LIVES: EACH CRASH SHEDS BLOOD & WEAKENS CAT 9 TIMES!",
        )
        for idx, line in enumerate(lines):
            col = COLOR_RED if idx == 4 else (COLOR_CYAN if idx == 1 else (COLOR_GOLD if idx == 0 else COLOR_WHITE))
            lbl = self._txt(self.font_xs, line, col)
            surface.blit(lbl, (box.x + 28, box.y + 170 + idx * 24))

        cta = pygame.Rect(box.centerx - 195, box.bottom - 50, 390, 36)
        self._draw_panel(surface, cta, fill=COLOR_GREEN, border=COLOR_WHITE, shadow=COLOR_WHITE, shadow_off=4)
        if state.gesture_start_hold > 0.02:
            prog = min(1.0, state.gesture_start_hold / 0.55)
            pygame.draw.rect(surface, COLOR_GOLD, (cta.x + 2, cta.y + 2, int((cta.width - 4) * prog), cta.height - 4))
        cta_txt = self._txt(self.font_sm, "RAISE BOTH HANDS OR PRESS ENTER TO START", COLOR_BG)
        surface.blit(cta_txt, (cta.centerx - cta_txt.get_width() // 2, cta.centery - cta_txt.get_height() // 2))

    def _draw_end_overlay(
        self, surface: pygame.Surface, w: int, h: int, state: GameState, busted: bool
    ) -> None:
        cx = (w - 310) // 2
        accent = COLOR_RED if busted else COLOR_GOLD
        box = pygame.Rect(cx - 275, h // 2 - 140, 550, 265)
        self._draw_panel(surface, box, fill=COLOR_BG, border=accent, shadow=accent, shadow_off=8)

        headline = f"ALL 9 LIVES LOST // LVL {state.level:02d}" if busted else "MUSEUM HEIST COMPLETE!"
        h_lbl = self._txt(self.font_lg, headline, accent)
        surface.blit(h_lbl, (box.centerx - h_lbl.get_width() // 2, box.y + 20))

        reason_line = state.busted_reason if busted else "ALL EGYPTIAN RELICS & THE GOLDEN LAMP STOLEN!"
        stats_lines = (
            reason_line,
            f"SCORE: {state.score} | RELICS: {state.coins} | BOUNCE: {state.bounce_multiplier:.2f}x",
            f"LIVES LEFT: {state.lives_remaining}/9 ({int(state.cat_strength * 100)}% STR) | LVL {state.level:02d}",
        )
        for idx, line in enumerate(stats_lines):
            col = COLOR_AMBER if idx == 0 else (COLOR_GREEN if idx == 2 else COLOR_WHITE)
            lbl = self._txt(self.font_sm, line, col)
            surface.blit(lbl, (box.centerx - lbl.get_width() // 2, box.y + 78 + idx * 30))

        prompt = self._txt(self.font_sm, "RAISE PALM / ENTER / R: RETRY  |  TAB: SCORES  |  ESC: MENU", COLOR_WHITE)
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
        self._draw_panel(surface, box, fill=COLOR_BG, border=COLOR_WHITE, shadow=COLOR_GOLD, shadow_off=8)

        title = self._txt(self.font_lg, "LEADERBOARD", COLOR_GOLD)
        surface.blit(title, (box.x + 24, box.y + 18))

        summary = self._txt(
            self.font_xs,
            f"RUNS: {stats.total_runs}  |  WIN RATE: {stats.win_rate:.1f}%  |  BEST: {stats.best_score}",
            COLOR_GREEN,
        )
        surface.blit(summary, (box.x + 24, box.y + 60))

        header = self._txt(self.font_xs, "RK  CALLSIGN       SCORE    DIST     STREAK  MODE   OUTCOME", (160, 160, 175))
        surface.blit(header, (box.x + 24, box.y + 90))
        pygame.draw.line(surface, COLOR_WHITE, (box.x + 24, box.y + 110), (box.right - 24, box.y + 110), 2)

        if not leaderboard:
            empty = self._txt(self.font_sm, "NO RUNS RECORDED YET", COLOR_WHITE)
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
                lbl = self._txt(self.font_sm, row_txt, col)
                surface.blit(lbl, (box.x + 24, ry))

        close_lbl = self._txt(self.font_xs, "PRESS TAB OR ESC TO CLOSE", COLOR_WHITE)
        surface.blit(close_lbl, (box.centerx - close_lbl.get_width() // 2, box.bottom - 28))

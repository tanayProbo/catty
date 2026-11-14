"""Neobrutalist & Cyberpunk Pygame renderer for LASER_PAW_PATROL."""

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
from game_state import GameState, Phase


class UIRenderer:
    """Renders the Neobrutalist Cyberpunk stealth arcade UI and vector characters."""

    def __init__(self) -> None:
        self.font_xs = pygame.font.SysFont("consolas", 13, bold=True)
        self.font_sm = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_md = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 34, bold=True)
        self.font_xl = pygame.font.SysFont("consolas", 52, bold=True)

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

        self._draw_corridor_grid(surface, w, h, state)
        self._draw_floor_track(surface, w, h, state)
        self._draw_fish_bowl(surface, w, h, state)
        self._draw_characters(surface, w, h, state)
        self._draw_lasers(surface, w, h, state)

        # Red alarm flash overlay when busted
        if state.phase == Phase.BUSTED and state.flash_timer > 0.0:
            pulse = abs(math.sin(time.perf_counter() * 14.0))
            alpha = int(110 * pulse * min(1.0, state.flash_timer))
            if alpha > 0:
                overlay = pygame.Surface((w, h), pygame.SRCALPHA)
                overlay.fill((*COLOR_RED, alpha))
                surface.blit(overlay, (0, 0))

        self._draw_top_hud(surface, w, state, api_online)
        self._draw_cctv_feed(surface, w, state, cv_sample)
        self._draw_phase_banner(surface, w, h, state, cv_sample)

        if state.phase == Phase.MENU:
            self._draw_menu_overlay(surface, w, h, state, cv_sample)
        elif state.phase == Phase.BUSTED:
            self._draw_end_overlay(surface, w, h, state, busted=True)
        elif state.phase == Phase.VICTORY:
            self._draw_end_overlay(surface, w, h, state, busted=False)

        if state.show_leaderboard:
            self._draw_leaderboard_modal(surface, w, h, leaderboard, stats)

    def _draw_corridor_grid(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        floor_y = int(h * 0.78)
        grid_color = (26, 26, 32)
        if state.phase == Phase.FREEZE or state.phase == Phase.BUSTED:
            grid_color = (45, 14, 18)
        elif state.phase == Phase.CLEAR:
            grid_color = (12, 42, 24)

        # Vertical vault pillars
        spacing = max(90, w // 10)
        for x in range(0, w + spacing, spacing):
            pygame.draw.line(surface, grid_color, (x, 90), (x, floor_y), 1)
        # Horizontal perspective lines
        for y in range(110, floor_y, 55):
            pygame.draw.line(surface, grid_color, (0, y), (w, y), 1)

    def _draw_floor_track(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        floor_y = int(h * 0.78)
        pygame.draw.rect(surface, COLOR_GRAY_DARK, (0, floor_y, w, h - floor_y))
        pygame.draw.line(surface, COLOR_WHITE, (0, floor_y), (w, floor_y), 3)

        # Distance tick marks along floor
        start_x = int(w * 0.10)
        end_x = int(w * 0.88)
        span = max(1, end_x - start_x)
        for i in range(0, 11):
            tx = start_x + int((span * i) / 10)
            pygame.draw.line(surface, COLOR_GRAY_MID, (tx, floor_y + 4), (tx, floor_y + 18), 2)
            lbl = self.font_xs.render(f"{int(state.goal_meters * i / 10)}M", True, COLOR_GRAY_MID)
            surface.blit(lbl, (tx - lbl.get_width() // 2, floor_y + 22))

        # Bottom controls hint bar
        hint = "[ENTER] START/RETRY  |  [SPACE] SIM MOTION  |  [M] TOGGLE SIM  |  [ / ] SENSITIVITY  |  [TAB] LEADERBOARD"
        hint_surf = self.font_xs.render(hint, True, (150, 150, 160))
        surface.blit(hint_surf, (max(16, (w - hint_surf.get_width()) // 2), h - 26))

    def _draw_fish_bowl(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        floor_y = int(h * 0.78)
        bx = int(w * 0.89)
        by = floor_y - 38

        # Pulsing golden aura
        pulse = 0.75 + 0.25 * math.sin(time.perf_counter() * 4.5)
        aura_r = int(48 * pulse)
        aura = pygame.Surface((aura_r * 2 + 4, aura_r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(aura, (*COLOR_GOLD, 42), (aura_r + 2, aura_r + 2), aura_r)
        surface.blit(aura, (bx - aura_r - 2, by - aura_r - 2))

        # Pedestal
        ped_rect = pygame.Rect(bx - 24, floor_y - 16, 48, 16)
        pygame.draw.rect(surface, COLOR_GRAY_MID, ped_rect)
        pygame.draw.rect(surface, COLOR_WHITE, ped_rect, 2)

        # Glass bowl & golden water
        pygame.draw.circle(surface, (35, 80, 110), (bx, by), 26)
        pygame.draw.circle(surface, COLOR_GOLD, (bx, by + 4), 21)
        pygame.draw.circle(surface, COLOR_WHITE, (bx, by), 26, 3)
        # Rim
        pygame.draw.ellipse(surface, COLOR_WHITE, (bx - 18, by - 30, 36, 10), 2)

        # Swimming golden fish inside
        fx = bx + int(math.sin(time.perf_counter() * 3.2) * 7)
        fy = by + 4 + int(math.cos(time.perf_counter() * 4.0) * 3)
        pygame.draw.ellipse(surface, COLOR_AMBER, (fx - 8, fy - 4, 14, 8))
        pygame.draw.polygon(surface, COLOR_AMBER, [(fx - 7, fy), (fx - 13, fy - 5), (fx - 13, fy + 5)])
        pygame.draw.circle(surface, COLOR_BG, (fx + 3, fy - 1), 1)

    def _draw_characters(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        floor_y = int(h * 0.78)
        start_x = int(w * 0.10)
        end_x = int(w * 0.85)
        span = max(1, end_x - start_x)

        cat_norm = state.progress_ratio + (
            state.cat_chase_offset if state.phase == Phase.BUSTED else 0.0
        )
        cat_x = start_x + int(span * cat_norm)
        bob = int(abs(math.sin(state.walk_cycle)) * 4) if state.phase in (Phase.CREEPING, Phase.CLEAR, Phase.BUSTED) else 0
        cat_y = floor_y - 24 - bob

        self._draw_chubby_cat(surface, cat_x, cat_y, state.walk_cycle, state.phase == Phase.FREEZE)

        if state.phase == Phase.BUSTED:
            dog_x = start_x + int(span * state.dog_chase_x)
            dog_y = floor_y - 30 - int(abs(math.cos(state.walk_cycle)) * 6)
            self._draw_guard_dog(surface, dog_x, dog_y, state.walk_cycle)

    def _draw_chubby_cat(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        walk_cycle: float,
        frozen: bool,
    ) -> None:
        # Animated tail
        tail_wag = math.sin(walk_cycle * 0.8) * 8 if not frozen else 0
        pygame.draw.line(
            surface,
            COLOR_WHITE,
            (cx - 24, cy + 2),
            (cx - 42, int(cy - 14 + tail_wag)),
            5,
        )

        # Legs
        stride = int(math.sin(walk_cycle) * 7) if not frozen else 0
        for lx, off in [(-14, stride), (-4, -stride), (8, -stride), (18, stride)]:
            pygame.draw.rect(surface, COLOR_WHITE, (cx + lx - 3, cy + 12, 6, 12 + max(-4, off // 2)))

        # Chubby body
        body_rect = pygame.Rect(cx - 28, cy - 16, 52, 34)
        pygame.draw.ellipse(surface, COLOR_WHITE, body_rect)
        pygame.draw.ellipse(surface, COLOR_BG, body_rect, 2)

        # Head
        hx, hy = cx + 22, cy - 12
        pygame.draw.circle(surface, COLOR_WHITE, (hx, hy), 16)
        pygame.draw.circle(surface, COLOR_BG, (hx, hy), 16, 2)

        # Pointed ears
        pygame.draw.polygon(surface, COLOR_WHITE, [(hx - 12, hy - 10), (hx - 6, hy - 26), (hx, hy - 14)])
        pygame.draw.polygon(surface, COLOR_WHITE, [(hx + 2, hy - 14), (hx + 10, hy - 26), (hx + 14, hy - 10)])

        # Cyberpunk night-vision goggles
        goggle_color = COLOR_RED if frozen else COLOR_GREEN
        pygame.draw.rect(surface, COLOR_BG, (hx - 2, hy - 7, 17, 9))
        pygame.draw.rect(surface, goggle_color, (hx, hy - 5, 6, 5))
        pygame.draw.rect(surface, goggle_color, (hx + 8, hy - 5, 6, 5))

        # Whiskers
        pygame.draw.line(surface, COLOR_BG, (hx + 12, hy + 2), (hx + 22, hy), 1)
        pygame.draw.line(surface, COLOR_BG, (hx + 12, hy + 5), (hx + 22, hy + 6), 1)

    def _draw_guard_dog(
        self, surface: pygame.Surface, dx: int, dy: int, walk_cycle: float
    ) -> None:
        stride = int(math.sin(walk_cycle) * 10)
        for lx, off in [(-20, stride), (-8, -stride), (10, -stride), (22, stride)]:
            pygame.draw.line(
                surface,
                (170, 120, 90),
                (dx + lx, dy + 12),
                (dx + lx + off // 2, dy + 28),
                5,
            )

        # Muscular body
        pygame.draw.rect(surface, (150, 95, 65), (dx - 28, dy - 14, 56, 30), border_radius=6)
        pygame.draw.rect(surface, COLOR_WHITE, (dx - 28, dy - 14, 56, 30), 2, border_radius=6)

        # Spiked security collar
        pygame.draw.rect(surface, COLOR_RED, (dx + 16, dy - 14, 8, 24))

        # Head & snout
        pygame.draw.rect(surface, (150, 95, 65), (dx + 22, dy - 22, 26, 22), border_radius=4)
        pygame.draw.rect(surface, (110, 65, 45), (dx + 40, dy - 14, 16, 12), border_radius=3)
        # Glowing red cyber eye
        pygame.draw.circle(surface, COLOR_RED, (dx + 38, dy - 16), 4)
        # Ears
        pygame.draw.polygon(surface, (90, 50, 35), [(dx + 24, dy - 22), (dx + 28, dy - 34), (dx + 34, dy - 22)])

    def _draw_lasers(
        self, surface: pygame.Surface, w: int, h: int, state: GameState
    ) -> None:
        if state.phase not in (Phase.WARNING, Phase.FREEZE, Phase.CLEAR, Phase.BUSTED):
            return

        floor_y = int(h * 0.78)
        if state.phase == Phase.CLEAR:
            beam_color = COLOR_GREEN
            glow_color = COLOR_GREEN_DIM
        elif state.phase == Phase.WARNING:
            beam_color = COLOR_AMBER
            glow_color = (90, 60, 0)
        else:
            beam_color = COLOR_RED
            glow_color = COLOR_RED_DIM

        # Sweeping vertical & diagonal security laser grid
        num_beams = 6
        for i in range(num_beams):
            offset = ((i / num_beams) + state.laser_sweep_offset * 0.25) % 1.0
            x_top = int(w * (0.12 + offset * 0.76))
            x_bot = int(w * (0.08 + ((offset + 0.08) % 1.0) * 0.84))
            pygame.draw.line(surface, glow_color, (x_top, 88), (x_bot, floor_y), 8)
            pygame.draw.line(surface, beam_color, (x_top, 88), (x_bot, floor_y), 2)

        # Horizontal cross-beams
        for row_frac in (0.34, 0.52, 0.66):
            ly = int(h * row_frac)
            pygame.draw.line(surface, glow_color, (0, ly), (w, ly), 6)
            pygame.draw.line(surface, beam_color, (0, ly), (w, ly), 2)

    def _draw_top_hud(
        self, surface: pygame.Surface, w: int, state: GameState, api_online: bool
    ) -> None:
        hud_w = max(460, w - 300)
        hud_rect = pygame.Rect(18, 16, hud_w, 66)
        self._draw_brutalist_box(surface, hud_rect, fill=COLOR_BG, border=COLOR_WHITE, shadow=COLOR_WHITE, shadow_off=5)

        # Score
        score_lbl = self.font_xs.render("SCORE", True, (170, 170, 180))
        score_val = self.font_md.render(f"{state.score:06d}", True, COLOR_WHITE)
        surface.blit(score_lbl, (34, 24))
        surface.blit(score_val, (34, 42))

        # Distance & progress bar
        dist_lbl = self.font_xs.render(
            f"HEIST DISTANCE: {state.distance_meters:04.1f}M / {state.goal_meters:.0f}M",
            True,
            COLOR_GOLD,
        )
        surface.blit(dist_lbl, (165, 24))
        bar_rect = pygame.Rect(165, 45, max(120, hud_w - 310), 18)
        pygame.draw.rect(surface, COLOR_GRAY_DARK, bar_rect)
        fill_w = int(bar_rect.width * state.progress_ratio)
        if fill_w > 0:
            pygame.draw.rect(surface, COLOR_GREEN, (bar_rect.x, bar_rect.y, fill_w, bar_rect.height))
        pygame.draw.rect(surface, COLOR_WHITE, bar_rect, 2)

        # Streak & Backend badge
        right_x = hud_rect.right - 132
        streak_lbl = self.font_xs.render(f"STREAK x{state.streak}", True, COLOR_GREEN if state.streak > 0 else COLOR_WHITE)
        api_text = "API: ONLINE" if api_online else "DB: LOCAL"
        api_color = COLOR_GREEN if api_online else COLOR_AMBER
        api_lbl = self.font_xs.render(api_text, True, api_color)
        surface.blit(streak_lbl, (right_x, 25))
        surface.blit(api_lbl, (right_x, 47))

    def _draw_cctv_feed(
        self,
        surface: pygame.Surface,
        w: int,
        state: GameState,
        cv_sample: MotionSample,
    ) -> None:
        box_w, box_h = 248, 186
        box_x = w - box_w - 22
        box_y = 16
        border_col = COLOR_RED if state.phase == Phase.FREEZE else COLOR_WHITE
        shadow_col = COLOR_RED if state.phase == Phase.FREEZE else COLOR_WHITE

        box_rect = pygame.Rect(box_x, box_y, box_w, box_h + 44)
        self._draw_brutalist_box(surface, box_rect, fill=COLOR_BG, border=border_col, shadow=shadow_col, shadow_off=5)

        # Blit live OpenCV / simulation RGB frame
        try:
            frame_surf = pygame.surfarray.make_surface(np.transpose(cv_sample.frame_rgb, (1, 0, 2)))
            scaled = pygame.transform.scale(frame_surf, (box_w - 8, box_h - 8))
            surface.blit(scaled, (box_x + 4, box_y + 4))
        except Exception:
            pygame.draw.rect(surface, COLOR_GRAY_DARK, (box_x + 4, box_y + 4, box_w - 8, box_h - 8))

        # Blinking REC indicator
        blink = int(time.perf_counter() * 2.5) % 2 == 0
        if blink:
            pygame.draw.circle(surface, COLOR_RED, (box_x + 18, box_y + 18), 6)
        rec_txt = "SIM" if cv_sample.is_simulation else "REC"
        rec_lbl = self.font_xs.render(f"{rec_txt} // {cv_sample.status_label}", True, COLOR_WHITE)
        pygame.draw.rect(surface, COLOR_BG, (box_x + 28, box_y + 10, rec_lbl.get_width() + 8, 16))
        surface.blit(rec_lbl, (box_x + 32, box_y + 11))

        # Motion delta meter at bottom of CCTV box
        meter_y = box_y + box_h + 4
        meter_rect = pygame.Rect(box_x + 8, meter_y + 16, box_w - 16, 14)
        delta_lbl = self.font_xs.render(
            f"MOTION: {cv_sample.motion_score:04.1f} / THR {state.config.motion_threshold:.0f}",
            True,
            COLOR_RED if cv_sample.motion_score > state.config.motion_threshold else COLOR_GREEN,
        )
        surface.blit(delta_lbl, (box_x + 8, meter_y - 1))

        pygame.draw.rect(surface, COLOR_GRAY_DARK, meter_rect)
        fill_ratio = min(1.0, cv_sample.motion_score / 40.0)
        fill_col = COLOR_RED if cv_sample.motion_score > state.config.motion_threshold else COLOR_GREEN
        pygame.draw.rect(
            surface,
            fill_col,
            (meter_rect.x, meter_rect.y, int(meter_rect.width * fill_ratio), meter_rect.height),
        )
        # Threshold tick line
        thr_x = meter_rect.x + int(meter_rect.width * min(1.0, state.config.motion_threshold / 40.0))
        pygame.draw.line(surface, COLOR_WHITE, (thr_x, meter_rect.y - 2), (thr_x, meter_rect.bottom + 2), 2)
        pygame.draw.rect(surface, COLOR_WHITE, meter_rect, 2)

    def _draw_phase_banner(
        self,
        surface: pygame.Surface,
        w: int,
        h: int,
        state: GameState,
        cv_sample: MotionSample,
    ) -> None:
        if state.phase == Phase.WARNING:
            rect = pygame.Rect(w // 2 - 260, 108, 520, 62)
            self._draw_brutalist_box(surface, rect, fill=COLOR_BG, border=COLOR_AMBER, shadow=COLOR_AMBER)
            txt = self.font_md.render("!! LASER SWEEP INCOMING - FREEZE !!", True, COLOR_AMBER)
            surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        elif state.phase == Phase.FREEZE:
            rect = pygame.Rect(w // 2 - 210, 104, 420, 86)
            self._draw_brutalist_box(surface, rect, fill=COLOR_BG, border=COLOR_RED, shadow=COLOR_RED)
            t1 = self.font_lg.render(f"FREEZE! {max(0.0, state.phase_timer):.1f}s", True, COLOR_RED)
            sub = "RELEASE [SPACE]!" if cv_sample.sim_motion_active else "STAY COMPLETELY STILL"
            t2 = self.font_sm.render(sub, True, COLOR_WHITE)
            surface.blit(t1, (rect.centerx - t1.get_width() // 2, rect.y + 12))
            surface.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + 54))

        elif state.phase == Phase.CLEAR:
            rect = pygame.Rect(w // 2 - 210, 108, 420, 58)
            self._draw_brutalist_box(surface, rect, fill=COLOR_BG, border=COLOR_GREEN, shadow=COLOR_GREEN)
            txt = self.font_md.render("ZONE CLEAR // STEALTH BONUS!", True, COLOR_GREEN)
            surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

    def _draw_menu_overlay(
        self,
        surface: pygame.Surface,
        w: int,
        h: int,
        state: GameState,
        cv_sample: MotionSample,
    ) -> None:
        box = pygame.Rect(w // 2 - 310, h // 2 - 155, 620, 290)
        self._draw_brutalist_box(surface, box, fill=COLOR_BG, border=COLOR_WHITE, shadow=COLOR_GREEN, shadow_off=8)

        title = self.font_xl.render("LASER_PAW_PATROL", True, COLOR_WHITE)
        sub = self.font_sm.render("CV ARCADE STEALTH // DON'T MOVE ON RED LASERS", True, COLOR_GREEN)
        surface.blit(title, (box.centerx - title.get_width() // 2, box.y + 24))
        surface.blit(sub, (box.centerx - sub.get_width() // 2, box.y + 84))

        lines = [
            f"CALLSIGN: {state.config.callsign}   |   CV MODE: {cv_sample.status_label}",
            f"MOTION THRESHOLD: {state.config.motion_threshold:.0f} (PRESS [ OR ] TO ADJUST)",
            "1. CAT AUTO-WALKS TOWARD THE GOLDEN FISH BOWL.",
            "2. WHEN RED LASERS SWEEP, FREEZE FOR 3.0 SECONDS.",
            "3. NO WEBCAM? PRESS [M] FOR SIM MODE & HOLD [SPACE] TO TEST MOTION.",
        ]
        for idx, line in enumerate(lines):
            col = COLOR_AMBER if idx == 4 else COLOR_WHITE
            lbl = self.font_xs.render(line, True, col)
            surface.blit(lbl, (box.x + 28, box.y + 124 + idx * 24))

        cta = pygame.Rect(box.centerx - 150, box.bottom - 52, 300, 38)
        self._draw_brutalist_box(surface, cta, fill=COLOR_GREEN, border=COLOR_WHITE, shadow=COLOR_WHITE, shadow_off=4)
        cta_txt = self.font_md.render("PRESS [ENTER] TO START", True, COLOR_BG)
        surface.blit(cta_txt, (cta.centerx - cta_txt.get_width() // 2, cta.centery - cta_txt.get_height() // 2))

    def _draw_end_overlay(
        self, surface: pygame.Surface, w: int, h: int, state: GameState, busted: bool
    ) -> None:
        accent = COLOR_RED if busted else COLOR_GOLD
        box = pygame.Rect(w // 2 - 260, h // 2 - 125, 520, 230)
        self._draw_brutalist_box(surface, box, fill=COLOR_BG, border=accent, shadow=accent, shadow_off=8)

        headline = "!! BUSTED BY GUARD DOG !!" if busted else "HEIST COMPLETE // FISH SECURED!"
        h_lbl = self.font_lg.render(headline, True, accent)
        surface.blit(h_lbl, (box.centerx - h_lbl.get_width() // 2, box.y + 24))

        stats_lines = [
            f"FINAL SCORE: {state.score}   |   MAX STREAK: x{state.max_streak}",
            f"DISTANCE: {state.distance_meters:.1f}M / {state.goal_meters:.0f}M   |   PEAK MOTION: {state.motion_peak:.1f}",
            "RUN SAVED TO BACKEND DATABASE.",
        ]
        for idx, line in enumerate(stats_lines):
            col = COLOR_GREEN if idx == 2 else COLOR_WHITE
            lbl = self.font_sm.render(line, True, col)
            surface.blit(lbl, (box.centerx - lbl.get_width() // 2, box.y + 82 + idx * 28))

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

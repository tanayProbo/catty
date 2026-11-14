from __future__ import annotations

import sys
from typing import Optional
import pygame

from audio_engine import AudioEngine
from backend.client import BackendService
from backend.models import SettingsPayload
from config import (
    APP_TITLE,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MAX_LEVEL,
    MIN_HEIGHT,
    MIN_WIDTH,
    TARGET_FPS,
    GameConfig,
)
from cv_tracker import MotionTracker, PostureType
from game_state import GameState, Phase
from ui_renderer import UIRenderer

LEVEL_KEY_MAP = {
    pygame.K_1: 1,
    pygame.K_2: 2,
    pygame.K_3: 3,
    pygame.K_4: 4,
    pygame.K_5: 5,
    pygame.K_6: 6,
    pygame.K_7: 7,
    pygame.K_8: 8,
    pygame.K_9: 9,
    pygame.K_0: 10,
}


def _sync_settings(backend: BackendService, cfg: GameConfig) -> None:
    payload = SettingsPayload(
        motion_threshold=cfg.motion_threshold,
        freeze_duration=cfg.freeze_duration,
        audio_enabled=cfg.audio_enabled,
        simulation_mode=cfg.simulation_mode,
        callsign=cfg.callsign,
    )
    backend.save_settings_async(payload)


def main() -> int:
    backend = BackendService()
    backend.start()

    saved = backend.settings
    cfg = GameConfig(
        motion_threshold=saved.motion_threshold,
        freeze_duration=saved.freeze_duration,
        audio_enabled=saved.audio_enabled,
        simulation_mode=saved.simulation_mode,
        callsign=saved.callsign,
    )

    pygame.init()
    screen = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption(APP_TITLE)
    clock = pygame.time.Clock()

    audio = AudioEngine(enabled=cfg.audio_enabled)
    tracker = MotionTracker(force_simulation=cfg.simulation_mode)
    tracker.start()

    renderer = UIRenderer()
    state = GameState(
        config=cfg,
        on_sound=audio.play,
        on_reset_cv=tracker.reset_reference,
        on_submit_run=backend.submit_run_async,
    )

    kb_lane: Optional[int] = None
    running = True
    try:
        while running:
            dt = clock.tick(TARGET_FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    new_w = max(MIN_WIDTH, event.w)
                    new_h = max(MIN_HEIGHT, event.h)
                    try:
                        screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)
                    except pygame.error:
                        pass

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        state.show_leaderboard = not state.show_leaderboard
                        if state.show_leaderboard:
                            backend.refresh_async()

                    elif event.key == pygame.K_ESCAPE:
                        if state.show_leaderboard:
                            state.show_leaderboard = False
                        elif state.phase != Phase.MENU:
                            state.return_to_menu()
                        else:
                            running = False

                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if state.show_leaderboard:
                            state.show_leaderboard = False
                        elif state.phase in (Phase.MENU, Phase.BUSTED, Phase.VICTORY):
                            kb_lane = None
                            state.start_heist()

                    elif event.key == pygame.K_r:
                        if state.phase in (Phase.BUSTED, Phase.VICTORY):
                            kb_lane = None
                            state.start_heist()

                    elif event.key in LEVEL_KEY_MAP:
                        chosen = LEVEL_KEY_MAP[event.key]
                        state.select_starting_level(chosen)
                        if state.phase not in (Phase.MENU, Phase.BUSTED, Phase.VICTORY):
                            kb_lane = None
                            state.start_heist(start_level=chosen)

                    elif event.key == pygame.K_MINUS:
                        state.select_starting_level(state.config.starting_level - 1)

                    elif event.key == pygame.K_EQUALS:
                        state.select_starting_level(min(MAX_LEVEL, state.config.starting_level + 1))

                    elif event.key in (pygame.K_w, pygame.K_UP, pygame.K_SPACE):
                        if state.phase != Phase.FREEZE:
                            state.trigger_jump()

                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        state.trigger_crouch_roll()

                    elif event.key in (pygame.K_a, pygame.K_LEFT):
                        state.shift_lane(-1)
                        kb_lane = state.target_lane

                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        state.shift_lane(1)
                        kb_lane = state.target_lane

                    elif event.key == pygame.K_c:
                        tracker.calibrate_neutral()
                        state.action_banner = "CAMERA NEUTRAL POSTURE CALIBRATED"
                        state.action_banner_timer = 1.5

                    elif event.key == pygame.K_m:
                        cfg.simulation_mode = not cfg.simulation_mode
                        tracker.set_simulation_mode(cfg.simulation_mode)
                        _sync_settings(backend, cfg)

                    elif event.key == pygame.K_LEFTBRACKET:
                        cfg.motion_threshold = max(2.0, cfg.motion_threshold - 2.0)
                        _sync_settings(backend, cfg)

                    elif event.key == pygame.K_RIGHTBRACKET:
                        cfg.motion_threshold = min(60.0, cfg.motion_threshold + 2.0)
                        _sync_settings(backend, cfg)

            keys = pygame.key.get_pressed()
            tracker.set_simulated_motion(bool(keys[pygame.K_SPACE]))

            posture_ovr: Optional[PostureType] = None
            if keys[pygame.K_w] or keys[pygame.K_UP] or (keys[pygame.K_SPACE] and state.phase != Phase.FREEZE):
                posture_ovr = "JUMP"
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                posture_ovr = "CROUCH"
            tracker.set_pose_overrides(kb_lane, posture_ovr)

            cv_sample = tracker.get_sample()
            active_lane = kb_lane if (kb_lane is not None and cv_sample.is_simulation) else cv_sample.human_lane
            state.update(
                dt,
                cv_sample.motion_score,
                cv_sample.is_simulation,
                human_lane=active_lane,
                human_posture=cv_sample.human_posture,
            )

            leaderboard, stats, api_online = backend.get_snapshot()
            try:
                renderer.render(screen, state, cv_sample, leaderboard, stats, api_online)
                pygame.display.flip()
            except pygame.error:
                pass
    finally:
        tracker.stop()
        backend.stop()
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Main entry point for LASER_PAW_PATROL — CV Arcade Stealth Game."""

from __future__ import annotations

import sys
import pygame

from audio_engine import AudioEngine
from backend.client import BackendService
from backend.models import SettingsPayload
from config import (
    APP_TITLE,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MIN_HEIGHT,
    MIN_WIDTH,
    TARGET_FPS,
    GameConfig,
)
from cv_tracker import MotionTracker
from game_state import GameState, Phase
from ui_renderer import UIRenderer


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
                            state.start_heist()

                    elif event.key == pygame.K_r:
                        if state.phase in (Phase.BUSTED, Phase.VICTORY):
                            state.start_heist()

                    elif event.key == pygame.K_m:
                        cfg.simulation_mode = not cfg.simulation_mode
                        tracker.set_simulation_mode(cfg.simulation_mode)
                        _sync_settings(backend, cfg)

                    elif event.key == pygame.K_a:
                        cfg.audio_enabled = not cfg.audio_enabled
                        audio.enabled = cfg.audio_enabled
                        _sync_settings(backend, cfg)

                    elif event.key == pygame.K_LEFTBRACKET:
                        cfg.motion_threshold = max(2.0, cfg.motion_threshold - 2.0)
                        _sync_settings(backend, cfg)

                    elif event.key == pygame.K_RIGHTBRACKET:
                        cfg.motion_threshold = min(60.0, cfg.motion_threshold + 2.0)
                        _sync_settings(backend, cfg)

            keys = pygame.key.get_pressed()
            tracker.set_simulated_motion(bool(keys[pygame.K_SPACE]))

            cv_sample = tracker.get_sample()
            state.update(dt, cv_sample.motion_score, cv_sample.is_simulation)

            leaderboard, stats, api_online = backend.get_snapshot()
            try:
                renderer.render(screen, state, cv_sample, leaderboard, stats, api_online)
                pygame.display.flip()
            except pygame.error:
                # Recover gracefully from transient surface/display errors during resize
                pass
    finally:
        tracker.stop()
        backend.stop()
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())

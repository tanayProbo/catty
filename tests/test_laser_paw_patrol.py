"""Automated test suite for LASER_PAW_PATROL (CV tracker, game state, renderer, and backend)."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pygame
from fastapi.testclient import TestClient

from audio_engine import AudioEngine
from backend.database import DatabaseManager
from backend.models import RunCreate, SettingsPayload, TelemetryEventIn
from backend.server import create_app
from config import GameConfig
from cv_tracker import MotionTracker
from game_state import GameState, Phase
from ui_renderer import UIRenderer


class TestLaserPawPatrol(unittest.TestCase):
    """End-to-end unit and integration tests for LASER_PAW_PATROL."""

    def test_database_and_fastapi_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_heist.db")
            app = create_app(db_path=db_path)
            client = TestClient(app)

            # Health check
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "online")

            # Create run with telemetry
            payload = RunCreate(
                callsign="shadow_cat",
                score=1850,
                distance_meters=50.0,
                max_streak=4,
                motion_peak=9.5,
                cv_mode="simulation",
                outcome="VICTORY",
                duration_sec=28.4,
                telemetry=[
                    TelemetryEventIn(
                        event_type="LASER_SWEEP",
                        motion_delta=2.1,
                        heist_distance=15.0,
                        elapsed_sec=7.0,
                    ),
                    TelemetryEventIn(
                        event_type="FREEZE_PASS",
                        motion_delta=3.4,
                        heist_distance=15.0,
                        elapsed_sec=10.0,
                    ),
                ],
            )
            post_resp = client.post("/api/runs", json=payload.model_dump())
            self.assertEqual(post_resp.status_code, 201)
            created = post_resp.json()
            self.assertEqual(created["callsign"], "SHADOW_CAT")
            self.assertEqual(created["score"], 1850)

            # Leaderboard & run detail
            lb_resp = client.get("/api/leaderboard")
            self.assertEqual(lb_resp.status_code, 200)
            self.assertEqual(len(lb_resp.json()), 1)

            detail_resp = client.get(f"/api/runs/{created['id']}")
            self.assertEqual(detail_resp.status_code, 200)
            self.assertEqual(len(detail_resp.json()["telemetry"]), 2)

            # Stats summary
            stats_resp = client.get("/api/stats")
            self.assertEqual(stats_resp.status_code, 200)
            self.assertEqual(stats_resp.json()["victories"], 1)
            self.assertEqual(stats_resp.json()["win_rate"], 100.0)

            # Settings update
            new_settings = SettingsPayload(
                motion_threshold=20.0,
                freeze_duration=3.0,
                audio_enabled=False,
                simulation_mode=True,
                callsign="cyber_kitty",
            )
            put_resp = client.put("/api/settings", json=new_settings.model_dump())
            self.assertEqual(put_resp.status_code, 200)
            self.assertEqual(put_resp.json()["callsign"], "CYBER_KITTY")
            self.assertEqual(put_resp.json()["motion_threshold"], 20.0)

    def test_cv_motion_tracker_simulation_and_absdiff(self) -> None:
        tracker = MotionTracker(force_simulation=True)
        # Static frames should produce zero/low motion delta after baseline
        static_a = tracker._generate_sim_frame(motion_active=False)
        static_b = tracker._generate_sim_frame(motion_active=False)
        tracker._compute_motion(static_a)
        score_still, rgb = tracker._compute_motion(static_b)
        self.assertEqual(score_still, 0.0)
        self.assertEqual(rgb.shape, (tracker.height, tracker.width, 3))

        # Simulated motion frame should trigger significant absdiff delta
        moving = tracker._generate_sim_frame(motion_active=True)
        score_moving, _ = tracker._compute_motion(moving)
        self.assertGreater(score_moving, 5.0)

    def test_game_state_freeze_pass_and_busted(self) -> None:
        cfg = GameConfig(motion_threshold=14.0, freeze_duration=3.0)
        submitted: list[RunCreate] = []
        state = GameState(config=cfg, on_submit_run=submitted.append)

        state.start_heist()
        self.assertEqual(state.phase, Phase.CREEPING)

        # Force warning -> freeze transition
        state.next_sweep_timer = 0.05
        state.update(0.1, motion_score=0.0, is_simulation=True)
        self.assertEqual(state.phase, Phase.WARNING)

        for _ in range(10):
            state.update(0.1, motion_score=0.0, is_simulation=True)
        self.assertEqual(state.phase, Phase.FREEZE)

        # Hold still through the 3.0s freeze window -> CLEAR phase and streak +1
        for _ in range(31):
            state.update(0.1, motion_score=2.0, is_simulation=True)
        self.assertEqual(state.phase, Phase.CLEAR)
        self.assertEqual(state.streak, 1)

        # Enter another freeze and exceed motion threshold -> BUSTED
        state._enter_freeze()
        state.update(0.2, motion_score=0.0, is_simulation=True)  # pass 0.15s grace
        state.update(0.1, motion_score=28.0, is_simulation=True)
        self.assertEqual(state.phase, Phase.BUSTED)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0].outcome, "BUSTED")

    def test_headless_ui_renderer_and_audio(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        try:
            surface = pygame.Surface((1280, 760))
            renderer = UIRenderer()
            audio = AudioEngine(enabled=True)
            audio.play("klaxon")

            tracker = MotionTracker(force_simulation=True)
            sample = tracker.get_sample()
            state = GameState()

            with tempfile.TemporaryDirectory() as tmpdir:
                db = DatabaseManager(os.path.join(tmpdir, "ui.db"))
                for phase in (
                    Phase.MENU,
                    Phase.CREEPING,
                    Phase.WARNING,
                    Phase.FREEZE,
                    Phase.CLEAR,
                    Phase.BUSTED,
                    Phase.VICTORY,
                ):
                    state.phase = phase
                    renderer.render(
                        surface,
                        state,
                        sample,
                        db.get_leaderboard(),
                        db.get_stats(),
                        api_online=True,
                    )
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()

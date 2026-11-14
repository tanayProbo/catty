from __future__ import annotations

import os
import tempfile
import unittest

import pygame
from fastapi.testclient import TestClient

from audio_engine import AudioEngine
from backend.database import DatabaseManager
from backend.models import RunCreate, SettingsPayload, TelemetryEventIn
from backend.server import create_app
from config import MAX_LEVEL, GameConfig
from cv_tracker import MotionTracker
from game_state import CorridorObstacle, GameState, Phase
from ui_renderer import UIRenderer


class TestLaserPawPatrol(unittest.TestCase):
    def test_database_and_fastapi_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_heist.db")
            app = create_app(db_path=db_path)
            client = TestClient(app)

            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "online")

            payload = RunCreate(
                callsign="shadow_cat",
                score=1850,
                distance_meters=80.0,
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

            lb_resp = client.get("/api/leaderboard")
            self.assertEqual(lb_resp.status_code, 200)
            self.assertEqual(len(lb_resp.json()), 1)

            detail_resp = client.get(f"/api/runs/{created['id']}")
            self.assertEqual(detail_resp.status_code, 200)
            self.assertEqual(len(detail_resp.json()["telemetry"]), 2)

            stats_resp = client.get("/api/stats")
            self.assertEqual(stats_resp.status_code, 200)
            self.assertEqual(stats_resp.json()["victories"], 1)

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

    def test_cv_motion_and_human_pose_tracker(self) -> None:
        tracker = MotionTracker(force_simulation=True)
        static_a = tracker._generate_sim_frame(motion_active=False)
        static_b = tracker._generate_sim_frame(motion_active=False)
        tracker._compute_motion(static_a)
        score_still, rgb = tracker._compute_motion(static_b)
        self.assertLessEqual(score_still, 2.0)
        self.assertEqual(rgb.shape, (tracker.height, tracker.width, 3))

        tracker.set_pose_overrides(lane=-1, posture="CROUCH")
        crouch_frame = tracker._generate_sim_frame(motion_active=False, lane_override=-1, posture_override="CROUCH")
        tracker._compute_motion(crouch_frame)
        sample = tracker.get_sample()
        self.assertEqual(sample.human_lane, -1)
        self.assertEqual(sample.human_posture, "CROUCH")

        tracker.set_pose_overrides(lane=None, posture=None)
        base_frame = tracker._generate_sim_frame(motion_active=False)
        tracker._compute_motion_and_pose(base_frame, is_sim=False, lane_override=None, posture_override=None)
        jump_frame = tracker._generate_sim_frame(motion_active=False, posture_override="JUMP")
        _, _, _, _, _, detected_posture = tracker._compute_motion_and_pose(
            jump_frame, is_sim=False, lane_override=None, posture_override=None
        )
        self.assertEqual(detected_posture, "JUMP")

    def test_cat_pov_bounce_jump_crouch_roll_and_obstacles(self) -> None:
        state = GameState()
        state.start_heist()
        state.obstacles = [
            CorridorObstacle(kind="HIGH_GATE", distance_m=1.0, lane=0),
            CorridorObstacle(kind="LOW_BEAM", distance_m=3.5, lane=0),
            CorridorObstacle(kind="SIDE_BLOCK", distance_m=6.0, lane=0),
        ]

        state.distance_meters = 0.95
        state.update(0.05, motion_score=0.0, is_simulation=True, human_lane=0, human_posture="CROUCH")
        self.assertTrue(state.is_rolling)
        self.assertTrue(state.obstacles[0].cleared)
        self.assertNotEqual(state.phase, Phase.BUSTED)

        state.distance_meters = 3.45
        state.update(0.05, motion_score=0.0, is_simulation=True, human_lane=0, human_posture="JUMP")
        self.assertGreater(state.cat_jump_height, 0.0)
        self.assertTrue(state.obstacles[1].cleared)
        self.assertNotEqual(state.phase, Phase.BUSTED)

        state.distance_meters = 5.90
        for _ in range(5):
            state.update(0.04, motion_score=0.0, is_simulation=True, human_lane=1, human_posture="STAND")
        self.assertTrue(state.obstacles[2].cleared)
        self.assertNotEqual(state.phase, Phase.BUSTED)

    def test_ten_level_progression_to_victory(self) -> None:
        state = GameState()
        state.start_heist(start_level=1)
        self.assertEqual(state.level, 1)
        self.assertEqual(MAX_LEVEL, 10)

        for expected_lvl in range(1, MAX_LEVEL):
            self.assertEqual(state.level, expected_lvl)
            state.distance_meters = state.goal_meters
            state.update(0.05, motion_score=0.0, is_simulation=True)
            self.assertEqual(state.level, expected_lvl + 1)
            self.assertEqual(state.phase, Phase.CREEPING)

        # Completing level 10 triggers VICTORY
        self.assertEqual(state.level, 10)
        state.distance_meters = state.goal_meters
        state.update(0.05, motion_score=0.0, is_simulation=True)
        self.assertEqual(state.phase, Phase.VICTORY)

    def test_game_state_freeze_pass_and_busted(self) -> None:
        cfg = GameConfig(motion_threshold=14.0, freeze_duration=2.6)
        submitted: list[RunCreate] = []
        state = GameState(config=cfg, on_submit_run=submitted.append)

        state.start_heist()
        self.assertEqual(state.phase, Phase.CREEPING)

        state.next_sweep_timer = 0.05
        state.update(0.1, motion_score=0.0, is_simulation=True)
        self.assertEqual(state.phase, Phase.WARNING)

        for _ in range(10):
            state.update(0.1, motion_score=0.0, is_simulation=True)
        self.assertEqual(state.phase, Phase.FREEZE)

        for _ in range(28):
            state.update(0.1, motion_score=2.0, is_simulation=True)
        self.assertEqual(state.phase, Phase.CLEAR)
        self.assertEqual(state.streak, 1)

        state._enter_freeze()
        state.update(0.2, motion_score=0.0, is_simulation=True)
        state.update(0.1, motion_score=28.0, is_simulation=True)
        self.assertEqual(state.phase, Phase.BUSTED)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0].outcome, "BUSTED")

    def test_headless_ui_renderer_cat_pov_and_2d_pip(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        try:
            surface = pygame.Surface((1280, 760))
            renderer = UIRenderer()
            audio = AudioEngine(enabled=True)
            audio.play("jump")
            audio.play("roll")
            audio.play("level_up")

            tracker = MotionTracker(force_simulation=True)
            sample = tracker.get_sample()
            state = GameState()
            state.start_heist()

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
                    state.trigger_crouch_roll()
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

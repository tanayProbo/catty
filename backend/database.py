from __future__ import annotations

from contextlib import contextmanager
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Iterator, Optional

from backend.models import (
    RunCreate,
    RunDetail,
    RunRecord,
    SettingsPayload,
    StatsSummary,
    TelemetryRecord,
)


class DatabaseManager:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    callsign TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    distance_meters REAL NOT NULL,
                    max_streak INTEGER NOT NULL DEFAULT 0,
                    motion_peak REAL NOT NULL DEFAULT 0.0,
                    cv_mode TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    duration_sec REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_runs_score
                    ON runs (score DESC, distance_meters DESC);

                CREATE TABLE IF NOT EXISTS telemetry_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    motion_delta REAL NOT NULL DEFAULT 0.0,
                    heist_distance REAL NOT NULL DEFAULT 0.0,
                    elapsed_sec REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    motion_threshold REAL NOT NULL DEFAULT 14.0,
                    freeze_duration REAL NOT NULL DEFAULT 3.0,
                    audio_enabled INTEGER NOT NULL DEFAULT 1,
                    simulation_mode INTEGER NOT NULL DEFAULT 0,
                    callsign TEXT NOT NULL DEFAULT 'AGENT_PAW',
                    updated_at TEXT NOT NULL
                );
                """
            )
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT OR IGNORE INTO settings
                    (id, motion_threshold, freeze_duration, audio_enabled, simulation_mode, callsign, updated_at)
                VALUES (1, 14.0, 3.0, 1, 0, 'AGENT_PAW', ?)
                """,
                (now,),
            )

    def create_run(self, payload: RunCreate) -> RunRecord:
        now = datetime.now(timezone.utc).isoformat()
        clean_callsign = payload.callsign.strip().upper()[:24] or "AGENT_PAW"
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs
                    (callsign, score, distance_meters, max_streak, motion_peak,
                     cv_mode, outcome, duration_sec, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_callsign,
                    payload.score,
                    round(payload.distance_meters, 2),
                    payload.max_streak,
                    round(payload.motion_peak, 2),
                    payload.cv_mode,
                    payload.outcome,
                    round(payload.duration_sec, 2),
                    now,
                ),
            )
            run_id = int(cursor.lastrowid)
            if payload.telemetry:
                conn.executemany(
                    """
                    INSERT INTO telemetry_events
                        (run_id, event_type, motion_delta, heist_distance, elapsed_sec, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            ev.event_type,
                            round(ev.motion_delta, 2),
                            round(ev.heist_distance, 2),
                            round(ev.elapsed_sec, 2),
                            now,
                        )
                        for ev in payload.telemetry
                    ],
                )
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return RunRecord(**dict(row))

    def get_leaderboard(self, limit: int = 10) -> list[RunRecord]:
        safe_limit = max(1, min(limit, 100))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM runs
                ORDER BY score DESC, distance_meters DESC, id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
            return [RunRecord(**dict(r)) for r in rows]

    def get_run_detail(self, run_id: int) -> Optional[RunDetail]:
        with self._lock, self._connect() as conn:
            run_row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run_row is None:
                return None
            tel_rows = conn.execute(
                "SELECT * FROM telemetry_events WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
            return RunDetail(
                **dict(run_row),
                telemetry=[TelemetryRecord(**dict(t)) for t in tel_rows],
            )

    def get_stats(self) -> StatsSummary:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_runs,
                    COALESCE(SUM(CASE WHEN outcome = 'VICTORY' THEN 1 ELSE 0 END), 0) AS victories,
                    COALESCE(SUM(CASE WHEN outcome = 'BUSTED' THEN 1 ELSE 0 END), 0) AS busted,
                    COALESCE(MAX(score), 0) AS best_score,
                    COALESCE(MAX(distance_meters), 0.0) AS longest_distance,
                    COALESCE(AVG(motion_peak), 0.0) AS avg_motion_peak
                FROM runs
                """
            ).fetchone()
            total = int(row["total_runs"])
            victories = int(row["victories"])
            win_rate = round((victories / total) * 100.0, 1) if total > 0 else 0.0
            return StatsSummary(
                total_runs=total,
                victories=victories,
                busted=int(row["busted"]),
                win_rate=win_rate,
                best_score=int(row["best_score"]),
                longest_distance=round(float(row["longest_distance"]), 2),
                avg_motion_peak=round(float(row["avg_motion_peak"]), 2),
            )

    def get_settings(self) -> SettingsPayload:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM settings WHERE id = 1").fetchone()
            return SettingsPayload(
                motion_threshold=float(row["motion_threshold"]),
                freeze_duration=float(row["freeze_duration"]),
                audio_enabled=bool(row["audio_enabled"]),
                simulation_mode=bool(row["simulation_mode"]),
                callsign=str(row["callsign"]),
            )

    def update_settings(self, payload: SettingsPayload) -> SettingsPayload:
        now = datetime.now(timezone.utc).isoformat()
        clean_callsign = payload.callsign.strip().upper()[:24] or "AGENT_PAW"
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE settings
                SET motion_threshold = ?,
                    freeze_duration = ?,
                    audio_enabled = ?,
                    simulation_mode = ?,
                    callsign = ?,
                    updated_at = ?
                WHERE id = 1
                """,
                (
                    round(payload.motion_threshold, 2),
                    round(payload.freeze_duration, 2),
                    1 if payload.audio_enabled else 0,
                    1 if payload.simulation_mode else 0,
                    clean_callsign,
                    now,
                ),
            )
        return self.get_settings()

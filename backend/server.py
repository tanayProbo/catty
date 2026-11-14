"""FastAPI REST backend for LASER_PAW_PATROL."""

from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.database import DatabaseManager
from backend.models import (
    RunCreate,
    RunDetail,
    RunRecord,
    SettingsPayload,
    StatsSummary,
)
from config import DB_FILENAME


def create_app(db_path: str | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    resolved_db = db_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        DB_FILENAME,
    )
    db = DatabaseManager(resolved_db)

    app = FastAPI(
        title="LASER_PAW_PATROL Backend API",
        version="1.0.0",
        description="Persistence, telemetry, and leaderboard API for LASER_PAW_PATROL.",
    )
    app.state.db = db

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "online", "service": "laser_paw_patrol_backend"}

    @app.get("/api/leaderboard", response_model=list[RunRecord])
    def get_leaderboard(limit: int = Query(default=10, ge=1, le=100)) -> list[RunRecord]:
        return db.get_leaderboard(limit=limit)

    @app.post("/api/runs", response_model=RunRecord, status_code=201)
    def create_run(payload: RunCreate) -> RunRecord:
        return db.create_run(payload)

    @app.get("/api/runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: int) -> RunDetail:
        detail = db.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return detail

    @app.get("/api/stats", response_model=StatsSummary)
    def get_stats() -> StatsSummary:
        return db.get_stats()

    @app.get("/api/settings", response_model=SettingsPayload)
    def get_settings() -> SettingsPayload:
        return db.get_settings()

    @app.put("/api/settings", response_model=SettingsPayload)
    def update_settings(payload: SettingsPayload) -> SettingsPayload:
        return db.update_settings(payload)

    return app


app = create_app()

from typing import Literal
from pydantic import BaseModel, Field


class TelemetryEventIn(BaseModel):
    event_type: Literal[
        "LASER_SWEEP",
        "FREEZE_PASS",
        "MOTION_DETECTED",
        "HEIST_COMPLETE",
        "CAMERA_FALLBACK",
    ]
    motion_delta: float = Field(default=0.0, ge=0.0)
    heist_distance: float = Field(default=0.0, ge=0.0)
    elapsed_sec: float = Field(default=0.0, ge=0.0)


class RunCreate(BaseModel):
    callsign: str = Field(default="AGENT_PAW", min_length=1, max_length=24)
    score: int = Field(ge=0)
    distance_meters: float = Field(ge=0.0)
    max_streak: int = Field(default=0, ge=0)
    motion_peak: float = Field(default=0.0, ge=0.0)
    cv_mode: Literal["webcam", "simulation"] = "webcam"
    outcome: Literal["VICTORY", "BUSTED"]
    duration_sec: float = Field(default=0.0, ge=0.0)
    telemetry: list[TelemetryEventIn] = Field(default_factory=list)


class RunRecord(BaseModel):
    id: int
    callsign: str
    score: int
    distance_meters: float
    max_streak: int
    motion_peak: float
    cv_mode: str
    outcome: str
    duration_sec: float
    created_at: str


class TelemetryRecord(BaseModel):
    id: int
    run_id: int
    event_type: str
    motion_delta: float
    heist_distance: float
    elapsed_sec: float
    created_at: str


class RunDetail(RunRecord):
    telemetry: list[TelemetryRecord] = Field(default_factory=list)


class StatsSummary(BaseModel):
    total_runs: int
    victories: int
    busted: int
    win_rate: float
    best_score: int
    longest_distance: float
    avg_motion_peak: float


class SettingsPayload(BaseModel):
    motion_threshold: float = Field(default=14.0, ge=2.0, le=60.0)
    freeze_duration: float = Field(default=3.0, ge=1.0, le=10.0)
    audio_enabled: bool = True
    simulation_mode: bool = False
    callsign: str = Field(default="AGENT_PAW", min_length=1, max_length=24)

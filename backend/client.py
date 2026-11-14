from __future__ import annotations

import json
import os
import socket
import threading
import urllib.error
import urllib.request
from typing import Optional

import uvicorn

from backend.database import DatabaseManager
from backend.models import RunCreate, RunRecord, SettingsPayload, StatsSummary
from backend.server import create_app
from config import BACKEND_HOST, BACKEND_PORT, DB_FILENAME


class BackendService:
    def __init__(
        self,
        host: str = BACKEND_HOST,
        port: int = BACKEND_PORT,
        db_path: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        resolved_db = db_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            DB_FILENAME,
        )
        self.db = DatabaseManager(resolved_db)
        self._lock = threading.Lock()
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

        self.api_online: bool = False
        self.leaderboard: list[RunRecord] = []
        self.stats: StatsSummary = self.db.get_stats()
        self.settings: SettingsPayload = self.db.get_settings()

    def start(self) -> None:
        if self._is_port_open():
            self.api_online = self._ping_health()
            self.refresh_async()
            return

        app = create_app(self.db.db_path)
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._run_server,
            name="LaserPawBackend",
            daemon=True,
        )
        self._thread.start()
        self.api_online = True
        self.refresh_async()

    def _run_server(self) -> None:
        try:
            if self._server is not None:
                self._server.run()
        except Exception:
            self.api_online = False

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True

    def _is_port_open(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            return sock.connect_ex((self.host, self.port)) == 0

    def _ping_health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=0.6) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _http_json(
        self,
        path: str,
        method: str = "GET",
        body: Optional[dict] = None,
    ) -> Optional[object]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                raw = resp.read().decode("utf-8")
                self.api_online = True
                return json.loads(raw)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None

    def refresh_async(self, limit: int = 8) -> None:
        threading.Thread(
            target=self._refresh_worker,
            args=(limit,),
            daemon=True,
        ).start()

    def _refresh_worker(self, limit: int) -> None:
        lb_data = self._http_json(f"/api/leaderboard?limit={limit}")
        stats_data = self._http_json("/api/stats")
        if isinstance(lb_data, list):
            records = [RunRecord(**item) for item in lb_data]
        else:
            records = self.db.get_leaderboard(limit=limit)

        if isinstance(stats_data, dict):
            stats = StatsSummary(**stats_data)
        else:
            stats = self.db.get_stats()

        with self._lock:
            self.leaderboard = records
            self.stats = stats

    def submit_run_async(self, payload: RunCreate) -> None:
        threading.Thread(
            target=self._submit_run_worker,
            args=(payload,),
            daemon=True,
        ).start()

    def _submit_run_worker(self, payload: RunCreate) -> None:
        res = self._http_json("/api/runs", method="POST", body=payload.model_dump())
        if res is None:
            self.db.create_run(payload)
        self._refresh_worker(limit=8)

    def save_settings_async(self, payload: SettingsPayload) -> None:
        with self._lock:
            self.settings = payload
        threading.Thread(
            target=self._save_settings_worker,
            args=(payload,),
            daemon=True,
        ).start()

    def _save_settings_worker(self, payload: SettingsPayload) -> None:
        res = self._http_json("/api/settings", method="PUT", body=payload.model_dump())
        if res is None:
            self.db.update_settings(payload)

    def get_snapshot(self) -> tuple[list[RunRecord], StatsSummary, bool]:
        with self._lock:
            return list(self.leaderboard), self.stats, self.api_online

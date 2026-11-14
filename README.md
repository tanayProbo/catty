# LASER_PAW_PATROL // CV Arcade Stealth Game

**LASER_PAW_PATROL** is a high-performance, Python-based computer vision arcade stealth game built with **Pygame-ce**, **OpenCV (`cv2`)**, **NumPy**, and an embedded **FastAPI + SQLite** leaderboard/telemetry backend.

---

## Core Features & Architecture

- **Real-Time Computer Vision (`cv_tracker.py`)**:
  - Hooks into `cv2.VideoCapture(0)` in a dedicated daemon thread and calculates frame-to-frame motion delta (`cv2.absdiff` + Gaussian blur + binary threshold percentage).
  - **Zero-Crash Simulation Fallback**: Automatically falls back to a synthetic CCTV simulation mode if no webcam is connected or if a camera disconnects mid-session.
- **Neobrutalist & Cyberpunk Aesthetic (`ui_renderer.py`, `config.py`)**:
  - Pure pitch-black (`#000000`) canvas, stark white (`#FFFFFF`) 3px borders with hard offset brutalist drop shadows, neon security red (`#FF3333`) laser grids, and matrix terminal green (`#00FF66`) clear zones.
  - Procedural vector-drawn chubby stealth cat (with night-vision goggles), animated guard dog chase sequence, glowing golden fish bowl, live corner CCTV feed with blinking `REC` indicator, and top HUD.
- **Procedural Audio Engine (`audio_engine.py`)**:
  - Synthesizes warning klaxons, busted sirens, clear triad chimes, and victory fanfares in memory using NumPy waveforms and `pygame.sndarray` — zero external audio files required.
- **Embedded FastAPI + SQLite Backend (`backend/`)**:
  - Automatically launches a background FastAPI REST server (`http://127.0.0.1:8765`) backed by a thread-safe WAL SQLite database (`laser_paw_patrol.db`) to persist heist runs, CV motion telemetry events, leaderboards, and sensitivity settings.

---

## Project Structure

```text
LASER_PAW_PATROL/
├── main.py                      # One-command entry point & Pygame event loop
├── config.py                    # Neobrutalist palette, CV & gameplay constants
├── cv_tracker.py                # OpenCV absdiff motion detector & mock CCTV fallback
├── game_state.py                # Heist state machine (CREEPING, WARNING, FREEZE, CLEAR, BUSTED, VICTORY)
├── ui_renderer.py               # Neobrutalist HUD, CCTV preview, laser grid & vector sprites
├── audio_engine.py              # Procedural NumPy/Pygame sound synthesizer
├── backend/
│   ├── __init__.py
│   ├── models.py                # Pydantic schemas for runs, telemetry, stats & settings
│   ├── database.py              # Thread-safe SQLite manager
│   ├── server.py                # FastAPI REST endpoints (/health, /api/runs, /api/leaderboard, /api/stats)
│   └── client.py                # Non-blocking background server launcher & HTTP client
├── tests/
│   └── test_laser_paw_patrol.py # Automated unit & integration tests
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Game

```bash
python main.py
```

### 3. Run Automated Tests

```bash
python -m unittest discover -s tests -v
```

---

## Controls

| Key | Action |
| :--- | :--- |
| **`ENTER`** | Start Heist / Play Again |
| **`SPACE` (Hold)** | Simulate motion in Simulation Mode (or test failing a freeze) |
| **`M`** | Toggle between Live Webcam (`CAM_00`) and Simulation Mode |
| **`[` / `]`** | Decrease / Increase CV Motion Threshold sensitivity |
| **`A`** | Toggle Procedural Audio Mute |
| **`TAB`** | Open / Close Live SQLite & FastAPI Leaderboard Modal |
| **`R`** | Quick Retry after `BUSTED` or `VICTORY` |
| **`ESC`** | Return to Menu / Quit |

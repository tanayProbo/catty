# LASER_PAW_PATROL // Cat 3D POV & 2D Mini-Window Stealth Runner

**LASER_PAW_PATROL** is a high-performance Python computer vision arcade stealth runner built with **Pygame-ce**, **OpenCV (`cv2`)**, **NumPy**, and an embedded **FastAPI + SQLite** leaderboard/telemetry backend.

---

## Key Gameplay & CV Highlights

- **Bouncing First-Person Cat POV (Main Viewport)**:
  - Run down a 3-lane cyberpunk security vault from the **Cat's First-Person POV** complete with bouncing front paws, twitching whiskers, cat-ear night-vision visor HUD, and perspective 3D obstacles rushing toward you.
  - **Crouch-Roll Under Obstacles**: When you crouch (or press `S`/`DOWN`), the Cat POV camera drops low to the floor with a 360° cyber barrel-roll spin as the cat rolls underneath overhead laser gates!
  - **Jump Over Floor Tripwires**: When you jump (or press `W`/`UP`), the Cat leaps high over floor laser beams.
  - **3-Lane Side-Step**: Move left/center/right in front of the webcam (or press `A`/`D` / `LEFT`/`RIGHT`) to dodge lane security pylons.
  - **Red Laser Sweep Freeze**: Periodically, full-corridor security lasers sweep across the screen — freeze completely still for 2.6 seconds so `cv2.absdiff` motion stays below threshold!
- **Synchronized 2D Side-View Environment Mini-Window (`2D ENV // SIDE-VIEW TRACKER`)**:
  - A live Picture-in-Picture window on the right side of the screen renders the 2D side-scrolling environment in real time: watch the chubby 2D cat bounce forward, leap over low beams, curl into a spinning ball rolling underneath high gates, and race toward the golden fish bowl.
- **Real-Time Webcam Human Pose + Motion Tracking (`cv_tracker.py`)**:
  - Tracks human vertical posture (`JUMP ^` vs `CROUCH/ROLL v` vs `STAND`) and horizontal lane (`LEFT | CENTER | RIGHT`) alongside `cv2.absdiff` frame-delta motion detection, with automatic zero-crash fallback to Simulation Mode if no camera is attached.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

### Run Automated Tests

```bash
python -m unittest discover -s tests -v
```

---

## Controls (Webcam Human Pose + Keyboard)

| Input / Action | Effect |
| :--- | :--- |
| **Human Jump** or **`W` / `UP`** | Cat Jumps over low floor laser beams |
| **Human Crouch** or **`S` / `DOWN`** | Cat Crouches & Rolls underneath high laser gates |
| **Human Move Left/Right** or **`A` / `D` / `LEFT` / `RIGHT`** | Shift Cat between Left, Center, and Right lanes |
| **`C`** | Calibrate current webcam posture as neutral standing height |
| **`ENTER` / `R`** | Start Heist / Quick Retry |
| **`SPACE` (Hold)** | Simulate motion burst during Freeze in Simulation Mode |
| **`M`** | Toggle Live Webcam (`CAM_00`) vs Simulation Mode |
| **`[` / `]`** | Decrease / Increase CV Motion Threshold sensitivity |
| **`TAB`** | Open / Close Live SQLite & FastAPI Leaderboard Modal |
| **`ESC`** | Return to Menu / Quit |

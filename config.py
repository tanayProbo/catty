"""Central configuration and Neobrutalist Cyberpunk palette for LASER_PAW_PATROL."""

from dataclasses import dataclass

APP_TITLE = "LASER_PAW_PATROL // CV STEALTH ARCADE"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 760
MIN_WIDTH = 900
MIN_HEIGHT = 600
TARGET_FPS = 60

# Neobrutalist & Cyberpunk Color Palette
COLOR_BG = (0, 0, 0)               # #000000 Pure pitch-black
COLOR_WHITE = (255, 255, 255)      # #FFFFFF Stark white typography & borders
COLOR_RED = (255, 51, 51)          # #FF3333 Neon security red
COLOR_RED_DIM = (110, 15, 20)
COLOR_GREEN = (0, 255, 102)        # #00FF66 Matrix terminal green
COLOR_GREEN_DIM = (0, 90, 40)
COLOR_GOLD = (255, 215, 0)         # #FFD700 Golden fish bowl
COLOR_AMBER = (255, 170, 0)        # #FFAA00 Warning amber
COLOR_GRAY_DARK = (18, 18, 22)
COLOR_GRAY_MID = (48, 48, 56)
COLOR_SHADOW = (255, 255, 255)     # Hard brutalist offset shadow

BORDER_WIDTH = 3                   # Sharp ~2.5-3px solid borders
SHADOW_OFFSET = 6                  # Heavy offset drop shadow

# Gameplay & CV Defaults
HEIST_GOAL_METERS = 50.0
CAT_WALK_SPEED_MPS = 2.2           # Meters per second during CREEPING
FREEZE_CLEAR_BONUS_MPS = 3.2       # Extra burst forward when surviving a freeze
FREEZE_DURATION_SEC = 3.0
WARNING_DURATION_SEC = 0.85        # Brief klaxon telegraph before freeze locks in
CLEAR_DURATION_SEC = 1.15          # Green laser celebration window
SWEEP_INTERVAL_MIN = 3.2
SWEEP_INTERVAL_MAX = 5.8

# Computer Vision Defaults
CV_CAMERA_INDEX = 0
CV_FRAME_WIDTH = 320
CV_FRAME_HEIGHT = 240
DEFAULT_MOTION_THRESHOLD = 14.0    # Mean absdiff / percentage metric (0 - 100 scale)
CV_BLUR_KERNEL = (15, 15)

# Backend Defaults
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8765
DB_FILENAME = "laser_paw_patrol.db"


@dataclass
class GameConfig:
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD
    freeze_duration: float = FREEZE_DURATION_SEC
    audio_enabled: bool = True
    simulation_mode: bool = False
    callsign: str = "AGENT_PAW"

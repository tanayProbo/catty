"""Central configuration and Neobrutalist Cyberpunk palette for LASER_PAW_PATROL (Cat POV + 2D PiP Edition)."""

from dataclasses import dataclass

APP_TITLE = "LASER_PAW_PATROL // CAT POV & 2D STEALTH RUNNER"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 760
MIN_WIDTH = 960
MIN_HEIGHT = 620
TARGET_FPS = 60

# Neobrutalist & Cyberpunk Color Palette
COLOR_BG = (0, 0, 0)               # #000000 Pure pitch-black
COLOR_WHITE = (255, 255, 255)      # #FFFFFF Stark white typography & borders
COLOR_RED = (255, 51, 51)          # #FF3333 Neon security red
COLOR_RED_DIM = (110, 15, 20)
COLOR_GREEN = (0, 255, 102)        # #00FF66 Matrix terminal green
COLOR_GREEN_DIM = (0, 90, 40)
COLOR_CYAN = (0, 225, 255)         # #00E1FF Cyber visor cyan
COLOR_GOLD = (255, 215, 0)         # #FFD700 Golden fish bowl
COLOR_AMBER = (255, 170, 0)        # #FFAA00 Warning amber
COLOR_GRAY_DARK = (18, 18, 24)
COLOR_GRAY_MID = (52, 52, 64)
COLOR_SHADOW = (255, 255, 255)     # Hard brutalist offset shadow

BORDER_WIDTH = 3                   # Sharp ~2.5-3px solid borders
SHADOW_OFFSET = 6                  # Heavy offset drop shadow

# Gameplay & Cat POV Physics Defaults
HEIST_GOAL_METERS = 80.0
CAT_WALK_SPEED_MPS = 5.2           # Forward speed during active POV run
CAT_ROLL_SPEED_BOOST_MPS = 7.4     # Speed burst when crouch-rolling under obstacles
FREEZE_CLEAR_BONUS_MPS = 7.8       # Extra burst forward when surviving a freeze
FREEZE_DURATION_SEC = 2.6
WARNING_DURATION_SEC = 0.85        # Brief klaxon telegraph before freeze locks in
CLEAR_DURATION_SEC = 1.10          # Green laser celebration window
SWEEP_INTERVAL_MIN = 7.5
SWEEP_INTERVAL_MAX = 11.5

# Cat Jump, Bounce & Crouch-Roll Physics
CAT_BOUNCE_FREQ = 9.5              # Natural bouncy paw-step cadence
CAT_JUMP_DURATION_SEC = 0.95       # Responsive, generous airtime when human jumps
CAT_ROLL_DURATION_SEC = 0.80       # Roll-under-obstacle duration when human crouches
OBSTACLE_SPAWN_MIN_SEC = 1.55
OBSTACLE_SPAWN_MAX_SEC = 2.50

# Computer Vision & Human Pose Zones
CV_CAMERA_INDEX = 0
CV_FRAME_WIDTH = 320
CV_FRAME_HEIGHT = 240
DEFAULT_MOTION_THRESHOLD = 14.0    # Mean absdiff / percentage metric (0 - 100 scale)
CV_BLUR_KERNEL = (15, 15)
CV_JUMP_Y_THRESHOLD = 0.42         # Normalized Y < 0.42 (or upward impulse) -> Human JUMP detected
CV_CROUCH_Y_THRESHOLD = 0.62       # Normalized Y > 0.62 -> Human CROUCH/ROLL detected
CV_LEFT_X_THRESHOLD = 0.38         # Normalized X < 0.38 -> Human LEFT lane
CV_RIGHT_X_THRESHOLD = 0.62        # Normalized X > 0.62 -> Human RIGHT lane

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

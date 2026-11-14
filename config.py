from dataclasses import dataclass

APP_TITLE = "Laser Paw Patrol"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 760
MIN_WIDTH = 960
MIN_HEIGHT = 620
TARGET_FPS = 60

COLOR_BG = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 51, 51)
COLOR_RED_DIM = (110, 15, 20)
COLOR_GREEN = (0, 255, 102)
COLOR_GREEN_DIM = (0, 90, 40)
COLOR_CYAN = (0, 225, 255)
COLOR_GOLD = (255, 215, 0)
COLOR_AMBER = (255, 170, 0)
COLOR_GRAY_DARK = (18, 18, 24)
COLOR_GRAY_MID = (52, 52, 64)
COLOR_SHADOW = (255, 255, 255)

BORDER_WIDTH = 3
SHADOW_OFFSET = 6

HEIST_GOAL_METERS = 50.0
CAT_WALK_SPEED_MPS = 5.0
CAT_ROLL_SPEED_BOOST_MPS = 7.2
FREEZE_CLEAR_BONUS_MPS = 7.6
FREEZE_DURATION_SEC = 2.5
WARNING_DURATION_SEC = 0.85
CLEAR_DURATION_SEC = 1.10
SWEEP_INTERVAL_MIN = 7.5
SWEEP_INTERVAL_MAX = 11.0

CAT_BOUNCE_FREQ = 9.5
CAT_JUMP_DURATION_SEC = 0.95
CAT_ROLL_DURATION_SEC = 0.80
OBSTACLE_SPAWN_MIN_SEC = 1.55
OBSTACLE_SPAWN_MAX_SEC = 2.45

CV_CAMERA_INDEX = 0
CV_FRAME_WIDTH = 320
CV_FRAME_HEIGHT = 240
DEFAULT_MOTION_THRESHOLD = 14.0
CV_BLUR_KERNEL = (15, 15)
CV_JUMP_Y_THRESHOLD = 0.42
CV_CROUCH_Y_THRESHOLD = 0.62
CV_LEFT_X_THRESHOLD = 0.38
CV_RIGHT_X_THRESHOLD = 0.62

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8765
DB_FILENAME = "laser_paw_patrol.db"


@dataclass(frozen=True, slots=True)
class LevelSpec:
    number: int
    name: str
    goal_meters: float
    walk_speed_mps: float
    roll_speed_mps: float
    freeze_duration: float
    sweep_min_sec: float
    sweep_max_sec: float
    obstacle_gap_min: float
    obstacle_gap_max: float
    grid_color: tuple[int, int, int]


LEVELS: tuple[LevelSpec, ...] = (
    LevelSpec(1, "LOBBY VENTS", 45.0, 4.6, 6.6, 2.2, 8.5, 11.5, 9.5, 13.5, (30, 46, 38)),
    LevelSpec(2, "SERVICE CORRIDOR", 55.0, 4.9, 6.9, 2.3, 8.0, 10.8, 9.0, 12.8, (28, 44, 52)),
    LevelSpec(3, "SERVER HALLWAY", 65.0, 5.2, 7.2, 2.4, 7.5, 10.2, 8.5, 12.0, (36, 38, 56)),
    LevelSpec(4, "CRYO STORAGE", 75.0, 5.5, 7.5, 2.5, 7.0, 9.6, 8.0, 11.4, (24, 52, 58)),
    LevelSpec(5, "SECURITY HUB", 85.0, 5.8, 7.9, 2.6, 6.6, 9.0, 7.6, 10.8, (48, 34, 52)),
    LevelSpec(6, "BIOLAB TUNNEL", 95.0, 6.1, 8.2, 2.7, 6.2, 8.5, 7.2, 10.2, (28, 54, 36)),
    LevelSpec(7, "REACTOR WALKWAY", 105.0, 6.4, 8.6, 2.8, 5.8, 8.0, 6.8, 9.6, (56, 42, 26)),
    LevelSpec(8, "EXECUTIVE FLOOR", 115.0, 6.8, 9.0, 2.9, 5.4, 7.5, 6.4, 9.2, (52, 28, 42)),
    LevelSpec(9, "DIAMOND ATRIUM", 125.0, 7.2, 9.4, 3.0, 5.0, 7.0, 6.0, 8.6, (38, 50, 62)),
    LevelSpec(10, "THE GOLDEN VAULT", 140.0, 7.6, 9.8, 3.2, 4.6, 6.5, 5.6, 8.0, (60, 50, 22)),
)

MAX_LEVEL = len(LEVELS)


def get_level_spec(level_num: int) -> LevelSpec:
    idx = max(0, min(len(LEVELS) - 1, level_num - 1))
    return LEVELS[idx]


@dataclass(slots=True)
class GameConfig:
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD
    freeze_duration: float = FREEZE_DURATION_SEC
    audio_enabled: bool = True
    simulation_mode: bool = False
    callsign: str = "AGENT_PAW"
    starting_level: int = 1

from dataclasses import dataclass

APP_TITLE = "Laser Paw Patrol"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 760
MIN_WIDTH = 960
MIN_HEIGHT = 620
TARGET_FPS = 60

COLOR_BG = (12, 14, 20)
COLOR_WHITE = (245, 247, 250)
COLOR_RED = (255, 68, 68)
COLOR_RED_DIM = (115, 24, 30)
COLOR_GREEN = (46, 230, 125)
COLOR_GREEN_DIM = (18, 92, 50)
COLOR_CYAN = (56, 210, 245)
COLOR_GOLD = (255, 208, 45)
COLOR_AMBER = (255, 168, 36)
COLOR_GRAY_DARK = (24, 28, 38)
COLOR_GRAY_MID = (64, 72, 90)
COLOR_SHADOW = (0, 0, 0)

BORDER_WIDTH = 2
SHADOW_OFFSET = 5

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

MAX_CAT_LIVES = 9
BOUNCE_ESCALATION_INTERVAL_SEC = 10.0
MAX_BOUNCE_TIER = 5
BOUNCE_TIER_STEP = 0.22
MAX_BOUNCE_MULTIPLIER = 1.0 + (MAX_BOUNCE_TIER * BOUNCE_TIER_STEP)

CV_CAMERA_INDEX = 0
CV_FRAME_WIDTH = 320
CV_FRAME_HEIGHT = 240
DEFAULT_MOTION_THRESHOLD = 14.0
CV_BLUR_KERNEL = (15, 15)
CV_JUMP_Y_THRESHOLD = 0.28
CV_CROUCH_Y_THRESHOLD = 0.72
CV_LEFT_X_THRESHOLD = 0.32
CV_RIGHT_X_THRESHOLD = 0.68

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
    LevelSpec(1, "Anubis Foyer", 45.0, 4.6, 6.6, 2.2, 8.5, 11.5, 9.5, 13.5, (76, 62, 38)),
    LevelSpec(2, "Hall of Pillars", 55.0, 4.9, 6.9, 2.3, 8.0, 10.8, 9.0, 12.8, (82, 66, 40)),
    LevelSpec(3, "Pharaoh's Gallery", 65.0, 5.2, 7.2, 2.4, 7.5, 10.2, 8.5, 12.0, (70, 56, 44)),
    LevelSpec(4, "Canopic Vault", 75.0, 5.5, 7.5, 2.5, 7.0, 9.6, 8.0, 11.4, (64, 62, 46)),
    LevelSpec(5, "Sarcophagus Wing", 85.0, 5.8, 7.9, 2.6, 6.6, 9.0, 7.6, 10.8, (86, 64, 36)),
    LevelSpec(6, "Papyrus Archive", 95.0, 6.1, 8.2, 2.7, 6.2, 8.5, 7.2, 10.2, (74, 68, 42)),
    LevelSpec(7, "Scarab Sanctum", 105.0, 6.4, 8.6, 2.8, 5.8, 8.0, 6.8, 9.6, (90, 70, 34)),
    LevelSpec(8, "Obelisk Chamber", 115.0, 6.8, 9.0, 2.9, 5.4, 7.5, 6.4, 9.2, (84, 54, 42)),
    LevelSpec(9, "Tomb of Horus", 125.0, 7.2, 9.4, 3.0, 5.0, 7.0, 6.0, 8.6, (78, 66, 48)),
    LevelSpec(10, "Golden Reliquary", 140.0, 7.6, 9.8, 3.2, 4.6, 6.5, 5.6, 8.0, (102, 80, 32)),
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

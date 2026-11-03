"""
AeroMind Ops - Core Constants & C-MAPSS Specifications
"""

INDEX_COLUMNS = ["unit_nr", "time_cycles"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Invariant sensors in FD001 (standard deviation == 0 or nearly 0 across cycles)
FD001_INVARIANT_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

# Informative sensors exhibiting degradation patterns in FD001
FD001_INFORMATIVE_SENSORS = [
    s for s in SENSOR_COLUMNS if s not in FD001_INVARIANT_SENSORS
]

# Standard clipping threshold for Remaining Useful Life (RUL)
DEFAULT_MAX_RUL = 125

# Standard sequence window size
DEFAULT_WINDOW_SIZE = 30

# Degradation stages
STAGE_HEALTHY = 0    # RUL > 80
STAGE_DEGRADED = 1   # 30 < RUL <= 80
STAGE_CRITICAL = 2   # RUL <= 30

STAGE_NAMES = {
    STAGE_HEALTHY: "Healthy",
    STAGE_DEGRADED: "Degraded",
    STAGE_CRITICAL: "Critical",
}

# Sub-dataset metadata
DATASET_METADATA = {
    "FD001": {
        "train_trajectories": 100,
        "test_trajectories": 100,
        "conditions": 1,
        "fault_modes": 1,
        "description": "Sea level, HPC Degradation",
    },
    "FD002": {
        "train_trajectories": 260,
        "test_trajectories": 259,
        "conditions": 6,
        "fault_modes": 1,
        "description": "Six operating conditions, HPC Degradation",
    },
    "FD003": {
        "train_trajectories": 100,
        "test_trajectories": 100,
        "conditions": 1,
        "fault_modes": 2,
        "description": "Sea level, HPC & Fan Degradation",
    },
    "FD004": {
        "train_trajectories": 248,
        "test_trajectories": 249,
        "conditions": 6,
        "fault_modes": 2,
        "description": "Six operating conditions, HPC & Fan Degradation",
    },
}


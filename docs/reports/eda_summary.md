# AeroMind Ops — Exploratory Data Analysis Report

## Dataset: FD001
- **Engines:** 100 trajectories
- **Cycle Lifespans:** Min: 128 | Max: 362 | Mean: 206.3 cycles
- **Invariant Sensors (6):** sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19
- **Top Informative Sensors (by RUL correlation):**
  - `sensor_11`: Corr = -0.7752 (Increases with wear)
  - `sensor_4`: Corr = -0.7572 (Increases with wear)
  - `sensor_12`: Corr = +0.7489 (Decreases with wear)
  - `sensor_7`: Corr = +0.7330 (Decreases with wear)
  - `sensor_15`: Corr = -0.7209 (Increases with wear)
  - `sensor_21`: Corr = +0.7073 (Decreases with wear)
- **Operating Regimes (Settings):**
  - `setting_1`: 158 distinct operating points (range: [-0.0087, 0.0087])
  - `setting_2`: 13 distinct operating points (range: [-0.0006, 0.0006])
  - `setting_3`: 1 distinct operating points (range: [100.0, 100.0])

## Dataset: FD004
- **Engines:** 249 trajectories
- **Cycle Lifespans:** Min: 128 | Max: 543 | Mean: 246.0 cycles
- **Invariant Sensors (0):** None
- **Top Informative Sensors (by RUL correlation):**
  - `sensor_14`: Corr = -0.1017 (Increases with wear)
  - `sensor_16`: Corr = -0.0796 (Increases with wear)
  - `sensor_11`: Corr = -0.0684 (Increases with wear)
  - `sensor_4`: Corr = -0.0555 (Increases with wear)
  - `sensor_3`: Corr = -0.0413 (Increases with wear)
  - `sensor_17`: Corr = -0.0412 (Increases with wear)
- **Operating Regimes (Settings):**
  - `setting_1`: 536 distinct operating points (range: [0.0, 42.008])
  - `setting_2`: 105 distinct operating points (range: [0.0, 0.842])
  - `setting_3`: 2 distinct operating points (range: [60.0, 100.0])


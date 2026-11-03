from typing import List, Tuple
import numpy as np
import pandas as pd

from aeromind.core.constants import (
    DEFAULT_MAX_RUL,
    STAGE_CRITICAL,
    STAGE_DEGRADED,
    STAGE_HEALTHY,
)


def compute_rul_target(
    df: pd.DataFrame,
    max_rul: int = DEFAULT_MAX_RUL,
    is_test: bool = False,
    rul_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Computes piecewise linear Remaining Useful Life (RUL) and categorical degradation stage.
    
    Piecewise linear formulation:
        RUL(i, t) = min(T_i - t, max_rul)
    
    Degradation stages:
        - Healthy (0): RUL > 80
        - Degraded (1): 30 < RUL <= 80
        - Critical (2): RUL <= 30
    """
    df = df.copy()

    if not is_test:
        # For training: maximum cycle observed per engine represents failure point T_i
        max_cycle_per_engine = df.groupby("unit_nr")["time_cycles"].max().reset_index()
        max_cycle_per_engine.rename(columns={"time_cycles": "max_cycle"}, inplace=True)
        df = df.merge(max_cycle_per_engine, on="unit_nr", how="left")
        
        # Calculate true RUL
        df["rul_raw"] = df["max_cycle"] - df["time_cycles"]
        # Apply piecewise clipping
        df["rul"] = np.minimum(df["rul_raw"], max_rul).astype(float)
        df.drop(columns=["max_cycle"], inplace=True)
    else:
        if rul_df is None:
            raise ValueError("rul_df containing ground-truth final RUL must be provided for test set.")
        
        # Test trajectories end prior to failure
        max_cycle_per_engine = df.groupby("unit_nr")["time_cycles"].max().reset_index()
        max_cycle_per_engine.rename(columns={"time_cycles": "max_observed_cycle"}, inplace=True)
        
        # Merge with ground truth final RUL
        test_meta = max_cycle_per_engine.merge(rul_df[["unit_nr", "rul_true"]], on="unit_nr", how="left")
        df = df.merge(test_meta, on="unit_nr", how="left")
        
        # True RUL at cycle t = rul_true + (max_observed_cycle - t)
        df["rul_raw"] = df["rul_true"] + (df["max_observed_cycle"] - df["time_cycles"])
        df["rul"] = np.minimum(df["rul_raw"], max_rul).astype(float)
        df.drop(columns=["max_observed_cycle", "rul_true"], inplace=True)

    # Compute 3-class degradation stage
    conditions = [
        df["rul"] > 80,
        (df["rul"] > 30) & (df["rul"] <= 80),
        df["rul"] <= 30,
    ]
    choices = [STAGE_HEALTHY, STAGE_DEGRADED, STAGE_CRITICAL]
    df["degradation_stage"] = np.select(conditions, choices, default=STAGE_HEALTHY).astype(int)

    return df


def engine_level_train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """
    Splits dataset at the engine level to strictly prevent temporal leakage.
    Returns:
        (train_df, val_df, train_engine_ids, val_engine_ids)
    """
    unique_engines = df["unit_nr"].unique()
    rng = np.random.default_rng(random_seed)
    shuffled_engines = rng.permutation(unique_engines)
    
    num_val = int(len(unique_engines) * val_ratio)
    val_engines = sorted(shuffled_engines[:num_val].tolist())
    train_engines = sorted(shuffled_engines[num_val:].tolist())

    train_df = df[df["unit_nr"].isin(train_engines)].copy().reset_index(drop=True)
    val_df = df[df["unit_nr"].isin(val_engines)].copy().reset_index(drop=True)

    return train_df, val_df, train_engines, val_engines


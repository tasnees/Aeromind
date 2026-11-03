import numpy as np
import pandas as pd
import pytest

from aeromind.core.constants import ALL_COLUMNS
from aeromind.data.ingestion import DataIngestionValidator
from aeromind.data.target import compute_rul_target, engine_level_train_val_split


def test_monotonic_cycles_validation():
    # Valid continuous data
    df_valid = pd.DataFrame({
        "unit_nr": [1, 1, 1, 2, 2],
        "time_cycles": [1, 2, 3, 1, 2],
    })
    for col in ALL_COLUMNS[2:]:
        df_valid[col] = 1.0

    res = DataIngestionValidator.validate_dataframe(df_valid, "test")
    assert res["passed"] is True

    # Invalid non-continuous data (cycle skip: 1, 3)
    df_invalid = pd.DataFrame({
        "unit_nr": [1, 1],
        "time_cycles": [1, 3],
    })
    for col in ALL_COLUMNS[2:]:
        df_invalid[col] = 1.0

    res_inv = DataIngestionValidator.validate_dataframe(df_invalid, "test")
    assert res_inv["passed"] is False


def test_piecewise_rul_target_engineering():
    # Engine 1 has 150 cycles -> failure at cycle 150
    cycles = list(range(1, 151))
    df = pd.DataFrame({
        "unit_nr": [1] * 150,
        "time_cycles": cycles,
    })
    for col in ALL_COLUMNS[2:]:
        df[col] = 0.5

    df_target = compute_rul_target(df, max_rul=125, is_test=False)
    
    # At cycle 1: raw RUL = 149 -> clipped to 125
    assert df_target["rul"].iloc[0] == 125.0
    # At cycle 25: raw RUL = 125 -> clipped to 125
    assert df_target["rul"].iloc[24] == 125.0
    # At cycle 100: raw RUL = 50 -> 50.0
    assert df_target["rul"].iloc[99] == 50.0
    # At cycle 150: raw RUL = 0 -> 0.0
    assert df_target["rul"].iloc[-1] == 0.0


def test_engine_level_splitting_prevents_leakage():
    engines = list(range(1, 21))  # 20 engines
    records = []
    for eng in engines:
        for c in range(1, 11):
            records.append({"unit_nr": eng, "time_cycles": c})
    df = pd.DataFrame(records)

    train_df, val_df, train_eng, val_eng = engine_level_train_val_split(df, val_ratio=0.25, random_seed=42)

    # Assert exact engine sets
    assert len(train_eng) == 15
    assert len(val_eng) == 5
    # Assert zero overlap between train and validation engines
    assert len(set(train_eng).intersection(set(val_eng))) == 0
    assert len(set(train_df["unit_nr"]).intersection(set(val_df["unit_nr"]))) == 0


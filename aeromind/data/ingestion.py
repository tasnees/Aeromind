from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from aeromind.core.constants import ALL_COLUMNS, INDEX_COLUMNS, SETTING_COLUMNS, SENSOR_COLUMNS


class DataIngestionValidator:
    """Validates raw C-MAPSS dataset integrity."""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, dataset_name: str) -> Dict[str, any]:
        results = {
            "dataset_name": dataset_name,
            "total_rows": len(df),
            "num_columns": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated(subset=INDEX_COLUMNS).sum()),
            "unique_engines": int(df["unit_nr"].nunique()),
            "min_cycle": int(df["time_cycles"].min()),
            "max_cycle": int(df["time_cycles"].max()),
            "passed": True,
            "errors": [],
        }

        if results["missing_values"] > 0:
            results["passed"] = False
            results["errors"].append(f"Found {results['missing_values']} missing values.")

        if results["duplicate_rows"] > 0:
            results["passed"] = False
            results["errors"].append(f"Found {results['duplicate_rows']} duplicate (unit_nr, time_cycles) pairs.")

        # Check cycle monotonicity per engine
        for unit_nr, group in df.groupby("unit_nr"):
            cycles = group["time_cycles"].values
            if not np.all(np.diff(cycles) == 1):
                results["passed"] = False
                results["errors"].append(f"Engine {unit_nr} does not have strictly sequential continuous cycles.")
                break

        return results


def load_raw_dataset(
    dataset_name: str = "FD001",
    raw_dir: Path = Path("data/raw"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, test, and ground-truth RUL files for a given C-MAPSS sub-dataset.
    
    Returns:
        (train_df, test_df, rul_df)
    """
    search_dirs = [
        Path(raw_dir),
        Path("data/raw"),
        Path("/app/data/raw"),
        Path("../data/raw"),
        Path("."),
    ]

    train_path, test_path, rul_path = None, None, None
    for d in search_dirs:
        candidate_test = d / f"test_{dataset_name}.txt"
        if candidate_test.exists():
            test_path = candidate_test
            train_path = d / f"train_{dataset_name}.txt"
            rul_path = d / f"RUL_{dataset_name}.txt"
            break

    if test_path is None or not test_path.exists():
        raise FileNotFoundError(f"Test dataset test_{dataset_name}.txt not found in search paths: {search_dirs}")
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not rul_path.exists():
        raise FileNotFoundError(f"RUL ground truth file not found: {rul_path}")

    # Load space-delimited data files
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul_true"])

    # Cast integer identifiers
    train_df["unit_nr"] = train_df["unit_nr"].astype(int)
    train_df["time_cycles"] = train_df["time_cycles"].astype(int)
    test_df["unit_nr"] = test_df["unit_nr"].astype(int)
    test_df["time_cycles"] = test_df["time_cycles"].astype(int)
    rul_df["unit_nr"] = np.arange(1, len(rul_df) + 1, dtype=int)

    return train_df, test_df, rul_df


import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from aeromind.core.constants import DEFAULT_MAX_RUL, DEFAULT_WINDOW_SIZE
from aeromind.data.dataset import CMAPSSDataset, create_dataloaders, generate_sliding_windows
from aeromind.data.ingestion import DataIngestionValidator, load_raw_dataset
from aeromind.data.preprocessing import CMAPSSPreprocessor
from aeromind.data.target import compute_rul_target, engine_level_train_val_split


def run_data_pipeline(
    dataset_name: str = "FD001",
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    window_size: int = DEFAULT_WINDOW_SIZE,
    max_rul: int = DEFAULT_MAX_RUL,
    val_ratio: float = 0.2,
    condition_aware: bool = False,
    random_seed: int = 42,
) -> Tuple[Dict[str, CMAPSSDataset], CMAPSSPreprocessor]:
    """
    Complete reproducible data engineering pipeline:
    1. Loads raw C-MAPSS files and validates data integrity.
    2. Computes piecewise linear RUL target (clipped at max_rul) and degradation stages.
    3. Splits at the engine level (train vs validation) preventing cycle leakage.
    4. Fits normalization scaler strictly on training engines.
    5. Generates sliding sequence windows (W, D) and target tensors.
    6. Saves scaler parameters and returns PyTorch Dataset objects.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw datasets
    train_df, test_df, rul_df = load_raw_dataset(dataset_name, raw_dir=raw_dir)

    # Validation
    val_train = DataIngestionValidator.validate_dataframe(train_df, f"{dataset_name}_train")
    val_test = DataIngestionValidator.validate_dataframe(test_df, f"{dataset_name}_test")
    
    validation_report = {
        "dataset_name": dataset_name,
        "train_validation": val_train,
        "test_validation": val_test,
    }
    with open(processed_dir / f"validation_report_{dataset_name}.json", "w") as f:
        json.dump(validation_report, f, indent=2)

    if not val_train["passed"] or not val_test["passed"]:
        raise ValueError(f"Dataset integrity validation failed: {validation_report}")

    # 2. Compute RUL and degradation stages
    train_df = compute_rul_target(train_df, max_rul=max_rul, is_test=False)
    test_df = compute_rul_target(test_df, max_rul=max_rul, is_test=True, rul_df=rul_df)

    # 3. Engine-level split
    train_split, val_split, train_engines, val_engines = engine_level_train_val_split(
        train_df, val_ratio=val_ratio, random_seed=random_seed
    )

    # 4. Preprocessing & Scaling (Fit on train split only!)
    is_multi_condition = dataset_name in ["FD002", "FD004"]
    preprocessor = CMAPSSPreprocessor(
        method="standard",
        condition_aware=condition_aware or is_multi_condition,
    )
    train_scaled = preprocessor.fit_transform(train_split, dataset_name=dataset_name)
    val_scaled = preprocessor.transform(val_split)
    test_scaled = preprocessor.transform(test_df)

    # Save scaler parameters
    preprocessor.save_parameters(processed_dir / f"scaler_params_{dataset_name}.json")

    # 5. Sliding-window generation
    feature_cols = preprocessor.selected_features

    X_train, y_train, s_train, eng_train, cyc_train = generate_sliding_windows(
        train_scaled, feature_cols, window_size=window_size
    )
    X_val, y_val, s_val, eng_val, cyc_val = generate_sliding_windows(
        val_scaled, feature_cols, window_size=window_size
    )
    # Test set: extract the final window per engine for benchmark evaluation
    X_test, y_test, s_test, eng_test, cyc_test = generate_sliding_windows(
        test_scaled, feature_cols, window_size=window_size, last_window_only=True
    )

    datasets = {
        "train": CMAPSSDataset(X_train, y_train, s_train, eng_train, cyc_train),
        "val": CMAPSSDataset(X_val, y_val, s_val, eng_val, cyc_val),
        "test": CMAPSSDataset(X_test, y_test, s_test, eng_test, cyc_test),
    }

    print(
        f"[{dataset_name} Pipeline Summary]\n"
        f"  Features ({len(feature_cols)}): {feature_cols}\n"
        f"  Train: {len(datasets['train'])} windows (from {len(train_engines)} engines)\n"
        f"  Val:   {len(datasets['val'])} windows (from {len(val_engines)} engines)\n"
        f"  Test:  {len(datasets['test'])} windows (final window of {test_df['unit_nr'].nunique()} engines)"
    )

    return datasets, preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run C-MAPSS Preprocessing Pipeline")
    parser.add_argument("--dataset", type=str, default="FD001", help="Dataset name (FD001-FD004)")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--max-rul", type=int, default=DEFAULT_MAX_RUL)
    args = parser.parse_args()

    run_data_pipeline(dataset_name=args.dataset, window_size=args.window_size, max_rul=args.max_rul)


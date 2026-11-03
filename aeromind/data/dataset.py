from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from aeromind.core.constants import DEFAULT_WINDOW_SIZE


class CMAPSSDataset(Dataset):
    """
    PyTorch Dataset for sliding-window sequence data.
    Yields:
        - sequence: Tensor of shape (window_size, num_features)
        - rul: Tensor scalar (Remaining Useful Life)
        - stage: Tensor scalar (Categorical degradation stage: 0, 1, 2)
        - metadata: Dictionary containing engine_id and end_cycle
    """

    def __init__(
        self,
        features: np.ndarray,
        ruls: np.ndarray,
        stages: np.ndarray,
        engine_ids: np.ndarray,
        cycles: np.ndarray,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.ruls = torch.tensor(ruls, dtype=torch.float32)
        self.stages = torch.tensor(stages, dtype=torch.long)
        self.engine_ids = torch.tensor(engine_ids, dtype=torch.long)
        self.cycles = torch.tensor(cycles, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        return (
            self.features[idx],
            self.ruls[idx],
            self.stages[idx],
            {
                "engine_id": self.engine_ids[idx],
                "cycle": self.cycles[idx],
            },
        )


def generate_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = 1,
    last_window_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms engine time-series into sliding-window sequences.
    
    If last_window_only is True, extracts only the final window of each engine
    (standard protocol for C-MAPSS test evaluation).
    """
    feature_windows = []
    rul_targets = []
    stage_targets = []
    engine_ids = []
    end_cycles = []

    for unit_nr, group in df.groupby("unit_nr"):
        group = group.sort_values("time_cycles").reset_index(drop=True)
        num_cycles = len(group)

        if num_cycles < window_size:
            # If an engine trajectory is shorter than window_size, pad with first observation
            pad_len = window_size - num_cycles
            pad_features = np.repeat(group[feature_cols].iloc[[0]].values, pad_len, axis=0)
            full_features = np.vstack([pad_features, group[feature_cols].values])
            
            feature_windows.append(full_features)
            rul_targets.append(group["rul"].iloc[-1])
            stage_targets.append(group["degradation_stage"].iloc[-1])
            engine_ids.append(unit_nr)
            end_cycles.append(group["time_cycles"].iloc[-1])
            continue

        if last_window_only:
            # Only the final window of the engine
            window = group[feature_cols].iloc[-window_size:].values
            feature_windows.append(window)
            rul_targets.append(group["rul"].iloc[-1])
            stage_targets.append(group["degradation_stage"].iloc[-1])
            engine_ids.append(unit_nr)
            end_cycles.append(group["time_cycles"].iloc[-1])
        else:
            for start_idx in range(0, num_cycles - window_size + 1, stride):
                end_idx = start_idx + window_size
                window = group[feature_cols].iloc[start_idx:end_idx].values
                target_rul = group["rul"].iloc[end_idx - 1]
                target_stage = group["degradation_stage"].iloc[end_idx - 1]
                target_cycle = group["time_cycles"].iloc[end_idx - 1]

                feature_windows.append(window)
                rul_targets.append(target_rul)
                stage_targets.append(target_stage)
                engine_ids.append(unit_nr)
                end_cycles.append(target_cycle)

    return (
        np.array(feature_windows, dtype=np.float32),
        np.array(rul_targets, dtype=np.float32),
        np.array(stage_targets, dtype=np.int64),
        np.array(engine_ids, dtype=np.int64),
        np.array(end_cycles, dtype=np.int64),
    )


def create_dataloaders(
    train_dataset: CMAPSSDataset,
    val_dataset: CMAPSSDataset,
    test_dataset: Optional[CMAPSSDataset] = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Creates PyTorch DataLoaders for train, validation, and optional test datasets."""
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }
    if test_dataset is not None:
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    return dataloaders


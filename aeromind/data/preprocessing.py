import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

from aeromind.core.constants import FD001_INVARIANT_SENSORS, SENSOR_COLUMNS, SETTING_COLUMNS


class CMAPSSPreprocessor:
    """
    Handles feature selection, normalization, and condition-aware scaling for C-MAPSS data.
    Ensures zero data leakage by fitting exclusively on training engines.
    """

    def __init__(
        self,
        method: str = "standard",  # "standard" or "minmax"
        condition_aware: bool = False,
        invariant_threshold: float = 1e-4,
    ):
        self.method = method
        self.condition_aware = condition_aware
        self.invariant_threshold = invariant_threshold
        self.selected_features: List[str] = []
        self.scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        self.condition_clusterer: Optional[KMeans] = None
        self.condition_scalers: Dict[int, any] = {}
        self.is_fitted: bool = False
        self.stats: Dict[str, Dict[str, float]] = {}

    def fit(self, train_df: pd.DataFrame, dataset_name: str = "FD001") -> "CMAPSSPreprocessor":
        # Identify invariant sensors
        sensor_stds = train_df[SENSOR_COLUMNS].std()
        invariant_sensors = sensor_stds[sensor_stds < self.invariant_threshold].index.tolist()
        
        # In FD001, also include standard known invariant sensors
        if dataset_name == "FD001":
            for s in FD001_INVARIANT_SENSORS:
                if s not in invariant_sensors:
                    invariant_sensors.append(s)

        self.selected_features = [s for s in SENSOR_COLUMNS if s not in invariant_sensors]
        
        # If dataset has multiple operating conditions, include settings
        if dataset_name in ["FD002", "FD004"]:
            self.selected_features = SETTING_COLUMNS + self.selected_features

        if self.condition_aware and dataset_name in ["FD002", "FD004"]:
            # Cluster operating conditions into regimes
            self.condition_clusterer = KMeans(n_clusters=6, random_state=42, n_init=10)
            clusters = self.condition_clusterer.fit_predict(train_df[SETTING_COLUMNS])
            
            # Fit separate scaler per condition regime
            for c in range(6):
                idx = np.where(clusters == c)[0]
                if len(idx) > 0:
                    scaler = StandardScaler() if self.method == "standard" else MinMaxScaler()
                    scaler.fit(train_df.iloc[idx][self.selected_features])
                    self.condition_scalers[c] = scaler
        else:
            self.scaler.fit(train_df[self.selected_features])

        # Record statistics
        for feat in self.selected_features:
            self.stats[feat] = {
                "mean": float(train_df[feat].mean()),
                "std": float(train_df[feat].std()),
                "min": float(train_df[feat].min()),
                "max": float(train_df[feat].max()),
            }

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        df = df.copy()

        if self.condition_aware and self.condition_clusterer is not None:
            clusters = self.condition_clusterer.predict(df[SETTING_COLUMNS])
            scaled_features = np.zeros((len(df), len(self.selected_features)))
            
            for c, scaler in self.condition_scalers.items():
                idx = np.where(clusters == c)[0]
                if len(idx) > 0:
                    scaled_features[idx] = scaler.transform(df.iloc[idx][self.selected_features])
            
            df[self.selected_features] = scaled_features
        else:
            df[self.selected_features] = self.scaler.transform(df[self.selected_features])

        return df

    def fit_transform(self, train_df: pd.DataFrame, dataset_name: str = "FD001") -> pd.DataFrame:
        return self.fit(train_df, dataset_name=dataset_name).transform(train_df)

    def save_parameters(self, file_path: Path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "method": self.method,
            "condition_aware": self.condition_aware,
            "selected_features": self.selected_features,
            "feature_dim": len(self.selected_features),
            "stats": self.stats,
        }
        with open(file_path, "w") as f:
            json.dump(params, f, indent=2)

    @classmethod
    def load_parameters(cls, file_path: Path) -> Dict:
        with open(file_path, "r") as f:
            return json.load(f)


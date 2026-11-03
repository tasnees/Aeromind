from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    sub_dataset: str = "FD001"
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    window_size: int = 30
    stride: int = 1
    max_rul: int = 125
    val_ratio: float = 0.2
    random_seed: int = 42
    feature_cols: Optional[List[str]] = None


@dataclass
class ModelConfig:
    name: str = "lstm"  # "lstm", "tcn", "transformer", "multitask_transformer"
    input_dim: int = 14
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    num_classes: int = 3
    quantiles: List[float] = field(default_factory=lambda: [0.10, 0.50, 0.90])


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    patience: int = 10
    loss_weights: dict = field(
        default_factory=lambda: {
            "rul": 1.0,
            "classification": 0.5,
            "quantile": 0.5,
        }
    )
    checkpoint_dir: Path = Path("models/checkpoints")
    mlflow_experiment_name: str = "AeroMind_RUL_Research"
    mlflow_tracking_uri: Optional[str] = None


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


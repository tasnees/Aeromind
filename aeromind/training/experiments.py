import argparse
import json
from pathlib import Path
from typing import Dict
import torch

from aeromind.core.constants import DEFAULT_MAX_RUL, DEFAULT_WINDOW_SIZE
from aeromind.data.dataset import create_dataloaders
from aeromind.data.pipeline import run_data_pipeline
from aeromind.mlops.registry import ModelRegistry
from aeromind.mlops.tracker import ExperimentTracker
from aeromind.models.lstm import LSTMRegressor
from aeromind.models.multitask_transformer import ConditionAwareMultiTaskTransformer
from aeromind.models.tcn import TCNRegressor
from aeromind.models.transformer import TransformerRegressor
from aeromind.training.trainer import ModelTrainer


def run_experiment(
    exp_id: str,
    dataset_name: str = "FD001",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    window_size: int = DEFAULT_WINDOW_SIZE,
    max_rul: int = DEFAULT_MAX_RUL,
) -> Dict[str, any]:
    """
    Executes a controlled research experiment:
    - EXP-01: LSTM baseline
    - EXP-02: TCN
    - EXP-03: Temporal Transformer
    - EXP-04: Multi-Task Transformer (RUL + Stage + Quantiles)
    - EXP-05: Condition-aware Multi-Task Transformer
    - EXP-06: Multi-Task Transformer without Classification Head (Ablation)
    - EXP-07: Multi-Task Transformer without Quantile Head (Ablation)
    """
    tracker = ExperimentTracker(experiment_name="AeroMind_RUL_Research")
    run_id = tracker.start_run(run_name=exp_id)
    registry = ModelRegistry()

    # 1. Run Data Engineering Pipeline
    condition_aware = exp_id == "EXP-05" or dataset_name in ["FD002", "FD004"]
    datasets, preprocessor = run_data_pipeline(
        dataset_name=dataset_name,
        window_size=window_size,
        max_rul=max_rul,
        condition_aware=condition_aware,
    )
    dataloaders = create_dataloaders(
        datasets["train"], datasets["val"], datasets["test"], batch_size=batch_size
    )

    feature_dim = len(preprocessor.selected_features)

    # 2. Instantiate Model
    if exp_id == "EXP-01":
        model = LSTMRegressor(input_dim=feature_dim, hidden_dim=64, num_layers=2)
        arch_name = "LSTM"
    elif exp_id == "EXP-02":
        model = TCNRegressor(input_dim=feature_dim, num_channels=[32, 64, 64], kernel_size=3)
        arch_name = "TCN"
    elif exp_id == "EXP-03":
        model = TransformerRegressor(input_dim=feature_dim, d_model=64, nhead=4, num_layers=2)
        arch_name = "Transformer"
    elif exp_id == "EXP-04":
        model = ConditionAwareMultiTaskTransformer(
            input_dim=feature_dim,
            d_model=64,
            nhead=4,
            num_layers=2,
            enable_classification=True,
            enable_quantiles=True,
        )
        arch_name = "MultiTaskTransformer"
    elif exp_id == "EXP-05":
        model = ConditionAwareMultiTaskTransformer(
            input_dim=feature_dim,
            d_model=64,
            nhead=4,
            num_layers=2,
            enable_classification=True,
            enable_quantiles=True,
        )
        arch_name = "ConditionAwareMultiTaskTransformer"
    elif exp_id == "EXP-06":  # Ablation without classification
        model = ConditionAwareMultiTaskTransformer(
            input_dim=feature_dim,
            d_model=64,
            nhead=4,
            num_layers=2,
            enable_classification=False,
            enable_quantiles=True,
        )
        arch_name = "MultiTaskTransformer_NoClass"
    elif exp_id == "EXP-07":  # Ablation without quantiles
        model = ConditionAwareMultiTaskTransformer(
            input_dim=feature_dim,
            d_model=64,
            nhead=4,
            num_layers=2,
            enable_classification=True,
            enable_quantiles=False,
        )
        arch_name = "MultiTaskTransformer_NoQuantile"
    else:
        raise ValueError(f"Unknown experiment ID: {exp_id}")

    # Log Parameters
    hyperparams = {
        "exp_id": exp_id,
        "architecture": arch_name,
        "dataset": dataset_name,
        "window_size": window_size,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "feature_dim": feature_dim,
    }
    tracker.log_params(hyperparams)

    # 3. Train Model
    trainer = ModelTrainer(
        model=model,
        model_name=f"{exp_id}_{arch_name}",
        learning_rate=lr,
    )
    train_results = trainer.fit(
        dataloaders["train"], dataloaders["val"], epochs=epochs, patience=7
    )

    # 4. Final Benchmark Evaluation on Test Set (Final observed window of test fleet)
    test_metrics = trainer.evaluate(dataloaders["test"])
    print(
        f"\n[{exp_id} - {arch_name} Final Test Evaluation]\n"
        f"  Test MAE:        {test_metrics['mae']:.2f}\n"
        f"  Test RMSE:       {test_metrics['rmse']:.2f}\n"
        f"  Test NASA Score: {test_metrics['nasa_score']:.1f}"
    )
    if "picp_80" in test_metrics:
        print(f"  Test PICP (80%): {test_metrics['picp_80']:.2%}")

    # Log Metrics & Artifacts
    tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
    tracker.log_artifact(train_results["best_checkpoint"])
    tracker.end_run()

    # 5. Register in Model Registry
    version_num = registry.register_model_version(
        architecture=arch_name,
        checkpoint_path=train_results["best_checkpoint"],
        metrics=test_metrics,
        hyperparameters=hyperparams,
    )

    # Evaluate promotion gate
    gate_result = registry.evaluate_promotion_gate(version_num)
    print(f"Promotion gate evaluation: {gate_result}")

    return {
        "exp_id": exp_id,
        "architecture": arch_name,
        "version": version_num,
        "test_metrics": test_metrics,
        "checkpoint": train_results["best_checkpoint"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AeroMind DL Experiments")
    parser.add_argument("--exp", type=str, default="EXP-01", help="Experiment ID (EXP-01 to EXP-07)")
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    run_experiment(
        exp_id=args.exp,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


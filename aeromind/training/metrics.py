from typing import Dict, List, Optional
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Computes MAE, RMSE, and the asymmetric NASA prognostics score."""
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # NASA Asymmetric Score
    # Late predictions (d >= 0) penalize exp(d/10) - 1
    # Early predictions (d < 0) penalize exp(-d/13) - 1
    s_late = np.exp(errors[errors >= 0] / 10.0) - 1.0
    s_early = np.exp(-errors[errors < 0] / 13.0) - 1.0
    nasa_score = float(np.sum(s_late) + np.sum(s_early))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "nasa_score": round(nasa_score, 2),
    }


def calculate_uncertainty_metrics(
    y_true: np.ndarray,
    quantiles_pred: np.ndarray,
    quantiles: List[float] = [0.10, 0.50, 0.90],
) -> Dict[str, float]:
    """
    Computes PICP (Prediction Interval Coverage Probability) and MPIW (Mean Prediction Interval Width)
    for the 80% interval [q_0.10, q_0.90].
    """
    metrics = {}
    if quantiles_pred is None or len(quantiles) < 2:
        return metrics

    q_low_idx = 0
    q_high_idx = len(quantiles) - 1

    q_low = quantiles_pred[:, q_low_idx]
    q_high = quantiles_pred[:, q_high_idx]

    # Coverage: indicator whether true RUL falls inside interval [q_low, q_high]
    covered = (y_true >= q_low) & (y_true <= q_high)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(q_high - q_low))

    metrics["picp_80"] = round(picp, 4)
    metrics["mpiw_80"] = round(mpiw, 4)

    return metrics


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Computes categorical accuracy and macro F1 score for degradation stages."""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "stage_accuracy": round(acc, 4),
        "stage_macro_f1": round(f1, 4),
    }


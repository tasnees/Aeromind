from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aeromind.training.metrics import calculate_regression_metrics


class SensorAblationAnalyzer:
    """
    Evaluates individual sensor importance via feature ablation.
    Replaces each sensor channel in test windows with its training baseline mean,
    measuring the delta in RMSE and NASA score.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        device: Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.feature_names = feature_names
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def compute_ablation_impact(
        self,
        test_loader: DataLoader,
    ) -> Dict[str, Dict[str, float]]:
        # 1. Baseline performance
        all_y_true = []
        all_y_pred_base = []
        all_x = []

        for batch in test_loader:
            x, y_rul, _, _ = batch
            all_x.append(x)
            all_y_true.append(y_rul)
            
            x_dev = x.to(self.device)
            if hasattr(self.model, "stage_head"):
                pred = self.model(x_dev)["rul"]
            else:
                pred = self.model(x_dev)
            all_y_pred_base.append(pred.cpu())

        y_true = torch.cat(all_y_true).numpy()
        y_pred_base = torch.cat(all_y_pred_base).numpy()
        x_all = torch.cat(all_x)  # (N, W, D)

        base_metrics = calculate_regression_metrics(y_true, y_pred_base)
        base_rmse = base_metrics["rmse"]
        base_nasa = base_metrics["nasa_score"]

        ablation_results = {}

        # 2. Iterate through each sensor
        for feat_idx, feat_name in enumerate(self.feature_names):
            x_ablated = x_all.clone()
            # Zero out (in standard scaled space, 0 corresponds to the mean)
            x_ablated[:, :, feat_idx] = 0.0

            x_ablated_dev = x_ablated.to(self.device)
            if hasattr(self.model, "stage_head"):
                pred_ablated = self.model(x_ablated_dev)["rul"]
            else:
                pred_ablated = self.model(x_ablated_dev)
            
            y_pred_abl = pred_ablated.cpu().numpy()
            abl_metrics = calculate_regression_metrics(y_true, y_pred_abl)

            delta_rmse = abl_metrics["rmse"] - base_rmse
            delta_nasa = abl_metrics["nasa_score"] - base_nasa

            ablation_results[feat_name] = {
                "base_rmse": base_rmse,
                "ablated_rmse": abl_metrics["rmse"],
                "delta_rmse": round(float(delta_rmse), 4),
                "delta_nasa_score": round(float(delta_nasa), 2),
            }

        return ablation_results


class IntegratedGradientsAnalyzer:
    """
    Computes Integrated Gradients attribution scores for temporal sequences:
        IG_i(x) = (x_i - x'_i) * sum_{k=1}^m [ (1/m) * grad_i(x' + (k/m) * (x - x')) ]
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        num_steps: int = 50,
    ):
        self.model = model.eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_steps = num_steps

    def attribute(
        self,
        x: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Calculates attribution scores for an input window x of shape (1, window_size, input_dim).
        Returns:
            np.ndarray of shape (window_size, input_dim) with attribution weights.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(x).to(self.device)
        else:
            baseline = baseline.to(self.device)

        diff = x - baseline
        total_gradients = torch.zeros_like(x)

        for step in range(1, self.num_steps + 1):
            alpha = float(step) / self.num_steps
            interpolated = baseline + alpha * diff
            interpolated.requires_grad_(True)

            if hasattr(self.model, "stage_head"):
                pred = self.model(interpolated)["rul"]
            else:
                pred = self.model(interpolated)

            grad = torch.autograd.grad(pred.sum(), interpolated)[0]
            total_gradients += grad

        avg_gradients = total_gradients / self.num_steps
        attributions = (diff * avg_gradients).squeeze(0).detach().cpu().numpy()
        return attributions


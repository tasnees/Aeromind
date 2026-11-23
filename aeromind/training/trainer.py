import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from aeromind.training.losses import MultiTaskLoss
from aeromind.training.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_uncertainty_metrics,
)


class ModelTrainer:
    """
    Unified training and evaluation harness for AeroMind deep learning architectures.
    Supports single-task regression (LSTM, TCN, Transformer) and multi-task learning.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        huber_delta: float = 1.0,
        loss_weights: Optional[Dict[str, float]] = None,
        checkpoint_dir: Path = Path("models/checkpoints"),
    ):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.loss_weights = loss_weights or {
            "rul": 1.0,
            "classification": 0.5,
            "quantile": 0.5,
        }

        self.criterion = MultiTaskLoss(
            quantiles=[0.10, 0.50, 0.90],
            weight_rul=self.loss_weights.get("rul", 1.0),
            weight_stage=self.loss_weights.get("classification", 0.5),
            weight_quantile=self.loss_weights.get("quantile", 0.5),
            huber_delta=huber_delta,
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
        )

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "val_nasa_score": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch in dataloader:
            x, y_rul, y_stage, _ = batch
            x = x.to(self.device)
            y_rul = y_rul.to(self.device)
            y_stage = y_stage.to(self.device)

            self.optimizer.zero_grad()

            if hasattr(self.model, "stage_head"):
                # Multi-task model
                outputs = self.model(x)
                rul_pred = outputs["rul"]
                stage_logits = outputs.get("stage_logits")
                quantiles_pred = outputs.get("quantiles")

                loss_dict = self.criterion(
                    rul_pred=rul_pred,
                    rul_true=y_rul,
                    stage_logits=stage_logits,
                    stage_true=y_stage,
                    quantiles_pred=quantiles_pred,
                )
                loss = loss_dict["loss_total"]
            else:
                # Standard single-task regressor (LSTM, TCN, Transformer)
                rul_pred = self.model(x)
                loss_dict = self.criterion(rul_pred=rul_pred, rul_true=y_rul)
                loss = loss_dict["loss_rul"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0

        all_ruls_true = []
        all_ruls_pred = []
        all_stages_true = []
        all_stages_pred = []
        all_quantiles_pred = []

        is_multitask = hasattr(self.model, "stage_head")

        for batch in dataloader:
            x, y_rul, y_stage, _ = batch
            x = x.to(self.device)
            y_rul = y_rul.to(self.device)
            y_stage = y_stage.to(self.device)

            if is_multitask:
                outputs = self.model(x)
                rul_pred = outputs["rul"]
                stage_logits = outputs.get("stage_logits")
                quantiles_pred = outputs.get("quantiles")

                loss_dict = self.criterion(
                    rul_pred=rul_pred,
                    rul_true=y_rul,
                    stage_logits=stage_logits,
                    stage_true=y_stage,
                    quantiles_pred=quantiles_pred,
                )
                loss = loss_dict["loss_total"]

                if stage_logits is not None:
                    stage_preds = torch.argmax(stage_logits, dim=-1)
                    all_stages_pred.extend(stage_preds.cpu().numpy())
                    all_stages_true.extend(y_stage.cpu().numpy())

                if quantiles_pred is not None:
                    all_quantiles_pred.extend(quantiles_pred.cpu().numpy())
            else:
                rul_pred = self.model(x)
                loss_dict = self.criterion(rul_pred=rul_pred, rul_true=y_rul)
                loss = loss_dict["loss_rul"]

            total_loss += loss.item()
            all_ruls_pred.extend(rul_pred.cpu().numpy())
            all_ruls_true.extend(y_rul.cpu().numpy())

        y_true_arr = np.array(all_ruls_true)
        y_pred_arr = np.array(all_ruls_pred)

        metrics = calculate_regression_metrics(y_true_arr, y_pred_arr)
        metrics["loss"] = round(total_loss / max(1, len(dataloader)), 4)

        if len(all_stages_pred) > 0:
            stage_metrics = calculate_classification_metrics(
                np.array(all_stages_true), np.array(all_stages_pred)
            )
            metrics.update(stage_metrics)

        if len(all_quantiles_pred) > 0:
            unc_metrics = calculate_uncertainty_metrics(
                y_true_arr, np.array(all_quantiles_pred)
            )
            metrics.update(unc_metrics)

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        patience: int = 10,
    ) -> Dict[str, any]:
        best_val_rmse = float("inf")
        patience_counter = 0
        best_model_path = self.checkpoint_dir / f"{self.model_name}_best.pth"

        print(f"--- Training {self.model_name} on {self.device} (Max Epochs: {epochs}) ---")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]
            val_rmse = val_metrics["rmse"]

            self.scheduler.step(val_loss)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["val_rmse"].append(val_rmse)
            self.history["val_nasa_score"].append(val_metrics["nasa_score"])

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.2f} | "
                    f"Val NASA Score: {val_metrics['nasa_score']:.1f}"
                )

            # Checkpoint best model
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "model_name": self.model_name,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    best_model_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        # Load best checkpoint before returning
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Save training curves history
        history_path = self.checkpoint_dir / f"{self.model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return {
            "best_val_rmse": best_val_rmse,
            "best_checkpoint": str(best_model_path),
            "final_val_metrics": val_metrics,
        }


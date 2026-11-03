from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def nasa_scoring_function(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the official NASA C-MAPSS competition asymmetric scoring function.
    Late predictions (y_pred > y_true) are penalized exponentially more severely than early predictions.
    
    d_i = y_pred - y_true
    s_i = exp(-d_i / 13) - 1   if d_i < 0  (early)
          exp(d_i / 10) - 1    if d_i >= 0 (late)
    """
    diff = (y_pred - y_true).detach().cpu().numpy()
    score = 0.0
    for d in diff:
        if d < 0:
            score += float(torch.exp(torch.tensor(-d / 13.0)) - 1.0)
        else:
            score += float(torch.exp(torch.tensor(d / 10.0)) - 1.0)
    return score


class PinballLoss(nn.Module):
    """
    Quantile / Pinball loss function for uncertainty estimation.
    For quantile tau in (0, 1):
        L_tau(y, y_hat) = max(tau * (y - y_hat), (tau - 1) * (y - y_hat))
    """

    def __init__(self, quantiles: List[float] = [0.10, 0.50, 0.90]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: (batch_size, len(quantiles))
        y_true: (batch_size,) or (batch_size, 1)
        """
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(-1)
        
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = y_true - y_pred[:, i : i + 1]
            loss_q = torch.max((q * errors), ((q - 1.0) * errors))
            losses.append(loss_q.mean())
            
        return torch.stack(losses).sum()


class MultiTaskLoss(nn.Module):
    """
    Joint loss function combining:
    1. Huber Loss for continuous RUL regression
    2. Cross Entropy Loss for 3-class degradation stage classification
    3. Pinball Loss for multi-quantile uncertainty bounds
    """

    def __init__(
        self,
        quantiles: List[float] = [0.10, 0.50, 0.90],
        weight_rul: float = 1.0,
        weight_stage: float = 0.5,
        weight_quantile: float = 0.5,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.weight_rul = weight_rul
        self.weight_stage = weight_stage
        self.weight_quantile = weight_quantile

        self.huber_loss = nn.HuberLoss(delta=huber_delta)
        self.ce_loss = nn.CrossEntropyLoss()
        self.pinball_loss = PinballLoss(quantiles=quantiles)

    def forward(
        self,
        rul_pred: torch.Tensor,
        rul_true: torch.Tensor,
        stage_logits: Optional[torch.Tensor] = None,
        stage_true: Optional[torch.Tensor] = None,
        quantiles_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        l_rul = self.huber_loss(rul_pred, rul_true)
        total_loss = self.weight_rul * l_rul
        
        loss_dict = {"loss_rul": l_rul}

        if stage_logits is not None and stage_true is not None:
            l_stage = self.ce_loss(stage_logits, stage_true)
            total_loss = total_loss + self.weight_stage * l_stage
            loss_dict["loss_stage"] = l_stage

        if quantiles_pred is not None:
            l_quantile = self.pinball_loss(quantiles_pred, rul_true)
            total_loss = total_loss + self.weight_quantile * l_quantile
            loss_dict["loss_quantile"] = l_quantile

        loss_dict["loss_total"] = total_loss
        return loss_dict

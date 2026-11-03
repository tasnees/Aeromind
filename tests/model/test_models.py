import numpy as np
import pytest
import torch

from aeromind.models.lstm import LSTMRegressor
from aeromind.models.multitask_transformer import ConditionAwareMultiTaskTransformer
from aeromind.models.tcn import TCNRegressor
from aeromind.models.transformer import TransformerRegressor
from aeromind.training.losses import nasa_scoring_function
from aeromind.training.metrics import calculate_regression_metrics


def test_model_forward_passes():
    batch_size = 4
    window_size = 30
    feature_dim = 15

    dummy_input = torch.randn(batch_size, window_size, feature_dim)

    # 1. LSTM
    lstm = LSTMRegressor(input_dim=feature_dim, hidden_dim=32, num_layers=2)
    out_lstm = lstm(dummy_input)
    assert out_lstm.shape == (batch_size,)

    # 2. TCN
    tcn = TCNRegressor(input_dim=feature_dim, num_channels=[16, 32], kernel_size=3)
    out_tcn = tcn(dummy_input)
    assert out_tcn.shape == (batch_size,)

    # 3. Transformer
    trans = TransformerRegressor(input_dim=feature_dim, d_model=32, nhead=2, num_layers=2)
    out_trans = trans(dummy_input)
    assert out_trans.shape == (batch_size,)

    # 4. Multi-Task Transformer
    mt_trans = ConditionAwareMultiTaskTransformer(
        input_dim=feature_dim,
        d_model=32,
        nhead=2,
        num_layers=2,
        num_classes=3,
        quantiles=[0.10, 0.50, 0.90],
    )
    outputs = mt_trans(dummy_input)
    assert outputs["rul"].shape == (batch_size,)
    assert outputs["stage_logits"].shape == (batch_size, 3)
    assert outputs["quantiles"].shape == (batch_size, 3)


def test_nasa_asymmetric_scoring_penalty():
    # When prediction is early by 10 cycles (d = -10):
    # s = exp(10/13) - 1 ≈ 1.158
    y_true_early = torch.tensor([50.0])
    y_pred_early = torch.tensor([40.0])
    score_early = nasa_scoring_function(y_true_early, y_pred_early)

    # When prediction is late by 10 cycles (d = +10):
    # s = exp(10/10) - 1 ≈ 1.718
    y_true_late = torch.tensor([50.0])
    y_pred_late = torch.tensor([60.0])
    score_late = nasa_scoring_function(y_true_late, y_pred_late)

    # Late prediction must incur a strictly greater penalty than early prediction
    assert score_late > score_early
    assert score_early < 2.0
    assert score_late > 1.5


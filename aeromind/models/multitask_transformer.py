import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from aeromind.models.transformer import PositionalEncoding


class ConditionAwareMultiTaskTransformer(nn.Module):
    """
    Proposed Research Architecture: Condition-Aware Multi-Task Transformer.
    
    Shares a temporal self-attention backbone to concurrently optimize:
    1. Continuous RUL Regression Head: Predicted cycles until functional failure.
    2. Degradation Stage Classification Head: 3-class ordinal stage (Healthy, Degraded, Critical).
    3. Quantile Uncertainty Head: Pinball estimates for 10th, 50th, and 90th percentiles.
    
    Supports condition modulation for multi-regime environments and selective head disabling for ablations.
    """

    def __init__(
        self,
        input_dim: int = 15,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        num_classes: int = 3,
        quantiles: List[float] = [0.10, 0.50, 0.90],
        enable_classification: bool = True,
        enable_quantiles: bool = True,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.quantiles = quantiles
        self.enable_classification = enable_classification
        self.enable_quantiles = enable_quantiles

        # 1. Input Projection & Positional Encoding
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        # 2. Shared Temporal Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 3. Optional Condition Modulation (FiLM: Feature-wise Linear Modulation)
        self.condition_dim = condition_dim
        if condition_dim is not None and condition_dim > 0:
            self.condition_film = nn.Sequential(
                nn.Linear(condition_dim, d_model * 2),
                nn.ReLU(),
            )
        else:
            self.condition_film = None

        # 4. Multi-Task Heads
        # Head A: RUL Regression Head
        self.rul_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Head B: Degradation Classification Head
        if self.enable_classification:
            self.stage_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )
        else:
            self.stage_head = None

        # Head C: Quantile Uncertainty Head
        if self.enable_quantiles:
            self.quantile_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, len(quantiles)),
            )
        else:
            self.quantile_head = None

    def forward(
        self,
        x: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Input x: (batch_size, window_size, input_dim)
        Optional conditions: (batch_size, condition_dim)
        
        Returns:
            Dictionary with:
            - 'rul': (batch_size,) point estimate
            - 'stage_logits': (batch_size, num_classes) or None
            - 'quantiles': (batch_size, len(quantiles)) or None
            - 'encoded_features': (batch_size, d_model) for explainability
        """
        h = self.input_projection(x) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)
        encoded = self.transformer_encoder(h)
        
        # Temporal pooling / final cycle summary
        representation = encoded[:, -1, :]

        # Apply condition modulation if enabled
        if self.condition_film is not None and conditions is not None:
            film_params = self.condition_film(conditions)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            representation = representation * (1.0 + gamma) + beta

        outputs = {
            "rul": self.rul_head(representation).squeeze(-1),
            "encoded_features": representation,
        }

        if self.stage_head is not None:
            outputs["stage_logits"] = self.stage_head(representation)
        else:
            outputs["stage_logits"] = None

        if self.quantile_head is not None:
            outputs["quantiles"] = self.quantile_head(representation)
        else:
            outputs["quantiles"] = None

        return outputs


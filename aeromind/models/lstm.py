import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """
    Baseline recurrent architecture for turbofan Remaining Useful Life (RUL) estimation.
    Consists of input feature projection, 2-layer stacked LSTM, dropout, and an MLP regression head.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Recurrent backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        lstm_out_dim = hidden_dim * direction_factor

        # MLP Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (batch_size, window_size, input_dim)
        Output: (batch_size,) predicted RUL
        """
        lstm_out, _ = self.lstm(x)  # (batch_size, window_size, lstm_out_dim)
        last_step = lstm_out[:, -1, :]  # Take the representation at the final timestep
        out = self.regressor(last_step).squeeze(-1)
        return out


# src/model.py
import torch
import torch.nn as nn


class TemporalMeanMLP(nn.Module):
    """
    Baseline: promedia en el tiempo y pasa por un MLP.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (batch, T, D)
        x_mean = x.mean(dim=1)  # (batch, D)
        logits = self.net(x_mean)
        return logits


class LSTMSkeletonClassifier(nn.Module):
    """
    Modelo profundo principal: LSTM + MLP.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, x):
        # x: (batch, T, D)
        out, (h_n, c_n) = self.lstm(x)  # h_n: (num_layers * num_directions, batch, hidden)
        # Usamos el último hidden state de la última capa
        last_h = h_n[-1]  # (batch, hidden)
        logits = self.fc(last_h)
        return logits
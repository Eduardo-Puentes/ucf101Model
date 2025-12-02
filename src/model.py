import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


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

    def forward(self, x, lengths=None):
        """
        x: batch x T x (2V)
        lengths: batch (long) con la longitud real antes de padding.
        """
        if lengths is not None:
            device = x.device
            lengths = lengths.to(device)
            max_len = x.size(1)
            mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
            summed = (x * mask).sum(dim=1)
            denom = lengths.clamp(min=1).unsqueeze(-1).float()
            x_mean = summed / denom
        else:
            x_mean = x.mean(dim=1)
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
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * (2 if self.bidirectional else 1)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, x, lengths=None):
        """
        x: batch x T x (2V)
        lengths: batch (long) con la longitud real antes de padding.
        """
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),  
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)

        if self.bidirectional:
            last_forward = h_n[-2]
            last_backward = h_n[-1]
            last_h = torch.cat([last_forward, last_backward], dim=1)
        else:
            last_h = h_n[-1]

        logits = self.fc(last_h)
        return logits

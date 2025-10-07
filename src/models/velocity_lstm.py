"""Velocity Correction LSTM Model"""
import torch
import torch.nn as nn

class VelocityCorrectionLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        if hasattr(self.lstm, "flatten_parameters"):
            self.lstm.flatten_parameters()
        h, _ = self.lstm(x)
        return self.head(h)

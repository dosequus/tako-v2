"""Output heads for policy and value prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """Policy head: predicts move probabilities.

    Args:
        d_model: Model dimension
        action_size: Number of possible actions
    """

    def __init__(self, d_model: int, action_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, action_size, bias=False)

    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        """Predict policy from H-state.

        Args:
            z_H: H-module state [batch, seq_len, d_model]

        Returns:
            Log probabilities [batch, action_size]
        """
        # Pool over sequence (mean pooling)
        pooled = z_H.mean(dim=1)  # [batch, d_model]

        # Project to action logits
        logits = self.linear(pooled)  # [batch, action_size]

        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs


class ValueHead(nn.Module):
    """Value head: predicts outcome probabilities (W/D/L).

    Args:
        d_model: Model dimension
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, 3)

    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        """Predict value from H-state.

        Args:
            z_H: H-module state [batch, seq_len, d_model]

        Returns:
            Logits [batch, 3] for (Win, Draw, Loss)
        """
        pooled = z_H.mean(dim=1)  # [batch, d_model]
        hidden = F.relu(self.fc1(pooled))  # [batch, d_model]
        logits = self.fc2(hidden)  # [batch, 3]
        return logits

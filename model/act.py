"""Adaptive Computation Time (ACT) head for halting control."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ACTHead(nn.Module):
    """ACT head: decides whether to halt or continue reasoning.

    Args:
        d_model: Model dimension
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 2, bias=False)  # [Q_halt, Q_continue]

    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        """Compute halt/continue Q-values.

        Args:
            z_H: H-module state [batch, seq_len, d_model]

        Returns:
            Q-values [batch, 2] where [:, 0] = Q_halt, [:, 1] = Q_continue
        """
        # Pool over sequence (mean pooling)
        pooled = z_H.mean(dim=1)  # [batch, d_model]

        # Project to Q-values
        q_values = self.linear(pooled)  # [batch, 2]

        # Apply sigmoid to get values in [0, 1]
        q_values = torch.sigmoid(q_values)

        return q_values

    def should_halt(self, z_H: torch.Tensor) -> torch.Tensor:
        """Determine whether to halt for each batch element.

        Args:
            z_H: H-module state [batch, seq_len, d_model]

        Returns:
            Boolean tensor [batch] indicating whether to halt
        """
        q_values = self.forward(z_H)
        q_halt = q_values[:, 0]
        q_continue = q_values[:, 1]

        # Halt if Q_halt > Q_continue
        return q_halt > q_continue

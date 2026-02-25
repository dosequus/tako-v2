"""Hierarchical Reasoning Model (HRM) main implementation."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import random

from model.modules import InputNetwork, LModule, HModule
from model.heads import PolicyHead, ValueHead
from model.act import ACTHead


class HRM(nn.Module):
    """Hierarchical Reasoning Model (HRM).

    A ~27M parameter dual-module recurrent architecture that performs
    iterative reasoning within a single forward pass.

    Architecture:
        - L-module: fast tactical reasoning (updates every timestep)
        - H-module: slow strategic reasoning (updates every T timesteps)
        - 1-step gradient approximation (only final timestep has gradients)
        - Adaptive Computation Time (ACT) for dynamic segment control

    Args:
        vocab_size: Size of input token vocabulary
        action_size: Number of possible actions
        d_model: Model dimension (default: 512)
        n_layers: Number of transformer layers per module (default: 4, total 8 across both modules)
        n_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward dimension (default: 2048)
        N: Number of cycles per segment (default: 4)
        T: Number of timesteps per cycle (default: 4)
        dropout: Dropout probability (default: 0.0)
        max_seq_len: Maximum sequence length (default: 128)
    """

    def __init__(
        self,
        vocab_size: int,
        action_size: int,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 2048,
        N: int = 4,
        T: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 128
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.action_size = action_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.N = N
        self.T = T
        self.max_seq_len = max_seq_len

        # Input network
        self.input_net = InputNetwork(vocab_size, d_model, max_seq_len)

        # L-module and H-module (identical architecture, different update freq)
        self.L_module = LModule(d_model, n_heads, d_ff, n_layers, dropout)
        self.H_module = HModule(d_model, n_heads, d_ff, n_layers, dropout)

        # Output heads
        self.policy_head = PolicyHead(d_model, action_size)
        self.value_head = ValueHead(d_model)

        # ACT head
        self.act_head = ACTHead(d_model)

        # Fixed initialization for hidden states
        # Truncated normal (std=1, truncated at ±2)
        self._init_hidden_states()

    def _init_hidden_states(self):
        """Initialize fixed z_H and z_L from truncated normal."""
        # Truncated normal: sample from normal, reject outliers, renormalize
        def truncated_normal(size, std=1.0, trunc=2.0):
            """Sample from truncated normal distribution."""
            tensor = torch.randn(size) * std
            # Clip to [-trunc*std, trunc*std]
            tensor = torch.clamp(tensor, -trunc * std, trunc * std)
            return tensor

        # Initialize z_H and z_L: [1, max_seq_len, d_model]
        z_H_init = truncated_normal((1, self.max_seq_len, self.d_model))
        z_L_init = truncated_normal((1, self.max_seq_len, self.d_model))

        # Register as buffers (not parameters, but part of state)
        self.register_buffer("z_H_init", z_H_init, persistent=True)
        self.register_buffer("z_L_init", z_L_init, persistent=True)

    def optimize_for_inference(self, use_compile: bool = True, dtype: torch.dtype = torch.bfloat16) -> 'HRM':
        """Optimize model for inference with torch.compile and mixed precision.

        Args:
            use_compile: Whether to use torch.compile (default: True)
            dtype: Target dtype for mixed precision (default: bfloat16)

        Returns:
            Self (for chaining)
        """
        # Convert to target dtype
        if dtype is not None:
            self.to(dtype=dtype)
            print(f"✅ Model converted to {dtype}")

        # Apply torch.compile if available and requested
        if use_compile and hasattr(torch, 'compile'):
            # Note: We compile the forward method, not the entire module
            # This gives better control and avoids issues with dynamic behavior
            try:
                self.forward = torch.compile(self.forward, mode='reduce-overhead')
                print("✅ torch.compile enabled (mode='reduce-overhead')")
            except Exception as e:
                print(f"⚠️  torch.compile failed: {e}")

        return self

    def _get_initial_states(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial hidden states for given batch size and sequence length.

        Args:
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Tuple of (z_H, z_L), each [batch_size, seq_len, d_model]
        """
        # Expand and slice to match seq_len
        z_H = self.z_H_init[:, :seq_len, :].expand(batch_size, -1, -1).clone()
        z_L = self.z_L_init[:, :seq_len, :].expand(batch_size, -1, -1).clone()

        return z_H, z_L

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_cycles: Optional[int] = None,
        t_steps: Optional[int] = None,
        inference_mode: bool = False
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass with 1-step gradient approximation.

        Runs N×T timesteps total:
        - First N×T-1 steps in torch.no_grad() (no gradients)
        - Final step with gradients enabled (1-step approximation)

        Args:
            x: Input tokens [batch, seq_len]
            z: Optional initial states (z_H, z_L). If None, uses fixed init.
            n_cycles: Number of cycles (default: self.N)
            t_steps: Timesteps per cycle (default: self.T)
            inference_mode: If True, skip gradient computation entirely (default: False)

        Returns:
            Tuple of:
                - (z_H, z_L): Final hidden states
                - policy: Log probabilities [batch, action_size]
                - value: Log probabilities [batch, 3] (W/D/L)
        """
        batch_size, seq_len = x.shape
        N = n_cycles if n_cycles is not None else self.N
        T = t_steps if t_steps is not None else self.T

        # Get initial states
        if z is None:
            z_H, z_L = self._get_initial_states(batch_size, seq_len)
        else:
            z_H, z_L = z

        # Embed input (only once, reused for all timesteps)
        x_emb = self.input_net(x)

        # Total timesteps
        total_steps = N * T

        # In inference mode or when gradients are disabled, run all steps without gradients
        # In training mode, use 1-step gradient approximation
        grad_enabled = torch.is_grad_enabled() and not inference_mode

        if grad_enabled:
            # Training mode: N×T-1 steps without gradients
            with torch.no_grad():
                for step in range(total_steps - 1):
                    # L-module updates every step
                    z_L = self.L_module(z_L, z_H, x_emb)

                    # H-module updates every T steps
                    if (step + 1) % T == 0:
                        z_H = self.H_module(z_H, z_L)

            # Final step WITH gradients (1-step gradient approximation)
            z_L = self.L_module(z_L, z_H, x_emb)
            # Check if final step is an H-update
            if total_steps % T == 0:
                z_H = self.H_module(z_H, z_L)
        else:
            # Inference mode: all steps without gradients
            for step in range(total_steps):
                # L-module updates every step
                z_L = self.L_module(z_L, z_H, x_emb)

                # H-module updates every T steps
                if (step + 1) % T == 0:
                    z_H = self.H_module(z_H, z_L)

        # Generate outputs from final z_H
        policy = self.policy_head(z_H)
        value = self.value_head(z_H)

        return (z_H, z_L), policy, value

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        use_act: bool = True,
        max_segments: int = 10,
        act_epsilon: float = 0.15,
        min_segments: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Predict with ACT-based dynamic segment control.

        Runs multiple segments (forward passes) until ACT decides to halt
        or max_segments is reached. Hidden states are detached between segments
        (deep supervision).

        Args:
            x: Input tokens [batch, seq_len]
            use_act: Whether to use ACT for halting (default: True)
            max_segments: Maximum number of segments (default: 10)
            act_epsilon: Probability of forcing >= 2 segments (default: 0.15)
            min_segments: Minimum segments before ACT can halt (default: 1)

        Returns:
            Tuple of:
                - policy: Log probabilities [batch, action_size]
                - value: Log probabilities [batch, 3] (W/D/L)
                - num_segments: Number of segments executed
        """
        batch_size, seq_len = x.shape
        z_H, z_L = self._get_initial_states(batch_size, seq_len)

        # Stochastic M_min sampling (ε chance of forcing 2+ segments)
        if use_act and random.random() < act_epsilon:
            effective_min_segments = max(2, min_segments)
        else:
            effective_min_segments = min_segments

        num_segments = 0
        for seg in range(max_segments):
            num_segments += 1

            # Run one segment (inference_mode=True to skip gradient computation)
            (z_H, z_L), policy, value = self.forward(x, z=(z_H, z_L), inference_mode=True)

            # Check ACT halting condition
            if use_act and num_segments >= effective_min_segments:
                should_halt = self.act_head.should_halt(z_H)
                # Halt if all batch elements agree to halt
                if should_halt.all():
                    break

            # Detach hidden states for next segment (deep supervision)
            z_H = z_H.detach()
            z_L = z_L.detach()

        return policy, value, num_segments

    def compute_act_loss(self, z_H: torch.Tensor, target_segments: int) -> torch.Tensor:
        """Compute ACT regularization loss.

        Encourages halting near target_segments (to prevent always running max).

        Args:
            z_H: H-module state [batch, seq_len, d_model]
            target_segments: Target number of segments

        Returns:
            Scalar loss
        """
        # Simple L2 penalty: encourage Q_halt to be high when near target
        # This is a placeholder - actual ACT loss is more sophisticated
        q_values = self.act_head(z_H)
        q_halt = q_values[:, 0]

        # Want Q_halt ≈ 1.0 to encourage halting
        target = torch.ones_like(q_halt)
        loss = torch.nn.functional.mse_loss(q_halt, target)

        return loss

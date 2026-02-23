"""Central learner for HRM training with deep supervision."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import os
import math
from pathlib import Path

from training.replay_buffer import ReplayBuffer


class Learner:
    """Central training loop with deep supervision.

    Implements:
    - Deep supervision: run M segments, detach states between segments
    - Multi-component loss: policy + value + ACT
    - Checkpoint management for opponent pool
    - Cosine learning rate schedule
    """

    def __init__(
        self,
        model,
        replay_buffer: ReplayBuffer,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize learner.

        Args:
            model: HRM model instance
            replay_buffer: ReplayBuffer instance
            config: Full config dict (must contain 'training' and 'checkpointing' sections)
            device: Device for training ('cpu' or 'cuda')
        """
        self.model = model
        self.replay_buffer = replay_buffer
        self.config = config
        self.device = device

        # Training config
        train_cfg = config['training']
        self.batch_size = train_cfg['batch_size']
        self.n_supervision = train_cfg['n_supervision']
        self.policy_weight = train_cfg['policy_weight']
        self.value_weight = train_cfg['value_weight']
        self.act_weight = train_cfg['act_weight']
        self.grad_clip = train_cfg['grad_clip']

        # Checkpointing config
        ckpt_cfg = config['checkpointing']
        self.checkpoint_dir = Path(ckpt_cfg['checkpoint_dir'])
        self.save_interval = ckpt_cfg['save_interval']
        self.keep_checkpoints = ckpt_cfg['keep_checkpoints']

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model.to(device)
        self.model.train()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay']
        )

        # Learning rate scheduler (cosine)
        # Note: We'll set total_steps when we know the full training schedule
        # For now, use a large number
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100000,  # Will be adjusted in training script
            eta_min=train_cfg['lr_min']
        )

        # Training state
        self.global_step = 0
        self.checkpoints_saved: List[Tuple[str, int]] = []  # (path, step)

    def train_step(self) -> Dict[str, float]:
        """Execute one training step with deep supervision.

        Samples batch from replay buffer, runs M segments with deep supervision,
        computes losses, and updates model weights.

        Returns:
            Dict of losses: {'policy_loss', 'value_loss', 'act_loss', 'total_loss', 'lr'}
        """
        if not self.replay_buffer.ready():
            raise RuntimeError("Replay buffer not ready for training")

        # Sample batch
        states, policies, values = self.replay_buffer.sample_batch(self.batch_size)

        # Move to device
        states = states.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)

        # Convert value scalars to W/D/L distributions
        value_targets = self._value_to_wdl(values)  # [batch_size, 3]

        # Deep supervision: run M segments
        batch_size, seq_len = states.shape
        z_H, z_L = self.model._get_initial_states(batch_size, seq_len)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_act_loss = 0.0

        self.optimizer.zero_grad()

        for seg in range(self.n_supervision):
            # Run one segment (forward pass)
            (z_H, z_L), policy_logits, value_logits = self.model(states, z=(z_H, z_L))

            # Mask illegal moves in policy loss
            # For now, we don't have legal move masks in the buffer
            # So we'll just use the full policy
            # TODO: Consider adding legal move masks to replay buffer

            # Policy loss (cross-entropy with soft targets from MCTS)
            # policies is already a probability distribution from MCTS visit counts
            policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
            policy_loss = -torch.sum(policies * policy_log_probs, dim=-1).mean()

            # Value loss (cross-entropy with W/D/L targets)
            value_log_probs = torch.log_softmax(value_logits, dim=-1)
            value_loss = -torch.sum(value_targets * value_log_probs, dim=-1).mean()

            # ACT loss (encourage halting near target)
            act_loss = self.model.compute_act_loss(z_H, target_segments=self.n_supervision)

            # Combined loss
            segment_loss = (
                self.policy_weight * policy_loss +
                self.value_weight * value_loss +
                self.act_weight * act_loss
            )

            # Backward (accumulate gradients)
            segment_loss.backward()

            # Track losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_act_loss += act_loss.item()

            # Detach states for next segment (deep supervision)
            z_H = z_H.detach()
            z_L = z_L.detach()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        # Increment step counter
        self.global_step += 1

        # Return average losses across segments
        return {
            'policy_loss': total_policy_loss / self.n_supervision,
            'value_loss': total_value_loss / self.n_supervision,
            'act_loss': total_act_loss / self.n_supervision,
            'total_loss': (total_policy_loss + total_value_loss + total_act_loss) / self.n_supervision,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def _value_to_wdl(self, values: torch.Tensor) -> torch.Tensor:
        """Convert scalar values to W/D/L probability distributions.

        Args:
            values: [batch_size] tensor of outcomes in {-1.0, 0.0, 1.0}

        Returns:
            [batch_size, 3] tensor of W/D/L probabilities
        """
        batch_size = values.shape[0]
        wdl = torch.zeros(batch_size, 3, device=values.device)

        # Map values to one-hot W/D/L
        # Win (+1.0) -> [1, 0, 0]
        # Draw (0.0) -> [0, 1, 0]
        # Loss (-1.0) -> [0, 0, 1]
        win_mask = (values == 1.0)
        draw_mask = (values == 0.0)
        loss_mask = (values == -1.0)

        wdl[win_mask, 0] = 1.0
        wdl[draw_mask, 1] = 1.0
        wdl[loss_mask, 2] = 1.0

        return wdl

    def save_checkpoint(self) -> str:
        """Save model checkpoint and manage checkpoint rotation.

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"

        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config
        }, checkpoint_path)

        # Track checkpoint
        self.checkpoints_saved.append((str(checkpoint_path), self.global_step))

        # Rotate old checkpoints (keep only last N)
        if len(self.checkpoints_saved) > self.keep_checkpoints:
            old_path, _ = self.checkpoints_saved.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']

    def get_opponent_pool(self) -> List[Tuple[str, int]]:
        """Return list of checkpoints for opponent pool.

        Returns:
            List of (checkpoint_path, step) tuples
        """
        return self.checkpoints_saved.copy()

    def set_lr_schedule_steps(self, total_steps: int):
        """Set total steps for learning rate scheduler.

        Should be called at start of training when total steps is known.

        Args:
            total_steps: Total number of training steps
        """
        # Recreate scheduler with correct T_max
        train_cfg = self.config['training']
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=train_cfg['lr_min']
        )

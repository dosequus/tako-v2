"""Tests for learner training step — focused on NaN detection."""

import torch
import numpy as np
import pytest

from model.hrm import HRM
from training.learner import Learner
from training.replay_buffer import ReplayBuffer


@pytest.fixture
def tiny_config():
    """Minimal config for fast learner tests."""
    return {
        'model': {
            'vocab_size': 16,
            'action_size': 9,
            'd_model': 32,
            'n_layers': 1,
            'n_heads': 2,
            'd_ff': 64,
            'N': 1,
            'T': 2,
            'max_seq_len': 16,
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 1e-3,
            'lr_min': 1e-5,
            'lr_schedule': 'cosine',
            'optimizer': 'adam',
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'policy_weight': 1.0,
            'value_weight': 1.0,
            'act_weight': 0.1,
            'max_segments': 2,
            'n_supervision': 2,
            'act_epsilon': 0.15,
            'mask_legal_moves': True,
        },
        'checkpointing': {
            'save_interval': 100,
            'keep_checkpoints': 2,
            'checkpoint_dir': '/tmp/tako_test_ckpt',
        },
    }


def _make_sample(seq_len, action_size, value, legal_indices):
    """Create a single replay buffer sample."""
    state = torch.randint(0, 16, (seq_len,))
    policy = np.zeros(action_size, dtype=np.float32)
    legal_mask = np.zeros(action_size, dtype=bool)
    for idx in legal_indices:
        legal_mask[idx] = True
    # Distribute policy uniformly over legal moves
    policy[legal_mask] = 1.0 / len(legal_indices)
    return {
        'state': state,
        'policy': policy,
        'value': value,
        'legal_mask': legal_mask,
    }


def _fill_buffer(buf, n_samples, seq_len, action_size):
    """Fill replay buffer with random samples."""
    samples = []
    for _ in range(n_samples):
        n_legal = np.random.randint(1, action_size + 1)
        legal_indices = np.random.choice(action_size, n_legal, replace=False).tolist()
        value = np.random.choice([-1.0, 0.0, 1.0])
        samples.append(_make_sample(seq_len, action_size, value, legal_indices))
    buf.add_samples(samples)


def test_train_step_no_nan(tiny_config):
    """Training step should never produce NaN losses."""
    model = HRM(**tiny_config['model'], optimize=False)
    buf = ReplayBuffer(capacity=200, min_size=10)
    _fill_buffer(buf, 50, seq_len=4, action_size=9)

    learner = Learner(model, buf, tiny_config, device='cpu')
    losses = learner.train_step()

    for name, val in losses.items():
        assert not np.isnan(val), f"{name} is NaN"
        assert not np.isinf(val), f"{name} is Inf"


def test_train_step_single_legal_move(tiny_config):
    """No NaN when only one move is legal (all others masked to -inf)."""
    model = HRM(**tiny_config['model'], optimize=False)
    buf = ReplayBuffer(capacity=200, min_size=10)

    # Every sample has exactly 1 legal move
    samples = [_make_sample(4, 9, 1.0, [0]) for _ in range(50)]
    buf.add_samples(samples)

    learner = Learner(model, buf, tiny_config, device='cpu')
    losses = learner.train_step()

    for name, val in losses.items():
        assert not np.isnan(val), f"{name} is NaN with single legal move"


def test_train_step_all_draws(tiny_config):
    """No NaN when all values are draws (0.0)."""
    model = HRM(**tiny_config['model'], optimize=False)
    buf = ReplayBuffer(capacity=200, min_size=10)

    samples = [_make_sample(4, 9, 0.0, [0, 1, 2]) for _ in range(50)]
    buf.add_samples(samples)

    learner = Learner(model, buf, tiny_config, device='cpu')
    losses = learner.train_step()

    for name, val in losses.items():
        assert not np.isnan(val), f"{name} is NaN with all-draw values"


def test_policy_loss_masked_moves_no_nan():
    """Directly test that policy loss handles masked illegal moves without NaN.

    Reproduces the 0 * (-inf) = NaN scenario:
    - logits are masked to -inf for illegal moves
    - MCTS policy has 0 probability on those moves
    - log_softmax(-inf) = -inf, and 0 * (-inf) = NaN without nan_to_num
    """
    batch_size = 8
    action_size = 9

    # Raw logits from model
    logits = torch.randn(batch_size, action_size)

    # Only 2 moves are legal per position
    legal_mask = torch.zeros(batch_size, action_size, dtype=torch.bool)
    legal_mask[:, 0] = True
    legal_mask[:, 1] = True

    # MCTS policy: probability only on legal moves
    policies = torch.zeros(batch_size, action_size)
    policies[:, 0] = 0.7
    policies[:, 1] = 0.3

    # Mask illegal moves
    masked_logits = logits.clone()
    masked_logits[~legal_mask] = float('-inf')

    # Compute policy loss (same as learner)
    policy_log_probs = torch.log_softmax(masked_logits, dim=-1)
    per_action = policies * policy_log_probs
    policy_loss = -torch.nan_to_num(per_action, nan=0.0).sum(dim=-1).mean()

    assert not torch.isnan(policy_loss), "Policy loss is NaN"
    assert not torch.isinf(policy_loss), "Policy loss is Inf"
    assert policy_loss.item() >= 0, "Policy loss should be non-negative"


def test_multiple_train_steps_stable(tiny_config):
    """Multiple consecutive training steps should not diverge to NaN."""
    model = HRM(**tiny_config['model'], optimize=False)
    buf = ReplayBuffer(capacity=200, min_size=10)
    _fill_buffer(buf, 50, seq_len=4, action_size=9)

    learner = Learner(model, buf, tiny_config, device='cpu')

    for step in range(10):
        losses = learner.train_step()
        for name, val in losses.items():
            assert not np.isnan(val), f"{name} is NaN at step {step}"

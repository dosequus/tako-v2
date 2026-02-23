"""Integration tests for training pipeline components."""

import pytest
import torch
import numpy as np
import tempfile
import yaml
from pathlib import Path

from model.hrm import HRM
from games.othello import OthelloGame
from training.replay_buffer import ReplayBuffer
from training.learner import Learner
from training.mcts import MCTS


@pytest.fixture
def config():
    """Create a minimal config for testing."""
    return {
        'model': {
            'vocab_size': 128,
            'action_size': 65,
            'd_model': 64,
            'n_layers': 2,
            'n_heads': 4,
            'd_ff': 256,
            'N': 2,
            'T': 2,
            'max_seq_len': 128
        },
        'training': {
            'batch_size': 8,
            'learning_rate': 1e-3,
            'lr_schedule': 'cosine',
            'lr_min': 1e-4,
            'max_segments': 5,
            'n_supervision': 3,
            'act_epsilon': 0.15,
            'optimizer': 'adam',
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'policy_weight': 1.0,
            'value_weight': 1.0,
            'act_weight': 0.1
        },
        'mcts': {
            'simulations': 10,
            'puct_c': 1.5,
            'dirichlet_alpha': 1.0,
            'dirichlet_epsilon': 0.25,
            'temperature': 1.0,
            'temperature_threshold': 30
        },
        'checkpointing': {
            'save_interval': 100,
            'keep_checkpoints': 5,
            'checkpoint_dir': None  # Will be set to temp dir
        }
    }


def test_full_training_step(config):
    """Test complete training step with synthetic data."""
    # Create model
    model = HRM(**config['model'])

    # Create replay buffer and fill with synthetic data
    replay_buffer = ReplayBuffer(capacity=100, min_size=10)

    for _ in range(20):
        samples = [{
            'state': torch.randint(0, 5, (65,)),
            'policy': np.random.dirichlet([1.0] * 65),
            'value': np.random.choice([-1.0, 0.0, 1.0])
        }]
        replay_buffer.add_samples(samples)

    assert replay_buffer.ready()

    # Create learner with temp checkpoint dir
    with tempfile.TemporaryDirectory() as tmpdir:
        config['checkpointing']['checkpoint_dir'] = tmpdir
        learner = Learner(model, replay_buffer, config, device='cpu')

        # Run training step
        losses = learner.train_step()

        # Check losses are finite
        assert all(np.isfinite(v) for v in losses.values())
        assert losses['policy_loss'] >= 0.0
        assert losses['value_loss'] >= 0.0
        assert losses['act_loss'] >= 0.0
        assert losses['total_loss'] >= 0.0


def test_loss_decreases_on_synthetic_data(config):
    """Test that loss decreases when training on fixed synthetic dataset."""
    model = HRM(**config['model'])

    # Create fixed synthetic dataset
    replay_buffer = ReplayBuffer(capacity=100, min_size=10)

    # Create consistent samples (same targets)
    torch.manual_seed(42)
    np.random.seed(42)

    for _ in range(50):
        samples = [{
            'state': torch.randint(0, 5, (65,)),
            'policy': np.random.dirichlet([1.0] * 65),
            'value': 1.0  # Fixed value
        }]
        replay_buffer.add_samples(samples)

    with tempfile.TemporaryDirectory() as tmpdir:
        config['checkpointing']['checkpoint_dir'] = tmpdir
        learner = Learner(model, replay_buffer, config, device='cpu')

        # Run multiple training steps
        initial_loss = None
        final_loss = None

        for step in range(10):
            losses = learner.train_step()

            if step == 0:
                initial_loss = losses['total_loss']
            if step == 9:
                final_loss = losses['total_loss']

        # Loss should decrease (model should overfit to fixed dataset)
        assert final_loss < initial_loss


def test_checkpoint_save_load(config):
    """Test checkpoint saving and loading."""
    model = HRM(**config['model'])
    replay_buffer = ReplayBuffer(capacity=100, min_size=10)

    # Add samples
    for _ in range(20):
        samples = [{
            'state': torch.randint(0, 5, (65,)),
            'policy': np.random.dirichlet([1.0] * 65),
            'value': np.random.choice([-1.0, 0.0, 1.0])
        }]
        replay_buffer.add_samples(samples)

    with tempfile.TemporaryDirectory() as tmpdir:
        config['checkpointing']['checkpoint_dir'] = tmpdir
        learner = Learner(model, replay_buffer, config, device='cpu')

        # Run a few steps
        for _ in range(3):
            learner.train_step()

        # Save checkpoint
        checkpoint_path = learner.save_checkpoint()
        assert Path(checkpoint_path).exists()

        # Create new learner and load checkpoint
        new_model = HRM(**config['model'])
        new_learner = Learner(new_model, replay_buffer, config, device='cpu')
        new_learner.load_checkpoint(checkpoint_path)

        # Check state was restored
        assert new_learner.global_step == learner.global_step

        # Check model weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


def test_opponent_pool_management(config):
    """Test opponent pool checkpoint management."""
    model = HRM(**config['model'])
    replay_buffer = ReplayBuffer(capacity=100, min_size=10)

    for _ in range(20):
        samples = [{
            'state': torch.randint(0, 5, (65,)),
            'policy': np.random.dirichlet([1.0] * 65),
            'value': 1.0
        }]
        replay_buffer.add_samples(samples)

    with tempfile.TemporaryDirectory() as tmpdir:
        config['checkpointing']['checkpoint_dir'] = tmpdir
        config['checkpointing']['keep_checkpoints'] = 3
        learner = Learner(model, replay_buffer, config, device='cpu')

        # Save multiple checkpoints
        checkpoints = []
        for _ in range(5):
            learner.train_step()
            ckpt = learner.save_checkpoint()
            checkpoints.append(ckpt)

        # Should only keep last 3
        opponent_pool = learner.get_opponent_pool()
        assert len(opponent_pool) == 3

        # Check that only last 3 checkpoint files exist
        for i, ckpt_path in enumerate(checkpoints):
            if i < 2:
                # First 2 should be deleted
                assert not Path(ckpt_path).exists()
            else:
                # Last 3 should exist
                assert Path(ckpt_path).exists()


def test_mcts_game_generation(config):
    """Test that MCTS can generate a complete game."""
    model = HRM(**config['model'])
    mcts = MCTS(model, OthelloGame, config['mcts'], device='cpu')

    game = OthelloGame()
    game.reset()

    moves_made = 0
    max_moves = 100  # Safety limit

    while not game.is_terminal() and moves_made < max_moves:
        policy = mcts.search(game, move_num=moves_made)

        # Select action
        action = np.argmax(policy)

        # Make move
        game.make_move(action)
        moves_made += 1

    # Game should reach terminal state
    assert game.is_terminal()
    assert moves_made < max_moves

    # Outcome should be valid
    outcome = game.outcome()
    assert outcome in [-1.0, 0.0, 1.0]


def test_value_to_wdl_conversion(config):
    """Test conversion of scalar values to W/D/L distributions."""
    model = HRM(**config['model'])
    replay_buffer = ReplayBuffer(capacity=100, min_size=10)

    for _ in range(20):
        samples = [{
            'state': torch.randint(0, 5, (65,)),
            'policy': np.random.dirichlet([1.0] * 65),
            'value': 1.0
        }]
        replay_buffer.add_samples(samples)

    with tempfile.TemporaryDirectory() as tmpdir:
        config['checkpointing']['checkpoint_dir'] = tmpdir
        learner = Learner(model, replay_buffer, config, device='cpu')

        # Test conversion
        values = torch.tensor([1.0, 0.0, -1.0, 1.0])
        wdl = learner._value_to_wdl(values)

        # Check shape
        assert wdl.shape == (4, 3)

        # Check conversions
        assert torch.allclose(wdl[0], torch.tensor([1.0, 0.0, 0.0]))  # Win
        assert torch.allclose(wdl[1], torch.tensor([0.0, 1.0, 0.0]))  # Draw
        assert torch.allclose(wdl[2], torch.tensor([0.0, 0.0, 1.0]))  # Loss
        assert torch.allclose(wdl[3], torch.tensor([1.0, 0.0, 0.0]))  # Win


def test_batch_shapes(config):
    """Test that all batch processing maintains correct shapes."""
    model = HRM(**config['model'])
    replay_buffer = ReplayBuffer(capacity=100, min_size=10)

    batch_size = config['training']['batch_size']

    # Add samples
    for _ in range(50):
        samples = [{
            'state': torch.randint(0, 5, (65,)),
            'policy': np.random.dirichlet([1.0] * 65),
            'value': np.random.choice([-1.0, 0.0, 1.0])
        }]
        replay_buffer.add_samples(samples)

    # Sample batch
    states, policies, values = replay_buffer.sample_batch(batch_size)

    # Check shapes
    assert states.shape[0] == batch_size
    assert policies.shape == (batch_size, 65)
    assert values.shape == (batch_size,)

    # Run through model
    (z_H, z_L), policy_logits, value_logits = model(states)

    # Check output shapes
    assert policy_logits.shape == (batch_size, 65)
    assert value_logits.shape == (batch_size, 3)
    assert z_H.shape == (batch_size, 65, 64)
    assert z_L.shape == (batch_size, 65, 64)

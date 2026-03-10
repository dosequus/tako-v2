"""Unit tests for Replay Buffer."""

import pytest
import torch
import numpy as np

from training.replay_buffer import ReplayBuffer


def _make_sample(state_len=65, action_size=65, value=1.0):
    """Helper to create a valid sample dict."""
    legal_mask = np.random.randint(0, 2, size=action_size).astype(bool)
    legal_mask[0] = True  # ensure at least one legal move
    return {
        'state': torch.randint(0, 5, (state_len,)),
        'policy': np.random.dirichlet([1.0] * action_size),
        'value': value,
        'legal_mask': legal_mask,
    }


@pytest.fixture
def buffer():
    """Create a small replay buffer for testing."""
    return ReplayBuffer(capacity=100, min_size=10)


def test_replay_buffer_init(buffer):
    """Test replay buffer initialization."""
    assert buffer.capacity == 100
    assert buffer.min_size == 10
    assert len(buffer) == 0
    assert not buffer.ready()


def test_add_samples(buffer):
    """Test adding samples to buffer."""
    samples = [_make_sample() for _ in range(5)]

    buffer.add_samples(samples)

    assert len(buffer) == 5
    assert not buffer.ready()  # Still below min_size


def test_buffer_ready_check(buffer):
    """Test that buffer becomes ready after min_size samples."""
    # Add samples until ready
    for _ in range(2):
        samples = [
            _make_sample(value=np.random.choice([-1.0, 0.0, 1.0]))
            for _ in range(5)
        ]
        buffer.add_samples(samples)

    assert len(buffer) == 10
    assert buffer.ready()


def test_circular_buffer_overwrite():
    """Test that buffer overwrites oldest samples when full."""
    buffer = ReplayBuffer(capacity=5, min_size=2)

    # Fill buffer
    for i in range(5):
        buffer.add_samples([{
            'state': torch.tensor([i], dtype=torch.long),
            'policy': np.ones(65) / 65,
            'value': 1.0,
            'legal_mask': np.ones(65, dtype=bool),
        }])

    assert len(buffer) == 5

    # Add more samples (should overwrite oldest)
    for i in range(5, 8):
        buffer.add_samples([{
            'state': torch.tensor([i], dtype=torch.long),
            'policy': np.ones(65) / 65,
            'value': 1.0,
            'legal_mask': np.ones(65, dtype=bool),
        }])

    # Buffer size should stay at capacity
    assert len(buffer) == 5

    # First element should be from index 5 (indices 0,1,2 were overwritten by 5,6,7)
    assert buffer.states[0][0].item() == 5


def test_sample_batch(buffer):
    """Test batch sampling."""
    # Add samples
    samples = [
        _make_sample(value=np.random.choice([-1.0, 0.0, 1.0]))
        for _ in range(20)
    ]
    buffer.add_samples(samples)

    # Sample batch
    states, policies, values, legal_masks = buffer.sample_batch(batch_size=10)

    # Check shapes
    assert states.shape[0] == 10
    assert policies.shape == (10, 65)
    assert values.shape == (10,)
    assert legal_masks.shape == (10, 65)
    assert legal_masks.dtype == torch.bool

    # Check values are valid
    assert torch.all((values == -1.0) | (values == 0.0) | (values == 1.0))
    assert torch.all(policies >= 0.0)


def test_sample_batch_too_large(buffer):
    """Test that sampling larger than buffer raises error."""
    samples = [_make_sample() for _ in range(5)]
    buffer.add_samples(samples)

    with pytest.raises(ValueError):
        buffer.sample_batch(batch_size=10)


def test_buffer_stats(buffer):
    """Test buffer statistics."""
    stats = buffer.stats()

    assert stats['size'] == 0
    assert stats['capacity'] == 100
    assert stats['utilization'] == 0.0
    assert not stats['ready']

    # Add some samples
    samples = [_make_sample() for _ in range(50)]
    buffer.add_samples(samples)

    stats = buffer.stats()
    assert stats['size'] == 50
    assert stats['utilization'] == 0.5
    assert stats['ready']


def test_value_validation(buffer):
    """Test that invalid values are rejected."""
    # Valid values should work
    buffer.add_samples([_make_sample(value=1.0)])
    assert len(buffer) == 1

    # Invalid value should raise assertion
    invalid_sample = _make_sample(value=0.5)
    invalid_sample['value'] = 0.5  # Override after creation
    with pytest.raises(AssertionError):
        buffer.add_samples([invalid_sample])


def test_uniform_sampling_distribution(buffer):
    """Test that sampling is approximately uniform."""
    # Add samples with identifiable states
    samples = [
        {
            'state': torch.tensor([i] * 65, dtype=torch.long),
            'policy': np.ones(65) / 65,
            'value': 1.0,
            'legal_mask': np.ones(65, dtype=bool),
        }
        for i in range(100)
    ]
    buffer.add_samples(samples)

    # Sample many times and check distribution
    sample_counts = {}
    num_samples = 1000

    for _ in range(num_samples):
        states, _, _, _ = buffer.sample_batch(batch_size=1)
        state_id = states[0, 0].item()
        sample_counts[state_id] = sample_counts.get(state_id, 0) + 1

    # Each state should be sampled roughly equally
    # With 100 states and 1000 samples, expect ~10 samples per state
    # Allow some variance
    for count in sample_counts.values():
        assert 0 < count < 30  # Loose bounds for randomness


def test_padding_variable_length_states():
    """Test that variable-length states are padded correctly."""
    buffer = ReplayBuffer(capacity=10, min_size=2)

    # Add states of different lengths
    samples = [
        {
            'state': torch.randint(0, 5, (50,)),  # Short
            'policy': np.ones(65) / 65,
            'value': 1.0,
            'legal_mask': np.ones(65, dtype=bool),
        },
        {
            'state': torch.randint(0, 5, (65,)),  # Long
            'policy': np.ones(65) / 65,
            'value': 1.0,
            'legal_mask': np.ones(65, dtype=bool),
        }
    ]
    buffer.add_samples(samples)

    # Sample batch
    states, _, _, _ = buffer.sample_batch(batch_size=2)

    # All states should be padded to same length
    assert states.shape[0] == 2
    assert states.shape[1] == 65  # Max length


def test_legal_mask_stored_correctly():
    """Test that legal masks are stored and returned correctly."""
    buffer = ReplayBuffer(capacity=10, min_size=1)

    # Create a specific legal mask
    legal_mask = np.zeros(65, dtype=bool)
    legal_mask[[0, 5, 10]] = True  # Only 3 legal moves

    buffer.add_samples([{
        'state': torch.randint(0, 5, (65,)),
        'policy': np.random.dirichlet([1.0] * 65),
        'value': 1.0,
        'legal_mask': legal_mask,
    }])

    _, _, _, masks = buffer.sample_batch(batch_size=1)
    assert masks.shape == (1, 65)
    assert masks.dtype == torch.bool
    assert masks[0, 0].item() is True
    assert masks[0, 5].item() is True
    assert masks[0, 10].item() is True
    assert masks[0, 1].item() is False

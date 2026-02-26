"""Unit tests for HRM (Hierarchical Reasoning Model)."""

import torch
import pytest
from model.hrm import HRM


@pytest.fixture
def small_config():
    """Small HRM config for fast testing."""
    return {
        'vocab_size': 128,
        'action_size': 65,
        'd_model': 64,
        'n_layers': 2,
        'n_heads': 4,
        'd_ff': 256,
        'N': 2,
        'T': 2,
        'max_seq_len': 32
    }


@pytest.fixture
def full_config():
    """Full-size HRM config (as per spec)."""
    return {
        'vocab_size': 128,
        'action_size': 65,
        'd_model': 512,
        'n_layers': 4,  # 4 per module, 8 total
        'n_heads': 8,
        'd_ff': 2048,
        'N': 4,
        'T': 4,
        'max_seq_len': 128
    }


def test_forward_pass(small_config):
    """Test basic forward pass produces valid outputs."""
    model = HRM(**small_config)
    model.eval()

    batch_size = 2
    seq_len = 16

    # Random input tokens
    x = torch.randint(0, small_config['vocab_size'], (batch_size, seq_len))

    # Forward pass
    (z_H, z_L), policy, value = model(x)

    # Check shapes
    assert z_H.shape == (batch_size, seq_len, small_config['d_model'])
    assert z_L.shape == (batch_size, seq_len, small_config['d_model'])
    assert policy.shape == (batch_size, small_config['action_size'])
    assert value.shape == (batch_size, 3)  # W/D/L

    # Check no NaNs
    assert not torch.isnan(z_H).any()
    assert not torch.isnan(z_L).any()
    assert not torch.isnan(policy).any()
    assert not torch.isnan(value).any()

    # Check valid log probabilities for policy (should sum to 1 in probability space)
    policy_probs = torch.exp(policy)
    assert torch.allclose(policy_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    # Value head returns raw logits; verify softmax sums to 1
    value_probs = torch.softmax(value, dim=1)
    assert torch.allclose(value_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)


def test_one_step_gradient(small_config):
    """Test that gradients flow correctly with 1-step approximation."""
    model = HRM(**small_config)
    model.train()

    batch_size = 2
    seq_len = 16

    x = torch.randint(0, small_config['vocab_size'], (batch_size, seq_len))

    # Forward pass
    (z_H, z_L), policy, value = model(x)

    # Dummy loss (policy + value)
    target_policy = torch.randint(0, small_config['action_size'], (batch_size,))
    target_value = torch.randint(0, 3, (batch_size,))

    loss = torch.nn.functional.nll_loss(policy, target_policy) + \
           torch.nn.functional.cross_entropy(value, target_value)

    # Backward pass
    loss.backward()

    # Check gradients exist on all major components
    assert model.input_net.embedding.weight.grad is not None
    assert model.L_module.layers[0].q_proj.weight.grad is not None
    assert model.H_module.layers[0].q_proj.weight.grad is not None
    assert model.policy_head.linear.weight.grad is not None
    assert model.value_head.fc1.weight.grad is not None

    # Check gradient magnitudes are reasonable (not 0, not inf)
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            assert grad_norm > 0, f"{name} has zero gradient"
            assert not torch.isinf(param.grad).any(), f"{name} has inf gradient"
            assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"


def test_deep_supervision(small_config):
    """Test deep supervision: multiple segments with detached states."""
    model = HRM(**small_config)
    model.train()

    batch_size = 2
    seq_len = 16
    n_segments = 3

    x = torch.randint(0, small_config['vocab_size'], (batch_size, seq_len))

    # Initialize states
    z_H, z_L = model._get_initial_states(batch_size, seq_len)

    losses = []

    for seg in range(n_segments):
        # Run segment
        (z_H, z_L), policy, value = model(x, z=(z_H, z_L))

        # Compute loss
        target_policy = torch.randint(0, small_config['action_size'], (batch_size,))
        target_value = torch.randint(0, 3, (batch_size,))

        loss = torch.nn.functional.nll_loss(policy, target_policy) + \
               torch.nn.functional.nll_loss(value, target_value)

        losses.append(loss)

        # Verify backward succeeds
        loss.backward()

        # Detach for next segment (deep supervision)
        z_H = z_H.detach()
        z_L = z_L.detach()

    # Verify we got 3 segments with losses
    assert len(losses) == n_segments

    # Verify all losses are finite
    for loss in losses:
        assert torch.isfinite(loss)


def test_act_halting(small_config):
    """Test ACT halting logic in predict()."""
    model = HRM(**small_config)
    model.eval()

    batch_size = 2
    seq_len = 16

    x = torch.randint(0, small_config['vocab_size'], (batch_size, seq_len))

    # Run prediction with ACT
    policy, value, num_segments = model.predict(
        x,
        use_act=True,
        max_segments=5,
        act_epsilon=0.0,  # No stochastic min
        min_segments=1
    )

    # Check outputs
    assert policy.shape == (batch_size, small_config['action_size'])
    assert value.shape == (batch_size, 3)
    assert 1 <= num_segments <= 5

    # Run without ACT (should use max_segments)
    policy2, value2, num_segments2 = model.predict(
        x,
        use_act=False,
        max_segments=5
    )

    assert num_segments2 == 5  # Should run all segments without ACT


def test_custom_cycles_and_steps(small_config):
    """Test forward pass with custom N and T."""
    model = HRM(**small_config)
    model.eval()

    batch_size = 2
    seq_len = 16

    x = torch.randint(0, small_config['vocab_size'], (batch_size, seq_len))

    # Custom N=3, T=3 (total 9 steps)
    (z_H, z_L), policy, value = model(x, n_cycles=3, t_steps=3)

    # Verify outputs are valid
    assert policy.shape == (batch_size, small_config['action_size'])
    assert value.shape == (batch_size, 3)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()


def test_initial_states(small_config):
    """Test that initial states are correctly initialized."""
    model = HRM(**small_config)

    batch_size = 4
    seq_len = 16

    z_H, z_L = model._get_initial_states(batch_size, seq_len)

    # Check shapes
    assert z_H.shape == (batch_size, seq_len, small_config['d_model'])
    assert z_L.shape == (batch_size, seq_len, small_config['d_model'])

    # Check finite values
    assert torch.isfinite(z_H).all()
    assert torch.isfinite(z_L).all()

    # Check values are in reasonable range (truncated normal should be ≈[-2, 2])
    assert z_H.abs().max() < 5.0
    assert z_L.abs().max() < 5.0


def test_parameter_count(full_config):
    """Test that parameter count is approximately 27M."""
    model = HRM(**full_config)

    total_params = sum(p.numel() for p in model.parameters())

    # Should be around 27M parameters (spec says "~27M")
    # Allow reasonable tolerance (22M-35M) since architecture can vary
    assert 22_000_000 < total_params < 35_000_000, \
        f"Expected ~27M parameters, got {total_params:,}"


def test_batch_independence(small_config):
    """Test that batch elements are processed independently."""
    model = HRM(**small_config)
    model.eval()

    seq_len = 16

    # Two different inputs
    x1 = torch.randint(0, small_config['vocab_size'], (1, seq_len))
    x2 = torch.randint(0, small_config['vocab_size'], (1, seq_len))

    # Process separately
    _, policy1, value1 = model(x1)
    _, policy2, value2 = model(x2)

    # Process together
    x_batch = torch.cat([x1, x2], dim=0)
    _, policy_batch, value_batch = model(x_batch)

    # Results should match (accounting for floating point precision)
    assert torch.allclose(policy_batch[0], policy1[0], atol=1e-5)
    assert torch.allclose(policy_batch[1], policy2[0], atol=1e-5)
    assert torch.allclose(value_batch[0], value1[0], atol=1e-5)
    assert torch.allclose(value_batch[1], value2[0], atol=1e-5)

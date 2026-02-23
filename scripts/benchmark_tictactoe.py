"""Benchmark TicTacToe forward pass performance."""

import yaml
import torch
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.hrm import HRM
from games.tictactoe import TicTacToeGame


def benchmark_forward_pass(config_path: str = "config/tictactoe.yaml"):
    """Benchmark forward pass time and estimate game generation time."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("TicTacToe Performance Benchmark")
    print("=" * 80)

    # Model configuration
    model_config = config['model']
    mcts_config = config['mcts']

    print(f"\nModel Configuration:")
    print(f"  d_model: {model_config['d_model']}")
    print(f"  n_layers: {model_config['n_layers']}")
    print(f"  n_heads: {model_config['n_heads']}")
    print(f"  d_ff: {model_config['d_ff']}")
    print(f"  N (cycles): {model_config['N']}")
    print(f"  T (steps): {model_config['T']}")
    print(f"  Timesteps per segment: {model_config['N'] * model_config['T']}")

    print(f"\nMCTS Configuration:")
    print(f"  Simulations: {mcts_config['simulations']}")
    print(f"  max_segments_inference: {mcts_config.get('max_segments_inference', 'NOT SET')}")

    # Create model
    print(f"\nInitializing model...")
    model = HRM(**model_config)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Create dummy game state
    game = TicTacToeGame()
    tokens = game.to_tokens().unsqueeze(0)  # [1, seq_len]

    print(f"\nInput shape: {tokens.shape}")

    # Warmup
    print(f"\nWarming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model.predict(
                tokens,
                use_act=True,
                max_segments=mcts_config.get('max_segments_inference', 1)
            )

    # Benchmark forward pass
    print(f"\nBenchmarking forward pass (20 iterations)...")
    times = []
    with torch.no_grad():
        for i in range(20):
            start = time.time()
            policy, value, _ = model.predict(
                tokens,
                use_act=True,
                max_segments=mcts_config.get('max_segments_inference', 1)
            )
            elapsed = time.time() - start
            times.append(elapsed)

            if i == 0:  # Print first result
                print(f"  First pass: {elapsed*1000:.1f}ms")
                print(f"    Policy shape: {policy.shape}")
                print(f"    Value shape: {value.shape}")

    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\nForward Pass Results:")
    print(f"  Average: {avg_time*1000:.1f}ms")
    print(f"  Min: {min_time*1000:.1f}ms")
    print(f"  Max: {max_time*1000:.1f}ms")

    # Estimate game generation time
    avg_moves_per_game = 7  # TicTacToe typically 5-9 moves
    sims = mcts_config['simulations']

    total_forward_passes = sims * avg_moves_per_game
    estimated_time_per_game = avg_time * total_forward_passes

    print(f"\nEstimated Game Generation Time:")
    print(f"  MCTS simulations: {sims}")
    print(f"  Avg moves per game: {avg_moves_per_game}")
    print(f"  Total forward passes: {total_forward_passes}")
    print(f"  Estimated time per game: {estimated_time_per_game:.2f}s")
    print(f"  Estimated games per hour: {3600 / estimated_time_per_game:.0f}")

    # Comparison to baseline (if we had 311ms per pass, 200 sims)
    baseline_time_per_pass = 0.311  # seconds
    baseline_sims = 200
    baseline_time_per_game = baseline_time_per_pass * baseline_sims * avg_moves_per_game

    speedup = baseline_time_per_game / estimated_time_per_game

    print(f"\nSpeedup vs Baseline (311ms/pass, 200 sims):")
    print(f"  Baseline time per game: {baseline_time_per_game:.1f}s ({baseline_time_per_game/60:.1f} min)")
    print(f"  New time per game: {estimated_time_per_game:.1f}s")
    print(f"  Speedup: {speedup:.1f}x faster")

    print("\n" + "=" * 80)

    return {
        'avg_forward_pass_ms': avg_time * 1000,
        'estimated_game_time_s': estimated_time_per_game,
        'speedup': speedup,
        'total_params': total_params
    }


if __name__ == "__main__":
    results = benchmark_forward_pass()

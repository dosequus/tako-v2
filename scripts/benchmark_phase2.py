"""Benchmark Phase 2 optimizations: GPU acceleration and MCTS batching."""

import yaml
import torch
import time
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.hrm import HRM
from games.tictactoe import TicTacToeGame
from training.mcts import MCTS


def benchmark_device(device: str, config: dict, num_games: int = 10):
    """Benchmark performance on a specific device."""
    print(f"\n{'='*80}")
    print(f"Benchmarking on {device.upper()}")
    print(f"{'='*80}")

    # Create model
    model = HRM(**config['model'])
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    # Test 1: Forward pass latency
    print(f"\n--- Test 1: Forward Pass Latency ---")
    game = TicTacToeGame()
    tokens = game.to_tokens().unsqueeze(0).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model.predict(tokens, use_act=True, max_segments=config['mcts']['max_segments_inference'])
            if device == 'cuda':
                torch.cuda.synchronize()

    # Benchmark single forward pass
    times = []
    with torch.no_grad():
        for _ in range(50):
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.time()
            policy, value, _ = model.predict(
                tokens, use_act=True,
                max_segments=config['mcts']['max_segments_inference']
            )

            if device == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start
            times.append(elapsed)

    avg_forward = sum(times) / len(times)
    print(f"  Single forward pass: {avg_forward*1000:.2f}ms (avg of 50)")

    # Test 2: Batched forward pass
    print(f"\n--- Test 2: Batched Forward Pass ---")
    batch_sizes = [1, 4, 8, 16, 32]

    for batch_size in batch_sizes:
        tokens_batch = game.to_tokens().unsqueeze(0).repeat(batch_size, 1).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model.predict(tokens_batch, use_act=True, max_segments=config['mcts']['max_segments_inference'])
                if device == 'cuda':
                    torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(20):
                if device == 'cuda':
                    torch.cuda.synchronize()

                start = time.time()
                policy, value, _ = model.predict(
                    tokens_batch, use_act=True,
                    max_segments=config['mcts']['max_segments_inference']
                )

                if device == 'cuda':
                    torch.cuda.synchronize()

                elapsed = time.time() - start
                times.append(elapsed)

        avg_batch = sum(times) / len(times)
        per_sample = avg_batch / batch_size
        speedup = avg_forward / per_sample

        print(f"  Batch size {batch_size:2d}: {avg_batch*1000:6.2f}ms total, {per_sample*1000:5.2f}ms/sample ({speedup:.1f}x speedup)")

    # Test 3: MCTS with and without batching
    print(f"\n--- Test 3: MCTS Performance ---")

    mcts_config = config['mcts'].copy()

    # Without batching
    print(f"\n  Testing MCTS WITHOUT batching...")
    mcts_no_batch = MCTS(model, TicTacToeGame, mcts_config, device=device)
    game = TicTacToeGame()

    times_no_batch = []
    for i in range(num_games):
        game = TicTacToeGame()
        move_count = 0
        start = time.time()

        while not game.is_terminal() and move_count < 9:
            policy = mcts_no_batch.search(game, move_count, use_batching=False)
            action = np.argmax(policy)
            game.make_move(action)
            move_count += 1

        elapsed = time.time() - start
        times_no_batch.append(elapsed)

    avg_no_batch = sum(times_no_batch) / len(times_no_batch)
    print(f"    Average game time: {avg_no_batch:.3f}s ({num_games} games)")

    # With batching
    print(f"\n  Testing MCTS WITH batching (batch_size={mcts_config['batch_size']})...")
    mcts_batch = MCTS(model, TicTacToeGame, mcts_config, device=device)

    times_batch = []
    for i in range(num_games):
        game = TicTacToeGame()
        move_count = 0
        start = time.time()

        while not game.is_terminal() and move_count < 9:
            policy = mcts_batch.search(game, move_count, use_batching=True)
            action = np.argmax(policy)
            game.make_move(action)
            move_count += 1

        elapsed = time.time() - start
        times_batch.append(elapsed)

    avg_batch = sum(times_batch) / len(times_batch)
    batch_speedup = avg_no_batch / avg_batch
    print(f"    Average game time: {avg_batch:.3f}s ({num_games} games)")
    print(f"    Batching speedup: {batch_speedup:.2f}x")

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary for {device.upper()}")
    print(f"{'='*80}")
    print(f"  Forward pass: {avg_forward*1000:.2f}ms")
    print(f"  Batch-16 speedup: {speedup:.1f}x (from batching alone)")
    print(f"  MCTS game time (no batch): {avg_no_batch:.3f}s")
    print(f"  MCTS game time (batched): {avg_batch:.3f}s")
    print(f"  MCTS batching speedup: {batch_speedup:.2f}x")
    print(f"  Est. games/hour (8 workers): {8 * 3600 / avg_batch:.0f}")
    print(f"{'='*80}\n")

    return {
        'device': device,
        'forward_pass_ms': avg_forward * 1000,
        'batch_speedup': speedup,
        'game_time_no_batch': avg_no_batch,
        'game_time_batched': avg_batch,
        'mcts_speedup': batch_speedup,
        'games_per_hour': 8 * 3600 / avg_batch
    }


def main():
    # Load config
    with open('config/tictactoe.yaml') as f:
        config = yaml.safe_load(f)

    print("="*80)
    print("Tako Phase 2 Optimization Benchmark")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: d_model={config['model']['d_model']}, n_layers={config['model']['n_layers']}, N={config['model']['N']}")
    print(f"  MCTS: {config['mcts']['simulations']} sims, batch_size={config['mcts']['batch_size']}")

    results = {}

    # Benchmark CPU
    print(f"\n\n{'#'*80}")
    print("# PART 1: CPU Baseline")
    print(f"{'#'*80}")
    results['cpu'] = benchmark_device('cpu', config, num_games=5)

    # Benchmark GPU if available
    if torch.cuda.is_available():
        print(f"\n\n{'#'*80}")
        print("# PART 2: CUDA GPU")
        print(f"{'#'*80}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        results['cuda'] = benchmark_device('cuda', config, num_games=10)

        # Compare
        print(f"\n\n{'#'*80}")
        print("# COMPARISON: CPU vs GPU")
        print(f"{'#'*80}")
        cpu_game_time = results['cpu']['game_time_batched']
        gpu_game_time = results['cuda']['game_time_batched']
        total_speedup = cpu_game_time / gpu_game_time

        print(f"\n  CPU game time: {cpu_game_time:.3f}s")
        print(f"  GPU game time: {gpu_game_time:.3f}s")
        print(f"  Total GPU speedup: {total_speedup:.1f}x")
        print(f"\n  CPU games/hour (8 workers): {results['cpu']['games_per_hour']:.0f}")
        print(f"  GPU games/hour (8 workers): {results['cuda']['games_per_hour']:.0f}")
        print(f"  Improvement: {results['cuda']['games_per_hour'] / results['cpu']['games_per_hour']:.1f}x more games\n")

    elif torch.backends.mps.is_available():
        print(f"\n\n{'#'*80}")
        print("# PART 2: Apple MPS (Metal)")
        print(f"{'#'*80}")
        results['mps'] = benchmark_device('mps', config, num_games=10)

        # Compare
        print(f"\n\n{'#'*80}")
        print("# COMPARISON: CPU vs MPS")
        print(f"{'#'*80}")
        cpu_game_time = results['cpu']['game_time_batched']
        mps_game_time = results['mps']['game_time_batched']
        total_speedup = cpu_game_time / mps_game_time

        print(f"\n  CPU game time: {cpu_game_time:.3f}s")
        print(f"  MPS game time: {mps_game_time:.3f}s")
        print(f"  Total MPS speedup: {total_speedup:.1f}x")
        print(f"\n  CPU games/hour (8 workers): {results['cpu']['games_per_hour']:.0f}")
        print(f"  MPS games/hour (8 workers): {results['mps']['games_per_hour']:.0f}")
        print(f"  Improvement: {results['mps']['games_per_hour'] / results['cpu']['games_per_hour']:.1f}x more games\n")
    else:
        print(f"\n⚠️  No GPU available. Install CUDA or use Apple Silicon for GPU acceleration.\n")

    print("="*80)
    print("Benchmark complete!")
    print("="*80)


if __name__ == "__main__":
    main()

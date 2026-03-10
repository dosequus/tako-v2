"""Profile self-play pipeline to identify performance bottlenecks.

Usage:
    uv run python scripts/profile_selfplay.py --config config/othello.yaml
    uv run python scripts/profile_selfplay.py --config config/tictactoe.yaml --games 5
"""

import argparse
import time
import torch
import yaml
import sys
import copy
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.hrm import HRM
from games.othello import OthelloGame
from games.tictactoe import TicTacToeGame
from training.mcts import MCTS

GAME_REGISTRY = {
    'othello': OthelloGame,
    'tictactoe': TicTacToeGame,
}


class Timer:
    """Accumulating timer for profiling code sections."""

    def __init__(self):
        self.timings = defaultdict(lambda: {'total': 0.0, 'count': 0})

    @contextmanager
    def __call__(self, name):
        if self.sync_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if self.sync_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.timings[name]['total'] += elapsed
        self.timings[name]['count'] += 1

    sync_cuda = False

    def report(self):
        print("\n" + "=" * 70)
        print(f"{'Section':<35} {'Total (s)':>10} {'Count':>8} {'Avg (ms)':>10}")
        print("-" * 70)
        for name, data in sorted(self.timings.items(), key=lambda x: -x[1]['total']):
            avg_ms = (data['total'] / data['count']) * 1000 if data['count'] > 0 else 0
            print(f"{name:<35} {data['total']:>10.3f} {data['count']:>8} {avg_ms:>10.3f}")
        print("=" * 70)


def profile_model_inference(model, game_class, config, device, timer, n_evals=100):
    """Profile raw model inference speed."""
    print(f"\n--- Model Inference ({n_evals} evaluations) ---")
    game = game_class()
    game.reset()
    tokens = game.to_tokens().unsqueeze(0).to(device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model.predict(tokens, use_act=True, max_segments=1)

    # Single inference
    for _ in range(n_evals):
        with timer('model.predict (single)'):
            with torch.no_grad():
                model.predict(tokens, use_act=True, max_segments=1)

    # Batched inference at multiple sizes
    for bs in [16, 32, 64, 128]:
        tokens_batch = tokens.expand(bs, -1)
        n_runs = max(3, n_evals // bs)
        # Warmup
        for _ in range(2):
            with torch.no_grad():
                model.predict(tokens_batch, use_act=True, max_segments=1)
        for _ in range(n_runs):
            with timer(f'model.predict (batch={bs})'):
                with torch.no_grad():
                    model.predict(tokens_batch, use_act=True, max_segments=1)


def profile_game_ops(game_class, timer, n_ops=1000):
    """Profile game operations."""
    print(f"\n--- Game Operations ({n_ops} iterations) ---")
    game = game_class()

    for _ in range(n_ops):
        with timer('game.reset'):
            game.reset()

        with timer('game.to_tokens'):
            game.to_tokens()

        with timer('game.legal_moves'):
            game.legal_moves()

        with timer('game.deepcopy'):
            copy.deepcopy(game)

        # Play a few moves
        legal = game.legal_moves()
        if legal:
            with timer('game.make_move'):
                game.make_move(legal[0])


def profile_mcts_search(model, game_class, config, device, timer, n_searches=10):
    """Profile MCTS search."""
    print(f"\n--- MCTS Search ({n_searches} searches, {config['mcts']['simulations']} sims each) ---")
    mcts = MCTS(model, game_class, config['mcts'], device=device)
    game = game_class()
    game.reset()

    for _ in range(n_searches):
        with timer('mcts.search (full)'):
            mcts.search(game, move_num=0)


def profile_full_game(model, game_class, config, device, timer, n_games=3):
    """Profile full self-play game generation."""
    print(f"\n--- Full Game Generation ({n_games} games) ---")
    mcts = MCTS(model, game_class, config['mcts'], device=device)

    for _ in range(n_games):
        game = game_class()
        game.reset()
        move_num = 0

        with timer('full_game'):
            while not game.is_terminal():
                with timer('game_move_mcts'):
                    policy = mcts.search(game, move_num)
                action = np.argmax(policy) if move_num >= config['mcts']['temperature_threshold'] else np.random.choice(len(policy), p=policy)
                game.make_move(action)
                move_num += 1

        print(f"  Game finished in {move_num} moves")


def profile_symmetry(game_class, timer, n_samples=500):
    """Profile symmetry augmentation."""
    print(f"\n--- Symmetry Augmentation ({n_samples} samples) ---")
    game = game_class()
    game.reset()

    # Play a few moves to get a non-trivial state
    for _ in range(4):
        legal = game.legal_moves()
        if legal:
            game.make_move(legal[0])

    state = game.to_tokens()
    policy = np.random.dirichlet([1.0] * game.action_size())

    if hasattr(game, 'get_symmetries'):
        for _ in range(n_samples):
            with timer('get_symmetries'):
                game.get_symmetries(state, policy)
    else:
        print("  Game does not support symmetries")


def main():
    parser = argparse.ArgumentParser(description='Profile Tako self-play pipeline')
    parser.add_argument('--config', type=str, default='config/othello.yaml')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--games', type=int, default=3, help='Number of full games to profile')
    parser.add_argument('--mcts-sims', type=int, default=None, help='Override MCTS simulations')
    parser.add_argument('--batch-size', type=int, default=None, help='Override MCTS batch size')
    parser.add_argument('--optimize', action='store_true', help='Enable torch.compile + bfloat16')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mcts_sims is not None:
        config['mcts']['simulations'] = args.mcts_sims
    if args.batch_size is not None:
        config['mcts']['batch_size'] = args.batch_size

    game_name = config.get('game', 'othello')
    game_class = GAME_REGISTRY[game_name]
    device = args.device

    print(f"Profiling: {game_name}")
    print(f"Device: {device}")
    print(f"MCTS sims: {config['mcts']['simulations']}")
    print(f"Batch size: {config['mcts'].get('batch_size', 16)}")

    # Create model
    model = HRM(**config['model'], optimize=False)
    model.to(device)
    model.eval()

    if args.optimize:
        if device == 'cuda' and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif device == 'cuda':
            dtype = torch.float16
        else:
            dtype = None
        model.optimize_for_inference(use_compile=True, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M parameters")

    timer = Timer()
    timer.sync_cuda = (device == 'cuda')

    # Run profiling sections
    profile_game_ops(game_class, timer)
    profile_model_inference(model, game_class, config, device, timer)
    profile_symmetry(game_class, timer)
    profile_mcts_search(model, game_class, config, device, timer)
    profile_full_game(model, game_class, config, device, timer, n_games=args.games)

    # Estimate full training cost
    timer.report()

    # Extrapolate
    print("\n--- Estimates ---")
    if 'full_game' in timer.timings:
        game_time = timer.timings['full_game']['total'] / timer.timings['full_game']['count']
        games_per_worker = config['selfplay']['games_per_worker']
        num_workers = config['selfplay']['num_workers']
        min_buffer = config['selfplay']['min_buffer_size']
        game_samples = 60 * (8 if config['selfplay'].get('use_symmetry', False) else 1)  # rough estimate
        games_for_bootstrap = min_buffer / game_samples
        bootstrap_batches = games_for_bootstrap / (games_per_worker * num_workers)

        print(f"Avg game time: {game_time:.1f}s")
        print(f"Games per batch: {games_per_worker * num_workers}")
        print(f"Est. samples/game: ~{game_samples}")
        print(f"Bootstrap games needed: ~{games_for_bootstrap:.0f}")
        print(f"Bootstrap batches: ~{bootstrap_batches:.0f}")
        print(f"Est. bootstrap time (serial): {games_for_bootstrap * game_time / 3600:.1f}h")
        print(f"Est. bootstrap time ({num_workers} workers): {games_for_bootstrap * game_time / num_workers / 3600:.1f}h")

    if 'model.predict (single)' in timer.timings:
        single = timer.timings['model.predict (single)']['total'] / timer.timings['model.predict (single)']['count']
        print(f"\nModel inference: {single*1000:.1f}ms/eval")
        sims = config['mcts']['simulations']
        print(f"Neural evals per game (~60 moves × {sims} sims): ~{60 * sims}")
        print(f"Est. neural eval time per game (unbatched): {60 * sims * single:.1f}s")


if __name__ == '__main__':
    main()

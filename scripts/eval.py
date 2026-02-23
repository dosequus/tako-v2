"""Evaluation script for Tako models."""

import argparse
import yaml
import torch
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from games.othello import OthelloGame
from games.tictactoe import TicTacToeGame
from training.evaluator import Evaluator

# Game registry
GAME_REGISTRY = {
    'othello': OthelloGame,
    'tictactoe': TicTacToeGame,
    # 'hex': HexGame,  # Future
    # 'chess': ChessGame,  # Future
}


def main():
    parser = argparse.ArgumentParser(description='Evaluate Tako model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint to evaluate')
    parser.add_argument('--opponent', type=str, default='random',
                        help='Opponent type: "random" or path to checkpoint')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games to play')
    parser.add_argument('--config', type=str, default='config/othello.yaml',
                        help='Path to config file (for MCTS settings)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for inference (cpu/cuda)')
    args = parser.parse_args()

    print(f"[Eval] Evaluating checkpoint: {args.checkpoint}")
    print(f"[Eval] Opponent: {args.opponent}")
    print(f"[Eval] Games: {args.games}")
    print(f"[Eval] Device: {args.device}")

    # Load config for MCTS settings
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Get game class from config
    game_name = config.get('game', 'othello')  # Default to othello for backward compatibility
    if game_name not in GAME_REGISTRY:
        raise ValueError(f"Unknown game: {game_name}. Available: {list(GAME_REGISTRY.keys())}")

    game_class = GAME_REGISTRY[game_name]
    print(f"[Eval] Game: {game_name}")

    # Use evaluation MCTS config (more simulations)
    mcts_config = config['mcts'].copy()
    if 'evaluation' in config:
        mcts_config['simulations'] = config['evaluation'].get('mcts_sims', mcts_config['simulations'])

    print(f"[Eval] MCTS simulations: {mcts_config['simulations']}")

    # Create evaluator
    evaluator = Evaluator(
        game_class=game_class,
        mcts_config=mcts_config,
        device=args.device
    )

    # Run evaluation
    if args.opponent == 'random':
        print(f"[Eval] Playing {args.games} games vs random player...")
        results = evaluator.vs_random(args.checkpoint, num_games=args.games)
    else:
        print(f"[Eval] Playing {args.games} games vs opponent checkpoint...")
        results = evaluator.head_to_head(args.checkpoint, args.opponent, num_games=args.games)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total games:  {args.games}")
    print(f"Wins:         {results['wins']}")
    print(f"Losses:       {results['losses']}")
    print(f"Draws:        {results['draws']}")
    print(f"Win rate:     {results['win_rate']:.3f}")
    print("=" * 50)

    # Sanity check
    if args.opponent == 'random':
        if results['win_rate'] < 0.6:
            print("\n⚠️  Warning: Win rate vs random is low. Model may need more training.")
        elif results['win_rate'] > 0.95:
            print("\n✓ Excellent: Model dominates random baseline!")
        else:
            print("\n✓ Good: Model is learning.")


if __name__ == '__main__':
    main()

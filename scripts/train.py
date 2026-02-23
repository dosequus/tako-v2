"""Main training script for Tako Phase 1: Othello self-play."""

import argparse
import yaml
import ray
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.hrm import HRM
from games.othello import OthelloGame
from games.tictactoe import TicTacToeGame
from training.replay_buffer import ReplayBuffer
from training.learner import Learner
from training.worker import SelfPlayWorker

# Game registry
GAME_REGISTRY = {
    'othello': OthelloGame,
    'tictactoe': TicTacToeGame,
    # 'hex': HexGame,  # Future
    # 'chess': ChessGame,  # Future
}


def main():
    parser = argparse.ArgumentParser(description='Tako Phase 1: Othello Self-Play Training')
    parser.add_argument('--config', type=str, default='config/othello.yaml',
                        help='Path to config file')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of self-play workers (overrides config)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for learner (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (overrides config)')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Get game class from config
    game_name = config.get('game', 'othello')  # Default to othello for backward compatibility
    if game_name not in GAME_REGISTRY:
        raise ValueError(f"Unknown game: {game_name}. Available: {list(GAME_REGISTRY.keys())}")

    game_class = GAME_REGISTRY[game_name]

    print(f"[Train] Starting Tako training")
    print(f"[Train] Game: {game_name}")
    print(f"[Train] Config: {args.config}")
    print(f"[Train] Device: {args.device}")

    # Override config with CLI args
    if args.num_workers is not None:
        config['selfplay']['num_workers'] = args.num_workers

    num_workers = config['selfplay']['num_workers']
    games_per_worker = config['selfplay']['games_per_worker']
    print(f"[Train] Workers: {num_workers}")
    print(f"[Train] Games per worker: {games_per_worker}")

    # Detect available accelerators (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        # NVIDIA GPUs
        num_gpus = torch.cuda.device_count()
        device_type = 'cuda'
        print(f"[Train] Detected {num_gpus} CUDA GPU(s)")
        for i in range(num_gpus):
            print(f"[Train]   GPU {i}: {torch.cuda.get_device_name(i)}")
    elif torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3) - MPS
        num_gpus = 1  # MPS typically only has one device
        device_type = 'mps'
        print(f"[Train] Detected Apple MPS (Metal Performance Shaders)")
        print(f"[Train]   Note: All workers will share the MPS device")
    else:
        # CPU fallback
        num_gpus = 0
        device_type = 'cpu'
        print(f"[Train] No GPU detected - using CPU")
        print(f"[Train]   Tip: Enable GPU in Colab (Runtime → Change runtime type → GPU)")

    # Initialize Ray
    if not ray.is_initialized():
        # Configure Ray to use the current Python environment
        # Note: Ray may show a VIRTUAL_ENV warning with uv - this is harmless
        # Workers will correctly inherit the active Python environment
        ray_init_kwargs = {
            'logging_level': logging.INFO,
            'log_to_driver': True,
            'ignore_reinit_error': True,
            'include_dashboard': True,
        }

        # Only set num_gpus for CUDA (not for MPS)
        if device_type == 'cuda':
            ray_init_kwargs['num_gpus'] = num_gpus

        ray_ctx = ray.init(**ray_init_kwargs)
        print("[Train] Ray initialized")
        print(f"[Train] View worker logs in the dashboard under 'Actors' tab")
        print(f"[Train] Note: VIRTUAL_ENV warning from Ray/uv is harmless and can be ignored")

    # Create model
    model = HRM(**config['model'])
    print(f"[Train] Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config['selfplay']['replay_buffer_size'],
        min_size=config['selfplay']['min_buffer_size']
    )
    print(f"[Train] Replay buffer: capacity={replay_buffer.capacity}, min_size={replay_buffer.min_size}")

    # Create learner
    learner = Learner(model, replay_buffer, config, device=args.device)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"[Train] Resuming from checkpoint: {args.resume}")
        learner.load_checkpoint(args.resume)

    # Create self-play workers with GPU acceleration
    print(f"[Train] Creating {num_workers} self-play workers...")
    workers = []
    for i in range(num_workers):
        # Assign device to worker
        if device_type == 'cuda':
            # Distribute workers across available GPUs
            worker_device = f'cuda:{i % num_gpus}'
        elif device_type == 'mps':
            # All workers share the MPS device
            worker_device = 'mps'
        else:
            # CPU fallback
            worker_device = 'cpu'

        worker = SelfPlayWorker.remote(
            worker_id=i,
            game_class=game_class,
            model_config=config['model'],
            mcts_config=config['mcts'],
            opponent_pool_config={'recent_weight': config['selfplay']['recent_weight']},
            device=worker_device
        )
        workers.append(worker)

    print(f"[Train] Workers created on {device_type} device(s)")

    # Bootstrap phase: collect initial data
    print(f"[Train] Bootstrap phase: Collecting {replay_buffer.min_size} positions...")
    bootstrap_games = 0
    with tqdm(total=replay_buffer.min_size, desc="Bootstrap", unit="samples") as pbar:
        while not replay_buffer.ready():
            # Generate games from all workers
            futures = [w.generate_batch.remote(games_per_worker) for w in workers]  # 10 games per worker
            results = ray.get(futures)

            # Add samples to buffer
            prev_size = len(replay_buffer)
            for samples in results:
                replay_buffer.add_samples(samples)

            bootstrap_games += games_per_worker * num_workers
            new_samples = len(replay_buffer) - prev_size
            pbar.update(new_samples)
            pbar.set_postfix({'games': bootstrap_games})

    print(f"[Train] Bootstrap complete. Starting training.")

    # Save initial checkpoint
    checkpoint_path = learner.save_checkpoint()
    print(f"[Train] Initial checkpoint saved: {checkpoint_path}")

    # Sync workers to initial checkpoint
    for w in workers:
        ray.get(w.load_checkpoint.remote(checkpoint_path, learner.global_step))
        ray.get(w.update_opponent_pool.remote(learner.get_opponent_pool()))

    # Training loop
    epoch = 0
    total_games_generated = bootstrap_games
    checkpoint_interval = config['checkpointing']['save_interval']

    # Create logs directory
    log_dir = Path(f"logs/{game_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = log_dir / "metrics.json"
    metrics = []

    # Determine number of epochs
    num_epochs = args.epochs if args.epochs is not None else 50  # Default to 50 epochs

    print(f"[Train] Training for {num_epochs} epochs")

    try:
        for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs", unit="epoch"):
            print(f"\n[Train] === Epoch {epoch} ===")

            # Generate self-play games
            print(f"[Train] Generating {games_per_worker * num_workers} games...")
            futures = [w.generate_batch.remote(games_per_worker) for w in workers]
            results = ray.get(futures)

            # Add samples to buffer
            total_samples = 0
            for samples in results:
                replay_buffer.add_samples(samples)
                total_samples += len(samples)

            total_games_generated += games_per_worker * num_workers
            print(f"[Train] Generated {total_samples} samples")
            print(f"[Train] Buffer: {replay_buffer.stats()}")

            # Training steps
            steps_per_epoch = checkpoint_interval
            print(f"[Train] Training for {steps_per_epoch} steps...")

            epoch_losses = {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'act_loss': 0.0,
                'total_loss': 0.0,
                'lr': 0.0
            }

            with tqdm(range(steps_per_epoch), desc="Training", unit="step") as pbar:
                for step in pbar:
                    losses = learner.train_step()

                    # Accumulate losses
                    for k, v in losses.items():
                        epoch_losses[k] += v

                    # Update progress bar
                    pbar.set_postfix({
                        'policy': f"{losses['policy_loss']:.4f}",
                        'value': f"{losses['value_loss']:.4f}",
                        'act': f"{losses['act_loss']:.4f}",
                        'lr': f"{losses['lr']:.2e}"
                    })

            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= steps_per_epoch

            print(f"[Train] Epoch {epoch} losses: {epoch_losses}")

            # Save checkpoint
            checkpoint_path = learner.save_checkpoint()
            print(f"[Train] Checkpoint saved: {checkpoint_path}")

            # Sync workers to latest checkpoint
            print(f"[Train] Syncing workers to checkpoint...")
            for w in workers:
                ray.get(w.load_checkpoint.remote(checkpoint_path, learner.global_step))
                ray.get(w.update_opponent_pool.remote(learner.get_opponent_pool()))

            # Log metrics
            metric = {
                'epoch': epoch,
                'step': learner.global_step,
                'total_games': total_games_generated,
                'buffer_size': len(replay_buffer),
                'losses': epoch_losses,
                'timestamp': datetime.now().isoformat()
            }
            metrics.append(metric)

            # Save metrics
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"[Train] Metrics saved to {metrics_file}")

            # Optionally run evaluation every N epochs
            # (Deferred - evaluation script is separate)

    except KeyboardInterrupt:
        print("\n[Train] Training interrupted by user")

    finally:
        # Save final checkpoint
        if learner.global_step > 0:
            final_checkpoint = learner.save_checkpoint()
            print(f"[Train] Final checkpoint saved: {final_checkpoint}")

        # Shutdown Ray
        ray.shutdown()
        print("[Train] Ray shutdown complete")

    print("[Train] Training complete!")


if __name__ == '__main__':
    main()

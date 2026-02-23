"""Self-play workers for distributed game generation."""

import ray
import torch
import numpy as np
import random
import logging
import sys
from typing import List, Dict, Tuple
import copy

from training.mcts import MCTS

# Configure logging for Ray workers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add handler to output to stdout (required for Ray log_to_driver)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                                 datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@ray.remote
class SelfPlayWorker:
    """Ray remote actor for self-play game generation.

    Each worker:
    - Maintains its own copy of the model (CPU inference)
    - Runs MCTS to generate self-play games
    - Samples opponents from checkpoint pool
    - Returns training samples (state, policy, value) tuples
    """

    def __init__(
        self,
        worker_id: int,
        game_class,
        model_config: Dict,
        mcts_config: Dict,
        opponent_pool_config: Dict,
        device: str = 'cpu'
    ):
        """Initialize self-play worker.

        Args:
            worker_id: Unique worker identifier
            game_class: Game class (e.g., OthelloGame)
            model_config: Model configuration dict
            mcts_config: MCTS configuration dict
            opponent_pool_config: Opponent pool config with 'recent_weight'
            device: Device for inference (default: 'cpu')
        """
        self.worker_id = worker_id
        self.game_class = game_class
        self.model_config = model_config
        self.mcts_config = mcts_config
        self.recent_weight = opponent_pool_config['recent_weight']
        self.device = device

        # Import HRM here (inside Ray worker)
        from model.hrm import HRM

        # Create model
        self.model = HRM(**model_config)
        self.model.to(device)
        self.model.eval()

        # Create MCTS
        self.mcts = MCTS(self.model, game_class, mcts_config, device=device)

        # Opponent pool state
        self.opponent_checkpoints: List[Tuple[str, int]] = []  # (path, step)
        self.current_checkpoint_step = 0

        # Log initialization
        logger.info(f"Worker {self.worker_id}: Initialized on {device} with {game_class.__name__}")

    def load_checkpoint(self, checkpoint_path: str, step: int):
        """Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            step: Step number of this checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_checkpoint_step = step
        logger.info(f"Worker {self.worker_id}: Loaded checkpoint step {step}")

    def update_opponent_pool(self, checkpoints: List[Tuple[str, int]]):
        """Update opponent pool with latest checkpoints.

        Args:
            checkpoints: List of (checkpoint_path, step) tuples
        """
        self.opponent_checkpoints = checkpoints.copy()
        logger.info(f"Worker {self.worker_id}: Opponent pool updated ({len(checkpoints)} checkpoints)")

    def generate_game(self) -> List[Dict]:
        """Generate one self-play game and return training samples.

        Returns:
            List of samples, each with keys:
                - 'state': torch.Tensor of tokens [seq_len]
                - 'policy': np.ndarray of MCTS visit counts [action_size]
                - 'value': float in {-1.0, 0.0, 1.0} (game outcome)
        """
        # Create new game
        game = self.game_class()
        game.reset()

        # Sample opponent (if pool is available)
        opponent_model = None
        opponent_step = None
        if len(self.opponent_checkpoints) > 1:
            opponent_model = self._sample_opponent()
            # Get opponent step from last sampled checkpoint
            if hasattr(self, '_last_opponent_step'):
                opponent_step = self._last_opponent_step

        # Decide which model plays as which player
        # Randomly assign current model to player 1 or 2
        current_model_player = random.choice([1, 2])

        logger.info(
            f"Worker {self.worker_id}: Starting game "
            f"(current_step={self.current_checkpoint_step}, "
            f"opponent_step={opponent_step if opponent_step else 'self'}, "
            f"playing_as={'P1' if current_model_player == 1 else 'P2'})"
        )

        # Collect samples for this game
        samples = []
        move_num = 0

        while not game.is_terminal():
            # Determine which model to use for current player
            if game.current_player == current_model_player:
                # Use current model
                model_to_use = self.mcts
                is_current_model = True
            else:
                # Use opponent (or self if no opponent)
                model_to_use = opponent_model if opponent_model else self.mcts
                is_current_model = False

            # Get MCTS policy
            policy = model_to_use.search(game, move_num)

            # Save sample (only from current model's perspective)
            if is_current_model:
                samples.append({
                    'state': game.to_tokens().clone(),
                    'policy': policy.copy(),
                    'move_num': move_num,
                    'player': game.current_player
                })

            # Sample action from policy (with temperature if early game)
            if move_num < self.mcts_config['temperature_threshold']:
                # Sample from distribution
                action = np.random.choice(len(policy), p=policy)
            else:
                # Argmax (deterministic)
                action = np.argmax(policy)

            # Log move periodically
            if move_num % 10 == 0:
                logger.debug(
                    f"Worker {self.worker_id}: Move {move_num}, "
                    f"player={game.current_player}, action={action}"
                )

            # Make move
            game.make_move(action)
            move_num += 1

        # Get game outcome
        # outcome() is from perspective of current player at terminal state
        # We need to assign correct outcome to each sample based on which player won
        final_outcome = game.outcome()
        final_player = game.current_player

        # Assign outcomes to samples
        for sample in samples:
            # If sample's player is the final player, use final_outcome
            # Otherwise flip sign
            if sample['player'] == final_player:
                sample['value'] = float(final_outcome)
            else:
                sample['value'] = float(-final_outcome)

            # Remove temporary fields
            del sample['move_num']
            del sample['player']

        # Log game result
        outcome_str = {1.0: "WIN", 0.0: "DRAW", -1.0: "LOSS"}[final_outcome]
        logger.info(
            f"Worker {self.worker_id}: Game complete - {outcome_str} "
            f"({move_num} moves, {len(samples)} samples)"
        )

        return samples

    def generate_batch(self, num_games: int) -> List[Dict]:
        """Generate multiple self-play games.

        Args:
            num_games: Number of games to generate

        Returns:
            List of all samples from all games
        """
        logger.info(f"Worker {self.worker_id}: Generating batch of {num_games} games...")
        all_samples = []

        for game_idx in range(num_games):
            samples = self.generate_game()
            all_samples.extend(samples)

            # Log progress every 10 games
            if (game_idx + 1) % 10 == 0:
                logger.info(
                    f"Worker {self.worker_id}: Generated {game_idx + 1}/{num_games} games "
                    f"({len(all_samples)} samples so far)"
                )

        logger.info(
            f"Worker {self.worker_id}: Batch complete - {num_games} games, "
            f"{len(all_samples)} samples"
        )
        return all_samples

    def _sample_opponent(self):
        """Sample an opponent from the checkpoint pool.

        Returns:
            MCTS instance with opponent model loaded
        """
        if not self.opponent_checkpoints:
            return None

        # Determine pool split point
        n_total = len(self.opponent_checkpoints)
        n_recent = max(1, int(n_total * 0.3))  # Top 30% are "recent"

        # Sample: 70% from recent, 30% from older
        if random.random() < self.recent_weight:
            # Sample from recent
            checkpoint_path, step = random.choice(self.opponent_checkpoints[-n_recent:])
            pool_type = "recent"
        else:
            # Sample from older (if available)
            if n_total > n_recent:
                checkpoint_path, step = random.choice(self.opponent_checkpoints[:-n_recent])
                pool_type = "older"
            else:
                checkpoint_path, step = random.choice(self.opponent_checkpoints)
                pool_type = "recent"

        # Store for logging
        self._last_opponent_step = step

        logger.debug(
            f"Worker {self.worker_id}: Sampled opponent from {pool_type} pool (step {step})"
        )

        # Create opponent model
        from model.hrm import HRM
        opponent_model = HRM(**self.model_config)
        opponent_model.to(self.device)
        opponent_model.eval()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        opponent_model.load_state_dict(checkpoint['model_state_dict'])

        # Create MCTS for opponent
        opponent_mcts = MCTS(opponent_model, self.game_class, self.mcts_config, device=self.device)

        return opponent_mcts

    def get_stats(self) -> Dict:
        """Return worker statistics.

        Returns:
            Dict with worker state info
        """
        return {
            'worker_id': self.worker_id,
            'current_checkpoint_step': self.current_checkpoint_step,
            'opponent_pool_size': len(self.opponent_checkpoints),
            'device': str(self.device)
        }

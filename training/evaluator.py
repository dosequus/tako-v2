"""Evaluation infrastructure for model strength assessment."""

import torch
import numpy as np
from typing import Dict, Optional
import random
from tqdm import tqdm

from training.mcts import MCTS
from training.minimax import MinimaxPlayer


class Evaluator:
    """Evaluate model strength through various benchmarks.

    Supports:
    - Head-to-head comparison between two models
    - Evaluation vs random baseline
    - Win rate statistics
    """

    def __init__(
        self,
        game_class,
        mcts_config: Dict,
        device: str = 'cpu'
    ):
        """Initialize evaluator.

        Args:
            game_class: Game class (e.g., OthelloGame)
            mcts_config: MCTS configuration dict
            device: Device for inference ('cpu' or 'cuda')
        """
        self.game_class = game_class
        self.mcts_config = mcts_config
        self.device = device

    def head_to_head(
        self,
        model1_path: str,
        model2_path: Optional[str] = None,
        num_games: int = 100
    ) -> Dict:
        """Evaluate model1 vs model2 in head-to-head games.

        Args:
            model1_path: Path to checkpoint for model 1
            model2_path: Path to checkpoint for model 2 (None = vs self)
            num_games: Number of games to play

        Returns:
            Dict with keys:
                - wins: Number of wins for model1
                - losses: Number of losses for model1
                - draws: Number of draws
                - win_rate: Win rate (wins / total games)
        """
        from model.hrm import HRM

        # Load model 1
        checkpoint1 = torch.load(model1_path, map_location=self.device)
        model1 = HRM(**checkpoint1['config']['model'])
        model1.load_state_dict(checkpoint1['model_state_dict'])
        model1.to(self.device)
        model1.eval()

        mcts1 = MCTS(model1, self.game_class, self.mcts_config, device=self.device)

        # Load model 2 (or use model1 for self-play)
        if model2_path is not None:
            checkpoint2 = torch.load(model2_path, map_location=self.device)
            model2 = HRM(**checkpoint2['config']['model'])
            model2.load_state_dict(checkpoint2['model_state_dict'])
            model2.to(self.device)
            model2.eval()
            mcts2 = MCTS(model2, self.game_class, self.mcts_config, device=self.device)
        else:
            mcts2 = mcts1

        # Play games
        wins = 0
        losses = 0
        draws = 0

        with tqdm(range(num_games), desc="Evaluating", unit="game") as pbar:
            for game_num in pbar:
                # Alternate who plays first
                model1_is_player1 = (game_num % 2 == 0)

                outcome = self._play_game(mcts1, mcts2, model1_is_player1)

                if outcome == 1.0:
                    wins += 1
                elif outcome == -1.0:
                    losses += 1
                else:
                    draws += 1

                # Update progress bar
                current_win_rate = wins / (game_num + 1)
                pbar.set_postfix({
                    'W': wins,
                    'L': losses,
                    'D': draws,
                    'WR': f"{current_win_rate:.3f}"
                })

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_games if num_games > 0 else 0.0
        }

    def vs_random(
        self,
        model_path: str,
        num_games: int = 100
    ) -> Dict:
        """Evaluate model against random baseline.

        Args:
            model_path: Path to model checkpoint
            num_games: Number of games to play

        Returns:
            Dict with win rate statistics
        """
        from model.hrm import HRM

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        model = HRM(**checkpoint['config']['model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        mcts = MCTS(model, self.game_class, self.mcts_config, device=self.device)

        # Play against random
        wins = 0
        losses = 0
        draws = 0

        with tqdm(range(num_games), desc="vs Random", unit="game") as pbar:
            for game_num in pbar:
                # Alternate who plays first
                model_is_player1 = (game_num % 2 == 0)

                outcome = self._play_game_vs_random(mcts, model_is_player1)

                if outcome == 1.0:
                    wins += 1
                elif outcome == -1.0:
                    losses += 1
                else:
                    draws += 1

                # Update progress bar
                current_win_rate = wins / (game_num + 1)
                pbar.set_postfix({
                    'W': wins,
                    'L': losses,
                    'D': draws,
                    'WR': f"{current_win_rate:.3f}"
                })

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_games if num_games > 0 else 0.0
        }

    def vs_minimax(
        self,
        model_path: str,
        num_games: int = 100,
        minimax_depth: int = -1,
    ) -> Dict:
        """Evaluate model against optimal minimax player.

        Best used with small games (TicTacToe) where minimax can fully
        solve the game tree. For larger games, set minimax_depth to limit
        search depth.

        Args:
            model_path: Path to model checkpoint
            num_games: Number of games to play
            minimax_depth: Max minimax search depth (-1 = unlimited/exact solve)

        Returns:
            Dict with win rate statistics
        """
        from model.hrm import HRM

        checkpoint = torch.load(model_path, map_location=self.device)
        model = HRM(**checkpoint['config']['model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        mcts = MCTS(model, self.game_class, self.mcts_config, device=self.device)
        minimax = MinimaxPlayer(max_depth=minimax_depth)

        wins = 0
        losses = 0
        draws = 0

        with tqdm(range(num_games), desc="vs Minimax", unit="game") as pbar:
            for game_num in pbar:
                model_is_player1 = (game_num % 2 == 0)
                outcome = self._play_game_vs_minimax(mcts, minimax, model_is_player1)

                if outcome == 1.0:
                    wins += 1
                elif outcome == -1.0:
                    losses += 1
                else:
                    draws += 1

                current_win_rate = wins / (game_num + 1)
                pbar.set_postfix({
                    'W': wins,
                    'L': losses,
                    'D': draws,
                    'WR': f"{current_win_rate:.3f}"
                })

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_games if num_games > 0 else 0.0
        }

    def _play_game(
        self,
        mcts1: MCTS,
        mcts2: MCTS,
        model1_is_player1: bool
    ) -> float:
        """Play one game between two MCTS instances.

        Args:
            mcts1: MCTS for model 1
            mcts2: MCTS for model 2
            model1_is_player1: Whether model1 plays as player 1

        Returns:
            Outcome from model1's perspective: 1.0 (win), 0.0 (draw), -1.0 (loss)
        """
        game = self.game_class()
        game.reset()

        move_num = 0

        while not game.is_terminal():
            # Determine which MCTS to use
            if game.current_player == 1:
                mcts = mcts1 if model1_is_player1 else mcts2
            else:
                mcts = mcts2 if model1_is_player1 else mcts1

            # Get policy from MCTS
            policy = mcts.search(game, move_num)

            # Select action (deterministic for evaluation)
            action = np.argmax(policy)

            # Make move
            game.make_move(action)
            move_num += 1

        # Get outcome
        final_outcome = game.outcome()  # From perspective of final player

        # Convert to model1's perspective
        if game.current_player == 1:
            # Model1 is player 1
            outcome = final_outcome if model1_is_player1 else -final_outcome
        else:
            # Model1 is player 2
            outcome = -final_outcome if model1_is_player1 else final_outcome

        return float(outcome)

    def _play_game_vs_random(
        self,
        mcts: MCTS,
        model_is_player1: bool
    ) -> float:
        """Play one game between MCTS model and random player.

        Args:
            mcts: MCTS for model
            model_is_player1: Whether model plays as player 1

        Returns:
            Outcome from model's perspective: 1.0 (win), 0.0 (draw), -1.0 (loss)
        """
        game = self.game_class()
        game.reset()

        move_num = 0

        while not game.is_terminal():
            # Determine if current player is the model
            is_model_turn = (game.current_player == 1 and model_is_player1) or \
                            (game.current_player == 2 and not model_is_player1)

            if is_model_turn:
                # Use MCTS
                policy = mcts.search(game, move_num)
                action = np.argmax(policy)
            else:
                # Random player
                legal_moves = game.legal_moves()
                action = random.choice(legal_moves)

            # Make move
            game.make_move(action)
            move_num += 1

        # Get outcome
        final_outcome = game.outcome()  # From perspective of final player

        # Convert to model's perspective
        if game.current_player == 1:
            outcome = final_outcome if model_is_player1 else -final_outcome
        else:
            outcome = -final_outcome if model_is_player1 else final_outcome

        return float(outcome)

    def _play_game_vs_minimax(
        self,
        mcts: MCTS,
        minimax: MinimaxPlayer,
        model_is_player1: bool
    ) -> float:
        """Play one game between MCTS model and minimax player.

        Args:
            mcts: MCTS for model
            minimax: MinimaxPlayer instance
            model_is_player1: Whether model plays as player 1

        Returns:
            Outcome from model's perspective: 1.0 (win), 0.0 (draw), -1.0 (loss)
        """
        game = self.game_class()
        game.reset()
        minimax.clear_cache()

        move_num = 0

        while not game.is_terminal():
            is_model_turn = (game.current_player == 1 and model_is_player1) or \
                            (game.current_player == 2 and not model_is_player1)

            if is_model_turn:
                policy = mcts.search(game, move_num)
                action = np.argmax(policy)
            else:
                action = minimax.best_move(game)

            game.make_move(action)
            move_num += 1

        final_outcome = game.outcome()

        if game.current_player == 1:
            outcome = final_outcome if model_is_player1 else -final_outcome
        else:
            outcome = -final_outcome if model_is_player1 else final_outcome

        return float(outcome)

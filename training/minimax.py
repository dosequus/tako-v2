"""Minimax player with alpha-beta pruning for convergence testing.

Provides an optimal opponent for small games (TicTacToe, small Othello).
Use this to verify that a trained model converges toward perfect play.
"""

import copy
from functools import lru_cache
from typing import Dict, Tuple

from games.base import BaseGame


class MinimaxPlayer:
    """Optimal minimax player with alpha-beta pruning and transposition table.

    Solves the game tree exactly — suitable for small games like TicTacToe.
    For larger games, use max_depth to limit search depth (becomes an
    approximation with heuristic eval = 0 at the depth cutoff).
    """

    def __init__(self, max_depth: int = -1):
        """Initialize minimax player.

        Args:
            max_depth: Maximum search depth. -1 = unlimited (full solve).
        """
        self.max_depth = max_depth
        self.transposition_table: Dict[bytes, Tuple[float, int]] = {}
        self.nodes_searched = 0

    def clear_cache(self):
        """Clear transposition table between games."""
        self.transposition_table.clear()
        self.nodes_searched = 0

    def best_move(self, game: BaseGame) -> int:
        """Return the optimal move for the current player.

        Args:
            game: Current game state (not modified).

        Returns:
            Best move index.
        """
        self.nodes_searched = 0
        legal = game.legal_moves()
        if len(legal) == 1:
            return legal[0]

        best_val = -2.0
        best_action = legal[0]

        for move in legal:
            child = copy.deepcopy(game)
            child.make_move(move)
            # negamax: opponent's value is negated
            val = -self._negamax(child, depth=1, alpha=-2.0, beta=2.0)
            if val > best_val:
                best_val = val
                best_action = move

        return best_action

    def evaluate(self, game: BaseGame) -> float:
        """Return the minimax value of the position for the current player.

        Args:
            game: Current game state.

        Returns:
            1.0 = current player wins with best play
            0.0 = draw with best play
           -1.0 = current player loses with best play
        """
        self.nodes_searched = 0
        return self._negamax(game, depth=0, alpha=-2.0, beta=2.0)

    def _game_key(self, game: BaseGame) -> bytes:
        """Create a hashable key from the game state for the transposition table."""
        return game.board.tobytes() + bytes([game.current_player])

    def _negamax(self, game: BaseGame, depth: int, alpha: float, beta: float) -> float:
        """Negamax with alpha-beta pruning.

        Returns value from the perspective of game.current_player.
        """
        self.nodes_searched += 1

        if game.is_terminal():
            return game.outcome()

        if self.max_depth >= 0 and depth >= self.max_depth:
            return 0.0  # heuristic: assume draw at depth cutoff

        key = self._game_key(game)
        if key in self.transposition_table:
            cached_val, cached_depth = self.transposition_table[key]
            remaining = self.max_depth - depth if self.max_depth >= 0 else float('inf')
            cached_remaining = self.max_depth - cached_depth if self.max_depth >= 0 else float('inf')
            if cached_remaining >= remaining:
                return cached_val

        best_val = -2.0
        for move in game.legal_moves():
            child = copy.deepcopy(game)
            child.make_move(move)
            val = -self._negamax(child, depth + 1, -beta, -alpha)
            best_val = max(best_val, val)
            alpha = max(alpha, val)
            if alpha >= beta:
                break

        self.transposition_table[key] = (best_val, depth)
        return best_val

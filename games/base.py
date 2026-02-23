"""Abstract base class for game environments."""

from abc import ABC, abstractmethod
from typing import List
import torch


class BaseGame(ABC):
    """Abstract interface for all game environments.

    All games must implement this interface to be compatible with the
    HRM training pipeline. The training loop is game-agnostic.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the game to its initial state."""
        pass

    @abstractmethod
    def legal_moves(self) -> List[int]:
        """Return list of legal move indices for the current player.

        Returns:
            List of integer move indices. Empty list if game is terminal.
        """
        pass

    @abstractmethod
    def make_move(self, move: int) -> None:
        """Execute a move and update the game state.

        Args:
            move: Integer move index (must be in legal_moves())

        Raises:
            ValueError: If move is not legal
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if the game has ended.

        Returns:
            True if game is over, False otherwise
        """
        pass

    @abstractmethod
    def outcome(self) -> float:
        """Return the game outcome from the current player's perspective.

        Returns:
            1.0 if current player won
            0.0 if draw
            -1.0 if current player lost

        Should only be called when is_terminal() is True.
        """
        pass

    @abstractmethod
    def to_tokens(self) -> torch.Tensor:
        """Convert current game state to token sequence for HRM input.

        Returns:
            Tensor of shape [seq_len] with integer token indices
        """
        pass

    @abstractmethod
    def action_size(self) -> int:
        """Return the total number of possible actions in this game.

        Returns:
            Integer action space size (includes invalid moves)
        """
        pass

    @property
    @abstractmethod
    def current_player(self) -> int:
        """Return the current player (1 or 2)."""
        pass

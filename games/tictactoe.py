"""Pure Python TicTacToe implementation."""

import numpy as np
import torch
from typing import List, Optional

from games.base import BaseGame


class TicTacToeGame(BaseGame):
    """3×3 TicTacToe game environment.

    Board representation:
        0 = empty
        1 = player 1 (X)
        2 = player 2 (O)

    Move encoding:
        - Integers 0-8: board positions (row * 3 + col)

    Rules:
        - Standard TicTacToe rules
        - First player to get 3-in-a-row wins (horizontal, vertical, diagonal)
        - Game ends in draw if board is full with no winner
    """

    BOARD_SIZE = 3
    EMPTY = 0
    PLAYER_1 = 1  # X
    PLAYER_2 = 2  # O

    def __init__(self):
        """Initialize a new TicTacToe game."""
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self._current_player = self.PLAYER_1
        self.reset()

    def reset(self) -> None:
        """Reset to empty board."""
        self.board.fill(self.EMPTY)
        self._current_player = self.PLAYER_1

    @property
    def current_player(self) -> int:
        """Return current player (1 or 2)."""
        return self._current_player

    def _opponent(self) -> int:
        """Return opponent player."""
        return self.PLAYER_2 if self._current_player == self.PLAYER_1 else self.PLAYER_1

    def _is_valid_pos(self, row: int, col: int) -> bool:
        """Check if position is on the board."""
        return 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE

    def _check_winner(self) -> Optional[int]:
        """Check if there's a winner.

        Returns:
            Player number (1 or 2) if there's a winner, None otherwise
        """
        # Check rows
        for row in range(self.BOARD_SIZE):
            if (self.board[row, :] == self.board[row, 0]).all() and self.board[row, 0] != self.EMPTY:
                return self.board[row, 0]

        # Check columns
        for col in range(self.BOARD_SIZE):
            if (self.board[:, col] == self.board[0, col]).all() and self.board[0, col] != self.EMPTY:
                return self.board[0, col]

        # Check diagonal (top-left to bottom-right)
        if (self.board[0, 0] == self.board[1, 1] == self.board[2, 2]) and self.board[0, 0] != self.EMPTY:
            return self.board[0, 0]

        # Check anti-diagonal (top-right to bottom-left)
        if (self.board[0, 2] == self.board[1, 1] == self.board[2, 0]) and self.board[0, 2] != self.EMPTY:
            return self.board[0, 2]

        return None

    def legal_moves(self) -> List[int]:
        """Return list of legal move indices.

        Returns:
            List of move indices (0-8 for empty board positions)
        """
        if self.is_terminal():
            return []

        moves = []
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row, col] == self.EMPTY:
                    moves.append(row * self.BOARD_SIZE + col)

        return moves

    def make_move(self, move: int) -> None:
        """Execute a move.

        Args:
            move: Move index (0-8 for board positions)

        Raises:
            ValueError: If move is not legal
        """
        legal = self.legal_moves()
        if move not in legal:
            raise ValueError(f"Illegal move: {move}. Legal moves: {legal}")

        row = move // self.BOARD_SIZE
        col = move % self.BOARD_SIZE

        # Place piece
        self.board[row, col] = self._current_player

        # Switch player
        self._current_player = self._opponent()

    def is_terminal(self) -> bool:
        """Check if game has ended.

        Game ends when:
        - A player has 3-in-a-row
        - Board is full (draw)
        """
        # Check for winner
        if self._check_winner() is not None:
            return True

        # Check if board is full
        if not (self.board == self.EMPTY).any():
            return True

        return False

    def outcome(self) -> float:
        """Return game outcome from current player's perspective.

        Returns:
            1.0 if current player won
            0.0 if draw
            -1.0 if current player lost
        """
        if not self.is_terminal():
            raise ValueError("Game is not terminal")

        winner = self._check_winner()

        if winner is None:
            return 0.0  # Draw

        # Return outcome from current player's perspective
        return 1.0 if winner == self._current_player else -1.0

    def to_tokens(self) -> torch.Tensor:
        """Convert board state to token sequence.

        Token encoding:
            0: empty square
            1: player 1 piece (X)
            2: player 2 piece (O)
            3: current player is player 1
            4: current player is player 2

        Returns:
            Tensor of shape [10]: 9 board positions + 1 current player token
        """
        # Flatten board (9 positions)
        board_tokens = torch.from_numpy(self.board.flatten()).long()

        # Add current player indicator
        player_token = torch.tensor(
            [3 if self._current_player == self.PLAYER_1 else 4],
            dtype=torch.long
        )

        # Concatenate: [9 board positions, 1 player indicator]
        tokens = torch.cat([board_tokens, player_token])

        return tokens

    def action_size(self) -> int:
        """Return action space size.

        Returns:
            9 (9 board positions)
        """
        return 9

    def __repr__(self) -> str:
        """String representation of the board."""
        symbols = {self.EMPTY: '.', self.PLAYER_1: 'X', self.PLAYER_2: 'O'}
        lines = []
        lines.append("  " + " ".join(str(i) for i in range(self.BOARD_SIZE)))
        for i, row in enumerate(self.board):
            lines.append(f"{i} " + " ".join(symbols[cell] for cell in row))
        lines.append(f"Current player: {'X (Player 1)' if self._current_player == self.PLAYER_1 else 'O (Player 2)'}")
        return "\n".join(lines)

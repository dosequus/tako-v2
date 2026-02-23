"""Pure Python Othello (Reversi) implementation."""

import numpy as np
import torch
from typing import List, Tuple, Optional

from games.base import BaseGame


class OthelloGame(BaseGame):
    """8×8 Othello game environment.

    Board representation:
        0 = empty
        1 = black (player 1)
        2 = white (player 2)

    Move encoding:
        - Integers 0-63: board positions (row * 8 + col)
        - Integer 64: pass move
        - Integer -1: invalid/null move

    Rules:
        - Standard Othello rules
        - Must flip at least one opponent piece
        - Pass if no legal moves available
        - Game ends when both players pass or board is full
    """

    BOARD_SIZE = 8
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    PASS_MOVE = 64

    # Eight directions: N, NE, E, SE, S, SW, W, NW
    DIRECTIONS = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    def __init__(self):
        """Initialize a new Othello game."""
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self._current_player = self.BLACK
        self._last_move_was_pass = False
        self.reset()

    def reset(self) -> None:
        """Reset to initial Othello position (4 pieces in center)."""
        self.board.fill(self.EMPTY)
        # Initial position: center 4 squares
        mid = self.BOARD_SIZE // 2
        self.board[mid - 1, mid - 1] = self.WHITE
        self.board[mid - 1, mid] = self.BLACK
        self.board[mid, mid - 1] = self.BLACK
        self.board[mid, mid] = self.WHITE

        self._current_player = self.BLACK
        self._last_move_was_pass = False

    @property
    def current_player(self) -> int:
        """Return current player (1=BLACK, 2=WHITE)."""
        return self._current_player

    def _opponent(self) -> int:
        """Return opponent player."""
        return self.WHITE if self._current_player == self.BLACK else self.BLACK

    def _is_valid_pos(self, row: int, col: int) -> bool:
        """Check if position is on the board."""
        return 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE

    def _find_flips(self, row: int, col: int, player: int) -> List[Tuple[int, int]]:
        """Find all pieces that would be flipped by placing at (row, col).

        Args:
            row, col: Position to place piece
            player: Player making the move

        Returns:
            List of (row, col) positions that would be flipped
        """
        if self.board[row, col] != self.EMPTY:
            return []

        opponent = self.WHITE if player == self.BLACK else self.BLACK
        flips = []

        for dr, dc in self.DIRECTIONS:
            potential_flips = []
            r, c = row + dr, col + dc

            # Walk in this direction
            while self._is_valid_pos(r, c):
                if self.board[r, c] == self.EMPTY:
                    break
                elif self.board[r, c] == opponent:
                    potential_flips.append((r, c))
                    r, c = r + dr, c + dc
                else:  # Found player's piece
                    if potential_flips:  # Must have opponent pieces in between
                        flips.extend(potential_flips)
                    break

        return flips

    def legal_moves(self) -> List[int]:
        """Return list of legal move indices.

        Returns:
            List of move indices (0-63 for board positions, 64 for pass)
        """
        moves = []

        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self._find_flips(row, col, self._current_player):
                    moves.append(row * self.BOARD_SIZE + col)

        # If no legal moves, only option is to pass
        if not moves:
            moves.append(self.PASS_MOVE)

        return moves

    def make_move(self, move: int) -> None:
        """Execute a move and flip pieces.

        Args:
            move: Move index (0-63 for positions, 64 for pass)

        Raises:
            ValueError: If move is not legal
        """
        legal = self.legal_moves()
        if move not in legal:
            raise ValueError(f"Illegal move: {move}. Legal moves: {legal}")

        if move == self.PASS_MOVE:
            # Pass move
            if not self._last_move_was_pass:
                self._last_move_was_pass = True
                self._current_player = self._opponent()
            # If both players pass consecutively, game ends (handled in is_terminal)
            return

        # Normal move
        self._last_move_was_pass = False
        row = move // self.BOARD_SIZE
        col = move % self.BOARD_SIZE

        flips = self._find_flips(row, col, self._current_player)
        if not flips:
            raise ValueError(f"Move {move} at ({row},{col}) doesn't flip any pieces")

        # Place piece and flip
        self.board[row, col] = self._current_player
        for r, c in flips:
            self.board[r, c] = self._current_player

        # Switch player
        self._current_player = self._opponent()

    def is_terminal(self) -> bool:
        """Check if game has ended.

        Game ends when:
        - Board is full
        - Both players pass consecutively
        - One player has no pieces left (and there are pieces on board)
        """
        # Check if board is full
        if not (self.board == self.EMPTY).any():
            return True

        # Check if both players passed (current legal moves is only pass,
        # and last move was also pass)
        legal = self.legal_moves()
        if legal == [self.PASS_MOVE] and self._last_move_was_pass:
            return True

        # Check if either player has no pieces (only if game has started)
        black_count = (self.board == self.BLACK).sum()
        white_count = (self.board == self.WHITE).sum()
        total_pieces = black_count + white_count

        # Only terminal if one player is eliminated and game has pieces
        if total_pieces > 0 and (black_count == 0 or white_count == 0):
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

        black_count = (self.board == self.BLACK).sum()
        white_count = (self.board == self.WHITE).sum()

        if black_count > white_count:
            winner = self.BLACK
        elif white_count > black_count:
            winner = self.WHITE
        else:
            return 0.0  # Draw

        return 1.0 if winner == self._current_player else -1.0

    def to_tokens(self) -> torch.Tensor:
        """Convert board state to token sequence.

        Token encoding:
            0: empty square
            1: black piece
            2: white piece
            3: current player is black
            4: current player is white

        Returns:
            Tensor of shape [65]: 64 board positions + 1 current player token
        """
        # Flatten board (64 positions)
        board_tokens = torch.from_numpy(self.board.flatten()).long()

        # Add current player indicator
        player_token = torch.tensor([3 if self._current_player == self.BLACK else 4], dtype=torch.long)

        # Concatenate: [64 board positions, 1 player indicator]
        tokens = torch.cat([board_tokens, player_token])

        return tokens

    def action_size(self) -> int:
        """Return action space size.

        Returns:
            65 (64 board positions + 1 pass move)
        """
        return 65

    def __repr__(self) -> str:
        """String representation of the board."""
        symbols = {self.EMPTY: '.', self.BLACK: 'X', self.WHITE: 'O'}
        lines = []
        lines.append("  " + " ".join(str(i) for i in range(self.BOARD_SIZE)))
        for i, row in enumerate(self.board):
            lines.append(f"{i} " + " ".join(symbols[cell] for cell in row))
        lines.append(f"Current player: {'BLACK' if self._current_player == self.BLACK else 'WHITE'}")
        return "\n".join(lines)

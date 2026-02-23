"""Unit tests for game environments."""

import torch
import pytest
import numpy as np
from games.othello import OthelloGame


def test_othello_init():
    """Test Othello initialization."""
    game = OthelloGame()

    # Check initial board setup (4 pieces in center)
    expected_board = np.zeros((8, 8), dtype=np.int8)
    expected_board[3, 3] = OthelloGame.WHITE
    expected_board[3, 4] = OthelloGame.BLACK
    expected_board[4, 3] = OthelloGame.BLACK
    expected_board[4, 4] = OthelloGame.WHITE

    assert np.array_equal(game.board, expected_board)

    # Check current player is BLACK
    assert game.current_player == OthelloGame.BLACK

    # Check 4 legal moves initially
    legal = game.legal_moves()
    assert len(legal) == 4

    # Known legal opening moves for Othello
    expected_moves = {19, 26, 37, 44}  # (2,3), (3,2), (4,5), (5,4)
    assert set(legal) == expected_moves


def test_othello_reset():
    """Test that reset() restores initial state."""
    game = OthelloGame()

    # Make some moves
    legal = game.legal_moves()
    game.make_move(legal[0])
    game.make_move(game.legal_moves()[0])

    # Reset
    game.reset()

    # Should be back to initial state
    assert game.current_player == OthelloGame.BLACK
    assert len(game.legal_moves()) == 4


def test_othello_move_validity():
    """Test that valid moves work and invalid moves fail."""
    game = OthelloGame()

    legal = game.legal_moves()
    first_move = legal[0]

    # Make valid move
    game.make_move(first_move)

    # Verify piece was placed
    row = first_move // 8
    col = first_move % 8
    assert game.board[row, col] == OthelloGame.BLACK

    # Verify some pieces were flipped
    black_count = (game.board == OthelloGame.BLACK).sum()
    assert black_count > 2  # More than initial 2 black pieces

    # Verify player switched
    assert game.current_player == OthelloGame.WHITE

    # Try invalid move (should raise)
    with pytest.raises(ValueError):
        game.make_move(0)  # Top-left corner, not legal


def test_othello_piece_flipping():
    """Test that pieces are correctly flipped."""
    game = OthelloGame()

    # Initial state: 2 black, 2 white
    assert (game.board == OthelloGame.BLACK).sum() == 2
    assert (game.board == OthelloGame.WHITE).sum() == 2

    # Make move at (2, 3) - should flip (3, 3)
    game.make_move(19)

    # Now should have 4 black, 1 white
    assert (game.board == OthelloGame.BLACK).sum() == 4
    assert (game.board == OthelloGame.WHITE).sum() == 1


def test_othello_pass_move():
    """Test pass move when no legal moves available."""
    game = OthelloGame()

    # Create a position with no legal moves for one player
    # This is tricky to set up, so we'll just test the pass mechanism
    game.board.fill(OthelloGame.EMPTY)
    game.board[0, 0] = OthelloGame.BLACK
    game.board[0, 1] = OthelloGame.WHITE
    game._current_player = OthelloGame.WHITE

    legal = game.legal_moves()

    # If only pass is available
    if legal == [OthelloGame.PASS_MOVE]:
        game.make_move(OthelloGame.PASS_MOVE)
        # Player should switch
        assert game.current_player == OthelloGame.BLACK


def test_othello_terminal():
    """Test terminal state detection."""
    game = OthelloGame()

    # Initially not terminal
    assert not game.is_terminal()

    # Fill board to make it terminal
    game.board.fill(OthelloGame.BLACK)

    # Should be terminal now
    assert game.is_terminal()


def test_othello_terminal_both_pass():
    """Test that game ends when both players pass."""
    game = OthelloGame()

    # Set up position where both players must pass
    game.board.fill(OthelloGame.EMPTY)
    game.board[0, 0] = OthelloGame.BLACK
    game._current_player = OthelloGame.BLACK

    # First player passes
    legal = game.legal_moves()
    if OthelloGame.PASS_MOVE in legal:
        game.make_move(OthelloGame.PASS_MOVE)

        # Second player passes
        legal2 = game.legal_moves()
        if legal2 == [OthelloGame.PASS_MOVE]:
            # Should be terminal before second pass
            # (because legal moves is only pass and last was pass)
            assert game.is_terminal()


def test_othello_outcome():
    """Test outcome calculation."""
    game = OthelloGame()

    # Create a finished game with known outcome - fill most of board
    game.board.fill(OthelloGame.BLACK)
    game.board[0, 0] = OthelloGame.WHITE
    game.board[0, 1] = OthelloGame.WHITE
    game.board[0, 2] = OthelloGame.WHITE

    game._current_player = OthelloGame.BLACK

    # Verify terminal (board is nearly full)
    assert game.is_terminal()

    # Black has many, White has 3 → Black wins
    outcome = game.outcome()
    # Current player is BLACK, and BLACK wins
    assert outcome == 1.0

    # Switch to WHITE's perspective
    game._current_player = OthelloGame.WHITE
    outcome = game.outcome()
    # Current player is WHITE, and WHITE loses
    assert outcome == -1.0


def test_othello_outcome_draw():
    """Test draw outcome."""
    game = OthelloGame()

    # Create a draw - fill board with equal pieces
    game.board.fill(OthelloGame.EMPTY)
    game.board[:4, :] = OthelloGame.BLACK  # 32 black pieces
    game.board[4:, :] = OthelloGame.WHITE  # 32 white pieces

    game._current_player = OthelloGame.BLACK

    assert game.is_terminal()  # Board full

    outcome = game.outcome()
    assert outcome == 0.0  # Draw


def test_othello_tokens():
    """Test tokenization of board state."""
    game = OthelloGame()

    tokens = game.to_tokens()

    # Should be 65 tokens: 64 board + 1 player indicator
    assert tokens.shape == (65,)

    # Check dtype
    assert tokens.dtype == torch.long

    # Check values are in valid range
    # Board tokens: 0 (empty), 1 (black), 2 (white)
    # Player tokens: 3 (black), 4 (white)
    assert tokens.min() >= 0
    assert tokens.max() <= 4

    # Player indicator should be last token
    player_token = tokens[-1].item()
    assert player_token in [3, 4]

    # If current player is BLACK, last token should be 3
    if game.current_player == OthelloGame.BLACK:
        assert player_token == 3
    else:
        assert player_token == 4


def test_othello_action_size():
    """Test action space size."""
    game = OthelloGame()

    assert game.action_size() == 65  # 64 positions + 1 pass


def test_othello_full_game():
    """Test playing a full game to completion."""
    game = OthelloGame()

    max_moves = 100  # Safety limit
    moves_made = 0

    while not game.is_terminal() and moves_made < max_moves:
        legal = game.legal_moves()
        assert len(legal) > 0, "No legal moves but game not terminal"

        # Make first legal move
        game.make_move(legal[0])
        moves_made += 1

    # Should eventually terminate
    assert game.is_terminal()

    # Outcome should be valid
    outcome = game.outcome()
    assert outcome in [-1.0, 0.0, 1.0]


def test_othello_string_representation():
    """Test that __repr__ works."""
    game = OthelloGame()

    repr_str = repr(game)

    # Should contain board visualization
    assert '.' in repr_str or 'X' in repr_str or 'O' in repr_str
    assert 'Current player' in repr_str


def test_othello_determinism():
    """Test that same moves produce same results."""
    game1 = OthelloGame()
    game2 = OthelloGame()

    moves = [19, 26, 37, 44]  # Sequence of moves

    for move in moves:
        if move in game1.legal_moves():
            game1.make_move(move)
        if move in game2.legal_moves():
            game2.make_move(move)

    # Both games should have identical boards
    assert np.array_equal(game1.board, game2.board)
    assert game1.current_player == game2.current_player

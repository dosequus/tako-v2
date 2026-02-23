"""Unit tests for TicTacToe game environment."""

import torch
import pytest
import numpy as np
from games.tictactoe import TicTacToeGame


def test_tictactoe_init():
    """Test TicTacToe initialization."""
    game = TicTacToeGame()

    # Check initial board is empty
    expected_board = np.zeros((3, 3), dtype=np.int8)
    assert np.array_equal(game.board, expected_board)

    # Check current player is Player 1
    assert game.current_player == TicTacToeGame.PLAYER_1

    # Check 9 legal moves initially (all squares empty)
    legal = game.legal_moves()
    assert len(legal) == 9
    assert set(legal) == set(range(9))


def test_tictactoe_reset():
    """Test that reset() restores initial state."""
    game = TicTacToeGame()

    # Make some moves
    game.make_move(0)  # Player 1 at top-left
    game.make_move(4)  # Player 2 at center

    # Verify board has pieces
    assert game.board[0, 0] == TicTacToeGame.PLAYER_1
    assert game.board[1, 1] == TicTacToeGame.PLAYER_2

    # Reset
    game.reset()

    # Should be back to initial state
    assert game.current_player == TicTacToeGame.PLAYER_1
    assert len(game.legal_moves()) == 9
    assert (game.board == TicTacToeGame.EMPTY).all()


def test_tictactoe_legal_moves():
    """Test legal moves returns correct empty squares."""
    game = TicTacToeGame()

    # Initially all 9 squares are legal
    assert set(game.legal_moves()) == set(range(9))

    # Make a move
    game.make_move(0)

    # Now 8 squares are legal (all except 0)
    assert set(game.legal_moves()) == set(range(1, 9))

    # Make another move
    game.make_move(4)

    # Now 7 squares are legal
    legal = game.legal_moves()
    assert len(legal) == 7
    assert 0 not in legal
    assert 4 not in legal


def test_tictactoe_make_move():
    """Test valid moves update board and switch players."""
    game = TicTacToeGame()

    # Player 1 makes first move
    assert game.current_player == TicTacToeGame.PLAYER_1
    game.make_move(0)

    # Check piece was placed
    assert game.board[0, 0] == TicTacToeGame.PLAYER_1

    # Check player switched
    assert game.current_player == TicTacToeGame.PLAYER_2

    # Player 2 makes move
    game.make_move(4)
    assert game.board[1, 1] == TicTacToeGame.PLAYER_2
    assert game.current_player == TicTacToeGame.PLAYER_1


def test_tictactoe_invalid_move():
    """Test that invalid moves raise ValueError."""
    game = TicTacToeGame()

    # Make a valid move
    game.make_move(0)

    # Try to make same move again (should fail)
    with pytest.raises(ValueError):
        game.make_move(0)

    # Try invalid move index (should fail)
    with pytest.raises(ValueError):
        game.make_move(10)


def test_tictactoe_terminal_win_horizontal():
    """Test horizontal win detection."""
    game = TicTacToeGame()

    # Create horizontal win for Player 1 in top row
    game.board[0, :] = TicTacToeGame.PLAYER_1

    assert game.is_terminal()
    assert game._check_winner() == TicTacToeGame.PLAYER_1


def test_tictactoe_terminal_win_vertical():
    """Test vertical win detection."""
    game = TicTacToeGame()

    # Create vertical win for Player 2 in left column
    game.board[:, 0] = TicTacToeGame.PLAYER_2

    assert game.is_terminal()
    assert game._check_winner() == TicTacToeGame.PLAYER_2


def test_tictactoe_terminal_win_diagonal():
    """Test diagonal win detection."""
    game = TicTacToeGame()

    # Create diagonal win for Player 1 (top-left to bottom-right)
    game.board[0, 0] = TicTacToeGame.PLAYER_1
    game.board[1, 1] = TicTacToeGame.PLAYER_1
    game.board[2, 2] = TicTacToeGame.PLAYER_1

    assert game.is_terminal()
    assert game._check_winner() == TicTacToeGame.PLAYER_1


def test_tictactoe_terminal_win_anti_diagonal():
    """Test anti-diagonal win detection."""
    game = TicTacToeGame()

    # Create anti-diagonal win for Player 2 (top-right to bottom-left)
    game.board[0, 2] = TicTacToeGame.PLAYER_2
    game.board[1, 1] = TicTacToeGame.PLAYER_2
    game.board[2, 0] = TicTacToeGame.PLAYER_2

    assert game.is_terminal()
    assert game._check_winner() == TicTacToeGame.PLAYER_2


def test_tictactoe_terminal_draw():
    """Test draw when board is full with no winner."""
    game = TicTacToeGame()

    # Create a draw board:
    # X O X
    # X O O
    # O X X
    game.board = np.array([
        [1, 2, 1],
        [1, 2, 2],
        [2, 1, 1]
    ], dtype=np.int8)

    assert game.is_terminal()
    assert game._check_winner() is None


def test_tictactoe_not_terminal():
    """Test that incomplete game is not terminal."""
    game = TicTacToeGame()

    # Make a few moves but don't complete
    game.make_move(0)
    game.make_move(1)

    assert not game.is_terminal()


def test_tictactoe_outcome_win():
    """Test outcome when current player's opponent won."""
    game = TicTacToeGame()

    # Set up a win for Player 1
    game.board[0, :] = TicTacToeGame.PLAYER_1
    game._current_player = TicTacToeGame.PLAYER_2  # Current player is 2

    assert game.is_terminal()

    # Player 1 won, current player is 2, so outcome is -1.0
    outcome = game.outcome()
    assert outcome == -1.0

    # Switch perspective to Player 1
    game._current_player = TicTacToeGame.PLAYER_1
    outcome = game.outcome()
    assert outcome == 1.0


def test_tictactoe_outcome_draw():
    """Test draw outcome."""
    game = TicTacToeGame()

    # Create a draw board
    game.board = np.array([
        [1, 2, 1],
        [1, 2, 2],
        [2, 1, 1]
    ], dtype=np.int8)

    game._current_player = TicTacToeGame.PLAYER_1

    assert game.is_terminal()

    outcome = game.outcome()
    assert outcome == 0.0


def test_tictactoe_outcome_not_terminal():
    """Test that outcome raises error if game not terminal."""
    game = TicTacToeGame()

    # Game just started, not terminal
    with pytest.raises(ValueError):
        game.outcome()


def test_tictactoe_to_tokens():
    """Test tokenization of board state."""
    game = TicTacToeGame()

    tokens = game.to_tokens()

    # Should be 10 tokens: 9 board + 1 player indicator
    assert tokens.shape == (10,)

    # Check dtype
    assert tokens.dtype == torch.long

    # Initially all board positions should be 0 (empty)
    assert (tokens[:9] == 0).all()

    # Player indicator should be 3 (Player 1)
    assert tokens[9].item() == 3

    # Make some moves and re-check
    game.make_move(0)  # Player 1 at position 0
    tokens = game.to_tokens()

    # Position 0 should now be 1
    assert tokens[0].item() == 1
    # Player indicator should now be 4 (Player 2)
    assert tokens[9].item() == 4


def test_tictactoe_to_tokens_values():
    """Test token values are in valid range."""
    game = TicTacToeGame()

    # Make some moves
    game.make_move(0)
    game.make_move(4)
    game.make_move(8)

    tokens = game.to_tokens()

    # Check values are in valid range
    # Board tokens: 0 (empty), 1 (player 1), 2 (player 2)
    # Player tokens: 3 (player 1), 4 (player 2)
    assert tokens.min() >= 0
    assert tokens.max() <= 4


def test_tictactoe_action_size():
    """Test action space size."""
    game = TicTacToeGame()

    assert game.action_size() == 9


def test_tictactoe_full_game_win():
    """Test playing a full game to Player 1 win."""
    game = TicTacToeGame()

    # Play a game where Player 1 wins
    # Player 1 (X) at 0, Player 2 (O) at 3, Player 1 at 1, Player 2 at 4, Player 1 at 2
    moves = [0, 3, 1, 4, 2]  # Top row for Player 1

    for move in moves:
        assert not game.is_terminal()
        game.make_move(move)

    # Game should be terminal with Player 1 winning
    assert game.is_terminal()
    assert game._check_winner() == TicTacToeGame.PLAYER_1


def test_tictactoe_full_game_draw():
    """Test playing a full game to draw."""
    game = TicTacToeGame()

    # Play a game that ends in draw
    # X O X
    # O O X
    # O X X
    moves = [0, 1, 2, 3, 5, 4, 6, 8, 7]

    for move in moves:
        game.make_move(move)

    assert game.is_terminal()
    assert game._check_winner() is None


def test_tictactoe_legal_moves_empty_when_terminal():
    """Test that legal_moves returns empty list when game is terminal."""
    game = TicTacToeGame()

    # Create a terminal state
    game.board[0, :] = TicTacToeGame.PLAYER_1

    assert game.is_terminal()
    assert game.legal_moves() == []


def test_tictactoe_string_representation():
    """Test that __repr__ works."""
    game = TicTacToeGame()

    repr_str = repr(game)

    # Should contain board visualization
    assert '.' in repr_str or 'X' in repr_str
    assert 'Current player' in repr_str


def test_tictactoe_determinism():
    """Test that same moves produce same results."""
    game1 = TicTacToeGame()
    game2 = TicTacToeGame()

    moves = [0, 4, 1, 3, 2]

    for move in moves:
        game1.make_move(move)
        game2.make_move(move)

    # Both games should have identical boards
    assert np.array_equal(game1.board, game2.board)
    assert game1.current_player == game2.current_player


def test_tictactoe_all_positions():
    """Test that all 9 positions can be played."""
    game = TicTacToeGame()

    # Play all 9 positions in a sequence that doesn't create early win
    # This sequence creates a draw
    moves = [0, 1, 2, 4, 3, 5, 7, 6, 8]

    for move in moves:
        assert move in game.legal_moves()
        game.make_move(move)

    # Board should be full
    assert game.is_terminal()
    assert (game.board != TicTacToeGame.EMPTY).all()

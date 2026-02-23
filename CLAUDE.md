# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Tako is a chess engine built around the **Hierarchical Reasoning Model (HRM)** — a ~27M parameter dual-module recurrent architecture that performs iterative reasoning within a single forward pass, rather than relying on a static board evaluator. The target is 2500+ Elo (GM level) via self-play RL with minimal compute.

The canonical reference is `tako_spec_v1.1.pdf` in the repo root.

## Tech Stack

- **Python 3.11+** — model, training, evaluation, inference server
- **uv** — Python package management. **All Python commands must be run through `uv run`** to ensure the correct virtual environment and dependencies are used (e.g. `uv run pytest`, `uv run python scripts/train.py`)
- **PyTorch 2.x** — HRM implementation, mixed precision (bfloat16), FlashAttention 2/3
- **Rust (Edition 2021)** — game logic crates (PyO3 bindings via Maturin), self-play workers (Phase 4+)
- **Ray** — distributed worker ↔ learner communication (Phases 1–3)
- **zmq** — Rust worker ↔ Python GPU inference server IPC (Phase 4+)

## Commands

```bash
# Python (always use uv run to avoid missing packages)
uv sync                                          # Install/sync all dependencies
uv run pytest tests/                              # Run all tests
uv run pytest tests/test_hrm.py                   # HRM unit tests
uv run pytest tests/test_games.py                 # Game environment tests
uv run pytest tests/test_mcts.py                  # MCTS tests
uv run pytest tests/test_bindings.py              # PyO3 integration tests
uv run pytest tests/test_hrm.py::test_name -x     # Run single test, stop on first failure

# Rust (from repo root — workspace manifest)
cargo build --release                             # Build all Rust crates
cargo test                                        # Run all Rust tests
cargo test -p tako-hex-core                       # Test single crate
uv run maturin develop --release                  # Build PyO3 extensions for Python

# Training & evaluation
uv run python scripts/train.py --config config/othello.yaml
uv run python scripts/eval.py
uv run python scripts/pretrain.py                 # Supervised pretraining on PGN (Phase 3)
```

## Architecture

### HRM (Hierarchical Reasoning Model)

Four learnable components operating over a two-level temporal hierarchy:

1. **Input Network (`f_I`)** — tokenized game state → dense embeddings (embedding + RoPE)
2. **L-module (`f_L`)** — fast tactical module, updates every timestep. Inputs: own previous state + current H-state + input embedding, combined via element-wise addition before a transformer block
3. **H-module (`f_H`)** — slow strategic module, updates once per cycle (every T L-steps). Identical transformer architecture to L-module, different update frequency only
4. **Output Network (`f_O`)** — reads final H-state → policy head (softmax over legal moves) + value head (W/D/L probabilities)

A forward pass (one "segment") runs N×T total timesteps. N×T-1 steps run inside `torch.no_grad()`, only the final step has gradients enabled (1-step gradient approximation from DEQ/IFT theory). Hidden states `z_H`, `z_L` are initialized from a fixed truncated normal.

**ACT (Adaptive Computation Time):** A Q-head on `z_H` decides halt vs. continue after each segment. Deep supervision runs multiple segments sequentially, detaching hidden states between them.

### Game Environments

All games implement `BaseGame` (in `games/base.py`): `reset()`, `legal_moves()`, `make_move()`, `is_terminal()`, `outcome()`, `to_tokens()`, `action_size()`. The training loop is game-agnostic.

### Training Pipeline

- **Self-play workers** generate games using MCTS + HRM evaluation
- **Central learner** samples from a replay buffer (~1M positions, rolling window)
- **Loss per segment:** α·PolicyLoss + β·ValueLoss + γ·ACT_Loss (summed across M segments)
- **Opponent pool:** past checkpoints as opponents (70% recent, 30% older) to prevent policy collapse
- Checkpoint sync every ~1000 learner steps; workers tolerate stale checkpoints

### Rust / Python Boundary

- **Phases 1–3:** Rust crates (`tako-hex-core`, `tako-chess-core`) expose game logic to Python via PyO3/Maturin as native `.so` modules. Python wrappers in `games/` call Rust transparently. Zero-copy numpy arrays via `PyArray1`.
- **Phase 4:** `tako-worker` is a standalone Rust binary. MCTS runs entirely in Rust; neural eval is batched IPC to a Python GPU inference server (`training/inference_server.py`) over zmq.

## Development Phases

| Phase | Focus | Key Validation |
|-------|-------|----------------|
| 0 | HRM core + pure Python Othello | Forward pass, 1-step gradient, deep supervision |
| 1 | Othello self-play (pure Python) | Beat Edax level 3; H/L dimensionality hierarchy emerges |
| 2 | Hex + Rust PyO3 bindings | `tako-hex-core` correctness parity; Maturin builds clean |
| 3 | Chess bootstrap | `tako-chess-core` passes perft tests; 1700+ Elo from pretraining |
| 4 | Rust workers + distributed | 64+ workers stable; 2200+ Elo |
| 5 | GM push | 2500+ Elo on Lichess blitz; UCI interface |

## Key Hyperparameters

- `d_model=512`, `n_layers=8`, `n_heads=8`, `d_ff=2048`
- `N=4` cycles, `T=4` steps/cycle → NT=16 effective depth per segment
- `M_max=10` (training), `M_max=20` (inference) segments
- MCTS: 400 sims (self-play), 800 (eval), PUCT c=1.5
- Optimizer: Adam, lr 2e-4→2e-5 cosine decay, batch 512, bfloat16

## Conventions

- Transformer blocks use RoPE, GLU (SwiGLU), RMSNorm (post-norm), no biases, FlashAttention
- Module inputs merge via element-wise addition (not concatenation)
- Config files are YAML (`config/othello.yaml`, `config/hex.yaml`, `config/chess.yaml`)
- Rust crates use a Cargo workspace at repo root

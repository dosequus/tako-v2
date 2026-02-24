# Tako HRM - Google Colab Notebooks

Interactive Jupyter notebooks for training and evaluating Tako HRM models on Google Colab with free GPU acceleration.

---

## 📚 Notebooks

**Simplified structure:** One notebook per task, separate cells for each game.

| Notebook | Description | Games |
|----------|-------------|-------|
| **train.ipynb** | Train models using self-play RL | TicTacToe, Othello, Hex, Chess |
| **eval.ipynb** | Evaluate trained models vs baselines | TicTacToe, Othello, Hex, Chess |
| **play.ipynb** | Play interactively against models | TicTacToe, Othello, Hex, Chess |

Each notebook has separate cells for each game - just run the cells for the game you want!

---

## 🚀 Quick Start

### 1. Open in Google Colab

Click the links below to open notebooks directly in Colab:

- [train.ipynb](https://colab.research.google.com/github/zfdupont/tako-v2/blob/main/notebooks/train.ipynb) - Train models
- [eval.ipynb](https://colab.research.google.com/github/zfdupont/tako-v2/blob/main/notebooks/eval.ipynb) - Evaluate performance
- [play.ipynb](https://colab.research.google.com/github/zfdupont/tako-v2/blob/main/notebooks/play.ipynb) - Interactive play

### 2. Setup (First Time Only)

**Before running notebooks:**

1. **Create GitHub Token:**
   - Visit: https://github.com/settings/tokens
   - Scopes: ✅ `repo`, ✅ `read:org`
   - For organization repos: Click "Configure SSO" → "Authorize"

2. **Add Token to Colab:**
   - Click 🔑 (Secrets icon) in Colab sidebar
   - Add: `GITHUB_TOKEN` = your token
   - Enable "Notebook access"

3. **Update ORG_NAME:**
   - In the notebook clone cell, set `ORG_NAME = "YOUR_ORG_NAME"`

See `ORG_REPO_SETUP.md` for detailed instructions.

### 3. Enable GPU

**Critical for performance!**

1. In Colab: **Runtime** → **Change runtime type**
2. Select **GPU** (T4, V100, or A100)
3. Click **Save**

### 4. Train and Play

1. **Open `train.ipynb`** in Colab
2. **Run setup cells** (clone repo, mount Drive, check GPU)
3. **Run training cell** for your game (e.g., TicTacToe section)
4. **Evaluate** with `eval.ipynb` or **play** with `play.ipynb`

**Workflow:**
```
train.ipynb → eval.ipynb → play.ipynb
     ↓            ↓             ↓
  Train TicTacToe → Test vs random → Play vs model
```

---

## 📊 Expected Performance

### GPU Benchmarks (TicTacToe)

| GPU Type | Forward Pass | Games/Hour (8 workers) | Training to 90% Win Rate |
|----------|--------------|------------------------|--------------------------|
| **T4** (Colab Free) | ~1-2ms | ~150,000 | ~20 min |
| **V100** (Colab Pro) | ~0.5-1ms | ~300,000 | ~10 min |
| **A100** (Colab Pro+) | ~0.3-0.5ms | ~500,000 | ~5 min |
| **CPU** (No GPU) | ~3-5ms | ~50,000 | ~60 min |

*Actual performance depends on system load and runtime availability*

---

## 💾 Saving Your Work

### Google Drive Integration

All notebooks automatically:
- Mount your Google Drive
- Save checkpoints to `MyDrive/tako_checkpoints/`
- Persist models across sessions

**Your models are safe even if Colab disconnects!**

---

## 🎓 Learning Path

### Beginner: Understand HRM Basics
1. Open `train.ipynb` and run setup cells
2. Run TicTacToe training cell for 1 epoch
3. Monitor training logs - see games being generated
4. Check training curve plot

### Intermediate: Train a Good Model
1. Train for 5 epochs in `train.ipynb` (TicTacToe section)
2. Evaluate in `eval.ipynb` (TicTacToe section) - target >90% vs random
3. Play in `play.ipynb` (TicTacToe section) - try to beat it!
4. Experiment: Modify MCTS simulations, model size in config

### Advanced: Multi-Game Training
1. Train Othello model (2-3 hours) - target: beat Edax level 3
2. Train Hex model (3-4 hours) - target: strong tactical play
3. Compare different games' learning curves
4. Scale to Chess (requires pretraining on PGN data)

---

## 🛠️ Configuration

### Modify Training Hyperparameters

Edit `config/tictactoe.yaml`:

```yaml
model:
  d_model: 128        # Embedding dimension (64, 128, 256)
  n_layers: 2         # Layers per module (1, 2, 4)
  N: 1                # Cycles per segment (1, 2, 4)

mcts:
  simulations: 25     # MCTS sims (10, 25, 50, 100)
  max_segments_inference: 1  # Segments during inference (1, 2, 5)

training:
  batch_size: 512     # Batch size (128, 256, 512)
  max_segments: 2     # Training segments (2, 5, 10)
```

### Reduce Memory Usage (if OOM errors)

```yaml
selfplay:
  num_workers: 4      # Reduce from 8 to 4 or 2

training:
  batch_size: 256     # Reduce from 512 to 256
  replay_buffer_size: 100000  # Reduce from 500000
```

---

## 🐛 Troubleshooting

### "No CUDA GPUs are available" (Ray workers)
**Error:** Ray workers crash with "RuntimeError: No CUDA GPUs are available"
**Fix:** Updated training script uses fractional GPU allocation
**Solution:** Pull latest changes, restart Colab runtime
**Details:** See `COLAB_GPU_FIX.md`

### "could not read Username" (Git clone)
**Error:** Git clone fails with authentication error
**Fix:** Use subprocess.run() instead of !git clone
**Solution:** Token needs `repo` + `read:org` scopes, SSO authorization
**Details:** See `TROUBLESHOOTING.md` and `ORG_REPO_SETUP.md`

### "CUDA out of memory"
- Reduce `num_workers` (8 → 4 → 2)
- Reduce `batch_size` (512 → 256)
- Restart runtime: **Runtime** → **Restart runtime**

### "No module named 'model'"
- Run setup cell again
- Check `sys.path` includes `/content/tako-v2`

### "No checkpoints found"
- Train a model first (run training cell in `train.ipynb`)
- Check Google Drive is mounted

### Training is slow
- Verify GPU is enabled (**Runtime** → **Change runtime type**)
- Check logs: should see "[Train] Workers created on cuda device(s)"
- CPU mode is 50-100x slower

### Colab disconnects during training
- Training resumes from latest checkpoint
- Checkpoints saved every 500 steps to Google Drive
- Just re-run training cell to continue

---

## 📖 Additional Resources

- **Spec:** See `tako_spec_v1.1.pdf` for HRM architecture details
- **Code:** Full source in `/model`, `/training`, `/games`
- **Optimization:** See `OPTIMIZATION_RESULTS.md` for Phase 1 results

---

## 🎯 Next Steps After TicTacToe

Once you achieve >90% win rate vs random:

1. **Othello** - Larger board (8×8), more complex strategy
   - Config: `config/othello.yaml`
   - Target: Beat Edax level 3

2. **Hex** - Connection game, Rust PyO3 bindings
   - Config: `config/hex.yaml`
   - Tests Rust integration

3. **Chess** - Ultimate goal (2500+ Elo)
   - Requires pretraining on PGN data
   - See Phase 3+ in roadmap

---

## 💡 Pro Tips

### Speed up experimentation
- Use `max_segments_inference: 1` for fast self-play
- Use `simulations: 10-25` for TicTacToe (tiny state space)
- Start with small models (d_model=64) to verify training loop

### Monitor training
- Check worker logs for "Game complete" messages
- Loss should decrease (total loss < 2.0 after 1000 steps)
- Win rate vs random should reach 90%+ quickly

### Save compute
- **Colab Free** (T4): ~9 hours/day, sufficient for TicTacToe
- **Colab Pro** (V100/A100): Faster, more stable for long training
- Use Drive to persist checkpoints across sessions

---

**Happy Training! 🚀**

For questions or issues, see the main repo README or open an issue on GitHub.

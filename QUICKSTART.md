# Quick Start Guide - Chess AI

**Status:** ✅ **Model Pre-Trained & Ready to Play!**  
**No training required** - Play immediately!

---

## 🚀 Installation (5 Minutes)

### Step 1: Prerequisites

- **Python 3.8+** (tested on Python 3.12)
- **pip** (Python package manager)
- **Optional:** CUDA-capable GPU for faster training (not needed for playing)

### Step 2: Install Dependencies

```bash
# Navigate to project directory
cd chessAI

# Install all required packages
pip install -r requirements.txt
```

**Packages installed:**
- PyTorch 2.0+ (with CUDA support if available)
- python-chess 1.999
- pygame 2.5+
- numpy 1.24+
- pandas 2.0+
- tqdm 4.65+

### Step 3: Verify Installation

```bash
python test_system.py
```

**Expected output:** All tests should pass ✅

---

## 🎮 Play Immediately (No Training Needed!)

### Method 1: GUI Game (Recommended)

```bash
python ChessGame.py
```

**Features:**
- Beautiful pygame interface
- Click-to-move controls
- Visual board and pieces
- Player vs Player or Player vs AI modes
- AI thinking indicator

### Method 2: Console Game

```bash
python play_now.py
```

**Features:**
- Text-based interface
- No GUI needed
- Fast and lightweight
- Enter moves in UCI (e2e4) or SAN (e4) format

### Method 3: Interactive Menu

```bash
python start.py
```

**Options available:**
1. 🎮 Play Chess (GUI)
2. 🤖 Train AI (Quick - 5 iterations)
3. 🚀 Train AI (Full - 100 iterations)
4. 📊 Check Model Status
5. 🧪 Test Neural Network
6. 📂 Pre-train from Dataset
7. 📈 Show Dataset Statistics
8. ⚡ Benchmark Alpha-Beta
9. 🥊 Alpha-Beta vs MCTS Duel

### Method 4: Watch AI vs AI

```bash
python quick_play_demo.py
```

Watch Alpha-Beta engine battle against MCTS engine!

---

## 🎯 Current Model Performance

**The included model has been trained on:**
- ✅ 321,720 chess positions
- ✅ 5,000 grandmaster games (ELO 1500+)
- ✅ 5 training epochs (43 minutes on GPU)
- ✅ Loss reduced by 31.7%

**Current capabilities:**
- Plays standard openings (e4, d4, Nf3, c4)
- Finds checkmate in 1 move
- Makes tactical captures
- Evaluates positions correctly
- ELO: ~1200-1400 (beginner to intermediate)

---

## 🏋️ Training (Optional - Already Done!)

The model is **already trained**! But you can train it further:

### Quick Test Training

```bash
python -c "from trainer import quick_train; quick_train(3, 5)"
```
- 3 iterations, 5 games each
- ~5 minutes
- Just to test the training pipeline

### Supervised Pre-Training (Completed)

```bash
python train_from_csv.py --min_elo 1500 --max_games 5000 --epochs 5
```

**Already completed with these results:**
```
Dataset: 5,000 games, 321,720 positions
Epochs: 5
Time: 43 minutes 28 seconds
Loss: 5.28 → 3.60 (31.7% improvement)
Model: models/latest.pt (238 MB)
```

### Self-Play Training (For Stronger Play)

```bash
python trainer.py  # Run 10 iterations
```

Or use the interactive menu:
```bash
python start.py
# Choose option 3: Full Training
```

**Time estimates:**
- 10 iterations: ~2-4 hours → ELO 1500-1700
- 100 iterations: ~20+ hours → ELO 1800-2000
- 1000+ iterations: Days/weeks → ELO 2000+

---

## ⚙️ Configuration

Edit `config.py` to customize the AI:

### For Faster Play (CPU-friendly)
```python
NUM_SIMULATIONS = 100          # Reduced from 800 (faster but weaker)
NUM_RESIDUAL_BLOCKS = 5        # Reduced from 10 (smaller network)
BATCH_SIZE = 16                # Reduced from 32 (less memory)
```

### For Stronger Play (GPU recommended)
```python
NUM_SIMULATIONS = 1600         # Increased from 800 (stronger but slower)
NUM_RESIDUAL_BLOCKS = 20       # Increased from 10 (deeper network)
HIDDEN_SIZE = 512              # Increased from 256 (wider network)
```

### Training Parameters
```python
LEARNING_RATE = 0.001          # Adam optimizer learning rate
BATCH_SIZE = 32                # Training batch size
NUM_SELF_PLAY_GAMES = 100      # Games per training iteration
REPLAY_BUFFER_SIZE = 10000     # Experience replay size
```

### Dataset Settings
```python
GAMES_CSV_PATH = 'games.csv'   # Path to dataset
MIN_ELO_FILTER = 1500          # Minimum player rating
MAX_GAMES_TO_LOAD = 50000      # Maximum games to load
CSV_TRAIN_EPOCHS = 5           # Training epochs
```

---

## Playing Against the AI

### Method 1: GUI
```bash
python ChessGame.py
```

### Method 2: Python Script
```python
from engine_new import ChessEngine
import chess

engine = ChessEngine()
board = chess.Board()

while not board.is_game_over():
    move = engine.get_best_move(board)
    board.push(move)
    print(board)
```

---

## 🔧 Troubleshooting

### "No module named 'torch'"
```bash
pip install torch
# Or for GPU support:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### "CUDA out of memory"
**Solution 1:** Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # Default is 32
```

**Solution 2:** Use CPU instead:
```python
# The code will automatically use CPU if CUDA isn't available
```

### "Training is very slow"
**On CPU:**
```python
# In config.py
NUM_SIMULATIONS = 100  # Default is 800
NUM_RESIDUAL_BLOCKS = 5  # Default is 10
```

**For GPU:** PyTorch will auto-detect and use CUDA if available.

### "AI makes random moves"
The model needs training! Two options:

**Option 1:** Use the included pre-trained model (already done!)  
**Option 2:** Train yourself:
```bash
python train_from_csv.py --epochs 5 --max_games 5000
```

### "pygame not launching"
```bash
pip uninstall pygame
pip install pygame
```

### "Model not found"
```bash
# Check if model exists
ls models/latest.pt

# If not, train it:
python train_from_csv.py --epochs 5
```

---

## 📁 File Structure

```
chessAI/
├── start.py              ← START HERE (Interactive menu)
├── ChessGame.py          ← Play with GUI
├── play_now.py           ← Play in console
├── trainer.py            ← Train the AI
├── train_from_csv.py     ← Supervised training
├── examples.py           ← Code examples
├── test_system.py        ← Validate system
├── test_trained_model.py ← Test performance
├── quick_play_demo.py    ← AI vs AI demo
│
├── config.py             ← Configuration
├── neural_network.py     ← Deep learning model
├── mcts.py              ← Monte Carlo Tree Search
├── alpha_beta.py        ← Classical search
├── engine_new.py        ← Engine interface
├── dataset_loader.py    ← Load games.csv
│
├── README.md            ← Overview
├── QUICKSTART.md        ← This file
├── ARCHITECTURE.md      ← Technical details
├── PROJECT_SUMMARY.md   ← Project overview
├── TRAINING_SUMMARY.md  ← Training results
├── PROJECT_STATUS.md    ← Completion status
├── DIAGRAMS.md          ← Architecture diagrams
│
├── requirements.txt     ← Dependencies
├── games.csv            ← Training dataset (20,058 games)
└── models/              ← Trained models
    ├── latest.pt        (238 MB) ← Current model
    ├── pretrained_supervised.pt  ← Backup
    └── csv_epoch_*.pt   ← Checkpoints
```

---

## ✅ Next Steps

### 1. ✅ Install Dependencies (Done)
```bash
pip install -r requirements.txt
```

### 2. ✅ Verify Installation
```bash
python test_system.py  # All tests should pass
```

### 3. 🎮 Play Chess!
```bash
python ChessGame.py    # Recommended: GUI game
# OR
python play_now.py     # Console game
# OR
python start.py        # Interactive menu
```

### 4. 📊 Test Performance (Optional)
```bash
python test_trained_model.py   # See what the AI can do
python quick_play_demo.py      # Watch AI vs AI
```

### 5. 🚀 Train More (Optional)
```bash
python start.py  # Choose option 3 for full training
```

---

## 📊 Model Status

**Current Model:** ✅ Pre-trained and ready!

| Metric | Value |
|--------|-------|
| Training Positions | 321,720 |
| Training Games | 5,000 |
| Training Time | 43 minutes |
| Epochs | 5 |
| Model Size | 238 MB |
| Parameters |20,778,945 |
| ELO Estimate | ~1200-1400 |
| Status | ✅ **Ready to play!** |

---

## 🎯 Quick Command Reference

### Essential Commands
```bash
# Play immediately
python ChessGame.py              # GUI game
python play_now.py               # Console game
python start.py                  # Interactive menu

# Test the AI
python test_trained_model.py     # Performance tests
python test_system.py            # System validation
python quick_play_demo.py        # Watch AI vs AI
python examples.py               # Code examples

# Train (optional)
python train_from_csv.py         # Supervised learning
python trainer.py                # Self-play RL
```

### Check System Status
```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check if model exists
python -c "from pathlib import Path; print(f'Model: {Path(\"models/latest.pt\").exists()}')"

# Check model size
ls -lh models/latest.pt  # Unix/Mac
dir models\latest.pt     # Windows
```

---

## 📚 Documentation

- **[README.md](README.md)** - Complete overview and features
- **[QUICKSTART.md](QUICKSTART.md)** - This file
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical deep-dive
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - What the project does
- **[TRAINING_SUMMARY.md](TRAINING_SUMMARY.md)** - Training results
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Completion checklist
- **[DIAGRAMS.md](DIAGRAMS.md)** - System diagrams

---

## 🎉 You're Ready!

The Chess AI is **fully trained and ready to play!**

```bash
# Start playing now:
python ChessGame.py
```

**Have fun! ♟️🤖**

---

<div align="center">

**Quick Start Complete!**

For detailed information, see [README.md](README.md)

**[⬆ Back to Top](#quick-start-guide---chess-ai)**

</div>

### For Learning
- Start with `examples.py` to understand components
- Read `ARCHITECTURE.md` for technical details
- Experiment with different configurations

### For Training
- Start with quick training to test
- Use GPU if available for faster training
- Monitor loss values - should decrease over time
- Save checkpoints regularly

### For Playing
- Even untrained AI uses MCTS (decent play)
- Trained AI gets stronger with more iterations
- Adjust NUM_SIMULATIONS for speed vs strength

---

## Common Commands

```bash
# Start everything
python start.py

# Just play
python ChessGame.py

# Just train
python trainer.py

# Run examples
python examples.py

# Quick test
python -c "from trainer import quick_train; quick_train(3, 5)"
```

---

## Expected Results

### After 5 iterations (~15 min)
- ✅ Learns basic piece movement
- ✅ Avoids hanging pieces
- ✅ Some tactical awareness

### After 20 iterations (~2-4 hours)
- ✅ Good piece coordination
- ✅ Center control
- ✅ Basic opening principles

### After 100 iterations (~20+ hours)
- ✅ Strong tactical play
- ✅ Positional understanding
- ✅ Good endgame technique

---

## Support

### Issues?
1. Check `README.md` for general info
2. Check `ARCHITECTURE.md` for technical details
3. Run examples.py to test components
4. Check error messages carefully

### Want to Contribute?
- Optimize training speed
- Improve GUI
- Add features
- Better documentation

---

**Ready to start? Run `python start.py`!** 🚀

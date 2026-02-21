# System Diagrams - Chess AI

## Overall System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Chess AI System                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │   GUI Layer  │────▶│ Engine Layer │────▶│  AI Layer   │ │
│  │ (ChessGame)  │     │(engine_new)  │     │(MCTS + NN)  │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│         │                     │                     │        │
│         │                     │                     │        │
│         ▼                     ▼                     ▼        │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │   pygame     │     │ python-chess │     │   PyTorch   │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Neural Network Architecture

```
Input Board (14 × 8 × 8)
         │
         │ Each piece type = 1 plane
         │ White: planes 0-5
         │ Black: planes 6-11
         │ Castling: plane 12
         │ En passant: plane 13
         │
         ▼
┌─────────────────────┐
│  Conv2D(3×3, 256)   │  Input convolution
│  BatchNorm + ReLU   │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Residual Block 1   │  ┐
└─────────────────────┘  │
         │                │
         ▼                │
┌─────────────────────┐  │
│  Residual Block 2   │  │  10 blocks
└─────────────────────┘  │  total
         │                │
        ...               │
         │                │
         ▼                │
┌─────────────────────┐  │
│  Residual Block 10  │  ┘
└─────────────────────┘
         │
         ├─────────────────────────────┐
         │                             │
         ▼                             ▼
┌─────────────────────┐   ┌─────────────────────┐
│    Policy Head      │   │     Value Head      │
│                     │   │                     │
│  Conv2D(1×1, 32)    │   │  Conv2D(1×1, 32)    │
│  BatchNorm + ReLU   │   │  BatchNorm + ReLU   │
│  Flatten            │   │  Flatten            │
│  FC(2048, 4096)     │   │  FC(2048, 256)      │
│  LogSoftmax         │   │  FC(256, 1)         │
│                     │   │  Tanh               │
└─────────────────────┘   └─────────────────────┘
         │                             │
         ▼                             ▼
    4096 moves                   Value [-1, 1]
  (from-to pairs)              (win probability)
```

---

## Residual Block Detail

```
        Input
          │
          ├──────────────────────┐
          │                      │
          ▼                      │ Skip Connection
    ┌─────────────┐             │
    │ Conv2D(3×3) │             │
    │  BatchNorm  │             │
    │    ReLU     │             │
    └─────────────┘             │
          │                      │
          ▼                      │
    ┌─────────────┐             │
    │ Conv2D(3×3) │             │
    │  BatchNorm  │             │
    └─────────────┘             │
          │                      │
          └──────(+)◀────────────┘
                  │
                  ▼
                ReLU
                  │
                  ▼
               Output
```

---

## MCTS Algorithm Flow

```
Start: Root Node (Current Position)
         │
         ▼
    ┌─────────────────┐
    │  1. SELECTION   │  Choose path using UCT scores
    │                 │  until reaching unexpanded node
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │  2. EXPANSION   │  Get NN predictions
    │                 │  Create children for legal moves
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │  3. EVALUATION  │  Get value from neural network
    │                 │  V = NN(position)
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │ 4. BACKPROP     │  Update visit counts and values
    │                 │  up to root (flipping signs)
    └─────────────────┘
         │
         ▼
    Repeat 800× (or NUM_SIMULATIONS)
         │
         ▼
    Select move with most visits
```

---

## MCTS Tree Structure

```
                      Root (current position)
                     N=800, V=0.15
                      /    |    \
                    /      |      \
                  /        |        \
              e4          d4         Nf3
            N=450       N=250       N=100
            V=0.20      V=0.12      V=0.08
            /  |  \      /  \         |
           /   |   \    /    \        |
         e5   d6   Nf6 d5   Nf6      d5
       N=300 N=100 N=50 ...  ...    ...
       V=0.18 ...  ...

UCT Score = Q(node) + c × P(node) × √(N_parent) / (1 + N(node))
            ↑         ↑              ↑
         Average    Prior from    Exploration
          Value       NN            Bonus
```

---

## Training Loop

```
┌─────────────────────────────────────────────────────────┐
│                   Training Iteration                     │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                      │
        ▼                                      ▼
┌──────────────────┐                ┌──────────────────┐
│   SELF-PLAY      │                │   TRAINING       │
│                  │                │                  │
│ Play 100 games   │                │ Sample batches   │
│ using MCTS       │───────────────▶│ from buffer      │
│                  │                │                  │
│ Store:           │                │ Train NN on:     │
│ - Position       │                │ - Policy loss    │
│ - MCTS policy    │                │ - Value loss     │
│ - Game result    │                │                  │
└──────────────────┘                └──────────────────┘
        │                                      │
        │                                      │
        ▼                                      ▼
┌──────────────────┐                ┌──────────────────┐
│  Replay Buffer   │                │  Updated Model   │
│  (10000 max)     │                │                  │
└──────────────────┘                └──────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  Save Checkpoint        │
              │  Repeat Iteration       │
              └─────────────────────────┘
```

---

## Data Flow During Move Selection

```
┌──────────────────┐
│  Chess Position  │  Current board state
└──────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│        Board → Tensor Encoding               │
│  - 14 planes × 8 × 8                         │
│  - Piece positions (one-hot)                 │
│  - Game state (castling, en passant)         │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│        Neural Network Forward Pass           │
│  - Convolutional layers                      │
│  - Residual connections                      │
│  - Policy and value heads                    │
└──────────────────────────────────────────────┘
         │
         ├─────────────────────┬────────────────┐
         │                     │                │
         ▼                     ▼                ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Move Probs   │    │ Position Val │    │ Legal Moves  │
│ (4096 dims)  │    │ (-1 to +1)   │    │ from board   │
└──────────────┘    └──────────────┘    └──────────────┘
         │                     │                │
         └─────────────────────┴────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │    Filter to Legal Moves Only      │
         │    Normalize probabilities         │
         └────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │         Run MCTS (800 sims)        │
         │    Build search tree using probs   │
         │    and value from NN               │
         └────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │    Select Move Based on Visits     │
         │    Apply temperature if needed     │
         └────────────────────────────────────┘
                              │
                              ▼
                        Final Move
```

---

## Loss Calculation

```
Game Data from Self-Play
         │
         ├─ Position (board state)
         ├─ Policy (MCTS visit distribution)
         └─ Result (win/loss/draw)
         │
         ▼
┌──────────────────────────────────────────┐
│        Neural Network Prediction         │
│                                          │
│  Input: board_tensor                     │
│  Output: predicted_policy, predicted_val │
└──────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│            Calculate Losses              │
│                                          │
│  Policy Loss (Cross Entropy):            │
│    L_p = -Σ target_π(a) × log(pred_p(a))│
│                                          │
│  Value Loss (MSE):                       │
│    L_v = (target_z - pred_v)²           │
│                                          │
│  Total Loss:                             │
│    L = L_p + L_v                         │
└──────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         Backpropagation                  │
│  - Compute gradients                     │
│  - Update network weights (Adam)         │
└──────────────────────────────────────────┘
```

---

## Complete System Interaction

```
┌─────────────┐
│    User     │
└──────┬──────┘
       │
       │ Clicks move
       ▼
┌──────────────────┐
│   ChessGame.py   │  Pygame GUI
│   (GUI Layer)    │
└────────┬─────────┘
         │
         │ get_best_move(board)
         ▼
┌──────────────────┐
│  engine_new.py   │  Engine wrapper
└────────┬─────────┘
         │
         │ AlphaZeroPlayer.select_move()
         ▼
┌──────────────────┐
│     mcts.py      │  Monte Carlo Tree Search
│                  │
│  For 800 sims:   │
│  1. Select       │───┐
│  2. Expand       │   │
│  3. Evaluate     │◀──┤
│  4. Backprop     │   │
└────────┬─────────┘   │
         │              │
         │ predict()    │
         ▼              │
┌──────────────────┐   │
│neural_network.py │   │
│                  │───┘
│  ChessModel:     │  Neural network predictions
│  - board → tensor│
│  - forward pass  │
│  - policy, value │
└──────────────────┘
```

---

## File Dependencies

```
start.py  ──────────────┐
                        │
ChessGame.py ───────────┤
                        ├──▶ engine_new.py
examples.py ────────────┤         │
                        │         ├──▶ neural_network.py ──▶ config.py
trainer.py ─────────────┤         │            │
                        │         │            └──▶ torch, chess
                        │         │
                        │         └──▶ mcts.py
                        │                 │
                        │                 └──▶ neural_network.py
                        │
test_system.py ─────────┘

requirements.txt ──▶ All dependencies
```

---

## User Workflows

### Workflow 1: Quick Play
```
User starts
    │
    ├─▶ python ChessGame.py
    │
    ├─▶ Select "Play vs AI"
    │
    ├─▶ Make moves by clicking
    │
    └─▶ AI responds (using MCTS + NN)
```

### Workflow 2: Training
```
User starts
    │
    ├─▶ python start.py
    │
    ├─▶ Choose "Train AI"
    │
    ├─▶ Self-play begins
    │       │
    │       ├─ Game 1
    │       ├─ Game 2
    │       └─ ...Game 100
    │
    ├─▶ Training begins
    │       │
    │       ├─ Epoch 1
    │       ├─ Epoch 2
    │       └─ ...Epoch 10
    │
    ├─▶ Model saved
    │
    └─▶ Ready to play!
```

### Workflow 3: Development
```
Developer
    │
    ├─▶ Read ARCHITECTURE.md
    │
    ├─▶ Run examples.py
    │
    ├─▶ Modify config.py
    │
    ├─▶ Run test_system.py
    │
    ├─▶ Train with new config
    │
    └─▶ Compare results
```

---

## State Machine

```
           ┌──────────────┐
           │  UNTRAINED   │
           └──────┬───────┘
                  │
                  │ start training
                  ▼
           ┌──────────────┐
    ┌─────▶│  SELF-PLAY   │────┐
    │      └──────────────┘    │
    │              │            │ N games complete
    │              │            │
    │              ▼            │
    │      ┌──────────────┐    │
    │      │  TRAINING    │◀───┘
    │      └──────┬───────┘
    │             │
    │             │ epoch complete
    │             ▼
    │      ┌──────────────┐
    │      │ CHECKPOINT   │
    │      └──────┬───────┘
    │             │
    │             │ iteration complete
    └─────────────┘
                  │
                  │ training done
                  ▼
           ┌──────────────┐
           │   TRAINED    │
           └──────┬───────┘
                  │
                  │ user plays
                  ▼
           ┌──────────────┐
           │   PLAYING    │
           └──────────────┘
```

---

These diagrams show:
1. Overall architecture
2. Neural network structure
3. MCTS algorithm
4. Training loop
5. Data flow
6. Loss calculation
7. System interactions
8. File dependencies
9. User workflows
10. State machine

Use these to understand how everything fits together!

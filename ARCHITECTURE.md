# Chess AI Architecture - Deep Reinforcement Learning

## Overview

This chess AI uses **Deep Reinforcement Learning** to learn chess from scratch through self-play, inspired by DeepMind's AlphaZero. The system combines:

1. **Deep Neural Network** - Evaluates positions and suggests moves
2. **Monte Carlo Tree Search (MCTS)** - Plans ahead using tree search
3. **Self-Play** - Learns by playing against itself
4. **Reinforcement Learning** - Improves through experience

---

## System Components

### 1. Neural Network (`neural_network.py`)

#### Architecture: ResNet-based Dual-Head Network

```
Input (14 x 8 x 8)
    ↓
Conv Layer + BatchNorm + ReLU
    ↓
[Residual Block] × 10
    ↓
    ├─→ Policy Head → 4096 outputs (move probabilities)
    └─→ Value Head → 1 output (position evaluation)
```

#### Input Representation (14 planes)
- **Planes 0-5**: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- **Planes 6-11**: Black pieces
- **Plane 12**: Castling rights (encoded 0-1)
- **Plane 13**: En passant square

#### Output Heads
1. **Policy Head**
   - 4096-dimensional vector (64 from-squares × 64 to-squares)
   - Probability distribution over all possible moves
   - Used to guide MCTS exploration

2. **Value Head**
   - Single scalar value between -1 and 1
   - Estimates win probability from current position
   - +1 = white winning, -1 = black winning, 0 = draw

#### Key Components

**ResidualBlock**
```python
Conv2d(3x3) → BatchNorm → ReLU → Conv2d(3x3) → BatchNorm
     ↓                                           ↓
     └─────────────── Add ──────────────────────┘
                       ↓
                     ReLU
```

Benefits:
- Allows deeper networks (10+ layers)
- Better gradient flow during training
- Learns both low-level (piece positions) and high-level (strategy) features

---

### 2. Monte Carlo Tree Search (`mcts.py`)

#### MCTS Algorithm

```
1. Selection
   - Start at root
   - Choose child with highest UCT score
   - Repeat until reaching unexpanded node

2. Expansion
   - Use neural network to get move probabilities
   - Create child nodes for all legal moves

3. Evaluation
   - Use neural network to evaluate position
   - Get estimated value of position

4. Backpropagation
   - Update statistics up the tree
   - Flip value sign at each level (alternating players)
```

#### UCT Score (Upper Confidence Bound for Trees)

```
UCT(node) = Q(node) + c_puct × P(node) × √(N_parent) / (1 + N(node))
            ↑                   ↑           ↑
         Exploitation      Prior Prob    Exploration
```

Where:
- **Q(node)**: Average value from simulations (exploitation)
- **P(node)**: Prior probability from neural network
- **N**: Visit counts (exploration bonus for unvisited nodes)
- **c_puct**: Exploration constant (typically 1.0-2.0)

#### Why MCTS?

1. **Balances exploration & exploitation**: UCT formula
2. **Asymmetric tree growth**: Focuses on promising moves
3. **Anytime algorithm**: Can stop after any number of simulations
4. **Guided by neural network**: More efficient than random playouts

---

### 3. Training System (`trainer.py`)

#### Training Loop (AlphaZero Style)

```
For each iteration:
    1. Self-Play Phase
       ├─ Play N games against self
       ├─ Use MCTS to select moves
       ├─ Store (position, policy, outcome)
       └─ Add to replay buffer
    
    2. Training Phase
       ├─ Sample batches from replay buffer
       ├─ Train neural network
       │   ├─ Policy loss: Cross-entropy
       │   └─ Value loss: Mean squared error
       └─ Update weights
    
    3. Checkpoint
       └─ Save model weights
```

#### Replay Buffer

- Stores game positions for training
- Maximum size (default: 10,000 positions)
- Random sampling for mini-batches
- Improves training stability

#### Loss Function

```
Total Loss = Policy Loss + Value Loss

Policy Loss = -∑ π(move) × log(p(move))
              Target    Prediction
              
Value Loss = (z - v)²
             ↑    ↑
           Target Prediction
```

Where:
- **π**: Target policy from MCTS visit counts
- **p**: Predicted policy from neural network
- **z**: Actual game outcome (+1, 0, -1)
- **v**: Predicted value from neural network

---

### 4. Chess Engine (`engine_new.py`)

#### ChessEngine Class

Wraps the entire system for easy use:

```python
engine = ChessEngine()
move = engine.get_best_move(board)
```

Features:
- Automatic model loading
- Configurable MCTS simulations
- Position evaluation
- Move probability distribution

---

## Training Process

### Phase 1: Self-Play

```python
# AI plays against itself
player = AlphaZeroPlayer(model, num_simulations=800)

for move in game:
    1. Run MCTS from current position
    2. Select move based on visit counts
    3. Record (board, policy, _)
    4. Make move
```

#### Move Selection with Temperature

```
Early game (moves 1-30): temperature = 1.0
    → More exploration, diverse games
    
Late game (moves 31+): temperature = 0.0
    → Greedy, best move always chosen
```

### Phase 2: Training

```python
for epoch in range(num_epochs):
    for batch in replay_buffer:
        1. boards → neural network → (policy, value)
        2. Calculate losses
        3. Backpropagation
        4. Update weights
```

### Phase 3: Iteration

```
Iteration 1 → Iteration 2 → ... → Iteration N
    ↓             ↓                    ↓
Better AI → Harder games → Better training data
```

Key insight: **AI improves by playing against progressively stronger versions of itself**

---

## Key Algorithms

### 1. MCTS Node Selection

```python
def select_child(self):
    best_score = -inf
    for child in self.children:
        score = child.uct_score(parent_visits)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child
```

### 2. Move Probability from Visit Counts

```python
# After MCTS search
visits = [child.visit_count for child in root.children]

# Apply temperature
probs = visits ** (1 / temperature)
probs = probs / sum(probs)

# Sample move
move = sample(probs)
```

### 3. Position Evaluation

```python
def evaluate_board(board):
    tensor = board_to_tensor(board)
    policy, value = neural_network(tensor)
    return policy, value
```

---

## Training Progression

### Stage 1: Random Play (Iteration 0-5)
- Network outputs are nearly random
- MCTS provides some structure
- Games are chaotic, many illegal move attempts filtered

### Stage 2: Basic Tactics (Iteration 5-20)
- Learns piece values
- Captures hanging pieces
- Avoids losing material

### Stage 3: Positional Understanding (Iteration 20-50)
- Develops pieces
- Controls center
- King safety concepts

### Stage 4: Strategic Play (Iteration 50-100+)
- Opening principles
- Middlegame plans
- Endgame technique

### Stage 5: Expert Level (Iteration 100-1000+)
- Deep tactical calculations
- Strategic sacrifices
- Subtle positional play

---

## Hyperparameters

### Neural Network
- **Residual Blocks**: 10 (more = stronger but slower)
- **Hidden Channels**: 256 (network width)
- **Learning Rate**: 0.001 (Adam optimizer)
- **Batch Size**: 32

### MCTS
- **Simulations**: 800 (more = stronger play)
- **C_PUCT**: 1.0 (exploration constant)
- **Temperature**: 1.0 early, 0.0 late
- **Dirichlet Noise**: α=0.3, ε=0.25 (root exploration)

### Training
- **Self-Play Games**: 100 per iteration
- **Training Epochs**: 10 per iteration
- **Replay Buffer**: 10,000 positions
- **Total Iterations**: 1000+

---

## Performance Considerations

### Speed Optimizations
1. **GPU Acceleration**: Use CUDA for neural network
2. **Batch Inference**: Process multiple positions at once
3. **Parallel Self-Play**: Multiple games simultaneously
4. **MCTS Virtual Loss**: Prevent thread collision

### Memory Management
1. **Replay Buffer Limit**: Prevent memory overflow
2. **MCTS Tree Pruning**: Discard old branches
3. **Model Checkpointing**: Save periodically

### Training Time Estimates

**On CPU (Intel i7)**:
- 1 game: ~5 minutes
- 100 games: ~8 hours
- 1 iteration: ~10 hours

**On GPU (NVIDIA RTX 3080)**:
- 1 game: ~1 minute
- 100 games: ~2 hours
- 1 iteration: ~2.5 hours

---

## Comparison to AlphaZero

### Similarities
- ✅ ResNet architecture
- ✅ MCTS with neural network guidance
- ✅ Self-play training
- ✅ Policy + value dual heads

### Differences
- ❌ Smaller network (our: 256 channels, AZ: 256-512)
- ❌ Fewer simulations (our: 800, AZ: 1600+)
- ❌ Less parallelization
- ❌ No distributed training
- ❌ Simpler move representation

---

## Future Improvements

### Architecture
- [ ] Attention mechanisms
- [ ] Larger networks (more channels, blocks)
- [ ] Squeeze-and-excitation blocks
- [ ] Better move encoding (queen moves, promotions)

### Training
- [ ] Distributed training across multiple GPUs
- [ ] Prioritized experience replay
- [ ] Curriculum learning
- [ ] Transfer learning from Stockfish games

### MCTS
- [ ] Parallel MCTS (virtual loss)
- [ ] Progressive widening
- [ ] Transposition tables
- [ ] Opening book integration

### Infrastructure
- [ ] Evaluation against standard engines
- [ ] ELO rating tracking
- [ ] Cloud training support
- [ ] Model versioning and comparison

---

## References

1. **AlphaZero Paper**: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (Silver et al., 2017)

2. **AlphaGo Zero Paper**: "Mastering the game of Go without human knowledge" (Silver et al., 2017)

3. **Leela Chess Zero**: Open-source implementation of AlphaZero for chess

4. **python-chess**: Chess library for move generation and rules

---

## Conclusion

This chess AI demonstrates the power of deep reinforcement learning:

- **No human knowledge** needed (learns from scratch)
- **Self-improvement** through self-play
- **Deep understanding** emerges from simple objectives
- **Scales** with compute and training time

The same principles can be applied to other games and decision-making problems!

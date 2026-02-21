# Configuration for Chess AI Training

# Neural Network
BOARD_SIZE = 8
NUM_PLANES = 14  # 6 piece types * 2 colors + 2 for game state
HIDDEN_SIZE = 256
NUM_RESIDUAL_BLOCKS = 10

# MCTS
NUM_SIMULATIONS = 800  # Number of MCTS simulations per move
C_PUCT = 1.0  # Exploration constant
TEMPERATURE = 1.0  # Temperature for move selection
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_SELF_PLAY_GAMES = 100
NUM_TRAINING_ITERATIONS = 1000
REPLAY_BUFFER_SIZE = 10000

# Game
MAX_MOVES = 200  # Maximum moves before declaring draw

# Model saving
MODEL_PATH = 'models/'
CHECKPOINT_INTERVAL = 10

# Dataset (games.csv) - Supervised Pre-training
GAMES_CSV_PATH = 'games.csv'
MIN_ELO_FILTER = 1500        # Only use games where both players are >= this rating
MAX_GAMES_TO_LOAD = 50000    # Cap to avoid running out of memory
CSV_TRAIN_EPOCHS = 5         # Epochs for supervised pre-training pass
CSV_BATCH_SIZE = 64          # Larger batch for supervised phase

import os
import chess
from neural_network import ChessModel
from mcts import AlphaZeroPlayer
from config import NUM_SIMULATIONS, MODEL_PATH


class ChessEngine:
    """
    Chess AI engine using deep reinforcement learning.
    Uses MCTS with neural network guidance (AlphaZero-style).
    """
    
    def __init__(self, model_path=None, num_simulations=NUM_SIMULATIONS):
        """
        Initialize the chess engine.
        
        Args:
            model_path: path to trained model (uses latest if None)
            num_simulations: number of MCTS simulations per move
        """
        self.model = ChessModel()
        
        # Load trained model if available
        if model_path is None:
            latest_path = os.path.join(MODEL_PATH, 'latest.pt')
            if os.path.exists(latest_path):
                model_path = latest_path
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model.load(model_path)
        else:
            print("No trained model found - using untrained network")
            print("Run trainer.py to train the model first!")
        
        self.player = AlphaZeroPlayer(self.model, num_simulations, temperature=0.1)
    
    def get_best_move(self, board):
        """
        Get the best move for the current position using MCTS.
        
        Args:
            board: chess.Board object
            
        Returns:
            move: chess.Move object
        """
        if not isinstance(board, chess.Board):
            raise ValueError("Board must be a chess.Board object")
        
        move = self.player.select_move(board)
        return move
    
    def evaluate_position(self, board):
        """
        Evaluate the current position.
        
        Args:
            board: chess.Board object
            
        Returns:
            value: position evaluation (-1 to 1)
        """
        _, value = self.model.predict(board)
        return value
    
    def get_move_probabilities(self, board):
        """
        Get probability distribution over all legal moves.
        
        Args:
            board: chess.Board object
            
        Returns:
            move_probs: dict mapping moves to probabilities
        """
        move_probs, _ = self.model.predict(board)
        return move_probs

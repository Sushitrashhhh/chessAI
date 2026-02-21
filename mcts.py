import math
import numpy as np
import chess
from copy import deepcopy
from config import *


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.
    Each node represents a game state.
    """
    
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # Prior probability from neural network
        
        self.children = {}  # Map of move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    def value(self):
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def uct_score(self, parent_visit_count, c_puct=C_PUCT):
        """
        Calculate Upper Confidence Bound for Trees (UCT) score.
        Balances exploitation (value) and exploration (prior).
        """
        if self.visit_count == 0:
            q_value = 0.0
        else:
            q_value = self.value()
        
        # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        u_value = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        return q_value + u_value
    
    def select_child(self):
        """Select child with highest UCT score."""
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        for move, child in self.children.items():
            score = child.uct_score(self.visit_count)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        
        return best_move, best_child
    
    def expand(self, move_probs):
        """
        Expand this node by creating children for all legal moves.
        
        Args:
            move_probs: dict mapping moves to prior probabilities
        """
        self.is_expanded = True
        
        for move, prob in move_probs.items():
            if move not in self.children:
                child_board = self.board.copy()
                child_board.push(move)
                self.children[move] = MCTSNode(child_board, parent=self, move=move, prior=prob)
    
    def update(self, value):
        """
        Update this node's statistics after a simulation.
        Value is from perspective of current player.
        """
        self.visit_count += 1
        self.value_sum += value
    
    def backpropagate(self, value):
        """
        Backpropagate value up the tree, flipping sign at each level
        (because value is from perspective of current player).
        """
        self.update(value)
        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    """
    Monte Carlo Tree Search for chess.
    Uses neural network to guide search.
    """
    
    def __init__(self, model, num_simulations=NUM_SIMULATIONS):
        self.model = model
        self.num_simulations = num_simulations
        
    def search(self, board, temperature=TEMPERATURE):
        """
        Perform MCTS from the given board position.
        
        Args:
            board: chess.Board object
            temperature: controls exploration vs exploitation
                         - Higher temp = more exploration
                         - Lower temp = more exploitation
        
        Returns:
            move: selected move
            move_probs: dict of move -> visit probability
        """
        root = MCTSNode(board)
        
        # Add Dirichlet noise to root node for exploration
        move_probs, _ = self.model.predict(board)
        
        if len(move_probs) == 0:
            return None, {}
        
        # Add Dirichlet noise to encourage exploration
        moves = list(move_probs.keys())
        probs = np.array([move_probs[m] for m in moves])
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
        probs = (1 - DIRICHLET_EPSILON) * probs + DIRICHLET_EPSILON * noise
        
        # Normalize
        probs = probs / probs.sum()
        move_probs = {move: prob for move, prob in zip(moves, probs)}
        
        root.expand(move_probs)
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root)
        
        # Select move based on visit counts
        move_visits = {move: child.visit_count for move, child in root.children.items()}
        total_visits = sum(move_visits.values())
        
        if total_visits == 0:
            # Fallback to prior if no visits
            return max(move_probs.items(), key=lambda x: x[1])[0], move_probs
        
        # Calculate visit-based probabilities with temperature
        if temperature == 0:
            # Greedy: pick most visited
            best_move = max(move_visits.items(), key=lambda x: x[1])[0]
            visit_probs = {move: 0.0 for move in move_visits}
            visit_probs[best_move] = 1.0
        else:
            # Apply temperature
            visits = np.array([move_visits[m] for m in moves])
            visits_temp = visits ** (1.0 / temperature)
            visits_temp = visits_temp / visits_temp.sum()
            visit_probs = {move: prob for move, prob in zip(moves, visits_temp)}
        
        # Sample move according to probabilities
        moves = list(visit_probs.keys())
        probs = [visit_probs[m] for m in moves]
        selected_move = np.random.choice(moves, p=probs)
        
        return selected_move, visit_probs
    
    def _simulate(self, node):
        """
        Run one MCTS simulation from the given node.
        
        Steps:
        1. Selection: traverse tree using UCT until reaching unexpanded node
        2. Expansion: expand node using neural network
        3. Evaluation: evaluate position with neural network
        4. Backpropagation: update statistics up the tree
        """
        # Check for terminal state
        if node.board.is_game_over():
            result = node.board.result()
            if result == "1-0":  # White wins
                value = 1.0 if node.board.turn == chess.WHITE else -1.0
            elif result == "0-1":  # Black wins
                value = -1.0 if node.board.turn == chess.WHITE else 1.0
            else:  # Draw
                value = 0.0
            node.backpropagate(value)
            return
        
        # Expansion
        if not node.is_expanded:
            move_probs, value = self.model.predict(node.board)
            
            if len(move_probs) == 0:
                # No legal moves (shouldn't happen if game not over)
                value = 0.0
            
            node.expand(move_probs)
            # Value is from perspective of current player
            node.backpropagate(value)
            return
        
        # Selection: choose best child using UCT
        move, child = node.select_child()
        
        if child is None:
            # No children (shouldn't happen)
            node.backpropagate(0.0)
            return
        
        # Recurse on selected child
        self._simulate(child)


class AlphaZeroPlayer:
    """
    Chess player using MCTS with neural network guidance.
    """
    
    def __init__(self, model, num_simulations=NUM_SIMULATIONS, temperature=TEMPERATURE):
        self.model = model
        self.mcts = MCTS(model, num_simulations)
        self.temperature = temperature
        
    def select_move(self, board):
        """
        Select a move using MCTS.
        
        Args:
            board: chess.Board object
            
        Returns:
            move: chess.Move
        """
        move, _ = self.mcts.search(board, self.temperature)
        return move
    
    def get_move_with_policy(self, board):
        """
        Select a move and return the policy for training.
        
        Returns:
            move: selected move
            policy: dict of move -> probability (for training)
        """
        return self.mcts.search(board, self.temperature)


def play_game_with_mcts(model1, model2=None, num_simulations=NUM_SIMULATIONS):
    """
    Play a game using MCTS-guided players.
    
    Args:
        model1: ChessModel for white (or both if model2 is None)
        model2: ChessModel for black (optional, uses model1 if None)
        num_simulations: number of MCTS simulations per move
        
    Returns:
        game_data: list of (board, policy, result) tuples for training
        result: game result (1.0 for white win, -1.0 for black win, 0.0 for draw)
    """
    if model2 is None:
        model2 = model1
    
    player_white = AlphaZeroPlayer(model1, num_simulations)
    player_black = AlphaZeroPlayer(model2, num_simulations)
    
    board = chess.Board()
    game_data = []
    
    move_count = 0
    while not board.is_game_over() and move_count < MAX_MOVES:
        # Select player
        if board.turn == chess.WHITE:
            player = player_white
        else:
            player = player_black
        
        # Get move and policy
        move, policy = player.get_move_with_policy(board)
        
        if move is None:
            break
        
        # Store position and policy for training
        game_data.append((board.copy(), policy, 0.0))  # Result will be filled later
        
        # Make move
        board.push(move)
        move_count += 1
    
    # Determine result
    if board.is_checkmate():
        # Player who just moved won
        result = 1.0 if board.turn == chess.BLACK else -1.0
    else:
        # Draw
        result = 0.0
    
    # Update game data with actual result
    # Result is from perspective of player who made the move
    updated_game_data = []
    for i, (board_state, policy, _) in enumerate(game_data):
        # Flip result for black's moves
        if board_state.turn == chess.WHITE:
            updated_game_data.append((board_state, policy, result))
        else:
            updated_game_data.append((board_state, policy, -result))
    
    return updated_game_data, result

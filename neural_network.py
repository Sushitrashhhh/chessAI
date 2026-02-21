import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
from config import *


class ResidualBlock(nn.Module):
    """Residual block for the neural network."""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    """
    Neural network for chess that outputs:
    1. Policy: probability distribution over all possible moves
    2. Value: estimated win probability for current position
    
    Architecture inspired by AlphaZero.
    """
    
    def __init__(self, num_res_blocks=NUM_RESIDUAL_BLOCKS, hidden_channels=HIDDEN_SIZE):
        super(ChessNet, self).__init__()
        
        # Input: board representation (14 planes x 8 x 8)
        self.conv_input = nn.Conv2d(NUM_PLANES, hidden_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(hidden_channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)  # 64*64 possible moves (from-to)
        
        # Value head
        self.value_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Input processing
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


def board_to_tensor(board):
    """
    Convert a chess.Board to a tensor representation.
    
    Creates 14 planes:
    - 6 planes for white pieces (P, N, B, R, Q, K)
    - 6 planes for black pieces (p, n, b, r, q, k)
    - 1 plane for castling rights
    - 1 plane for en passant
    
    Returns: tensor of shape (14, 8, 8)
    """
    tensor = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    
    # Piece placement
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        piece_type = piece.piece_type - 1  # 0-5
        if piece.color == chess.WHITE:
            plane = piece_type
        else:
            plane = piece_type + 6
            
        tensor[plane, rank, file] = 1.0
    
    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 0.25
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[12, :, :] += 0.25
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[12, :, :] += 0.25
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[12, :, :] += 0.25
    
    # En passant
    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        tensor[13, rank, file] = 1.0
    
    return tensor


def move_to_index(move):
    """Convert a chess.Move to an index (0-4095)."""
    from_square = move.from_square
    to_square = move.to_square
    return from_square * 64 + to_square


def index_to_move(index, board):
    """
    Convert an index to a chess.Move.
    Note: This might not always produce a legal move.
    """
    from_square = index // 64
    to_square = index % 64
    
    # Try to create a move, handling promotion
    move = chess.Move(from_square, to_square)
    
    # Check if it's a pawn promotion
    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(to_square)
        if (piece.color == chess.WHITE and to_rank == 7) or \
           (piece.color == chess.BLACK and to_rank == 0):
            move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
    
    return move


class ChessModel:
    """Wrapper class for the neural network with training utilities."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = ChessNet().to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )
        
    def predict(self, board):
        """
        Predict policy and value for a given board position.
        
        Returns:
            policy: dict mapping legal moves to probabilities
            value: estimated value of position (-1 to 1)
        """
        self.model.eval()
        with torch.no_grad():
            tensor = board_to_tensor(board)
            tensor = torch.FloatTensor(tensor).unsqueeze(0).to(self.device)
            
            log_policy, value = self.model(tensor)
            policy = torch.exp(log_policy).cpu().numpy()[0]
            value = value.cpu().item()
            
            # Map to legal moves only
            legal_moves = list(board.legal_moves)
            move_probs = {}
            total_prob = 0.0
            
            for move in legal_moves:
                idx = move_to_index(move)
                if idx < len(policy):
                    move_probs[move] = policy[idx]
                    total_prob += policy[idx]
            
            # Normalize
            if total_prob > 0:
                for move in move_probs:
                    move_probs[move] /= total_prob
            else:
                # Uniform distribution if all zero
                for move in legal_moves:
                    move_probs[move] = 1.0 / len(legal_moves)
            
            return move_probs, value
    
    def train_step(self, boards, target_policies, target_values):
        """
        Perform one training step.
        
        Args:
            boards: list of chess.Board objects
            target_policies: list of target policy distributions (dicts)
            target_values: list of target values (-1 to 1)
        """
        self.model.train()
        
        # Convert boards to tensors
        board_tensors = []
        policy_targets = []
        value_targets = []
        
        for board, target_policy, target_value in zip(boards, target_policies, target_values):
            tensor = board_to_tensor(board)
            board_tensors.append(tensor)
            
            # Convert target_policy dict to full array
            policy_array = np.zeros(4096, dtype=np.float32)
            for move, prob in target_policy.items():
                idx = move_to_index(move)
                if idx < 4096:
                    policy_array[idx] = prob
            policy_targets.append(policy_array)
            value_targets.append(target_value)
        
        # Stack into batches
        board_tensors = torch.FloatTensor(np.array(board_tensors)).to(self.device)
        policy_targets = torch.FloatTensor(np.array(policy_targets)).to(self.device)
        value_targets = torch.FloatTensor(np.array(value_targets)).unsqueeze(1).to(self.device)
        
        # Forward pass
        log_policy_pred, value_pred = self.model(board_tensors)
        
        # Calculate loss
        policy_loss = -torch.sum(policy_targets * log_policy_pred) / len(boards)
        value_loss = F.mse_loss(value_pred, value_targets)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def save(self, path):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

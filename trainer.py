import os
import random
from collections import deque
import numpy as np
import torch
from tqdm import tqdm
import chess

from neural_network import ChessModel
from mcts import play_game_with_mcts
from config import (
    REPLAY_BUFFER_SIZE, BATCH_SIZE, NUM_SELF_PLAY_GAMES,
    NUM_SIMULATIONS, NUM_TRAINING_ITERATIONS, MODEL_PATH,
    CHECKPOINT_INTERVAL, LEARNING_RATE, WEIGHT_DECAY,
    GAMES_CSV_PATH, MIN_ELO_FILTER, MAX_GAMES_TO_LOAD,
    CSV_TRAIN_EPOCHS, CSV_BATCH_SIZE,
)


class ReplayBuffer:
    """Store game positions for training."""
    
    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
        
    def add_game(self, game_data):
        """
        Add all positions from a game to the buffer.
        
        Args:
            game_data: list of (board, policy, result) tuples
        """
        self.buffer.extend(game_data)
        
    def sample(self, batch_size):
        """Sample a batch of positions randomly."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        boards = [item[0] for item in batch]
        policies = [item[1] for item in batch]
        values = [item[2] for item in batch]
        
        return boards, policies, values
    
    def __len__(self):
        return len(self.buffer)


class Trainer:
    """
    Trainer for the chess AI using self-play and reinforcement learning.
    Similar to AlphaZero training approach.
    """
    
    def __init__(self, model=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        if model is None:
            self.model = ChessModel(device=device)
        else:
            self.model = model
            
        self.replay_buffer = ReplayBuffer()
        
        # Create models directory if it doesn't exist
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
    
    def self_play(self, num_games=NUM_SELF_PLAY_GAMES):
        """
        Generate training data through self-play.
        
        Args:
            num_games: number of games to play
            
        Returns:
            statistics: dict with game statistics
        """
        print(f"\nStarting self-play: {num_games} games")
        
        wins_white = 0
        wins_black = 0
        draws = 0
        total_moves = 0
        
        for game_num in tqdm(range(num_games), desc="Self-play"):
            # Play game with some randomness early on
            game_data, result = play_game_with_mcts(
                self.model, 
                num_simulations=NUM_SIMULATIONS
            )
            
            # Add to replay buffer
            self.replay_buffer.add_game(game_data)
            
            # Update statistics
            total_moves += len(game_data)
            if result > 0:
                wins_white += 1
            elif result < 0:
                wins_black += 1
            else:
                draws += 1
        
        stats = {
            'wins_white': wins_white,
            'wins_black': wins_black,
            'draws': draws,
            'avg_moves': total_moves / num_games if num_games > 0 else 0,
            'buffer_size': len(self.replay_buffer)
        }
        
        print(f"Self-play complete: W={wins_white} B={wins_black} D={draws}")
        print(f"Avg moves per game: {stats['avg_moves']:.1f}")
        print(f"Replay buffer size: {stats['buffer_size']}")
        
        return stats
    
    def train(self, num_epochs=10, batch_size=BATCH_SIZE):
        """
        Train the neural network on data from replay buffer.
        
        Args:
            num_epochs: number of training epochs
            batch_size: batch size for training
            
        Returns:
            losses: dict with loss statistics
        """
        if len(self.replay_buffer) < batch_size:
            print("Not enough data in replay buffer for training")
            return {}
        
        print(f"\nTraining for {num_epochs} epochs with batch size {batch_size}")
        
        total_losses = []
        policy_losses = []
        value_losses = []
        
        num_batches = len(self.replay_buffer) // batch_size
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for _ in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Sample batch
                boards, policies, values = self.replay_buffer.sample(batch_size)
                
                # Train on batch
                losses = self.model.train_step(boards, policies, values)
                epoch_losses.append(losses['total_loss'])
                total_losses.append(losses['total_loss'])
                policy_losses.append(losses['policy_loss'])
                value_losses.append(losses['value_loss'])
            
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        stats = {
            'total_loss': np.mean(total_losses),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses)
        }
        
        print(f"Training complete - Total Loss: {stats['total_loss']:.4f}")
        
        return stats
    
    def train_iteration(self, iteration, num_self_play_games=NUM_SELF_PLAY_GAMES):
        """
        Perform one full training iteration:
        1. Self-play to generate data
        2. Train on generated data
        3. Save checkpoint
        
        Args:
            iteration: current iteration number
            num_self_play_games: number of self-play games to generate
        """
        print(f"\n{'='*60}")
        print(f"Training Iteration {iteration}")
        print(f"{'='*60}")
        
        # Self-play
        self_play_stats = self.self_play(num_self_play_games)
        
        # Train
        train_stats = self.train()
        
        # Save checkpoint
        if iteration % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(MODEL_PATH, f'checkpoint_{iteration}.pt')
            self.model.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Also save as latest
        latest_path = os.path.join(MODEL_PATH, 'latest.pt')
        self.model.save(latest_path)
        
        return {**self_play_stats, **train_stats}
    
    def pretrain_from_csv(
        self,
        csv_path: str = GAMES_CSV_PATH,
        min_elo: int = MIN_ELO_FILTER,
        max_games: int = MAX_GAMES_TO_LOAD,
    ) -> dict:
        """
        Phase 0: Supervised pre-training from games.csv.

        Loads real human games into the replay buffer (filtered by ELO),
        then trains the network on those positions before any self-play.
        This gives the network a strong baseline policy and value function,
        dramatically accelerating the subsequent self-play phase.

        Args:
            csv_path  : path to the CSV file
            min_elo   : minimum rating required for both players
            max_games : maximum number of games to load

        Returns:
            dict with 'loading_stats' and 'training_stats' keys
        """
        from dataset_loader import load_games_into_buffer

        print(f"\n{'='*60}")
        print("Phase 0: Supervised Pre-training from Dataset")
        print(f"{'='*60}")
        print(f"  CSV path : {csv_path}")
        print(f"  Min ELO  : {min_elo}")
        print(f"  Max games: {max_games:,}")

        loading_stats = load_games_into_buffer(
            self.replay_buffer,
            csv_path=csv_path,
            min_elo=min_elo,
            max_games=max_games,
        )

        if len(self.replay_buffer) == 0:
            print("[Trainer] No positions loaded — skipping supervised training.")
            return {'loading_stats': loading_stats, 'training_stats': {}}

        print(f"[Trainer] Starting supervised training "
              f"({CSV_TRAIN_EPOCHS} epochs, batch {CSV_BATCH_SIZE}) ...")
        training_stats = self.train(
            num_epochs=CSV_TRAIN_EPOCHS,
            batch_size=CSV_BATCH_SIZE,
        )

        # Save a checkpoint after supervised pre-training
        pretrain_path = os.path.join(MODEL_PATH, 'pretrained_supervised.pt')
        self.model.save(pretrain_path)
        print(f"[Trainer] Supervised checkpoint saved → {pretrain_path}")

        return {'loading_stats': loading_stats, 'training_stats': training_stats}

    def full_training(
        self,
        num_iterations: int = NUM_TRAINING_ITERATIONS,
        pretrain_csv: bool = True,
    ) -> list:
        """
        Run the full training pipeline:
          Phase 0 (optional) — Supervised pre-training from games.csv
          Phase 1..N         — AlphaZero-style self-play iterations

        Args:
            num_iterations: number of self-play training iterations
            pretrain_csv  : if True, run supervised pre-training first
        """
        print(f"\nStarting full training: {num_iterations} iterations")
        print(f"Device: {self.device}")
        print(f"Self-play games per iteration: {NUM_SELF_PLAY_GAMES}")
        print(f"MCTS simulations per move: {NUM_SIMULATIONS}")
        print(f"Supervised pre-training: {'ON' if pretrain_csv else 'OFF'}")

        all_stats = []

        # ------------------------------------------------------------------
        # Phase 0: Supervised pre-training from games.csv
        # ------------------------------------------------------------------
        if pretrain_csv:
            pretrain_stats = self.pretrain_from_csv()
            all_stats.append({'phase': 'supervised', **pretrain_stats})

        # ------------------------------------------------------------------
        # Phase 1..N: Self-play reinforcement learning
        # ------------------------------------------------------------------
        for iteration in range(1, num_iterations + 1):
            stats = self.train_iteration(iteration)
            all_stats.append(stats)

            # Print summary
            print(f"\nIteration {iteration} Summary:")
            print(f"  Buffer size: {stats['buffer_size']}")
            print(f"  Total loss: {stats.get('total_loss', 'N/A')}")

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")

        return all_stats


def quick_train(num_iterations=5, num_games_per_iter=10):
    """
    Quick training function for testing (fewer games/iterations).
    
    Args:
        num_iterations: number of training iterations
        num_games_per_iter: number of self-play games per iteration
    """
    print("Starting quick training mode...")
    trainer = Trainer()
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n--- Quick Training Iteration {iteration}/{num_iterations} ---")
        
        # Self-play with fewer games
        trainer.self_play(num_games_per_iter)
        
        # Train with fewer epochs
        trainer.train(num_epochs=3)
        
        # Save
        trainer.model.save(os.path.join(MODEL_PATH, f'quick_checkpoint_{iteration}.pt'))
    
    print("\nQuick training complete!")
    return trainer


if __name__ == '__main__':
    # Run full training
    trainer = Trainer()
    trainer.full_training(num_iterations=10)

"""
Quick Start Script for Chess AI

This script helps you get started with the Chess AI quickly.
Choose from different modes to train or play.
"""

import os
import sys


def print_banner():
    print("=" * 60)
    print("  Chess AI - Deep Reinforcement Learning")
    print("  AlphaZero-style Self-Play Training")
    print("=" * 60)
    print()


def check_dependencies():
    """Check if all required packages are installed."""
    required = ['torch', 'chess', 'pygame', 'numpy', 'tqdm']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing dependencies:", ', '.join(missing))
        print("\nInstall them with:")
        print("  pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True


def show_menu():
    """Show main menu."""
    print("\nWhat would you like to do?\n")
    print("1. 🎮 Play Chess (with GUI)")
    print("2. 🤖 Train AI (Quick - 5 iterations, ~10 minutes)")
    print("3. 🚀 Train AI (Full - 100 iterations, several hours)")
    print("4. 📊 Check Model Status")
    print("5. 🧪 Test Neural Network")
    print("6. 📂 Pre-train from Dataset (games.csv)")
    print("7. 📈 Show Dataset Statistics")
    print("8. ⚡ Benchmark Alpha-Beta Engine")
    print("9. 🥊 Alpha-Beta vs MCTS (1 game duel)")
    print("0. ❌ Exit")
    print()

    choice = input("Enter your choice (0-9): ").strip()
    return choice


def play_game():
    """Launch the chess game."""
    print("\n🎮 Launching Chess Game...")
    print("Close the game window to return to this menu.\n")
    
    try:
        from ChessGame import ChessGame
        game = ChessGame()
        game.play()
    except Exception as e:
        print(f"Error launching game: {e}")
        import traceback
        traceback.print_exc()


def train_quick():
    """Quick training mode."""
    print("\n🤖 Starting Quick Training...")
    print("This will train for 5 iterations with 10 games each.")
    print("Estimated time: 10-20 minutes (depending on hardware)\n")
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    try:
        from trainer import quick_train
        trainer = quick_train(num_iterations=5, num_games_per_iter=10)
        print("\n✅ Quick training complete!")
        print("You can now play against the trained AI.")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


def train_full():
    """Full training mode."""
    print("\n🚀 Starting Full Training...")
    print("This will train for 100 iterations.")
    print("Estimated time: Several hours to days (depending on hardware)")
    print("\nConfiguration:")
    print("  - 100 self-play games per iteration")
    print("  - 800 MCTS simulations per move")
    print("  - ResNet with 10 residual blocks\n")
    
    confirm = input("This will take a long time. Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    try:
        from trainer import Trainer
        trainer = Trainer()
        trainer.full_training(num_iterations=100)
        print("\n✅ Full training complete!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


def check_model_status():
    """Check if trained model exists and show info."""
    print("\n📊 Model Status:")
    print("-" * 40)
    
    from config import MODEL_PATH
    
    if not os.path.exists(MODEL_PATH):
        print("❌ No models directory found")
        print("Train the AI first using option 2 or 3")
        return
    
    # Check for latest model
    latest_path = os.path.join(MODEL_PATH, 'latest.pt')
    if os.path.exists(latest_path):
        size_mb = os.path.getsize(latest_path) / (1024 * 1024)
        print(f"✅ Latest model found: {size_mb:.2f} MB")
    else:
        print("❌ No trained model found")
    
    # List all checkpoints
    checkpoints = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pt')]
    if checkpoints:
        print(f"\n📁 Checkpoints found: {len(checkpoints)}")
        for ckpt in sorted(checkpoints):
            print(f"   - {ckpt}")
    
    print()


def pretrain_from_csv():
    """Pre-train from games.csv using supervised learning."""
    print("\n📂 Supervised Pre-training from games.csv")
    print("-" * 45)
    print("This loads real Lichess games, replays every position,")
    print("and trains the network before any self-play begins.")
    print("\nDefaults (edit config.py to change):")

    from config import (
        GAMES_CSV_PATH, MIN_ELO_FILTER, MAX_GAMES_TO_LOAD,
        CSV_TRAIN_EPOCHS, CSV_BATCH_SIZE,
    )
    print(f"  CSV file : {GAMES_CSV_PATH}")
    print(f"  Min ELO  : {MIN_ELO_FILTER}")
    print(f"  Max games: {MAX_GAMES_TO_LOAD:,}")
    print(f"  Epochs   : {CSV_TRAIN_EPOCHS}")
    print(f"  Batch    : {CSV_BATCH_SIZE}\n")

    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    try:
        from trainer import Trainer
        trainer = Trainer()
        stats = trainer.pretrain_from_csv()
        print("\n✅ Supervised pre-training complete!")
        loading = stats.get('loading_stats', {})
        print(f"   Games loaded   : {loading.get('games_loaded', 'N/A'):,}")
        print(f"   Positions added: {loading.get('positions_added', 'N/A'):,}")
        print("You can now continue with self-play training (option 3).")
    except Exception as e:
        print(f"Error during pre-training: {e}")
        import traceback
        traceback.print_exc()


def show_dataset_stats():
    """Show statistics about the games.csv dataset."""
    print("\n📈 Dataset Statistics (games.csv)")
    print("-" * 40)
    try:
        from dataset_loader import get_dataset_stats
        get_dataset_stats()
    except Exception as e:
        print(f"Error reading dataset: {e}")
        import traceback
        traceback.print_exc()


def test_network():
    """Test neural network with a sample position."""
    print("\n🧪 Testing Neural Network...")
    
    try:
        import chess
        from neural_network import ChessModel
        
        print("Creating model...")
        model = ChessModel()
        
        print("Testing with starting position...")
        board = chess.Board()
        
        move_probs, value = model.predict(board)
        
        print(f"\n✅ Network working!")
        print(f"   Position value: {value:.3f}")
        print(f"   Number of legal moves: {len(move_probs)}")
        
        # Show top 3 moves
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        print("\n   Top 3 predicted moves:")
        for i, (move, prob) in enumerate(sorted_moves[:3], 1):
            print(f"   {i}. {move.uci()}: {prob*100:.2f}%")
        
        print()
        
    except Exception as e:
        print(f"❌ Error testing network: {e}")
        import traceback
        traceback.print_exc()


def benchmark_alpha_beta():
    """Benchmark the Alpha-Beta engine on the starting position."""
    print("\n⚡ Alpha-Beta Engine Benchmark")
    print("-" * 40)
    print("Runs a depth-5 search on the starting position.")
    print("No trained model needed — uses piece-square heuristics.\n")
    try:
        from alpha_beta import benchmark
        benchmark(depth=5)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def ab_vs_mcts_duel():
    """Play one game: Alpha-Beta (White) vs MCTS (Black)."""
    print("\n🥊 Alpha-Beta vs MCTS Duel")
    print("-" * 40)
    print("Alpha-Beta plays White (depth=4, 2s/move)")
    print("MCTS plays Black (100 simulations/move)")
    print("WARNING: may take a few minutes.\n")

    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    try:
        from alpha_beta import play_ab_vs_mcts
        result = play_ab_vs_mcts(ab_depth=4, ab_time=2.0, mcts_sims=100, verbose=True)
        if result > 0:
            print("\n⚔️  Alpha-Beta wins!")
        elif result < 0:
            print("\n🧠 MCTS wins!")
        else:
            print("\n🤝 Draw!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():

    """Main entry point."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install dependencies first.")
        return
    
    # Main loop
    while True:
        choice = show_menu()

        if choice == '1':
            play_game()
        elif choice == '2':
            train_quick()
        elif choice == '3':
            train_full()
        elif choice == '4':
            check_model_status()
        elif choice == '5':
            test_network()
        elif choice == '6':
            pretrain_from_csv()
        elif choice == '7':
            show_dataset_stats()
        elif choice == '8':
            benchmark_alpha_beta()
        elif choice == '9':
            ab_vs_mcts_duel()
        elif choice == '0':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Please enter 0-9.")

        input("\nPress Enter to continue...")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

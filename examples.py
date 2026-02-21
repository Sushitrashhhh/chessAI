"""
Example usage of the Chess AI system.
Demonstrates how to use different components.
"""

import chess
from neural_network import ChessModel, board_to_tensor
from mcts import MCTS, AlphaZeroPlayer, play_game_with_mcts
from engine_new import ChessEngine
from trainer import Trainer


def example_1_neural_network():
    """Example: Using the neural network directly."""
    print("\n" + "="*60)
    print("Example 1: Neural Network Predictions")
    print("="*60)
    
    # Create model
    model = ChessModel()
    
    # Create a chess position
    board = chess.Board()
    board.push_san("e4")  # 1. e4
    board.push_san("e5")  # 1... e5
    
    print(f"\nPosition after 1. e4 e5:")
    print(board)
    
    # Get predictions
    move_probs, value = model.predict(board)
    
    print(f"\nPosition evaluation: {value:.3f}")
    print("(+1 = winning for white, -1 = winning for black, 0 = equal)")
    
    print("\nTop 5 predicted moves:")
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (move, prob) in enumerate(sorted_moves[:5], 1):
        print(f"  {i}. {move.uci()}: {prob*100:.1f}%")


def example_2_mcts():
    """Example: Using MCTS for move selection."""
    print("\n" + "="*60)
    print("Example 2: MCTS Search")
    print("="*60)
    
    model = ChessModel()
    mcts = MCTS(model, num_simulations=100)
    
    board = chess.Board()
    
    print(f"\nStarting position:")
    print(board)
    
    print("\nRunning MCTS with 100 simulations...")
    move, policy = mcts.search(board, temperature=1.0)
    
    print(f"\nSelected move: {move.uci()}")
    print("\nMove probabilities from MCTS:")
    sorted_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)
    for i, (m, prob) in enumerate(sorted_moves[:5], 1):
        print(f"  {i}. {m.uci()}: {prob*100:.1f}%")


def example_3_engine():
    """Example: Using the Chess Engine."""
    print("\n" + "="*60)
    print("Example 3: Chess Engine")
    print("="*60)
    
    # Create engine (will load trained model if available)
    engine = ChessEngine(num_simulations=50)
    
    board = chess.Board()
    
    print("\nPlaying a short game against itself:")
    print(board)
    print()
    
    for move_num in range(1, 6):  # Play 5 moves
        # White's move
        print(f"\n{move_num}. ", end="")
        move = engine.get_best_move(board)
        print(f"{move.uci()} ", end="")
        board.push(move)
        
        if board.is_game_over():
            break
        
        # Black's move
        move = engine.get_best_move(board)
        print(f"{move.uci()}")
        board.push(move)
        
        if board.is_game_over():
            break
    
    print("\nFinal position:")
    print(board)


def example_4_self_play():
    """Example: Self-play game."""
    print("\n" + "="*60)
    print("Example 4: Self-Play Game")
    print("="*60)
    
    model = ChessModel()
    
    print("\nPlaying a self-play game...")
    print("(This uses MCTS to select moves)")
    
    game_data, result = play_game_with_mcts(model, num_simulations=50)
    
    print(f"\nGame finished!")
    print(f"Result: {result:.1f} (1.0=white wins, -1.0=black wins, 0.0=draw)")
    print(f"Number of moves: {len(game_data)}")
    print(f"Training positions generated: {len(game_data)}")


def example_5_training():
    """Example: Training loop (short version)."""
    print("\n" + "="*60)
    print("Example 5: Training (Demo)")
    print("="*60)
    
    print("\nThis demonstrates the training process.")
    print("For real training, use trainer.py or start.py")
    
    trainer = Trainer()
    
    # Generate some self-play games
    print("\nGenerating 3 self-play games...")
    stats = trainer.self_play(num_games=3)
    
    print(f"\nResults:")
    print(f"  White wins: {stats['wins_white']}")
    print(f"  Black wins: {stats['wins_black']}")
    print(f"  Draws: {stats['draws']}")
    print(f"  Avg moves: {stats['avg_moves']:.1f}")
    print(f"  Buffer size: {stats['buffer_size']}")
    
    # Train on the data
    if stats['buffer_size'] >= 32:
        print("\nTraining neural network...")
        train_stats = trainer.train(num_epochs=2, batch_size=32)
        print(f"\nTraining loss: {train_stats['total_loss']:.4f}")


def example_6_board_representation():
    """Example: Board tensor representation."""
    print("\n" + "="*60)
    print("Example 6: Board Representation")
    print("="*60)
    
    board = chess.Board()
    board.push_san("e4")
    
    print("\nPosition after 1. e4:")
    print(board)
    
    # Convert to tensor
    tensor = board_to_tensor(board)
    
    print(f"\nBoard tensor shape: {tensor.shape}")
    print(f"Number of planes: {tensor.shape[0]}")
    print("  - Planes 0-5: White pieces (P, N, B, R, Q, K)")
    print("  - Planes 6-11: Black pieces")
    print("  - Plane 12: Castling rights")
    print("  - Plane 13: En passant square")
    
    # Show white pawn positions (plane 0)
    print("\nWhite pawns (Plane 0):")
    print(tensor[0])


def example_7_alpha_beta():
    """Example: Alpha-Beta engine."""
    print("\n" + "="*60)
    print("Example 7: Alpha-Beta Engine")
    print("="*60)

    from alpha_beta import AlphaBetaPlayer

    player = AlphaBetaPlayer(max_depth=4, time_limit=None, verbose=True)

    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")

    print(f"\nPosition after 1. e4 e5 2. Nf3:")
    print(board)

    print("\nRunning Alpha-Beta (depth=4) ...")
    move = player.select_move(board)
    print(f"\nBest move: {move.uci() if move else 'None'}")


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# Chess AI Examples")
    print("# Demonstrating Neural Network, MCTS, and Training")
    print("#"*60)
    
    examples = [
        ("Neural Network",     example_1_neural_network),
        ("MCTS Search",        example_2_mcts),
        ("Chess Engine",       example_3_engine),
        ("Self-Play",          example_4_self_play),
        ("Training Demo",      example_5_training),
        ("Board Representation", example_6_board_representation),
        ("Alpha-Beta Engine",  example_7_alpha_beta),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. Run all examples")

    choice = input("\nWhich example to run? (0-7): ").strip()
    
    try:
        if choice == '0':
            for name, func in examples:
                func()
                input("\nPress Enter to continue to next example...")
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                examples[idx][1]()
            else:
                print("Invalid choice!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "#"*60)
    print("# Examples complete!")
    print("#"*60)


if __name__ == '__main__':
    main()

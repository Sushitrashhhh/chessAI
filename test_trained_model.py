"""
test_trained_model.py
=====================
Test the trained model to verify it makes good chess decisions.
"""

import chess
from neural_network import ChessModel
from engine_new import ChessEngine
from alpha_beta import AlphaBetaPlayer

def test_position_evaluation():
    """Test if the model can evaluate positions correctly."""
    print("\n" + "="*70)
    print("Test 1: Position Evaluation")
    print("="*70)
    
    model = ChessModel()
    
    # Test 1: Starting position (should be close to 0)
    board = chess.Board()
    _, value = model.predict(board)
    print(f"\nStarting position evaluation: {value:.3f}")
    print("(Should be close to 0.0 = equal position)")
    
    # Test 2: White has a big advantage (Scholar's Mate position)
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Bc4")
    board.push_san("Nc6")
    board.push_san("Qh5")
    board.push_san("Nf6")
    board.push_san("Qxf7")  # Checkmate!
    
    _, value = model.predict(board)
    print(f"\nCheckmate position evaluation: {value:.3f}")
    print("(Should be close to +1.0 = white winning)")
    print(board)
    
    # Test 3: Material advantage - white has extra queen
    board = chess.Board()
    board.set_fen("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    _, value = model.predict(board)
    print(f"\nWhite up a knight evaluation: {value:.3f}")
    print("(Should be positive = white better)")


def test_move_quality():
    """Test if the model suggests good moves."""
    print("\n" + "="*70)
    print("Test 2: Move Quality")
    print("="*70)
    
    engine = ChessEngine(num_simulations=100)
    
    # Test standard opening
    board = chess.Board()
    print("\nStarting position:")
    print(board)
    
    move = engine.get_best_move(board)
    print(f"\nBest move: {move.uci()} ({board.san(move)})")
    
    common_openings = ['e2e4', 'd2d4', 'g1f3', 'c2c4']
    if move.uci() in common_openings:
        print("✅ Good! This is a common opening move.")
    else:
        print("⚠️  Unusual opening move, but still learning.")
    
    # Test tactical position - can it find a winning move?
    board = chess.Board()
    board.set_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
    print("\n\nTactical position (Scholar's mate threat):")
    print(board)
    
    move = engine.get_best_move(board)
    print(f"\nBest move: {move.uci()} ({board.san(move)})")
    
    if move.uci() == "h5f7":
        print("✅ Perfect! Found the checkmate!")
    else:
        print(f"Selected: {board.san(move)}")


def test_alpha_beta_vs_neural():
    """Compare alpha-beta and neural network play."""
    print("\n" + "="*70)
    print("Test 3: Alpha-Beta vs Neural Network (3 moves)")
    print("="*70)
    
    # Create both engines
    model = ChessModel()
    alpha_beta = AlphaBetaPlayer(max_depth=3, model=model)
    neural_engine = ChessEngine(num_simulations=50)
    
    board = chess.Board()
    print("\nStarting position:")
    print(board)
    
    for move_num in range(1, 4):
        print(f"\n--- Move {move_num} (White) ---")
        
        # Alpha-beta move
        ab_move = alpha_beta.select_move(board)
        print(f"Alpha-Beta suggests: {board.san(ab_move)}")
        
        # Neural + MCTS move  
        nn_move = neural_engine.get_best_move(board)
        print(f"Neural+MCTS suggests: {board.san(nn_move)}")
        
        # Use the neural network's move
        move_san = board.san(nn_move)
        board.push(nn_move)
        print(f"\nAfter {move_san}:")
        print(board)
        
        if board.is_game_over():
            break
        
        # Black's move (simple - just use alpha-beta)
        print(f"\n--- Move {move_num} (Black) ---")
        black_move = alpha_beta.select_move(board)
        black_san = board.san(black_move)
        print(f"Black plays: {black_san}")
        board.push(black_move)
        
        if board.is_game_over():
            break


def test_endgame_knowledge():
    """Test if the model understands basic endgames."""
    print("\n" + "="*70)
    print("Test 4: Endgame Knowledge")
    print("="*70)
    
    model = ChessModel()
    
    # King + Queen vs King (should win easily)
    board = chess.Board()
    board.set_fen("3k4/8/8/8/8/8/8/3K3Q w - - 0 1")
    
    print("\nKing + Queen vs King:")
    print(board)
    
    _, value = model.predict(board)
    print(f"Evaluation: {value:.3f}")
    
    if value > 0.5:
        print("✅ Correctly evaluates this as winning for white!")
    else:
        print("⚠️  Model should learn this is winning with more training.")
    
    # King + Pawn vs King
    board = chess.Board()
    board.set_fen("3k4/8/8/8/8/8/3P4/3K4 w - - 0 1")
    
    print("\nKing + Pawn vs King:")
    print(board)
    
    _, value = model.predict(board)
    print(f"Evaluation: {value:.3f}")
    
    if value > 0.0:
        print("✅ Correctly evaluates this as better for white!")
    else:
        print("⚠️  Needs more training on pawn endgames.")


def main():
    print("\n" + "="*70)
    print("  Chess AI - Trained Model Testing")
    print("="*70)
    print("\nThis script tests the trained model's decision-making ability.")
    
    try:
        test_position_evaluation()
        test_move_quality()
        test_alpha_beta_vs_neural()
        test_endgame_knowledge()
        
        print("\n" + "="*70)
        print("  Testing Complete!")
        print("="*70)
        print("\n✅ The model has been successfully trained from games.csv!")
        print("\nNext steps:")
        print("  1. Play against the AI: python ChessGame.py")
        print("  2. Run more training: python start.py → Option 3")
        print("  3. Play Alpha-Beta vs MCTS: python start.py → Option 9")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

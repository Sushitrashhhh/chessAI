"""
quick_play_demo.py
==================
Quick demo showing Alpha-Beta vs MCTS with the trained model.
"""

import chess
from alpha_beta import AlphaBetaPlayer
from engine_new import ChessEngine
from neural_network import ChessModel

def play_alpha_beta_vs_mcts(max_moves=10):
    """
    Play a short game between Alpha-Beta and MCTS engines.
    
    Alpha-Beta: Fast tactical search with trained neural network evaluation
    MCTS: Monte Carlo Tree Search with neural network guidance
    """
    print("\n" + "="*70)
    print("  Alpha-Beta (White) vs Neural+MCTS (Black)")
    print("="*70)
    
    # Create engines
    model = ChessModel()
    alpha_beta = AlphaBetaPlayer(max_depth=4, model=model, time_limit=2.0)
    mcts_engine = ChessEngine(num_simulations=100)
    
    board = chess.Board()
    
    print("\nStarting position:")
    print(board)
    print()
    
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        # White (Alpha-Beta)
        print(f"Move {board.fullmove_number} (White - Alpha-Beta)")
        white_move = alpha_beta.select_move(board)
        print(f"  {board.san(white_move)}")
        board.push(white_move)
        
        if board.is_game_over():
            break
        
        # Black (MCTS)
        print(f"Move {board.fullmove_number-1} (Black - MCTS)")
        black_move = mcts_engine.get_best_move(board)
        print(f"  {board.san(black_move)}")
        board.push(black_move)
        
        move_count += 1
    
    print("\n" + "-"*70)
    print("Final position:")
    print(board)
    print()
    
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        print(f"Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw by insufficient material")
    else:
        print(f"Game stopped after {move_count} moves")
    
    print("="*70)


def show_best_moves():
    """Show best moves for various positions."""
    print("\n" + "="*70)
    print("  Best Move Analysis")
    print("="*70)
    
    model = ChessModel()
    alpha_beta = AlphaBetaPlayer(max_depth=4, model=model)
    
    positions = [
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
        ("Tactical Puzzle", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1"),
    ]
    
    for name, fen in positions:
        board = chess.Board(fen)
        print(f"\n{name}:")
        print(board)
        
        move = alpha_beta.select_move(board)
        print(f"\n✓ Best move: {board.san(move)}")
        print()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  Chess AI - Quick Play Demo")
    print("  Trained model with Alpha-Beta + MCTS")
    print("="*70)
    
    play_alpha_beta_vs_mcts(max_moves=10)
    show_best_moves()
    
    print("\n" + "="*70)
    print("  Demo Complete!")
    print("="*70)
    print("\nThe trained model is working with both search methods:")
    print("  • Alpha-Beta: Fast tactical search (good for quick decisions)")
    print("  • MCTS: Monte Carlo Tree Search (good for strategic planning)")
    print("\nBoth use the trained neural network for evaluation!")
    print("="*70 + "\n")

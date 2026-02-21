"""
play_now.py
===========
Quickest way to play against your trained Chess AI!
"""

import chess
from engine_new import ChessEngine

def play_console_game():
    """
    Play a console-based game against the trained AI.
    Simple text interface - no GUI needed.
    """
    print("\n" + "="*70)
    print("  Chess AI - Quick Console Game")
    print("  You are WHITE. AI is BLACK.")
    print("="*70)
    print("\nEnter moves in UCI format (e.g., 'e2e4') or SAN (e.g., 'e4')")
    print("Type 'quit' to exit\n")
    
    # Create the AI engine
    engine = ChessEngine(num_simulations=200)  # Fast but strong
    board = chess.Board()
    
    while not board.is_game_over():
        # Show current position
        print("\n" + "─"*70)
        print(f"Move {board.fullmove_number}")
        print("─"*70)
        print(board)
        print()
        
        if board.turn == chess.WHITE:
            # Human's turn
            while True:
                try:
                    move_input = input("Your move: ").strip()
                    
                    if move_input.lower() == 'quit':
                        print("\nThanks for playing!")
                        return
                    
                    # Try to parse as UCI or SAN
                    try:
                        move = chess.Move.from_uci(move_input)
                    except:
                        move = board.parse_san(move_input)
                    
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("❌ Illegal move! Try again.")
                        
                except Exception as e:
                    print(f"❌ Invalid input: {e}")
                    print("   Examples: 'e2e4' or 'e4' or 'Nf3'")
        else:
            # AI's turn
            print("🤖 AI is thinking...")
            move = engine.get_best_move(board)
            print(f"AI plays: {board.san(move)} ({move.uci()})")
            board.push(move)
    
    # Game over
    print("\n" + "="*70)
    print("  GAME OVER")
    print("="*70)
    print(board)
    print()
    
    if board.is_checkmate():
        winner = "BLACK (AI)" if board.turn == chess.WHITE else "WHITE (You)"
        print(f"Checkmate! {winner} wins! 🎉")
    elif board.is_stalemate():
        print("Stalemate - It's a draw!")
    elif board.is_insufficient_material():
        print("Draw - Insufficient material")
    elif board.is_fifty_moves():
        print("Draw - Fifty move rule")
    else:
        print("Draw - Threefold repetition")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    play_console_game()

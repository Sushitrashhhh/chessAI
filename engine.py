import random

def get_all_possible_moves(board, is_white_turn):
    moves = []
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece == '.':
                continue

            # Example: white’s turn
            if is_white_turn and piece.isupper():
                # TODO: add real logic later
                moves.append(((row, col), (row - 1, col)))
            elif not is_white_turn and piece.islower():
                moves.append(((row, col), (row + 1, col)))
    return moves


def get_best_move(board, is_white_turn):
    """Placeholder AI — picks a random valid move for now."""
    possible_moves = get_all_possible_moves(board, is_white_turn)
    if not possible_moves:
        return None
    return random.choice(possible_moves)

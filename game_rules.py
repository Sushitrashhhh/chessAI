# game_rules.py

def move_piece(board, row1, col1, row2, col2):
    """Move a piece on the board."""
    piece = board[row1][col1]
    board[row1][col1] = '.'
    board[row2][col2] = piece


def is_valid_move(board, row1, col1, row2, col2):
    """Check if a move is valid (currently supports pawns only)."""
    piece = board[row1][col1]
    target = board[row2][col2]

    if piece == '.':
        return False

    # White pawn
    if piece == 'P':
        # Move forward one square
        if col1 == col2 and row2 == row1 - 1 and target == '.':
            return True
        # Capture diagonally
        if abs(col2 - col1) == 1 and row2 == row1 - 1 and target.islower():
            return True

    # Black pawn
    if piece == 'p':
        # Move forward one square
        if col1 == col2 and row2 == row1 + 1 and target == '.':
            return True
        # Capture diagonally
        if abs(col2 - col1) == 1 and row2 == row1 + 1 and target.isupper():
            return True

    # TODO: Add rules for R, N, B, Q, K
    return False


def get_square_clicked(pos, SQUARE_SIZE):
    """Convert mouse click (x, y) to board coordinates (row, col)."""
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col

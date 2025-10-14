import pygame
from game_rules import move_piece, is_valid_move, get_square_clicked

# Initialize pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 640, 640
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('CHESS GAME')

ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Board representation
board = [
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
]


def draw_board():
    """Draw the chess board and pieces."""
    for row in range(ROWS):
        for col in range(COLS):
            color = BLACK if (row + col) % 2 == 0 else WHITE
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            piece = board[row][col]
            if piece != '.':
                # Placeholder for pieces (colored circles)
                pygame.draw.circle(
                    screen,
                    (200, 0, 0) if piece.isupper() else (0, 0, 200),
                    (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                    25
                )


def main():
    selected_square = None
    running = True

    while running:
        draw_board()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                row, col = get_square_clicked(event.pos, SQUARE_SIZE)

                if selected_square is None:
                    selected_square = (row, col)
                else:
                    row1, col1 = selected_square
                    if is_valid_move(board, row1, col1, row, col):
                        move_piece(board, row1, col1, row, col)
                    selected_square = None

    pygame.quit()


if __name__ == "__main__":
    main()

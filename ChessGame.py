import pygame
import chess
import chess.svg
import os
import threading

from engine_new import ChessEngine


class ChessGame:
    """
    Chess game with pygame GUI that can play against the ML-based AI.
    """
    
    def __init__(self):
        pygame.init()
        
        # Screen setup
        self.WIDTH, self.HEIGHT = 640, 720
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Chess AI - Deep RL')
        
        self.BOARD_SIZE = 640
        self.SQUARE_SIZE = self.BOARD_SIZE // 8
        self.INFO_HEIGHT = 80
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.LIGHT_SQUARE = (240, 217, 181)
        self.DARK_SQUARE = (181, 136, 99)
        self.HIGHLIGHT = (186, 202, 43)
        self.SELECTED = (246, 246, 105)
        
        # Chess board
        self.board = chess.Board()
        
        # AI Engine
        self.engine = None
        self.ai_thinking = False
        
        # Game state
        self.selected_square = None
        self.legal_moves = []
        self.play_mode = None  # 'pvp' or 'ai'
        self.player_color = chess.WHITE  # Player plays as white by default
        self._pending_ai_move = None   # Set by AI thread when move is ready
        self._ai_thread: threading.Thread | None = None
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
    def draw_board(self):
        """Draw the chess board."""
        for row in range(8):
            for col in range(8):
                color = self.LIGHT_SQUARE if (row + col) % 2 == 0 else self.DARK_SQUARE
                rect = pygame.Rect(col * self.SQUARE_SIZE, row * self.SQUARE_SIZE, 
                                  self.SQUARE_SIZE, self.SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                
                # Highlight selected square
                square = chess.square(col, 7 - row)
                if self.selected_square == square:
                    pygame.draw.rect(self.screen, self.SELECTED, rect)
                
                # Highlight legal move squares
                if self.selected_square is not None:
                    for move in self.legal_moves:
                        if move.to_square == square:
                            pygame.draw.circle(self.screen, self.HIGHLIGHT,
                                             rect.center, self.SQUARE_SIZE // 6)
    
    def draw_pieces(self):
        """Draw pieces using Unicode chess symbols."""
        piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        piece_font = pygame.font.Font(None, 60)
        
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                piece = self.board.piece_at(square)
                
                if piece:
                    symbol = piece_symbols.get(piece.symbol(), piece.symbol())
                    color = self.WHITE if piece.color == chess.WHITE else self.BLACK
                    text = piece_font.render(symbol, True, color)
                    text_rect = text.get_rect(center=(
                        col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
                        row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
                    ))
                    self.screen.blit(text, text_rect)
    
    def draw_info(self):
        """Draw game information panel."""
        info_rect = pygame.Rect(0, self.BOARD_SIZE, self.WIDTH, self.INFO_HEIGHT)
        pygame.draw.rect(self.screen, (50, 50, 50), info_rect)
        
        # Game status
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            text = f"Checkmate! {winner} wins!"
        elif self.board.is_stalemate():
            text = "Stalemate!"
        elif self.board.is_check():
            text = "Check!"
        elif self.ai_thinking:
            text = "AI is thinking..."
        else:
            turn = "White" if self.board.turn == chess.WHITE else "Black"
            text = f"{turn} to move"
        
        text_surface = self.font.render(text, True, self.WHITE)
        text_rect = text_surface.get_rect(center=(self.WIDTH // 2, self.BOARD_SIZE + 30))
        self.screen.blit(text_surface, text_rect)
        
        # Move count
        move_text = f"Move: {self.board.fullmove_number}"
        move_surface = self.small_font.render(move_text, True, self.WHITE)
        self.screen.blit(move_surface, (10, self.BOARD_SIZE + 50))
    
    def get_square_from_mouse(self, pos):
        """Convert mouse position to chess square."""
        x, y = pos
        if y > self.BOARD_SIZE:
            return None
        col = x // self.SQUARE_SIZE
        row = y // self.SQUARE_SIZE
        return chess.square(col, 7 - row)
    
    def handle_click(self, pos):
        """Handle mouse click on board."""
        if self.board.is_game_over():
            return
        
        # Don't allow moves during AI turn
        if self.play_mode == 'ai' and self.board.turn != self.player_color:
            return
        
        square = self.get_square_from_mouse(pos)
        if square is None:
            return
        
        # If no square selected, select piece
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.legal_moves = [move for move in self.board.legal_moves 
                                   if move.from_square == square]
        else:
            # Try to make a move
            move = None
            promotion_piece = chess.QUEEN   # Auto-promote to queen
            for legal_move in self.legal_moves:
                if legal_move.to_square == square:
                    # Prefer the queen promotion if multiple promotions exist
                    if legal_move.promotion in (None, chess.QUEEN):
                        move = legal_move
                        break
                    # Fallback: accept any promotion
                    move = legal_move
            
            if move:
                self.board.push(move)
                print(f"Move: {move.uci()}")
            elif square == self.selected_square:
                # Clicked same square → deselect
                pass
            # Always deselect after attempting a move
            self.selected_square = None
            self.legal_moves = []
    
    def ai_move(self):
        """
        Trigger the AI move in a background thread so the UI stays responsive.
        When the thread finishes it sets self._pending_ai_move; the main loop
        applies it on the next frame.
        """
        if self.engine is None or self.board.is_game_over():
            return

        if self.board.turn == self.player_color:
            return  # Human's turn

        # Apply completed AI move
        if self._pending_ai_move is not None:
            move = self._pending_ai_move
            self._pending_ai_move = None
            if move and self.board.is_legal(move):
                self.board.push(move)
                print(f"AI Move: {move.uci()}")
            self.ai_thinking = False
            return

        # Start AI thread if not already running
        if self._ai_thread is None or not self._ai_thread.is_alive():
            self.ai_thinking = True

            def _run():
                move = self.engine.get_best_move(self.board)
                self._pending_ai_move = move

            self._ai_thread = threading.Thread(target=_run, daemon=True)
            self._ai_thread.start()
    
    def draw(self):
        """Draw entire game state."""
        self.draw_board()
        self.draw_pieces()
        self.draw_info()
    
    def show_menu(self):
        """Show game mode selection menu."""
        self.screen.fill(self.BLACK)
        
        title = self.font.render("Chess AI - Deep Reinforcement Learning", True, self.WHITE)
        title_rect = title.get_rect(center=(self.WIDTH // 2, 100))
        self.screen.blit(title, title_rect)
        
        # Buttons
        button_width, button_height = 300, 60
        button_x = (self.WIDTH - button_width) // 2
        
        pvp_button = pygame.Rect(button_x, 250, button_width, button_height)
        ai_button = pygame.Rect(button_x, 350, button_width, button_height)
        train_button = pygame.Rect(button_x, 450, button_width, button_height)
        
        pygame.draw.rect(self.screen, (0, 100, 0), pvp_button)
        pygame.draw.rect(self.screen, (0, 0, 150), ai_button)
        pygame.draw.rect(self.screen, (150, 0, 0), train_button)
        
        pvp_text = self.small_font.render("Player vs Player", True, self.WHITE)
        ai_text = self.small_font.render("Play vs AI", True, self.WHITE)
        train_text = self.small_font.render("Train AI", True, self.WHITE)
        
        self.screen.blit(pvp_text, pvp_text.get_rect(center=pvp_button.center))
        self.screen.blit(ai_text, ai_text.get_rect(center=ai_button.center))
        self.screen.blit(train_text, train_text.get_rect(center=train_button.center))
        
        pygame.display.flip()
        
        # Wait for selection
        while self.play_mode is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    if pvp_button.collidepoint(pos):
                        self.play_mode = 'pvp'
                    elif ai_button.collidepoint(pos):
                        self.play_mode = 'ai'
                        print("Initializing AI engine...")
                        self.engine = ChessEngine(num_simulations=200)
                    elif train_button.collidepoint(pos):
                        print("Starting training mode...")
                        self.start_training()
                        return False
        
        return True
    
    def start_training(self):
        """Start training the AI."""
        from trainer import quick_train
        
        # Close pygame window
        pygame.quit()
        
        # Run training
        print("\nStarting AI training...")
        print("This will take some time. The AI learns by playing against itself.")
        quick_train(num_iterations=5, num_games_per_iter=5)
        
        print("\nTraining complete! Run the game again to play against the trained AI.")
    
    def play(self):
        """Main game loop."""
        # Show menu
        if not self.show_menu():
            return
        
        running = True
        clock = pygame.time.Clock()
        
        while running:
            clock.tick(30)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Reset game
                        self.board.reset()
                        self.selected_square = None
                        self.legal_moves = []
            
            # AI move
            if self.play_mode == 'ai':
                self.ai_move()
            
            # Draw
            self.draw()
            pygame.display.flip()
        
        pygame.quit()


if __name__ == "__main__":
    game = ChessGame()
    game.play()

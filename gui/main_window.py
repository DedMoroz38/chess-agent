from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QStatusBar, QLabel, QMessageBox, QPushButton)
from PyQt6.QtCore import Qt
from gui.board_widget import BoardWidget
from gui.move_history_widget import MoveHistoryWidget
from gui.game_controller import GameController
from samples import sample0


class ChessMainWindow(QMainWindow):
    """Main window for Chess Fragment GUI"""
    
    def __init__(self):
        super().__init__()
        self.controller = GameController()
        self.board_widget = None
        self.move_history = None
        
        self.setup_ui()
        self.connect_signals()
        
        # Start new game automatically
        self.controller.start_new_game(sample0)
        
    def setup_ui(self):
        """Create the user interface"""
        self.setWindowTitle("Chess Fragment - 5x5 Chess")
        self.setStyleSheet("QMainWindow { background-color: #312E2B; }")
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout (horizontal)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left side: Board
        board_container = QVBoxLayout()
        
        # Title
        title = QLabel("Chess Fragment")
        title.setStyleSheet("color: white; font-size: 20px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        board_container.addWidget(title)
        
        # Board widget
        self.board_widget = BoardWidget()
        board_container.addWidget(self.board_widget)
        
        # Button style
        button_style = """
            QPushButton {
                background-color: #4A4A4A;
                color: white;
                border: 1px solid #666666;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #5A5A5A;
            }
            QPushButton:pressed {
                background-color: #3A3A3A;
            }
        """
        
        # New game button
        new_game_btn = QPushButton("New Game")
        new_game_btn.setStyleSheet(button_style)
        new_game_btn.clicked.connect(self.new_game)
        board_container.addWidget(new_game_btn)
        
        main_layout.addLayout(board_container)
        
        # Right side: Move history
        right_container = QVBoxLayout()
        
        history_label = QLabel("Move History")
        history_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; padding: 10px;")
        history_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_container.addWidget(history_label)
        
        self.move_history = MoveHistoryWidget()
        self.move_history.setMinimumWidth(300)
        right_container.addWidget(self.move_history)
        
        main_layout.addLayout(right_container)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #2A2A2A;
                color: white;
                font-size: 14px;
                padding: 5px;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Set window size
        self.resize(1000, 700)
        
    def connect_signals(self):
        """Connect controller signals to UI updates"""
        # Board updates
        self.controller.board_updated.connect(self.on_board_updated)
        
        # Move completion
        self.controller.move_completed.connect(self.on_move_completed)
        
        # Game end
        self.controller.game_ended.connect(self.on_game_ended)
        
        # Status changes
        self.controller.status_changed.connect(self.on_status_changed)
        
        # AI thinking indicator
        self.controller.ai_thinking.connect(self.on_ai_thinking)
        
        # Square clicks
        self.board_widget.square_clicked.connect(self.on_square_clicked)
        
    def on_board_updated(self, board):
        """Update board display"""
        self.board_widget.update_board(board)
        
    def on_move_completed(self, description, piece, from_pos, to_pos):
        """Handle move completion"""
        self.move_history.add_move(description)
        
    def on_game_ended(self, result):
        """Show game end dialog"""
        QMessageBox.information(self, "Game Over", f"Game ended: {result}")
        self.status_bar.showMessage(f"Game Over: {result}")
        
    def on_status_changed(self, status):
        """Update status bar"""
        self.status_bar.showMessage(status)
        
    def on_ai_thinking(self, thinking):
        """Update UI when AI is thinking"""
        if thinking:
            self.status_bar.showMessage("AI is thinking...")
        
    def on_square_clicked(self, row, col):
        """Handle square click from board"""
        # Only respond if it's human's turn
        if not self.controller.is_human_turn():
            return
            
        # Get piece at clicked position
        piece = self.controller.get_piece_at(col, row)
        
        # If no piece selected yet
        if self.board_widget.selected_piece is None:
            if piece and hasattr(piece, 'player'):
                # Check if piece belongs to current player
                current_player = self.controller.get_current_player()
                if piece.player == current_player:
                    # Select this piece
                    self.board_widget.selected_piece = piece
                    self.board_widget.selected_position = (col, row)
                    self.board_widget.select_square(row, col)
                    
                    # Get and highlight legal moves
                    legal_moves = self.controller.get_legal_moves_for_piece(piece)
                    self.board_widget.highlight_legal_moves(legal_moves)
        else:
            # Piece already selected, try to move
            selected_piece = self.board_widget.selected_piece
            
            # Check if clicked square is a legal move
            legal_moves = self.controller.get_legal_moves_for_piece(selected_piece)
            target_move = None
            
            for move in legal_moves:
                if hasattr(move, 'position') and move.position.x == col and move.position.y == row:
                    target_move = move
                    break
            
            if target_move:
                # Execute the move
                self.controller.execute_move(selected_piece, target_move)
            
            # Deselect
            self.board_widget.selected_piece = None
            self.board_widget.selected_position = None
            self.board_widget.clear_highlights()
            
    def new_game(self):
        """Start a new game"""
        # Clear history
        self.move_history.clear_history()
        
        # Clear board selection
        self.board_widget.selected_piece = None
        self.board_widget.selected_position = None
        self.board_widget.clear_highlights()
        
        # Start new game with settings from controller's __init__
        self.controller.start_new_game(sample0)

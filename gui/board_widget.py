from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QColor, QFont


class SquareWidget(QWidget):
    """Individual square on the chess board"""
    clicked = pyqtSignal(int, int)  # row, col
    
    # Unicode chess symbols
    PIECE_SYMBOLS = {
        'King': {'white': '♔', 'black': '♚'},
        'Queen': {'white': '♕', 'black': '♛'},
        'Bishop': {'white': '♗', 'black': '♝'},
        'Knight': {'white': '♘', 'black': '♞'},
        'Pawn': {'white': '♙', 'black': '♟'},
        'Right': {'white': 'R', 'black': 'r'},
    }
    
    def __init__(self, row, col, is_light):
        super().__init__()
        self.row = row
        self.col = col
        self.is_light = is_light
        self.piece = None
        self.is_highlighted = False
        self.is_selected = False
        
        # Colors
        self.light_color = QColor("#F0D9B5")
        self.dark_color = QColor("#B58863")
        self.highlight_color = QColor("#90EE90")
        self.select_color = QColor("#FFFF00")
        
        # Fixed size
        self.setFixedSize(120, 120)
        self.setMouseTracking(True)
        
    def set_piece(self, piece):
        """Update piece on this square"""
        self.piece = piece
        self.update()
        
    def set_highlight(self, state):
        """Set highlight state for legal moves"""
        self.is_highlighted = state
        self.update()
        
    def set_selected(self, state):
        """Set selected state"""
        self.is_selected = state
        self.update()
        
    def get_piece_symbol(self):
        """Get Unicode symbol for current piece"""
        if not self.piece:
            return ""
            
        piece_name = self.piece.name if hasattr(self.piece, 'name') else str(self.piece).split()[0]
        player_name = self.piece.player.name if hasattr(self.piece.player, 'name') else 'white'
        
        # Handle Right piece specially
        if 'Right' in piece_name:
            return 'R' if player_name == 'white' else 'r'
        
        # Get symbol from dict
        for key, symbols in self.PIECE_SYMBOLS.items():
            if key.lower() in piece_name.lower():
                return symbols.get(player_name, '?')
        
        return '?'
    
    def paintEvent(self, event):
        """Custom drawing for square"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        
        # Draw square background
        base_color = self.light_color if self.is_light else self.dark_color
        painter.fillRect(rect, base_color)
        
        # Draw highlight overlay if needed
        if self.is_highlighted:
            painter.fillRect(rect, QColor(self.highlight_color.red(), 
                                         self.highlight_color.green(), 
                                         self.highlight_color.blue(), 128))
        
        # Draw selection border
        if self.is_selected:
            painter.setPen(Qt.GlobalColor.yellow)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(4):
                painter.drawRect(rect.adjusted(i, i, -i, -i))
        
        # Draw piece
        if self.piece:
            symbol = self.get_piece_symbol()
            font = QFont("Arial", 60)
            painter.setFont(font)
            painter.setPen(Qt.GlobalColor.black if hasattr(self.piece.player, 'name') and 
                          self.piece.player.name == 'white' else Qt.GlobalColor.darkGray)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, symbol)
    
    def mousePressEvent(self, event):
        """Handle mouse click"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.row, self.col)


class BoardWidget(QWidget):
    """5x5 Chess board display"""
    square_clicked = pyqtSignal(int, int)  # row, col
    move_requested = pyqtSignal(object, object)  # piece, move_option
    
    def __init__(self):
        super().__init__()
        self.board = None
        self.squares = []
        self.selected_piece = None
        self.selected_position = None
        self.legal_moves = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the board grid"""
        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create 5x5 grid of squares
        for row in range(5):
            row_squares = []
            for col in range(5):
                # Checkerboard pattern
                is_light = (row + col) % 2 == 0
                square = SquareWidget(row, col, is_light)
                square.clicked.connect(self.on_square_clicked)
                layout.addWidget(square, row, col)
                row_squares.append(square)
            self.squares.append(row_squares)
        
        # Add coordinate labels
        # Column labels (top)
        for col in range(5):
            label = QLabel(str(col))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: white; font-weight: bold;")
            layout.addWidget(label, 5, col)
        
        # Row labels (right)
        for row in range(5):
            label = QLabel(str(row))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: white; font-weight: bold;")
            layout.addWidget(label, row, 5)
        
        self.setLayout(layout)
        
    def update_board(self, board):
        """Update display from board state"""
        self.board = board
        if not board:
            return
            
        # Clear all squares first
        for row in range(5):
            for col in range(5):
                self.squares[row][col].set_piece(None)
        
        # Place pieces using board.get_pieces()
        try:
            for piece in board.get_pieces():
                if hasattr(piece, 'position'):
                    x, y = piece.position.x, piece.position.y
                    if 0 <= y < 5 and 0 <= x < 5:
                        self.squares[y][x].set_piece(piece)
        except Exception as e:
            print(f"Error updating board: {e}")
    
    def on_square_clicked(self, row, col):
        """Handle square click"""
        self.square_clicked.emit(row, col)
        
    def clear_highlights(self):
        """Remove all highlighting"""
        for row in range(5):
            for col in range(5):
                self.squares[row][col].set_highlight(False)
                self.squares[row][col].set_selected(False)
    
    def highlight_legal_moves(self, moves):
        """Highlight squares with legal moves"""
        self.clear_highlights()
        self.legal_moves = moves
        
        for move in moves:
            if hasattr(move, 'position'):
                x, y = move.position.x, move.position.y
                if 0 <= y < 5 and 0 <= x < 5:
                    self.squares[y][x].set_highlight(True)
    
    def select_square(self, row, col):
        """Mark a square as selected"""
        self.clear_highlights()
        if 0 <= row < 5 and 0 <= col < 5:
            self.squares[row][col].set_selected(True)


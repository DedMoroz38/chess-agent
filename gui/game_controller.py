from PyQt6.QtCore import QObject, pyqtSignal, QThread
from itertools import cycle
from chessmaker.chess.base import Board
from extension.board_utils import copy_piece_move
from extension.board_rules import get_result
from samples import white, black, sample0
from agents.agent_minimax_ab import agent


class AIWorker(QThread):
    """Worker thread to run AI computation without freezing UI"""
    move_ready = pyqtSignal(object, object)  # piece, move_option
    
    def __init__(self, board, player, ai_function):
        super().__init__()
        self.board = board
        self.player = player
        self.ai_function = ai_function
        
    def run(self):
        """Execute AI move calculation in separate thread"""
        piece, move = self.ai_function(self.board, self.player, None)
        self.move_ready.emit(piece, move)


class GameController(QObject):
    """Bridge between game logic and GUI"""
    board_updated = pyqtSignal(object)  # board
    move_completed = pyqtSignal(str, object, object, object)  # description, piece, from_pos, to_pos
    game_ended = pyqtSignal(str)  # result message
    ai_thinking = pyqtSignal(bool)  # True when AI is thinking
    status_changed = pyqtSignal(str)  # Status message
    
    def __init__(self):
        super().__init__()
        self.board = None
        self.players = None
        self.turn_order = None
        self.move_history = []
        self.white_player = None  # Human
        self.black_player = agent  # AI
        self.ai_thread = None
        self.game_active = False
        self.move_count = 0
        
    def start_new_game(self, board_sample=sample0, white_is_human=True):
        """Initialize a new game"""
        # Create board using same method as test.py
        players = [white, black]
        self.board = Board(
            squares=board_sample,
            players=players,
            turn_iterator=cycle(players),
        )
        self.players = players
        self.turn_order = cycle(players)
        self.white_player = None if white_is_human else agent
        self.move_history = []
        self.game_active = True
        self.move_count = 0
        
        # Emit initial board state
        self.board_updated.emit(self.board)
        self.status_changed.emit("White to move (Human)")
        
    def get_current_player(self):
        """Get the current player from board state"""
        return self.board.current_player if self.board else None
    
    def is_human_turn(self):
        """Check if it's human player's turn"""
        if not self.board or not self.game_active:
            return False
        current = self.get_current_player()
        if not current:
            return False
        # White is human if white_player is None, black is always AI
        return (current.name == "white" and self.white_player is None)
    
    def execute_move(self, piece, move_option):
        """Execute a move on the board"""
        if not self.game_active:
            return False
            
        try:
            # Store position before move
            from_pos = (piece.position.x, piece.position.y)
            
            # Execute the move
            piece.move(move_option)
            to_pos = (move_option.position.x, move_option.position.y)
            
            # Update move count
            self.move_count += 1
            
            # Create move description
            piece_name = piece.name if hasattr(piece, 'name') else str(piece)
            move_desc = f"{self.move_count}. {piece_name} ({from_pos[0]},{from_pos[1]}) → ({to_pos[0]},{to_pos[1]})"
            
            # Check for captures
            if hasattr(move_option, 'captures') and move_option.captures:
                caps = ", ".join(f"({c.x},{c.y})" for c in move_option.captures)
                if caps:
                    move_desc += f" [captures: {caps}]"
            
            self.move_history.append(move_desc)
            self.move_completed.emit(move_desc, piece, from_pos, to_pos)
            
            # Update board display
            self.board_updated.emit(self.board)
            
            # Check for game end
            result = get_result(self.board)
            if result:
                self.game_active = False
                self.game_ended.emit(result)
                return True
            
            # Trigger next turn
            self.next_turn()
            return True
            
        except Exception as e:
            print(f"Error executing move: {e}")
            return False
    
    def next_turn(self):
        """Handle the next turn (potentially AI)"""
        if not self.game_active:
            return
            
        current_player = self.get_current_player()
        if not current_player:
            return
            
        # Update status
        player_type = "Human" if self.is_human_turn() else "AI"
        self.status_changed.emit(f"{current_player.name.capitalize()} to move ({player_type})")
        
        # If it's AI turn, trigger AI move
        if not self.is_human_turn():
            self.make_ai_move()
    
    def make_ai_move(self):
        """Start AI move calculation in separate thread"""
        if not self.game_active or self.ai_thread is not None:
            return
            
        current_player = self.get_current_player()
        if not current_player:
            return
            
        # Clone board for AI
        temp_board = self.board.clone()
        
        # Determine which AI to use
        ai_function = self.black_player if current_player.name == "black" else self.white_player
        if ai_function is None:
            return  # Human player
            
        self.ai_thinking.emit(True)
        self.status_changed.emit(f"{current_player.name.capitalize()} thinking...")
        
        # Start AI worker thread
        self.ai_thread = AIWorker(temp_board, current_player, ai_function)
        self.ai_thread.move_ready.connect(self.on_ai_move_ready)
        self.ai_thread.finished.connect(self.on_ai_thread_finished)
        self.ai_thread.start()
    
    def on_ai_move_ready(self, p_piece, p_move_opt):
        """Handle AI move when ready"""
        if not self.game_active:
            return
            
        # Translate move from cloned board to real board
        board, piece, move_opt = copy_piece_move(self.board, p_piece, p_move_opt)
        
        if not piece or not move_opt:
            # AI couldn't make a move
            result = get_result(self.board)
            if result:
                self.game_ended.emit(result)
            else:
                current = self.get_current_player()
                self.game_ended.emit(f"{current.name} cannot make a legal move")
            self.game_active = False
            self.ai_thinking.emit(False)
            return
        
        # Execute the AI's move
        self.execute_move(piece, move_opt)
        self.ai_thinking.emit(False)
    
    def on_ai_thread_finished(self):
        """Clean up AI thread after completion"""
        if self.ai_thread:
            self.ai_thread.deleteLater()
            self.ai_thread = None
    
    def get_legal_moves_for_piece(self, piece):
        """Get list of legal move options for a piece"""
        if not piece or not self.game_active:
            return []
        try:
            return list(piece.get_move_options())
        except:
            return []
    
    def get_piece_at(self, x, y):
        """Get piece at board position (x, y)"""
        if not self.board:
            return None
        try:
            for piece in self.board.get_pieces():
                if hasattr(piece, 'position') and piece.position.x == x and piece.position.y == y:
                    return piece
            return None
        except:
            return None


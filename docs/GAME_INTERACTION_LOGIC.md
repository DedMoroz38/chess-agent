# Game Interaction Logic

This document explains how the chess game components interact in both the command-line test environment and the GUI environment.

## 1. Test Game Logic (test.py)

### Core Components Interaction

The `testgame` function orchestrates the interaction between the agent, opponent, and game board:

#### Game Setup
```python
def make_custom_board(board_sample):
    players = [white, black]  # From samples.py
    board = Board(
        squares=board_sample,
        players=players,
        turn_iterator=cycle(players),
    )
    return board, players
```

#### Game Loop Structure
The test game follows this interaction pattern:

1. **Turn Management**: Uses `cycle(players)` to alternate between white and black players
2. **Player Function Calls**: Depending on current player color, calls either:
   - `p_white(temp_board, player, var)` - typically the `agent` function  
   - `p_black(temp_board, player, var)` - typically the `opponent` function
3. **Move Translation**: Uses `copy_piece_move()` to translate moves from cloned board to real board
4. **Move Execution**: Executes the move and updates the board state
5. **Game End Detection**: Checks for game termination conditions after each move

### Agent vs Opponent Interaction

#### Agent Function (`agent.py`)
- **Purpose**: Implements the AI player logic
- **Strategy**: Uses `list_legal_moves_for()` to get all legal moves, then randomly selects one
- **Input**: Takes board state, current player, and optional variable
- **Output**: Returns a piece and move option tuple
- **Implementation**: Same logic for both white and black pieces

```python
def agent(board: Board, player, var):
    legal = list_legal_moves_for(board, player)
    if legal:
        piece, move_opt = random.choice(legal)
    return piece, move_opt
```

#### Opponent Function (`opponent.py`)
- **Purpose**: Implements the opposing AI player logic
- **Strategy**: Randomly selects a piece, then randomly selects a move for that piece
- **Different Approach**: Instead of getting all legal moves first, it:
  1. Randomly picks a piece belonging to the current player
  2. Gets move options for that specific piece
  3. Randomly selects from those moves
- **Potential Issue**: May be less efficient as it might pick pieces with no legal moves

### Key Differences Between Agent and Opponent

| Aspect | Agent | Opponent |
|--------|--------|----------|
| Move Selection | Global: Gets all legal moves first | Local: Picks piece first, then moves |
| Efficiency | More efficient (pre-filtered legal moves) | Less efficient (may pick immobile pieces) |
| Implementation | Uses `list_legal_moves_for()` utility | Uses `board.get_player_pieces()` directly |

## 2. GUI Game Logic

### Architecture Overview

The GUI system replaces the simple test loop with a sophisticated event-driven architecture:

```
User Input → BoardWidget → GameController → Game Logic → Board Updates → GUI Refresh
```

### Key Components

#### GameController (`gui/game_controller.py`)
- **Role**: Central coordinator that bridges game logic and GUI
- **Responsibilities**:
  - Manages game state and turn order
  - Handles both human and AI moves
  - Emits signals for UI updates
  - Controls AI thread execution

#### Main Differences from Test Game

| Test Game | GUI Game |
|-----------|----------|
| Synchronous loop | Asynchronous event-driven |
| Both players are AI functions | One human, one AI |
| Direct function calls | Signal/slot communication |
| Console output | Visual board updates |
| Immediate execution | Threaded AI execution |

### Human vs AI Player Interaction

#### Human Player Integration
When it's the human player's turn:

1. **Turn Detection**: `is_human_turn()` checks if current player is human
2. **Input Handling**: Board clicks are captured by `BoardWidget`
3. **Piece Selection**: Click selects piece and highlights legal moves
4. **Move Execution**: Second click attempts to move to target square
5. **Validation**: Move is validated against legal move options

```python
def on_square_clicked(self, row, col):
    if not self.controller.is_human_turn():
        return  # Ignore clicks during AI turn
    
    piece = self.controller.get_piece_at(col, row)
    
    if self.board_widget.selected_piece is None:
        # Select piece if it belongs to current player
        self.board_widget.selected_piece = piece
        legal_moves = self.controller.get_legal_moves_for_piece(piece)
        self.board_widget.highlight_legal_moves(legal_moves)
    else:
        # Try to execute move to clicked square
        self.controller.execute_move(selected_piece, target_move)
```

#### AI Player Integration
When it's the AI player's turn:

1. **Thread Creation**: AI calculation runs in separate `AIWorker` thread
2. **Non-blocking**: UI remains responsive during AI "thinking"
3. **Move Translation**: AI moves on cloned board are translated to real board
4. **Automatic Execution**: AI moves are executed automatically

```python
def make_ai_move(self):
    temp_board = self.board.clone()
    ai_function = self.black_player  # Uses same agent function as test.py
    
    self.ai_thread = AIWorker(temp_board, current_player, ai_function)
    self.ai_thread.move_ready.connect(self.on_ai_move_ready)
    self.ai_thread.start()
```

### How GUI Replaces Opponent with Human

#### Configuration Changes
```python
def start_new_game(self, board_sample=sample0, white_is_human=True):
    self.white_player = None if white_is_human else agent  # None = Human
    self.black_player = agent  # Always AI
```

#### Player Type Detection
```python
def is_human_turn(self):
    current = self.get_current_player()
    # White is human if white_player is None
    return (current.name == "white" and self.white_player is None)
```

#### Turn Handling Logic
```python
def next_turn(self):
    if self.is_human_turn():
        # Wait for user input via board clicks
        self.status_changed.emit(f"{current_player.name} to move (Human)")
    else:
        # Trigger AI move automatically
        self.make_ai_move()
```

## 3. Key Architectural Patterns

### Command Pattern
- Test game: Direct function calls
- GUI: Signal/slot pattern for loose coupling

### State Management
- Test game: Simple loop with local variables
- GUI: Centralized state in GameController with event notifications

### Concurrency
- Test game: Single-threaded, blocking
- GUI: Multi-threaded with AI worker threads

### User Interface
- Test game: ASCII art console output
- GUI: Rich visual interface with piece symbols, highlighting, and move history

## 4. Common Core Logic

Both systems share the same fundamental game logic:

- **Board State**: Same `chessmaker.chess.base.Board` class
- **Move Validation**: Same `piece.get_move_options()` system  
- **Game Rules**: Same `extension.board_rules.get_result()` function
- **AI Logic**: Same `agent` function (GUI just calls it in a thread)
- **Move Translation**: Same `copy_piece_move()` utility

The main difference is in the **control flow and user interaction model**:
- Test game: Programmatic AI vs AI with console output
- GUI: Interactive human vs AI with visual interface and event-driven updates

This architecture allows the same core game logic and AI to work in both environments while providing appropriate interfaces for each use case.

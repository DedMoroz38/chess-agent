# Chess Fragment Agent – Detailed Walkthrough

This document explains how the coursework scaffold works end to end, how the custom extensions interact with the [ChessMaker](https://wolfdwyc.github.io/ChessMaker) engine, and what happens during a simulated game. Use it as a reference when evolving the agent or introducing new tests.

## Core Concepts and External Engine
- **ChessMaker library** supplies the heavy lifting: `Board`, `Player`, `Piece`, `MoveOption`, and `Square` abstractions plus standard chess pieces and rules.
- The coursework narrows the game to a **5×5 board** with a curated piece set (King, Queen, Bishop, Knight, the custom `Right`, and directional Pawns that promote only to Queens).
- All gameplay code (agents, opponents, samples, rules) is pure Python. When `test.py` runs, it orchestrates the ChessMaker objects to simulate a turn-based game until a result condition fires.

## Module-by-Module Overview

### `test.py` – Simulation Driver
- `make_custom_board(board_sample)` wraps a 5×5 matrix of `Square` instances (see `samples.py`) inside a ChessMaker `Board`, assigns the canonical white/black players, and builds a `cycle` iterator so turns alternate forever.
- `testgame(p_white, p_black, board_sample)` is the main loop:
  1. Clone the board before every move (`temp_board = board.clone()`), so the decision logic receives an isolated snapshot.
  2. Call the current side’s callable (`agent` for white, `opponent` for black by default) with `(temp_board, player, var)`.
  3. Use `copy_piece_move` to translate the move that was chosen on the clone back onto the live board instance. This prevents the agent from mutating the authoritative board directly.
  4. If either the piece or the move is missing, ask `get_result` (see `extension/board_rules.py`) whether the game is over; otherwise declare an illegal-move loss.
  5. Execute the move (`piece.move(move_opt)`) and print board state afterwards.
  6. After each move, re-run `get_result`; if it returns a non-`None` string (checkmate, draw, stalemate, etc.), exit the loop.
- `var` is a free-form accumulator you can re-purpose to store agent state between turns (left `None` in the scaffold).

### `agent.py` – Player Under Development
- Current implementation imports `list_legal_moves_for` and chooses a random `(piece, move_option)` pair for **any** legal move. It does so separately for white/black to emphasize API usage, but both branches contain identical logic.
- To build a competitive agent, replace the random chooser with heuristics, search (e.g., minimax), policy networks, or any other strategy. Always return `(piece, move_option)` objects that belong to the *cloned* board that was passed in.

### `opponent.py` – Baseline Rival
- Mirrors the agent signature. Instead of enumerating all moves up front, it:
  1. Picks a random piece from the active player.
  2. Samples that piece’s legal move list (via `piece.get_move_options()`).
  3. Repeats until it finds a piece that can move.
- Intended as a trivial sparring partner; you can swap in stronger opponents to stress-test the agent.

### `samples.py` – Starting Positions
- Imports ChessMaker pieces plus the custom `Right` and `Pawn_Q` (see below).
- Defines global `Player` objects (`white`, `black`) reused across the project.
- `sample0` and `sample1` are 5×5 matrices of `Square` objects describing alternative initial layouts. Empty squares use `Square()`; occupied squares wrap freshly instantiated pieces bound to the relevant player.
- Modify or extend these matrices to test the agent in different tactical situations.

### `extension/board_utils.py`
- `print_board_ascii(board)` renders the 5×5 position. White pieces show as uppercase letters, black as lowercase; empty squares are dots. The column/row indices help when debugging moves manually.
- `list_legal_moves_for(board, player)` flattens the legal moves for every piece owned by `player` into a list of `(piece, move_option)` pairs – convenient for breadth-first strategies.
- `copy_piece_move(board, piece, move)` bridges the cloned board and the live board:
  * Finds the matching piece instance on the authoritative board by type and position.
  * Locates the equivalent move option on that piece (same destination coordinate).
  * Returns `(board, real_piece, real_move)` so the caller can safely execute the move.
  * Any mismatch (e.g., move no longer legal due to concurrent modifications) yields `(board, None, None)` to signal failure.

### `extension/board_rules.py`
- Enhances ChessMaker’s result detection:
  * Tracks repeated positions with `_update_repetition_count`; fivefold repetition is declared as a draw.
  * Chains built-in rule checks (`no_kings`, `checkmate`) plus custom ones (`cannot_move`, `only_2kings`).
  * `cannot_move` identifies stalemates—when the side to move has no legal actions.
  * `only_2kings` recognizes bare-king draws.
- `get_result(board)` is called both when a side fails to supply a move and after every legal move to see if the game reached a terminal state.

### `extension/piece_pawn.py`
- Defines `Pawn_Q(player)` factory that instantiates a ChessMaker `Pawn` pointing toward the opponent (using `Pawn.Direction.UP` or `DOWN`) with promotion restricted to a `Queen`. This keeps pawn behavior consistent with the 5×5 variant.

### `extension/piece_right.py`
- Introduces the custom `Right` piece by subclassing `Piece`.
- `name` property forces the display name `"Right"` so `print_board_ascii` can translate it to `"R"`.
- `_get_move_options` composes movement patterns:
  * Knight-like jumps derived from `knight.MOVE_OFFSETS`.
  * Straight-line sliding (rook-style) via `get_straight_until_blocked`.
  * `filter_uncapturable_positions` removes destinations blocked by friendly pieces.
  * `positions_to_move_options` converts legal coordinates to executable `MoveOption` objects recognised by the engine.
- `clone` provides the constructor hook ChessMaker expects when duplicating the board state.

## Game Flow at a Glance
1. **Setup**: Choose white/black controllers and a sample board in `if __name__ == "__main__"` inside `test.py`.
2. **Initialization**: `make_custom_board` instantiates the board and turn iterator.
3. **Per Turn**:
   - `Board.clone()` → pass snapshot to the current side’s function.
   - Strategy picks `(piece, move_option)` on the clone.
   - `copy_piece_move` resolves that choice against the live board.
   - `piece.move(move_option)` mutates the authoritative board; captures are automatically handled via `move_option.captures`.
   - `print_board_ascii` renders the result.
   - `get_result` checks for checkmate, stalemate, draws, or repetition.
4. **Termination**: When `get_result` returns a string, or a controller fails to supply a legal move, the loop exits and the reason is printed.

## Debugging and Extension Tips
- Use `print_board_ascii` before and after experimental logic to verify positions. Coordinates printed alongside the diagram map directly to `move_option.position.x` / `.y`.
- When implementing new strategies, rely on `list_legal_moves_for` or each piece’s `get_move_options` to avoid illegal moves.
- Preserve the clone-and-copy pattern: mutating the cloned board lets you perform look-ahead searches without altering the live state; only the mapped move should reach `piece.move`.
- `var` can carry state (transposition tables, evaluation caches, reinforcement-learning traces) between turns—initialise it in `testgame` and update it inside the agent.
- Extend `samples.py` with edge-case positions (e.g., near-promotion endgames) to regression-test agent improvements.

## Running a Game
```bash
python test.py
```
Adjust the `testgame` call in the `__main__` guard to pit different controllers or start positions against each other. Results (e.g., `"Draw - fivefold repetition"`, `"Checkmate - black loses"`) are printed automatically once the loop ends.


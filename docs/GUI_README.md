# Chess Fragment GUI

A graphical user interface for the 5x5 Chess Fragment game using PyQt6.

## Installation

1. Activate the conda environment:
```bash
conda activate comp2321
```

2. Install PyQt6:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org PyQt6
```

## Running the GUI

```bash
python gui_main.py
```

## Features

### Current Implementation
- ✅ 5x5 chess board with visual squares
- ✅ Unicode chess piece symbols (♔♕♖♗♘♙)
- ✅ Click-to-select, click-to-move interaction
- ✅ Legal move highlighting (green squares)
- ✅ Move history panel
- ✅ Human vs AI gameplay (you play as White)
- ✅ AI runs in separate thread (no UI freezing)
- ✅ Game end detection (checkmate, stalemate, draws)
- ✅ New Game button

### Game Modes
Currently configured for:
- **White:** Human (you)
- **Black:** AI (agent.py)
- **Board:** sample0

## How to Play

1. **Your turn (White):** 
   - Click on one of your pieces (white pieces: ♔♕♖♗♘♙)
   - Legal moves will highlight in green
   - Click on a green square to move
   - Click elsewhere to deselect

2. **AI turn (Black):**
   - The AI will automatically make its move
   - Status bar shows "AI is thinking..."
   - Wait for AI to complete its move

3. **Game end:**
   - A dialog box will appear when the game ends
   - Click "New Game" to start a fresh game

## Interface Layout

```
┌─────────────────────────────────────────┐
│ Chess Fragment                          │
├──────────────────┬──────────────────────┤
│                  │ Move History         │
│   5x5 Board      │                      │
│                  │  1. Knight (0,0)→... │
│   (with pieces)  │  2. Pawn (1,3)→...   │
│                  │  ...                 │
│  [New Game]      │                      │
├──────────────────┴──────────────────────┤
│ Status: White to move (Human)           │
└─────────────────────────────────────────┘
```

## Board Coordinates

The board uses a coordinate system:
- Columns: 0-4 (left to right)
- Rows: 0-4 (top to bottom)

Coordinates are displayed on the right and bottom edges of the board.

## Keyboard Shortcuts

Currently none implemented. Use mouse to interact.

## Troubleshooting

### GUI doesn't start
- Ensure PyQt6 is installed: `pip list | grep PyQt6`
- Check conda environment is activated: `conda activate comp2321`

### AI doesn't move
- Check terminal for error messages
- Ensure `agent.py` is in the same directory

### Pieces don't display correctly
- Some systems may not support all Unicode chess symbols
- The custom "Right" piece displays as 'R' (white) or 'r' (black)

## File Structure

```
chess-agent/
├── gui/
│   ├── __init__.py
│   ├── game_controller.py    # Game logic bridge
│   ├── board_widget.py        # Board display
│   ├── move_history_widget.py # Move list
│   └── main_window.py         # Main window
├── gui_main.py                # Entry point
└── GUI_README.md             # This file
```

## Future Enhancements

Potential additions:
- [ ] Settings dialog (choose AI, board sample, colors)
- [ ] Undo/redo moves
- [ ] Move animations
- [ ] Sound effects
- [ ] Save/load games
- [ ] Different piece graphics options
- [ ] AI vs AI mode
- [ ] Custom board positions

## Technical Details

- **Framework:** PyQt6
- **Python Version:** 3.12.9 (or 3.11+)
- **Game Engine:** ChessMaker library
- **Threading:** AI runs in QThread to prevent UI blocking

## Notes

- The GUI reuses all existing game logic from `test.py`, `agent.py`, and the `extension/` directory
- No modifications to existing coursework files were made
- The terminal version (`test.py`) continues to work independently


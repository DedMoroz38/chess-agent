# agent.py
from math import inf

from extension.board_utils import list_legal_moves_for, copy_piece_move
from extension.board_rules import get_result

NAME_VALUES = {
    "Queen": 9.0,
    "Right": 8.0,
    "Bishop": 3.0,
    "Knight": 3.0,
    "Pawn": 1.0,
    "King": 0.0,
}

def piece_value(piece) -> float:
    # Robust against different class/name representations
    n = getattr(piece, "name", None) or piece.__class__.__name__
    return NAME_VALUES.get(n, 0.0)

def center_bonus(move_option) -> float:
    # 5×5 board center is (2,2); use Manhattan distance for a tiny heuristic nudge
    cx, cy = 2, 2
    pos = move_option.position  # has .x and .y
    dist = abs(pos.x - cx) + abs(pos.y - cy)
    # max dist (corner to center) = 4 → small [0..0.4] bonus
    return (4 - dist) * 0.1

def greedy_score_after(board, piece, move_option) -> float:
    """Clone → map piece/move onto clone → make move → score."""
    temp = board.clone()
    temp, p2, m2 = copy_piece_move(temp, piece, move_option)
    if p2 is None or m2 is None:
        # Safety: if mapping failed, treat as terrible
        return -inf

    # Immediate capture value before/when executing the move
    capture = getattr(m2, "captures", None)
    score = 0.0
    if capture is not None:
        score += piece_value(capture)

    # Execute on the temporary board
    p2.move(m2)

    # If this produces a terminal winning state, prefer it decisively
    result = get_result(temp)
    if result and "Checkmate" in result:
        score += 1_000.0  # winning move > everything else

    # Small positional nudge toward the center
    score += center_bonus(m2)

    return score

def agent(board, player, var=None):
    """Greedy 1-ply: pick the move with the highest immediate heuristic score."""
    moves = list_legal_moves_for(board, player)
    if not moves:
        return None, None  # No legal move → framework will detect stalemate/terminal

    best = (None, None)
    best_score = -inf

    for piece, move_option in moves:
        s = greedy_score_after(board, piece, move_option)
        if s > best_score:
            best_score = s
            best = (piece, move_option)

    return best

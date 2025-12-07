from __future__ import annotations

from math import inf
from typing import Optional, Tuple

from extension.board_rules import get_result
from extension.board_utils import copy_piece_move, list_legal_moves_for

PIECE_VALUES = {
    "Queen": 9.0,
    "Right": 7.5,
    "Bishop": 3.25,
    "Knight": 3.1,
    "Pawn": 1.0,
    "King": 0.0,
}

CENTER_TABLE = (
    (0.00, 0.20, 0.30, 0.20, 0.00),
    (0.20, 0.45, 0.60, 0.45, 0.20),
    (0.30, 0.60, 0.80, 0.60, 0.30),
    (0.20, 0.45, 0.60, 0.45, 0.20),
    (0.00, 0.20, 0.30, 0.20, 0.00),
)

MATE_SCORE = 1_000_000.0


class BoardEvaluator:
    def __init__(self, mobility_weight: float = 0.15, material_weight: float = 10.0):
        self.mobility_weight = mobility_weight
        self.material_weight = material_weight

    def same_player(self, left, right) -> bool:
        if left is None or right is None:
            return False
        return left is right or getattr(left, "name", None) == getattr(right, "name", None)

    def opponent_for(self, board, player):
        for candidate in getattr(board, "players", []):
            if not self.same_player(candidate, player):
                return candidate
        return None

    def piece_value(self, piece) -> float:
        name = getattr(piece, "name", None) or piece.__class__.__name__
        return PIECE_VALUES.get(name, PIECE_VALUES.get(piece.__class__.__name__, 0.0))

    def terminal_score(self, board, perspective, depth_remaining: int):
        result = get_result(board)
        if not result:
            return None
        lower = result.lower()
        opponent = self.opponent_for(board, perspective)
        win_score = MATE_SCORE + depth_remaining
        lose_score = -MATE_SCORE - depth_remaining
        if "draw" in lower:
            return 0.0
        if opponent and opponent.name.lower() in lower and "loses" in lower:
            return win_score
        if perspective and perspective.name.lower() in lower and "loses" in lower:
            return lose_score
        if opponent and opponent.name.lower() in lower and "wins" in lower:
            return lose_score
        if perspective and perspective.name.lower() in lower and "wins" in lower:
            return win_score
        if "checkmate" in lower:
            current = getattr(board, "current_player", None)
            if self.same_player(current, perspective):
                return lose_score
            return win_score
        return 0.0

    def evaluate(self, board, perspective):
        pieces = list(board.get_pieces())
        score = 0.0
        for piece in pieces:
            sign = 1.0 if self.same_player(piece.player, perspective) else -1.0
            pos = piece.position
            center_bonus = CENTER_TABLE[pos.y][pos.x]
            score += sign * (self.piece_value(piece) * self.material_weight + center_bonus)

        opponent = self.opponent_for(board, perspective)
        if opponent:
            mobility_delta = len(list_legal_moves_for(board, perspective)) - len(list_legal_moves_for(board, opponent))
            score += mobility_delta * self.mobility_weight

        return score


class AlphaBetaAgent:
    def __init__(self, max_depth: int = 4, evaluator: Optional[BoardEvaluator] = None):
        self.max_depth = max_depth
        self.evaluator = evaluator or BoardEvaluator()

    def choose(self, board, player):
        board.current_player = player
        moves = list_legal_moves_for(board, player)
        if not moves:
            return None, None

        depth = self._adaptive_depth(board, len(moves))
        ordered = self._order_moves(moves)

        alpha, beta = -inf, inf
        best_score = -inf
        best_move: Tuple = (None, None)

        for piece, move in ordered:
            child = self._make_child(board, piece, move)
            if child is None:
                continue
            next_player = self._next_player(child, player)
            score = self._search(child, depth - 1, player, next_player, False, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = (piece, move)
            alpha = max(alpha, score)
            if beta <= alpha:
                break

        return best_move

    def _search(self, board, depth, maximizing_player, current_player, is_maximizing, alpha, beta):
        board.current_player = current_player
        terminal = self.evaluator.terminal_score(board, maximizing_player, depth)
        if terminal is not None:
            return terminal
        if depth == 0:
            return self.evaluator.evaluate(board, maximizing_player)

        moves = list_legal_moves_for(board, current_player)
        if not moves:
            return -MATE_SCORE - depth if is_maximizing else MATE_SCORE + depth

        ordered = self._order_moves(moves)

        if is_maximizing:
            value = -inf
            for piece, move in ordered:
                child = self._make_child(board, piece, move)
                if child is None:
                    continue
                next_player = self._next_player(child, current_player)
                score = self._search(child, depth - 1, maximizing_player, next_player, False, alpha, beta)
                value = max(value, score)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value

        value = inf
        for piece, move in ordered:
            child = self._make_child(board, piece, move)
            if child is None:
                continue
            next_player = self._next_player(child, current_player)
            score = self._search(child, depth - 1, maximizing_player, next_player, True, alpha, beta)
            value = min(value, score)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

    def _make_child(self, board, piece, move):
        clone = board.clone()
        clone, mapped_piece, mapped_move = copy_piece_move(clone, piece, move)
        if mapped_piece is None or mapped_move is None:
            return None
        mapped_piece.move(mapped_move)
        return clone

    def _next_player(self, board, just_moved):
        candidate = getattr(board, "current_player", None)
        if candidate and not self.evaluator.same_player(candidate, just_moved):
            return candidate
        opponent = self.evaluator.opponent_for(board, just_moved)
        return opponent or candidate or just_moved

    def _order_moves(self, moves):
        def key(item):
            piece, mv = item
            capture_weight = len(getattr(mv, "captures", []) or [])
            to_center = CENTER_TABLE[mv.position.y][mv.position.x]
            return capture_weight * 5 + to_center + self.evaluator.piece_value(piece) * 0.1
        return sorted(moves, key=key, reverse=True)

    def _adaptive_depth(self, board, move_count: int) -> int:
        piece_count = sum(1 for _ in board.get_pieces())
        if piece_count <= 6 or move_count <= 8:
            return min(self.max_depth + 1, 5)
        if piece_count <= 10 or move_count <= 14:
            return self.max_depth
        return max(2, self.max_depth - 1)


SEARCH_AGENT = AlphaBetaAgent()


def agent(board, player, var=None):
    return SEARCH_AGENT.choose(board, player)

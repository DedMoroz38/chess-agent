from __future__ import annotations

import random
from dataclasses import dataclass
from math import inf
from typing import Dict, Optional, Tuple

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

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

BOARD_SIZE = 5

PIECE_TYPES = ["King", "Queen", "Right", "Bishop", "Knight", "Pawn"]
PIECE_TYPE_INDEX = {name: i for i, name in enumerate(PIECE_TYPES)}


class ZobristHasher:
    def __init__(self, board_size: int = BOARD_SIZE, num_piece_types: int = len(PIECE_TYPES), num_players: int = 2, seed: int = 42):
        random.seed(seed)
        self.board_size = board_size
        
        self.piece_keys = [
            [
                [
                    [random.getrandbits(64) for _ in range(board_size)]
                    for _ in range(board_size)
                ]
                for _ in range(num_players)
            ]
            for _ in range(num_piece_types)
        ]
        
        self.side_to_move_key = random.getrandbits(64)
        
        self._player_index_cache: Dict[str, int] = {}
    
    def _get_player_index(self, player) -> int:
        name = getattr(player, "name", str(player))
        if name not in self._player_index_cache:
            self._player_index_cache[name] = len(self._player_index_cache) % 2
        return self._player_index_cache[name]
    
    def _get_piece_type_index(self, piece) -> int:
        name = getattr(piece, "name", None) or piece.__class__.__name__
        return PIECE_TYPE_INDEX.get(name, 0)
    
    def compute_hash(self, board, side_to_move) -> int:
        h = 0
        for piece in board.get_pieces():
            piece_type = self._get_piece_type_index(piece)
            player_idx = self._get_player_index(piece.player)
            x, y = piece.position.x, piece.position.y
            h ^= self.piece_keys[piece_type][player_idx][x][y]
        
        side_idx = self._get_player_index(side_to_move)
        if side_idx == 1:
            h ^= self.side_to_move_key
        
        return h


@dataclass(slots=True)
class TTEntry:
    hash_key: int
    value: float
    flag: int 
    age: int 
    best_signature: Optional[Tuple] = None


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


_ZOBRIST_HASHER = ZobristHasher()


class AlphaBetaAgent:
    def __init__(self, max_depth: int = 3, evaluator: Optional[BoardEvaluator] = None, table_size: int = 100_000):
        self.max_depth = max_depth
        self.evaluator = evaluator or BoardEvaluator()
        self.table_size = table_size
        self.transposition_table: Dict[int, TTEntry] = {}
        self.zobrist = _ZOBRIST_HASHER
        self.search_age = 0 

    def choose(self, board, player):
        self.search_age += 1
        
        board.current_player = player
        moves = list_legal_moves_for(board, player)
        if not moves:
            return None, None

        depth = self._adaptive_depth(board, len(moves))
        
        root_hash = self.zobrist.compute_hash(board, player)
        root_entry = self.transposition_table.get(root_hash)
        hint = root_entry.best_signature if root_entry and root_entry.hash_key == root_hash else None
        ordered = self._order_moves(moves, hint)

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

        hash_key = self.zobrist.compute_hash(board, current_player)
        entry = self.transposition_table.get(hash_key)
        
        if entry and entry.hash_key == hash_key and entry.depth >= depth:
            if entry.flag == TT_EXACT:
                return entry.value
            if entry.flag == TT_LOWER:
                alpha = max(alpha, entry.value)
            elif entry.flag == TT_UPPER:
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        alpha_orig, beta_orig = alpha, beta

        terminal = self.evaluator.terminal_score(board, maximizing_player, depth)
        if terminal is not None:
            self._store_tt(hash_key, depth, terminal, alpha_orig, beta_orig, None, force_flag=TT_EXACT)
            return terminal
        if depth == 0:
            eval_score = self.evaluator.evaluate(board, maximizing_player)
            self._store_tt(hash_key, depth, eval_score, alpha_orig, beta_orig, None, force_flag=TT_EXACT)
            return eval_score

        moves = list_legal_moves_for(board, current_player)
        if not moves:
            value = -MATE_SCORE - depth if is_maximizing else MATE_SCORE + depth
            self._store_tt(hash_key, depth, value, alpha_orig, beta_orig, None)
            return value

        hint = entry.best_signature if entry and entry.hash_key == hash_key else None
        ordered = self._order_moves(moves, hint)

        if is_maximizing:
            value = -inf
            best_sig = None
            for piece, move in ordered:
                child = self._make_child(board, piece, move)
                if child is None:
                    continue
                next_player = self._next_player(child, current_player)
                score = self._search(child, depth - 1, maximizing_player, next_player, False, alpha, beta)
                if score > value:
                    value = score
                    best_sig = self._move_signature(piece, move)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            self._store_tt(hash_key, depth, value, alpha_orig, beta_orig, best_sig)
            return value

        value = inf
        best_sig = None
        for piece, move in ordered:
            child = self._make_child(board, piece, move)
            if child is None:
                continue
            next_player = self._next_player(child, current_player)
            score = self._search(child, depth - 1, maximizing_player, next_player, True, alpha, beta)
            if score < value:
                value = score
                best_sig = self._move_signature(piece, move)
            beta = min(beta, value)
            if beta <= alpha:
                break
        self._store_tt(hash_key, depth, value, alpha_orig, beta_orig, best_sig)
        return value

    def _store_tt(self, hash_key: int, depth: int, value: float, alpha_orig: float, beta_orig: float, 
                  best_signature: Optional[Tuple], force_flag: Optional[int] = None):
        flag = force_flag
        if flag is None:
            if value <= alpha_orig:
                flag = TT_UPPER
            elif value >= beta_orig:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
        
        existing = self.transposition_table.get(hash_key)
        if existing:
            age_diff = self.search_age - existing.age
            should_replace = (
                depth > existing.depth or
                age_diff >= 2 or
                (depth == existing.depth and flag == TT_EXACT and existing.flag != TT_EXACT)
            )
            if not should_replace:
                return
        
        self.transposition_table[hash_key] = TTEntry(
            hash_key=hash_key,
            depth=depth,
            value=value,
            flag=flag,
            age=self.search_age,
            best_signature=best_signature
        )
        
        if len(self.transposition_table) > self.table_size:
            self._evict_entries()
    
    def _evict_entries(self):
        entries_to_remove = self.table_size // 10
        
        scored = [
            (key, (self.search_age - entry.age) * 100 - entry.depth)
            for key, entry in self.transposition_table.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        for key, _ in scored[:entries_to_remove]:
            del self.transposition_table[key]

    def _move_signature(self, piece, move):
        dest = getattr(move, "position", None)
        dest_coords = (getattr(dest, "x", None), getattr(dest, "y", None))
        return (
            piece.__class__.__name__,
            getattr(piece.player, "name", None),
            piece.position.x,
            piece.position.y,
            dest_coords,
        )

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

    def _order_moves(self, moves, tt_best_signature: Optional[Tuple] = None):
        def key(item):
            piece, mv = item
            capture_weight = len(getattr(mv, "captures", []) or [])
            to_center = CENTER_TABLE[mv.position.y][mv.position.x]
            return capture_weight * 5 + to_center + self.evaluator.piece_value(piece) * 0.1

        ordered = sorted(moves, key=key, reverse=True)
        if tt_best_signature is None:
            return ordered

        for idx, (piece, mv) in enumerate(ordered):
            if self._move_signature(piece, mv) == tt_best_signature:
                ordered.insert(0, ordered.pop(idx))
                break
        return ordered

    def _adaptive_depth(self, board, move_count: int) -> int:
        piece_count = sum(1 for _ in board.get_pieces())
        if piece_count <= 6 or move_count <= 8:
            return min(self.max_depth + 1, 5)
        if piece_count <= 10 or move_count <= 14:
            return self.max_depth
        return max(2, self.max_depth - 1)

    def clear_table(self):
        self.transposition_table.clear()
        self.search_age = 0


SEARCH_AGENT = AlphaBetaAgent()
_AGENT_CACHE: Dict[int, AlphaBetaAgent] = {}


def _select_agent(var) -> AlphaBetaAgent:
    if isinstance(var, dict):
        depth = var.get("depth") or var.get("max_depth")
        if depth is not None:
            depth_int = int(depth)
            cached = _AGENT_CACHE.get(depth_int)
            if cached is None:
                cached = AlphaBetaAgent(max_depth=depth_int)
                _AGENT_CACHE[depth_int] = cached
            return cached
    return SEARCH_AGENT


def agent(board, player, var=None):
    search_agent = _select_agent(var)
    return search_agent.choose(board, player)

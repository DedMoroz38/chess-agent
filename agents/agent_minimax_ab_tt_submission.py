from __future__ import annotations

import random
import time
from dataclasses import dataclass
from math import inf

from extension.board_rules import get_result
from extension.board_utils import copy_piece_move, list_legal_moves_for

# =============================================================================
# OPENING BOOK - Pre-computed best moves for opening positions
# =============================================================================
# Move signature format: (piece_class_name, player_name, from_x, from_y, (to_x, to_y))
# Board hash -> Move signature

# White's best first moves from starting positions
OPENING_BOOK_WHITE_FIRST = {
    0xd80281d4c746be2a: ('Pawn', 'white', 0, 3, (0, 2)),
    0x3fdd6166065dcd3e: ('Pawn', 'white', 4, 3, (4, 2)),
}

# Black's best responses to white's first moves
OPENING_BOOK_BLACK_RESPONSE = {
    0x91c0a8dee839ee8e: ('Pawn', 'black', 1, 1, (0, 2)),
    0xa510e854c1e87b0f: ('Pawn', 'black', 0, 1, (1, 2)),
    0x881ab5105beecbba: ('Pawn', 'black', 1, 1, (2, 2)),
    0x2da38c72c0373e69: ('Pawn', 'black', 2, 1, (3, 2)),
    0x1367f5c4311bfa07: ('Pawn', 'black', 3, 1, (3, 2)),
    0xa19c6ce0140c5641: ('Pawn', 'black', 0, 1, (1, 2)),
    0xdbdbdc65e938ef97: ('Pawn', 'black', 2, 1, (3, 2)),
    0xf4b81576f0008913: ('Pawn', 'black', 3, 1, (3, 2)),
    0x761f486c29229d9a: ('Pawn', 'black', 1, 1, (0, 2)),
    0x42cf08e600f3081b: ('Pawn', 'black', 0, 1, (1, 2)),
    0x6fc555a29af5b8ae: ('Pawn', 'black', 0, 1, (0, 2)),
    0xca7c6cc0012c4d7d: ('Pawn', 'black', 2, 1, (2, 2)),
    0xea9cd531e20bbb56: ('Pawn', 'black', 1, 1, (2, 2)),
    0xf95f3ec907479a88: ('Pawn', 'black', 1, 1, (0, 2)),
    0xf8c4f064ed2c720d: ('Pawn', 'black', 2, 1, (3, 2)),
}

# White's best second moves after black's response
OPENING_BOOK_WHITE_SECOND = {
    0x1be3bbf782ad3b90: ('Right', 'white', 0, 4, (0, 2)),
    0xe7d3abe0774af24c: ('Pawn', 'white', 2, 3, (3, 2)),
}

# Combined opening book for easy lookup
OPENING_BOOK = {
    **OPENING_BOOK_WHITE_FIRST,
    **OPENING_BOOK_BLACK_RESPONSE,
    **OPENING_BOOK_WHITE_SECOND,
}

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

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
    def __init__(self, board_size=BOARD_SIZE, num_piece_types=len(PIECE_TYPES), num_players=2, seed=42):
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
        
        self._player_index_cache = {}
    
    def _get_player_index(self, player):
        name = getattr(player, "name", str(player))
        if name not in self._player_index_cache:
            self._player_index_cache[name] = len(self._player_index_cache) % 2
        return self._player_index_cache[name]
    
    def _get_piece_type_index(self, piece):
        name = getattr(piece, "name", None) or piece.__class__.__name__
        return PIECE_TYPE_INDEX.get(name, 0)
    
    def compute_hash(self, board, side_to_move):
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


@dataclass
class TTEntry:
    hash_key = None
    depth = None
    value = None
    flag = None
    age = None
    best_signature = None


class BoardEvaluator:
    def __init__(self, mobility_weight=0.15, material_weight=10.0):
        self.mobility_weight = mobility_weight
        self.material_weight = material_weight

    def same_player(self, left, right):
        if left is None or right is None:
            return False
        return left is right or getattr(left, "name", None) == getattr(right, "name", None)

    def opponent_for(self, board, player):
        for candidate in getattr(board, "players", []):
            if not self.same_player(candidate, player):
                return candidate
        return None

    def piece_value(self, piece):
        name = getattr(piece, "name", None) or piece.__class__.__name__
        return PIECE_VALUES.get(name, PIECE_VALUES.get(piece.__class__.__name__, 0.0))

    def terminal_score(self, board, perspective, depth_remaining):
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
    def __init__(self, max_depth=3, evaluator=None, table_size=100_000, time_limit=5.0):
        self.max_depth = max_depth
        self.evaluator = evaluator or BoardEvaluator()
        self.table_size = table_size
        self.transposition_table = {}
        self.zobrist = _ZOBRIST_HASHER
        self.search_age = 0
        self.time_limit = time_limit
        self.search_cancelled = False
        self.search_start_time = 0
        self.nodes_explored = 0

    def choose(self, board, player):
        """Choose best move using iterative deepening with time management."""
        self.search_age += 1
        self.search_cancelled = False
        self.search_start_time = time.time()
        self.nodes_explored = 0
        
        board.current_player = player
        
        # Check opening book first for instant moves
        opening_move = self._get_opening_book_move(board, player)
        if opening_move:
            return opening_move
        
        moves = list_legal_moves_for(board, player)
        if not moves:
            return None, None
        
        # Single move - return immediately
        if len(moves) == 1:
            return moves[0]

        # Adaptive max depth based on position complexity
        max_depth = self._adaptive_depth(board, len(moves))
        
        # Get initial move ordering from TT
        root_hash = self.zobrist.compute_hash(board, player)
        root_entry = self.transposition_table.get(root_hash)
        hint = root_entry.best_signature if root_entry and root_entry.hash_key == root_hash else None
        ordered = self._order_moves(moves, hint)
        
        # Iterative deepening: search from depth 1 to max_depth
        best_move = ordered[0]  # Fallback to first move
        best_score = -inf
        
        for depth in range(1, max_depth + 1):
            if self._time_exceeded():
                break
            
            depth_best_move, depth_best_score, completed = self._search_root(board, player, ordered, depth)
            
            # Always update if we found a better move (even from partial search)
            if depth_best_move[0] is not None and depth_best_score > best_score:
                best_move = depth_best_move
                best_score = depth_best_score
            
            # Only reorder moves if depth fully completed
            if completed:
                ordered = self._reorder_with_best(ordered, best_move)
            else:
                # Partial search - stop iterating to deeper levels
                break
        
        return best_move
    
    def _search_root(self, board, player, ordered_moves, depth):
        """Search at root level for a specific depth.
        
        Returns: (best_move, best_score, completed)
        - completed is True only if ALL moves were searched at this depth
        """
        alpha, beta = -inf, inf
        best_score = -inf
        best_move = (None, None)
        completed = True
        
        for piece, move in ordered_moves:
            if self._time_exceeded():
                self.search_cancelled = True
                completed = False
                break
                
            child = self._make_child(board, piece, move)
            if child is None:
                continue
            next_player = self._next_player(child, player)
            score = self._search(child, depth - 1, player, next_player, False, alpha, beta)
            
            if self.search_cancelled:
                completed = False
                break
                
            if score > best_score:
                best_score = score
                best_move = (piece, move)
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        
        return best_move, best_score, completed
    
    def _time_exceeded(self):
        """Check if time limit has been exceeded."""
        return (time.time() - self.search_start_time) >= self.time_limit
    
    def _reorder_with_best(self, moves, best_move):
        """Reorder moves list with best move first."""
        if best_move[0] is None:
            return moves
        result = [best_move]
        for m in moves:
            if m != best_move:
                result.append(m)
        return result

    def _search(self, board, depth, maximizing_player, current_player, is_maximizing, alpha, beta):
        # Check for time cancellation more frequently (every 100 nodes)
        if self.nodes_explored % 100 == 0 and self._time_exceeded():
            self.search_cancelled = True
            return 0
        
        # Early exit if already cancelled
        if self.search_cancelled:
            return 0
        
        self.nodes_explored += 1
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

    def _store_tt(self, hash_key, depth, value, alpha_orig, beta_orig, best_signature, force_flag=None):
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
        
        entry = TTEntry()
        entry.hash_key = hash_key
        entry.depth = depth
        entry.value = value
        entry.flag = flag
        entry.age = self.search_age
        entry.best_signature = best_signature
        self.transposition_table[hash_key] = entry
        
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

    def _order_moves(self, moves, tt_best_signature=None):
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

    def _adaptive_depth(self, board, move_count):
        piece_count = sum(1 for _ in board.get_pieces())
        if piece_count <= 6 or move_count <= 8:
            return min(self.max_depth + 1, 5)
        if piece_count <= 10 or move_count <= 14:
            return self.max_depth
        return max(2, self.max_depth - 1)

    def _get_opening_book_move(self, board, player):
        """Look up the current position in the opening book.
        
        Returns (piece, move) if found, None otherwise.
        """
        if not OPENING_BOOK:
            return None
        
        board_hash = self.zobrist.compute_hash(board, player)
        move_sig = OPENING_BOOK.get(board_hash)
        
        if move_sig is None:
            return None
        
        # move_sig format: (piece_class_name, player_name, from_x, from_y, (to_x, to_y))
        piece_class, player_name, from_x, from_y, (to_x, to_y) = move_sig
        
        # Find the matching piece and move
        for piece in board.get_player_pieces(player):
            if (piece.__class__.__name__ == piece_class and
                piece.position.x == from_x and 
                piece.position.y == from_y):
                # Found the piece, now find the move
                for move in piece.get_move_options():
                    if move.position.x == to_x and move.position.y == to_y:
                        return (piece, move)
        
        return None

    def clear_table(self):
        self.transposition_table.clear()
        self.search_age = 0


SEARCH_AGENT = AlphaBetaAgent()
_AGENT_CACHE = {}


def _select_agent(var):
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

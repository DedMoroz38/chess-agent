# Chess Agent Algorithm Analysis

## Overview

This document provides a comprehensive analysis of the current Minimax with Alpha-Beta Pruning chess agent implementation, explaining how it works and detailing potential improvements.

**Current Implementation**: `agents/agent_minimax_ab.py`  
**Algorithm**: Minimax with Alpha-Beta Pruning  
**Default Search Depth**: 4 plies (configurable)

---

## How the Current Algorithm Works

### 1. Core Components

#### A. **BoardEvaluator Class**

The evaluator assigns a numeric score to board positions from the perspective of a specific player.

**Evaluation Function Components:**

1. **Material Score** (weight: 10.0)
   - Queen: 9.0 points
   - Right (Custom piece): 7.5 points
   - Bishop: 3.25 points
   - Knight: 3.1 points
   - Pawn: 1.0 points
   - King: 0.0 points (infinite value, handled separately)

2. **Positional Score** (Center Control)
   ```
   Center Table (5x5 board):
   0.00  0.20  0.30  0.20  0.00
   0.20  0.45  0.60  0.45  0.20
   0.30  0.60  0.80  0.60  0.30
   0.20  0.45  0.60  0.45  0.20
   0.00  0.20  0.30  0.20  0.00
   ```
   - Center squares (especially [2,2]) are valued highest
   - Encourages controlling the center of the board

3. **Mobility Score** (weight: 0.15)
   - Difference in legal move count between player and opponent
   - More moves = better position

4. **Terminal Scores**
   - Win: +1,000,000 + depth_remaining
   - Loss: -1,000,000 - depth_remaining
   - Draw: 0.0
   - Depth bonus encourages faster wins and slower losses

#### B. **AlphaBetaAgent Class**

Implements the minimax search with alpha-beta pruning optimization.

**Key Methods:**

1. **`choose(board, player)`** - Main entry point
   - Generates all legal moves
   - Determines search depth adaptively
   - Orders moves for better pruning
   - Searches each move and returns the best

2. **`_search(board, depth, maximizing_player, current_player, is_maximizing, alpha, beta)`**
   - Recursive minimax implementation
   - Alternates between maximizing and minimizing
   - Prunes branches using alpha-beta bounds
   - Returns evaluation score at leaf nodes

3. **`_order_moves(moves)`**
   - Sorts moves to improve pruning efficiency
   - Priority: captures (×5) + center control + piece value (×0.1)
   - Better moves examined first = more cutoffs

4. **`_adaptive_depth(board, move_count)`**
   - Adjusts search depth based on game complexity
   - Endgame (≤6 pieces) or few moves (≤8): depth+1 (max 5)
   - Mid-game (≤10 pieces or ≤14 moves): default depth
   - Complex positions: depth-1 (min 2)

### 2. Algorithm Flow

```
1. Generate all legal moves for current player
2. Determine search depth (adaptive)
3. Order moves (captures, center control, piece value)
4. For each move in order:
   a. Create child board state
   b. Recursively search with opponent's turn (minimizing)
   c. Track best score and alpha bound
   d. Prune if beta ≤ alpha
5. Return piece and move with highest score
```

### 3. Alpha-Beta Pruning Mechanism

**Alpha**: Best score maximizer can guarantee (lower bound)  
**Beta**: Best score minimizer can guarantee (upper bound)

**Pruning occurs when**: β ≤ α

- If maximizing: no need to explore further if current branch can't beat beta
- If minimizing: no need to explore further if current branch can't beat alpha

**Example:**
```
Maximizer at root (α=-∞, β=+∞)
  Child 1: returns 5 → α=5
  Child 2: Minimizer (α=5, β=+∞)
    Grandchild A: returns 3 → β=3
    Grandchild B: starts, but α(5) > β(3) → PRUNE!
  Child 3: continues...
```

---

## Current Strengths

1. ✅ **Efficient Pruning**: Alpha-beta reduces search space significantly
2. ✅ **Move Ordering**: Prioritizes captures and center control for better cutoffs
3. ✅ **Adaptive Depth**: Searches deeper in simpler positions
4. ✅ **Comprehensive Evaluation**: Material + position + mobility
5. ✅ **Terminal Detection**: Correctly identifies wins/losses/draws
6. ✅ **Configurable**: Easy to adjust weights and max depth

---

## Potential Improvements

### 1. **Transposition Table (Caching)** ⭐ HIGH IMPACT

**Problem**: Same positions reached via different move orders are evaluated multiple times.

**Solution**: Hash table to store previously evaluated positions.

```python
class TranspositionTable:
    def __init__(self, size_mb: int = 64):
        self.table = {}
        self.max_entries = (size_mb * 1024 * 1024) // 128  # ~128 bytes per entry
        
    def store(self, board_hash: int, depth: int, score: float, 
              flag: str, best_move=None):
        """
        flag: 'exact', 'lower', 'upper'
        - exact: score is exact at this depth
        - lower: score is alpha (lower bound)
        - upper: score is beta (upper bound)
        """
        if len(self.table) >= self.max_entries:
            # Simple replacement: remove random entry (or LRU)
            self.table.pop(next(iter(self.table)))
        
        self.table[board_hash] = {
            'depth': depth,
            'score': score,
            'flag': flag,
            'best_move': best_move
        }
    
    def lookup(self, board_hash: int, depth: int, alpha: float, beta: float):
        """Returns (score, best_move) or (None, None)"""
        entry = self.table.get(board_hash)
        if not entry or entry['depth'] < depth:
            return None, None
            
        if entry['flag'] == 'exact':
            return entry['score'], entry['best_move']
        elif entry['flag'] == 'lower' and entry['score'] >= beta:
            return entry['score'], entry['best_move']
        elif entry['flag'] == 'upper' and entry['score'] <= alpha:
            return entry['score'], entry['best_move']
            
        return None, entry.get('best_move')  # Return move hint even if no cutoff
```

**Integration into search:**
```python
def _search(self, board, depth, ...):
    board_hash = self._hash_board(board)
    cached_score, cached_move = self.tt.lookup(board_hash, depth, alpha, beta)
    if cached_score is not None:
        return cached_score
    
    # ... normal search ...
    
    # Store result
    flag = 'exact' if value > alpha and value < beta else \
           'lower' if value >= beta else 'upper'
    self.tt.store(board_hash, depth, value, flag, best_move)
```

**Expected Improvement**: 30-50% speed increase, allowing deeper searches.

### 2. **Zobrist Hashing** ⭐ HIGH IMPACT

**Problem**: Need fast, consistent way to hash board positions.

**Solution**: Pre-compute random values for each piece-square combination.

```python
import random

class ZobristHasher:
    def __init__(self):
        random.seed(42)  # Reproducible
        self.piece_keys = {}
        
        pieces = ['Pawn', 'Knight', 'Bishop', 'Right', 'Queen', 'King']
        players = ['white', 'black']
        
        for piece in pieces:
            for player in players:
                for y in range(5):
                    for x in range(5):
                        key = (piece, player, y, x)
                        self.piece_keys[key] = random.getrandbits(64)
        
        self.turn_key = random.getrandbits(64)
    
    def hash_board(self, board, current_player):
        h = 0
        for piece in board.get_pieces():
            key = (piece.name, piece.player.name, 
                   piece.position.y, piece.position.x)
            if key in self.piece_keys:
                h ^= self.piece_keys[key]
        
        if current_player.name == 'black':
            h ^= self.turn_key
            
        return h
```

### 3. **Iterative Deepening** ⭐ MEDIUM IMPACT

**Problem**: Fixed depth doesn't handle time constraints well.

**Solution**: Search depth 1, 2, 3... until time expires. Use best move from deepest completed search.

```python
def choose_with_time_limit(self, board, player, time_limit_ms: float):
    import time
    start = time.time()
    best_move = None
    
    for depth in range(1, self.max_depth + 1):
        if (time.time() - start) * 1000 >= time_limit_ms * 0.9:
            break  # 90% time used
            
        try:
            move = self.choose_at_depth(board, player, depth)
            if move[0] is not None:
                best_move = move
        except TimeoutError:
            break
    
    return best_move
```

**Benefits:**
- Always has a move (from depth 1)
- TT populated with shallow searches helps deeper searches
- Can handle strict time controls

### 4. **Quiescence Search** ⭐ MEDIUM-HIGH IMPACT

**Problem**: Horizon effect - stopping search at depth 0 during captures can misjudge position.

**Solution**: At leaf nodes, continue searching only capture moves until position is "quiet."

```python
def _quiescence_search(self, board, player, opponent, alpha, beta):
    """Search only captures until position stabilizes"""
    stand_pat = self.evaluator.evaluate(board, player)
    
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    
    # Only consider captures
    captures = [
        (pc, mv) for pc, mv in list_legal_moves_for(board, player)
        if len(getattr(mv, "captures", []) or []) > 0
    ]
    
    for piece, move in self._order_moves(captures):
        child = self._make_child(board, piece, move)
        if child is None:
            continue
        
        score = -self._quiescence_search(child, opponent, player, -beta, -alpha)
        
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    
    return alpha
```

Replace `evaluate()` call at depth 0 with `_quiescence_search()`.

### 5. **Killer Moves & History Heuristic** ⭐ MEDIUM IMPACT

**Problem**: Move ordering could be better.

**Solution**: Track moves that caused cutoffs and try them first.

```python
class KillerMoves:
    def __init__(self, max_depth: int):
        # Store 2 killer moves per depth level
        self.killers = [[None, None] for _ in range(max_depth + 1)]
    
    def store(self, depth: int, move):
        if self.killers[depth][0] != move:
            self.killers[depth][1] = self.killers[depth][0]
            self.killers[depth][0] = move
    
    def get(self, depth: int):
        return [m for m in self.killers[depth] if m is not None]

class HistoryTable:
    def __init__(self):
        self.history = {}  # {(piece_type, from_pos, to_pos): score}
    
    def update(self, piece, move, depth):
        """Reward moves that cause cutoffs"""
        key = (piece.name, piece.position, move.position)
        self.history[key] = self.history.get(key, 0) + depth * depth
    
    def score(self, piece, move):
        key = (piece.name, piece.position, move.position)
        return self.history.get(key, 0)
```

**Updated move ordering:**
1. Best move from TT
2. Captures (MVV-LVA)
3. Killer moves
4. History heuristic
5. Other moves

### 6. **Null Move Pruning** ⭐ LOW-MEDIUM IMPACT

**Problem**: Some branches are so good/bad they can be proven without full search.

**Solution**: Give opponent a free "pass" move. If position still too good, prune.

```python
def _search(self, board, depth, ...):
    # ... after terminal check ...
    
    # Null move pruning (not in PV, not in endgame, depth >= 3)
    if not is_pv and depth >= 3 and not is_endgame(board):
        # Give opponent a free move
        null_board = board.clone()
        null_board.current_player = self._next_player(null_board, current_player)
        
        # Search with reduced depth (R=2)
        null_score = -self._search(
            null_board, depth - 1 - 2, 
            maximizing_player, null_board.current_player,
            not is_maximizing, -beta, -beta + 1
        )
        
        if null_score >= beta:
            return beta  # Prune this branch
    
    # ... continue normal search ...
```

### 7. **Better Evaluation Function** ⭐ MEDIUM IMPACT

**Current weaknesses:**
- No king safety consideration
- No pawn structure evaluation
- Simple center control

**Improvements:**

```python
def evaluate_advanced(self, board, perspective):
    score = 0.0
    
    # 1. Material (existing)
    for piece in board.get_pieces():
        sign = 1.0 if self.same_player(piece.player, perspective) else -1.0
        score += sign * self.piece_value(piece) * self.material_weight
    
    # 2. Piece-square tables (position-specific)
    score += self._piece_square_evaluation(board, perspective)
    
    # 3. King safety
    score += self._king_safety(board, perspective) * 2.0
    
    # 4. Pawn structure
    score += self._pawn_structure(board, perspective) * 0.5
    
    # 5. Piece coordination
    score += self._piece_coordination(board, perspective) * 0.3
    
    # 6. Control of key squares
    score += self._square_control(board, perspective) * 0.4
    
    # 7. Mobility (existing)
    score += self._mobility_score(board, perspective)
    
    return score

def _king_safety(self, board, perspective):
    """Penalize exposed king"""
    king = self._find_king(board, perspective)
    if not king:
        return 0.0
    
    # Count friendly pieces near king
    nearby_defenders = 0
    for piece in board.get_player_pieces(perspective):
        if self._distance(king.position, piece.position) <= 1:
            nearby_defenders += 1
    
    # Count enemy pieces attacking king area
    enemy_attacks = 0
    opponent = self.opponent_for(board, perspective)
    for piece in board.get_player_pieces(opponent):
        if self._distance(king.position, piece.position) <= 2:
            enemy_attacks += 1
    
    return nearby_defenders * 0.5 - enemy_attacks * 0.8

def _pawn_structure(self, board, perspective):
    """Evaluate pawn chains, passed pawns, isolated pawns"""
    score = 0.0
    pawns = [p for p in board.get_player_pieces(perspective) 
             if p.name == 'Pawn']
    
    for pawn in pawns:
        # Passed pawn bonus
        if self._is_passed_pawn(board, pawn):
            score += 1.0
        
        # Isolated pawn penalty
        if self._is_isolated_pawn(board, pawn, pawns):
            score -= 0.5
    
    return score
```

### 8. **Aspiration Windows** ⭐ LOW IMPACT

**Problem**: Alpha-beta with (-inf, +inf) wastes work.

**Solution**: Search with narrow window around previous best score.

```python
def _search_with_aspiration(self, board, depth, player, prev_score):
    window = 50  # Initial window
    alpha = prev_score - window
    beta = prev_score + window
    
    while True:
        score = self._search(board, depth, player, player, True, alpha, beta)
        
        if alpha < score < beta:
            return score  # Success
        
        # Failed - widen window and retry
        if score <= alpha:
            alpha = -inf
        if score >= beta:
            beta = inf
        
        window *= 2  # Exponential widening
```

### 9. **Multi-PV (Principal Variation)** ⭐ LOW IMPACT (UX)

**Problem**: Only see one best line.

**Solution**: Track top N moves.

```python
def choose_multi_pv(self, board, player, n_pv: int = 3):
    """Return top N best moves with scores"""
    moves = list_legal_moves_for(board, player)
    scored_moves = []
    
    for piece, move in moves:
        child = self._make_child(board, piece, move)
        if child is None:
            continue
        score = self._search(child, self.max_depth - 1, ...)
        scored_moves.append((score, piece, move))
    
    scored_moves.sort(reverse=True)
    return scored_moves[:n_pv]
```

### 10. **Lazy SMP (Parallel Search)** ⭐ HIGH IMPACT (if multi-core)

**Problem**: Single-threaded search doesn't use modern CPUs.

**Solution**: Multiple threads search same position with shared TT.

```python
from concurrent.futures import ThreadPoolExecutor
import threading

class ParallelAgent(AlphaBetaAgent):
    def __init__(self, num_threads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.num_threads = num_threads
        self.tt_lock = threading.Lock()
    
    def choose(self, board, player):
        moves = list_legal_moves_for(board, player)
        if not moves:
            return None, None
        
        # Split moves among threads
        moves_per_thread = len(moves) // self.num_threads
        thread_moves = [
            moves[i:i + moves_per_thread]
            for i in range(0, len(moves), moves_per_thread)
        ]
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(self._search_moves, board, player, chunk)
                for chunk in thread_moves
            ]
            
            results = [f.result() for f in futures]
        
        # Find best across all threads
        best = max(results, key=lambda x: x[0])
        return best[1], best[2]  # piece, move
```

---

## Implementation Priority

### Phase 1: Foundation (Highest ROI)
1. **Zobrist Hashing** (1-2 hours) - Required for TT
2. **Transposition Table** (2-3 hours) - Biggest speed boost
3. **Quiescence Search** (2-3 hours) - Better leaf evaluation

### Phase 2: Optimization (Medium ROI)
4. **Killer Moves + History Heuristic** (2-3 hours)
5. **Improved Evaluation Function** (3-4 hours)
6. **Iterative Deepening** (1-2 hours)

### Phase 3: Advanced (Lower ROI or more complex)
7. **Null Move Pruning** (2 hours)
8. **Aspiration Windows** (1-2 hours)
9. **Parallel Search** (4-6 hours)
10. **Multi-PV** (1 hour)

---

## Expected Performance Gains

| Improvement | Speed Gain | Strength Gain | Implementation Time |
|-------------|-----------|---------------|-------------------|
| Transposition Table | 30-50% | ★★★★★ | 2-3 hours |
| Quiescence Search | -20% (slower)* | ★★★★☆ | 2-3 hours |
| Killer Moves | 10-15% | ★★★☆☆ | 2-3 hours |
| Better Eval | 0% | ★★★★☆ | 3-4 hours |
| Iterative Deepening | 0-10% | ★★☆☆☆ | 1-2 hours |
| Null Move Pruning | 15-20% | ★★★☆☆ | 2 hours |
| Parallel Search | 2-3x | ★★★★★ | 4-6 hours |

*Quiescence search is slower per move but stronger (can search deeper main line due to TT savings)

**Combined effect**: With TT + Quiescence + Killer moves, expect to search **2-3 ply deeper** at the same time cost, dramatically increasing playing strength.

---

## Testing Improvements

```python
# Benchmark script
import time

def benchmark_agent(agent, test_positions, depth):
    total_nodes = 0
    total_time = 0
    
    for board, player in test_positions:
        start = time.time()
        agent.max_depth = depth
        piece, move = agent.choose(board, player)
        elapsed = time.time() - start
        
        total_time += elapsed
        total_nodes += getattr(agent, 'nodes_searched', 0)
    
    print(f"Depth: {depth}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Nodes searched: {total_nodes}")
    print(f"Nodes per second: {total_nodes / total_time:.0f}")

# Compare agents
baseline = AlphaBetaAgent(max_depth=4)
improved = AlphaBetaAgentWithTT(max_depth=4)

benchmark_agent(baseline, test_positions, 4)
benchmark_agent(improved, test_positions, 4)
```

---

## Conclusion

The current implementation is solid with efficient alpha-beta pruning and adaptive depth. The most impactful improvements are:

1. **Transposition Table** - Massive speed boost
2. **Quiescence Search** - Eliminates horizon effect
3. **Better Evaluation** - Understands positions more deeply
4. **Parallel Search** - Leverages modern hardware

Implementing Phase 1 improvements would likely result in an agent that plays at least **2-3 ply deeper** (equivalent to ~500-800 Elo points in traditional chess).

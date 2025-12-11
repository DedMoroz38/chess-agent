# Chess Agent Optimization Strategies

## Problem Statement

The current Minimax Alpha-Beta agent with Transposition Table struggles with time constraints when search depth exceeds 2. The initial move search takes too long, causing failures under move time limits.

**Current Status:**
- ✅ Alpha-Beta pruning implemented
- ✅ Transposition Table with Zobrist hashing implemented
- ✅ Move ordering (captures, center control, TT hints)
- ✅ Adaptive depth based on game complexity

---

## Optimization Strategies

### 1. **Opening Book** ⭐⭐⭐ CRITICAL - Immediate Impact

**Problem:** Computing the best opening moves from scratch is wasteful when optimal openings are well-known for this 5x5 variant.

**Solution:** Pre-compute and store optimal opening moves.

**Implementation:**

```python
# Opening book structure: board_hash -> (piece_signature, move_destination)
OPENING_BOOK = {
    # White's first move (from starting position)
    # Format: board_hash -> best_move_signature
    0x123456789ABCDEF0: ("Pawn", "white", 2, 3, (2, 2)),  # Central pawn to center
    
    # Black responses to common white openings
    0xFEDCBA9876543210: ("Pawn", "black", 2, 1, (2, 2)),  # Mirror center control
}

def choose(self, board, player):
    # Check opening book first
    board_hash = self.zobrist.compute_hash(board, player)
    if board_hash in OPENING_BOOK:
        return self._move_from_signature(board, OPENING_BOOK[board_hash])
    
    # Fall back to search
    return self._search_best_move(board, player)
```

**Pre-computed Moves to Store:**
1. **White's first move** - Best opening (likely central pawn advance)
2. **Black's responses** - All ~10-15 possible white first moves
3. **White's second move** - Responses to common black replies

**Expected Impact:** Instant first moves, no search required for opening phase.

---

### 2. **Iterative Deepening with Time Management** ⭐⭐⭐ HIGH IMPACT

**Problem:** Fixed depth search may exceed time limits.

**Solution:** Search progressively deeper, always having a move ready.

```python
def choose_with_time_limit(self, board, player, time_limit_ms=5000):
    import time
    start = time.time()
    best_move = None
    
    for depth in range(1, self.max_depth + 1):
        elapsed_ms = (time.time() - start) * 1000
        
        # Stop if 70% of time used (leave buffer for final iteration)
        if elapsed_ms >= time_limit_ms * 0.7:
            break
        
        move = self._search_at_depth(board, player, depth)
        if move[0] is not None:
            best_move = move
            
        # Early exit if found winning move
        if self._is_winning_score(move):
            break
    
    return best_move
```

**Benefits:**
- Always returns a valid move (from depth 1)
- TT entries from shallow searches accelerate deeper searches
- Can adapt to varying time constraints

---

### 3. **Aspiration Windows** ⭐⭐ MEDIUM-HIGH IMPACT

**Problem:** Initial alpha=-∞, beta=+∞ means no early cutoffs at root.

**Solution:** Use previous iteration's score to set narrow bounds.

```python
def iterative_deepening_aspiration(self, board, player, max_depth):
    prev_score = 0
    window = 50  # Initial aspiration window
    best_move = None
    
    for depth in range(1, max_depth + 1):
        alpha = prev_score - window
        beta = prev_score + window
        
        score, move = self._search_root(board, player, depth, alpha, beta)
        
        # Re-search with full window if failed
        if score <= alpha or score >= beta:
            score, move = self._search_root(board, player, depth, -inf, inf)
        
        prev_score = score
        best_move = move
    
    return best_move
```

**Expected Impact:** 10-20% reduction in nodes searched.

---

### 4. **Principal Variation Search (PVS)** ⭐⭐ MEDIUM IMPACT

**Problem:** Full alpha-beta window for all moves is inefficient.

**Solution:** After first move, search with null window; re-search if needed.

```python
def _search_pvs(self, board, depth, alpha, beta, is_pv):
    # ... terminal/depth checks ...
    
    moves = self._order_moves(list_legal_moves_for(board, player))
    first_move = True
    
    for piece, move in moves:
        child = self._make_child(board, piece, move)
        
        if first_move:
            # Full window search for first move
            score = -self._search_pvs(child, depth-1, -beta, -alpha, True)
            first_move = False
        else:
            # Null window search (zero-width)
            score = -self._search_pvs(child, depth-1, -alpha-1, -alpha, False)
            
            # Re-search with full window if it might be better
            if alpha < score < beta:
                score = -self._search_pvs(child, depth-1, -beta, -score, True)
        
        if score > alpha:
            alpha = score
            best_move = (piece, move)
        
        if alpha >= beta:
            break
    
    return alpha
```

---

### 5. **Late Move Reduction (LMR)** ⭐⭐ MEDIUM IMPACT

**Problem:** Searching all moves to full depth is expensive.

**Solution:** Reduce depth for moves that are likely bad (searched late in ordering).

```python
def _search_with_lmr(self, board, depth, alpha, beta, ply):
    moves = self._order_moves(list_legal_moves_for(board, player))
    
    for idx, (piece, move) in enumerate(moves):
        child = self._make_child(board, piece, move)
        
        # LMR conditions: not first moves, depth >= 3, not captures
        is_capture = len(getattr(move, "captures", []) or []) > 0
        apply_lmr = idx >= 3 and depth >= 3 and not is_capture
        
        if apply_lmr:
            # Search with reduced depth
            reduction = 1 if idx < 6 else 2
            score = -self._search(child, depth - 1 - reduction, -beta, -alpha)
            
            # Re-search at full depth if promising
            if score > alpha:
                score = -self._search(child, depth - 1, -beta, -alpha)
        else:
            score = -self._search(child, depth - 1, -beta, -alpha)
        
        # ... rest of alpha-beta logic
```

**Expected Impact:** 20-40% reduction in nodes for depth >= 3.

---

### 6. **Move Ordering Improvements** ⭐⭐ MEDIUM IMPACT

**Current ordering:** Captures → Center control → Piece value → TT hint first

**Enhanced ordering:**

```python
def _order_moves_enhanced(self, moves, depth, tt_best=None, killers=None):
    def score(item):
        piece, mv = item
        score = 0
        
        # 1. TT best move (highest priority) - handled separately
        
        # 2. Winning captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
        captures = getattr(mv, "captures", []) or []
        if captures:
            victim_value = max(self.evaluator.piece_value(c) for c in captures)
            attacker_value = self.evaluator.piece_value(piece)
            score += 10000 + victim_value * 10 - attacker_value
        
        # 3. Killer moves (non-captures that caused cutoffs at this depth)
        if killers and self._move_signature(piece, mv) in killers:
            score += 5000
        
        # 4. History heuristic
        score += self.history_table.get(piece, mv, 0) * 0.01
        
        # 5. Center control
        score += CENTER_TABLE[mv.position.y][mv.position.x] * 100
        
        return score
    
    ordered = sorted(moves, key=score, reverse=True)
    
    # Put TT best move first
    if tt_best:
        for idx, (piece, mv) in enumerate(ordered):
            if self._move_signature(piece, mv) == tt_best:
                ordered.insert(0, ordered.pop(idx))
                break
    
    return ordered
```

**Add Killer Moves tracking:**

```python
class KillerMoves:
    def __init__(self, max_depth=10):
        self.killers = [set() for _ in range(max_depth)]
    
    def add(self, depth, move_sig):
        if len(self.killers[depth]) >= 2:
            self.killers[depth].pop()
        self.killers[depth].add(move_sig)
    
    def get(self, depth):
        return self.killers[depth] if depth < len(self.killers) else set()
```

---

### 7. **Lazy Evaluation** ⭐ LOW-MEDIUM IMPACT

**Problem:** Full evaluation at every leaf node is expensive.

**Solution:** Quick material-only eval first; full eval only if close to alpha/beta.

```python
def _lazy_evaluate(self, board, perspective, alpha, beta):
    # Quick material-only evaluation
    quick_score = self._material_score(board, perspective)
    
    # If clearly outside window, return quick score
    margin = 200  # Adjust based on evaluation scale
    if quick_score + margin < alpha or quick_score - margin > beta:
        return quick_score
    
    # Full evaluation for positions near the window
    return self.evaluator.evaluate(board, perspective)
```

---

### 8. **Incremental Zobrist Hashing** ⭐ LOW IMPACT (already partially done)

**Current:** Recompute full hash each position.

**Optimization:** Update hash incrementally when making moves.

```python
def _make_child_with_hash(self, board, piece, move, current_hash):
    # XOR out piece from old position
    new_hash = current_hash ^ self._piece_hash(piece, piece.position)
    
    # XOR in piece at new position
    new_hash ^= self._piece_hash(piece, move.position)
    
    # XOR out captured pieces
    for capture in getattr(move, "captures", []) or []:
        new_hash ^= self._piece_hash(capture, capture.position)
    
    # Toggle side to move
    new_hash ^= self.zobrist.side_to_move_key
    
    # Make the actual move
    clone = board.clone()
    # ... apply move ...
    
    return clone, new_hash
```

---

### 9. **Parallel Search (Future)** ⭐⭐ HIGH IMPACT (Complex)

**Problem:** Single-threaded search limits performance.

**Solution:** Search multiple root moves in parallel.

```python
from concurrent.futures import ThreadPoolExecutor

def choose_parallel(self, board, player, num_workers=4):
    moves = list_legal_moves_for(board, player)
    ordered = self._order_moves(moves)
    
    # Search first move on main thread (PV move)
    pv_move = ordered[0]
    pv_score = self._search_move(board, pv_move, -inf, inf)
    
    # Search remaining moves in parallel with PV bound
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(self._search_move, board, move, pv_score, inf)
            for move in ordered[1:]
        ]
        
        results = [f.result() for f in futures]
    
    # Find best
    all_scores = [(pv_move, pv_score)] + list(zip(ordered[1:], results))
    return max(all_scores, key=lambda x: x[1])
```

**Note:** Requires careful TT synchronization.

---

## Implementation Priority

| Priority | Optimization | Effort | Impact | When to Implement |
|----------|-------------|--------|--------|-------------------|
| 🔴 1 | Opening Book | Low | Very High | Immediate |
| 🔴 2 | Iterative Deepening | Medium | High | Immediate |
| 🟡 3 | Aspiration Windows | Low | Medium | After #2 |
| 🟡 4 | Late Move Reduction | Medium | High | After #3 |
| 🟡 5 | PVS | Medium | Medium | After #4 |
| 🟢 6 | Enhanced Move Ordering | Low | Medium | Anytime |
| 🟢 7 | Killer Moves | Low | Medium | With #6 |
| 🟢 8 | Lazy Evaluation | Low | Low | Optional |
| 🔵 9 | Parallel Search | High | High | Future |

---

## Opening Book Implementation Guide

### Step 1: Determine Starting Position Hash

```python
# In your agent, add this debug code:
def debug_opening(board, player):
    hasher = ZobristHasher()
    hash_val = hasher.compute_hash(board, player)
    print(f"Board hash for {player.name}: {hex(hash_val)}")
    
    for piece, move in list_legal_moves_for(board, player):
        sig = _move_signature(piece, move)
        print(f"  {piece.name} at ({piece.position.x},{piece.position.y}) -> ({move.position.x},{move.position.y})")
```

### Step 2: Analyze Best Openings

Run deep searches (depth 5-6) offline to determine:
1. Best white opening move
2. Best black responses to each white opening
3. Best white second moves

### Step 3: Store in Dictionary

```python
OPENING_BOOK_WHITE_FIRST = {
    # Starting position hash -> best move
    # "best move" format: (piece_type, player_name, from_x, from_y, (to_x, to_y))
}

OPENING_BOOK_BLACK_RESPONSE = {
    # Position after white's first move -> best black response
}

OPENING_BOOK_WHITE_SECOND = {
    # Position after black's first move -> best white response
}

def get_opening_move(self, board, player):
    board_hash = self.zobrist.compute_hash(board, player)
    
    # Check appropriate book based on move count
    move_count = self._estimate_move_count(board)
    
    if move_count == 0 and player.name.lower() == "white":
        return OPENING_BOOK_WHITE_FIRST.get(board_hash)
    elif move_count == 1 and player.name.lower() == "black":
        return OPENING_BOOK_BLACK_RESPONSE.get(board_hash)
    elif move_count == 2 and player.name.lower() == "white":
        return OPENING_BOOK_WHITE_SECOND.get(board_hash)
    
    return None
```

### Step 4: Simpler Alternative - Piece Count Based

Instead of hashing, detect opening by piece count:

```python
def is_opening_position(self, board):
    """Check if we're in the opening phase."""
    piece_count = sum(1 for _ in board.get_pieces())
    return piece_count == 20  # Full starting pieces for 5x5

def get_first_move_white(self, board, player):
    """Pre-defined best first move for white."""
    if not self.is_opening_position(board):
        return None
    if player.name.lower() != "white":
        return None
    
    # Find center pawn and move it forward
    for piece in board.get_player_pieces(player):
        if piece.name == "Pawn" and piece.position.x == 2 and piece.position.y == 3:
            for move in piece.get_move_options():
                if move.position.x == 2 and move.position.y == 2:
                    return (piece, move)
    
    return None
```

---

## Quick Win: Simplified Opening Book

For immediate implementation without complex hashing:

```python
def choose(self, board, player):
    # Quick opening book check
    opening_move = self._get_opening_book_move(board, player)
    if opening_move:
        return opening_move
    
    # Regular search
    return self._search_best_move(board, player)

def _get_opening_book_move(self, board, player):
    pieces = list(board.get_pieces())
    
    # Move 1: White - all 20 pieces present
    if len(pieces) == 20 and player.name.lower() == "white":
        # Move center pawn (2, 3) -> (2, 2)
        return self._find_move(board, player, "Pawn", (2, 3), (2, 2))
    
    # Move 2: Black - responding to white's opening
    if len(pieces) == 20 and player.name.lower() == "black":
        # Move center pawn (2, 1) -> (2, 2) to contest center
        return self._find_move(board, player, "Pawn", (2, 1), (2, 2))
    
    return None

def _find_move(self, board, player, piece_name, from_pos, to_pos):
    for piece in board.get_player_pieces(player):
        if piece.name == piece_name and (piece.position.x, piece.position.y) == from_pos:
            for move in piece.get_move_options():
                if (move.position.x, move.position.y) == to_pos:
                    return (piece, move)
    return None
```

---

## Benchmarking Recommendations

After implementing optimizations, measure:

1. **Nodes per second** - Raw search speed
2. **Time to depth N** - How fast each depth is reached
3. **TT hit rate** - Transposition table effectiveness
4. **Cutoff rate** - Alpha/beta pruning effectiveness
5. **Move quality** - Against known test positions

```python
def benchmark_search(agent, board, player, depth):
    import time
    
    agent.reset_stats()
    start = time.time()
    move = agent.choose(board, player)
    elapsed = time.time() - start
    
    stats = agent.get_search_statistics()
    print(f"Depth {depth}:")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Nodes: {stats['nodes_explored']}")
    print(f"  NPS: {stats['nodes_explored'] / elapsed:.0f}")
    print(f"  TT hits: {stats['tt_hits']}")
    print(f"  Cutoffs: {stats['alpha_cutoffs'] + stats['beta_cutoffs']}")
```

---

## Conclusion

The most impactful immediate changes are:

1. **Opening Book** - Eliminates slow first moves entirely
2. **Iterative Deepening** - Ensures time compliance
3. **Aspiration Windows** - Easy win with iterative deepening

These three alone should solve the time limit issues while maintaining strong play.

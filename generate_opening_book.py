"""
Opening Book Generator for Chess Agent

This script computes the best opening moves for the 5x5 chess variant
and generates an opening_book.py file that can be used by the agent.

Run this script once to generate the opening book, then import it in your agent.
"""

from __future__ import annotations

import time
from itertools import cycle
from chessmaker.chess.base import Board

from samples import white, black, sample0, sample1
from extension.board_utils import list_legal_moves_for, copy_piece_move
from agents.agent_minimax_ab_tt_no_test import AlphaBetaAgent, ZobristHasher, _ZOBRIST_HASHER


def make_custom_board(board_sample):
    """Create a fresh board from a sample configuration."""
    players = [white, black]
    board = Board(
        squares=board_sample,
        players=players,
        turn_iterator=cycle(players),
    )
    return board, players


def move_signature(piece, move):
    """Create a hashable signature for a move."""
    dest = getattr(move, "position", None)
    dest_coords = (getattr(dest, "x", None), getattr(dest, "y", None))
    return (
        piece.__class__.__name__,
        getattr(piece.player, "name", None),
        piece.position.x,
        piece.position.y,
        dest_coords,
    )


def compute_best_move(board, player, depth=4, description=""):
    """Compute the best move for a position using deep search."""
    print(f"\n{'='*60}")
    print(f"Computing: {description}")
    print(f"Player: {player.name}, Depth: {depth}")
    print(f"{'='*60}")
    
    agent = AlphaBetaAgent(max_depth=depth)
    
    start_time = time.time()
    piece, move = agent.choose(board, player)
    elapsed = time.time() - start_time
    
    if piece and move:
        sig = move_signature(piece, move)
        board_hash = _ZOBRIST_HASHER.compute_hash(board, player)
        
        print(f"Best move: {piece.name} ({piece.position.x},{piece.position.y}) -> ({move.position.x},{move.position.y})")
        print(f"Move signature: {sig}")
        print(f"Board hash: {hex(board_hash)}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Nodes explored: {agent.nodes_explored if hasattr(agent, 'nodes_explored') else 'N/A'}")
        
        return board_hash, sig, piece, move
    else:
        print("No valid move found!")
        return None, None, None, None


def apply_move(board, piece, move):
    """Apply a move to the board and return the new board state."""
    clone = board.clone()
    clone, mapped_piece, mapped_move = copy_piece_move(clone, piece, move)
    if mapped_piece and mapped_move:
        mapped_piece.move(mapped_move)
    return clone


def generate_opening_book(board_sample, sample_name, depth=4):
    """Generate opening book entries for a given starting position."""
    print(f"\n{'#'*70}")
    print(f"# Generating Opening Book for {sample_name}")
    print(f"{'#'*70}")
    
    results = {
        "white_first": {},
        "black_response": {},
        "white_second": {},
    }
    
    # =========================================================================
    # PHASE 1: White's First Move
    # =========================================================================
    board, players = make_custom_board(board_sample)
    board.current_player = white
    
    board_hash, best_sig, best_piece, best_move = compute_best_move(
        board, white, depth, 
        f"{sample_name} - White's first move"
    )
    
    if board_hash and best_sig:
        results["white_first"][board_hash] = best_sig
        
        # Apply white's best first move
        board_after_white1 = apply_move(board, best_piece, best_move)
        board_after_white1.current_player = black
        
        # =====================================================================
        # PHASE 2: Black's Response to White's Best Move
        # =====================================================================
        board_hash_b, black_sig, black_piece, black_move = compute_best_move(
            board_after_white1, black, depth,
            f"{sample_name} - Black's response to white's best opening"
        )
        
        if board_hash_b and black_sig:
            results["black_response"][board_hash_b] = black_sig
            
            # Apply black's response
            board_after_black1 = apply_move(board_after_white1, black_piece, black_move)
            board_after_black1.current_player = white
            
            # =================================================================
            # PHASE 3: White's Second Move
            # =================================================================
            board_hash_w2, white2_sig, _, _ = compute_best_move(
                board_after_black1, white, depth,
                f"{sample_name} - White's second move"
            )
            
            if board_hash_w2 and white2_sig:
                results["white_second"][board_hash_w2] = white2_sig
    
    # =========================================================================
    # PHASE 2b: Black's responses to ALL possible White first moves
    # =========================================================================
    print(f"\n{'='*60}")
    print("Computing Black's responses to all White first moves...")
    print(f"{'='*60}")
    
    board, players = make_custom_board(board_sample)
    board.current_player = white
    
    white_moves = list_legal_moves_for(board, white)
    print(f"White has {len(white_moves)} possible first moves")
    
    for i, (w_piece, w_move) in enumerate(white_moves):
        print(f"\n--- White move {i+1}/{len(white_moves)}: {w_piece.name} ({w_piece.position.x},{w_piece.position.y}) -> ({w_move.position.x},{w_move.position.y}) ---")
        
        # Apply this white move
        board_after_w = apply_move(board, w_piece, w_move)
        board_after_w.current_player = black
        
        # Compute black's best response
        b_hash, b_sig, _, _ = compute_best_move(
            board_after_w, black, depth,
            f"Black response to white move {i+1}"
        )
        
        if b_hash and b_sig:
            if b_hash not in results["black_response"]:
                results["black_response"][b_hash] = b_sig
    
    return results


def write_opening_book_file(all_results, output_path="opening_book.py"):
    """Write the opening book to a Python file."""
    
    # Merge results from all samples
    merged = {
        "white_first": {},
        "black_response": {},
        "white_second": {},
    }
    
    for sample_results in all_results.values():
        for phase, entries in sample_results.items():
            merged[phase].update(entries)
    
    with open(output_path, "w") as f:
        f.write('"""Opening Book for Chess Agent - Auto-generated"""\n\n')
        f.write("# Move signature format: (piece_class_name, player_name, from_x, from_y, (to_x, to_y))\n")
        f.write("# Board hash -> Move signature\n\n")
        
        # White's first moves
        f.write("# White's best first moves from starting positions\n")
        f.write("OPENING_BOOK_WHITE_FIRST = {\n")
        for board_hash, move_sig in merged["white_first"].items():
            f.write(f"    {hex(board_hash)}: {move_sig!r},\n")
        f.write("}\n\n")
        
        # Black's responses
        f.write("# Black's best responses to white's first moves\n")
        f.write("OPENING_BOOK_BLACK_RESPONSE = {\n")
        for board_hash, move_sig in merged["black_response"].items():
            f.write(f"    {hex(board_hash)}: {move_sig!r},\n")
        f.write("}\n\n")
        
        # White's second moves
        f.write("# White's best second moves after black's response\n")
        f.write("OPENING_BOOK_WHITE_SECOND = {\n")
        for board_hash, move_sig in merged["white_second"].items():
            f.write(f"    {hex(board_hash)}: {move_sig!r},\n")
        f.write("}\n\n")
        
        # Combined book
        f.write("# Combined opening book for easy lookup\n")
        f.write("OPENING_BOOK = {\n")
        f.write("    **OPENING_BOOK_WHITE_FIRST,\n")
        f.write("    **OPENING_BOOK_BLACK_RESPONSE,\n")
        f.write("    **OPENING_BOOK_WHITE_SECOND,\n")
        f.write("}\n\n\n")
        
        # Helper function
        f.write("def get_opening_move(board_hash):\n")
        f.write('    """Look up a move from the opening book."""\n')
        f.write("    return OPENING_BOOK.get(board_hash)\n")
    
    print(f"\n{'='*70}")
    print(f"Opening book written to: {output_path}")
    print(f"  - White first moves: {len(merged['white_first'])}")
    print(f"  - Black responses: {len(merged['black_response'])}")
    print(f"  - White second moves: {len(merged['white_second'])}")
    print(f"  - Total entries: {sum(len(v) for v in merged.values())}")
    print(f"{'='*70}")


def main():
    """Main entry point for opening book generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate chess opening book")
    parser.add_argument("--depth", type=int, default=4, help="Search depth (default: 4)")
    parser.add_argument("--output", type=str, default="opening_book.py", help="Output file path")
    parser.add_argument("--sample", type=str, choices=["0", "1", "both"], default="both",
                       help="Which sample board to use (default: both)")
    args = parser.parse_args()
    
    print(f"Opening Book Generator")
    print(f"Search depth: {args.depth}")
    print(f"Output file: {args.output}")
    print(f"Sample(s): {args.sample}")
    
    all_results = {}
    
    if args.sample in ["0", "both"]:
        all_results["sample0"] = generate_opening_book(sample0, "sample0", args.depth)
    
    if args.sample in ["1", "both"]:
        all_results["sample1"] = generate_opening_book(sample1, "sample1", args.depth)
    
    write_opening_book_file(all_results, args.output)
    
    print("\nDone! You can now import the opening book in your agent.")


if __name__ == "__main__":
    main()

"""Opening Book for Chess Agent - Auto-generated"""

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


def get_opening_move(board_hash):
    """Look up a move from the opening book."""
    return OPENING_BOOK.get(board_hash)

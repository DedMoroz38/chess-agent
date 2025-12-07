from __future__ import annotations

from copy import deepcopy
from itertools import cycle
from typing import Dict, List

from chessmaker.chess.base import Board

from samples import black, sample0, sample1, white


class TestScenarios:
    """Predefined chess positions and game setups for consistent benchmarking."""

    def __init__(self) -> None:
        self.samples = [sample0, sample1]

    def make_board(self, board_sample) -> Board:
        """Create a fresh board instance from a sample layout."""
        players = [white, black]
        squares = deepcopy(board_sample)
        board = Board(squares=squares, players=players, turn_iterator=cycle(players))
        board.current_player = white
        return board

    def _position(self, name: str, board: Board, player) -> Dict:
        return {"name": name, "board": board, "player": player}

    def opening_positions(self) -> List[Dict]:
        boards = [self.make_board(s) for s in self.samples]
        positions: List[Dict] = []
        if boards:
            positions.append(self._position("opening_sample0_white_to_move", boards[0], white))
        if len(boards) > 1:
            positions.append(self._position("opening_sample1_black_to_move", boards[1], black))
        return positions

    def midgame_positions(self) -> List[Dict]:
        # Reuse the first sample but treat it as a midgame-like snapshot for timing tests.
        board = self.make_board(self.samples[0])
        board.current_player = black
        return [self._position("midgame_activity_black_to_move", board, black)]

    def endgame_positions(self) -> List[Dict]:
        board = self.make_board(self.samples[1])
        board.current_player = white
        return [self._position("endgame_space_white_to_move", board, white)]

    def tactical_puzzles(self) -> List[Dict]:
        # Placeholder tactical scenario: reuse sample1 for consistent comparisons.
        board = self.make_board(self.samples[1])
        return [self._position("tactical_sample1_white_to_move", board, white)]

    def full_games(self) -> List[Board]:
        return [self.make_board(self.samples[0])]

    def position_suite(self) -> List[Dict]:
        """Combined suite across openings/mid/end positions."""
        return (
            self.opening_positions()
            + self.midgame_positions()
            + self.endgame_positions()
            + self.tactical_puzzles()
        )

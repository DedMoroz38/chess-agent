from __future__ import annotations

import csv
import json
import time
import tracemalloc
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Callable, Dict, List, Optional

from extension.board_rules import get_result
from extension.board_utils import copy_piece_move

from .metrics_collector import MetricsCollector
from .test_scenarios import TestScenarios


def _ensure_tracing() -> None:
    try:
        tracemalloc.start()
    except RuntimeError:
        # Already started.
        pass


def _current_memory_mb() -> Optional[float]:
    _ensure_tracing()
    try:
        current, _ = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)
    except RuntimeError:
        return None


class AgentWrapper:
    """Wrapper to execute an agent with metrics collection."""

    def __init__(
        self,
        agent_name: str,
        agent_func: Callable,
        config: Optional[Dict] = None,
        metrics_hook: Optional[Callable[[], Dict]] = None,
    ) -> None:
        self.agent_name = agent_name
        self.agent_func = agent_func
        self.config = config or {}
        self.metrics_hook = metrics_hook
        self.metrics = MetricsCollector()

    def reset(self) -> None:
        self.metrics.reset()

    def make_move(self, board, player):
        start_time = time.perf_counter()
        start_mem = _current_memory_mb()

        piece, move = self.agent_func(board, player, self.config)

        end_time = time.perf_counter()
        end_mem = _current_memory_mb()

        extras = self.metrics_hook() if self.metrics_hook else None
        mem_delta = None
        if start_mem is not None and end_mem is not None:
            mem_delta = end_mem - start_mem
        nodes = extras.get("nodes_explored") if extras else None
        depth = extras.get("max_depth_reached") if extras else None

        self.metrics.record_move(
            thinking_time=end_time - start_time,
            memory_delta=mem_delta,
            nodes=nodes,
            depth=depth,
            extras=extras,
        )
        return piece, move


class EfficiencyTester:
    """Runs efficiency experiments across registered agents and writes CSV summaries."""

    def __init__(self, scenarios: Optional[TestScenarios] = None, results_root: Path | str = "results") -> None:
        self.scenarios = scenarios or TestScenarios()
        self.results_root = Path(results_root)
        self.benchmarks_dir = self.results_root / "benchmarks"
        self.reports_dir = self.results_root / "reports"
        self.agent_wrappers: List[AgentWrapper] = []
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def register_agent(
        self,
        agent_name: str,
        agent_func: Callable,
        *,
        config: Optional[Dict] = None,
        metrics_hook: Optional[Callable[[], Dict]] = None,
    ) -> AgentWrapper:
        wrapper = AgentWrapper(agent_name, agent_func, config=config, metrics_hook=metrics_hook)
        self.agent_wrappers.append(wrapper)
        return wrapper

    def run_position_suite(self, positions: Optional[List[Dict]] = None) -> Path:
        """Evaluate each agent on a suite of single-move positions. Returns CSV path."""
        suite = positions or self.scenarios.position_suite()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        rows: List[Dict] = []

        for wrapper in self.agent_wrappers:
            for pos in suite:
                wrapper.reset()
                board = pos["board"].clone() if hasattr(pos["board"], "clone") else pos["board"]
                player = pos["player"]
                board.current_player = player
                wrapper.make_move(board, player)
                stats = wrapper.metrics.get_statistics()
                row = {
                    "timestamp": timestamp,
                    "test_type": "position",
                    "scenario": pos.get("name", "position"),
                    "agent": wrapper.agent_name,
                    "config": self._config_label(wrapper.config),
                }
                row.update(self._prefixed_stats("result", stats))
                rows.append(row)

        filename = f"{timestamp}_position_suite.csv"
        return self._write_csv(self.benchmarks_dir, filename, rows)

    def run_games(self, games_per_matchup: int = 1, max_moves: int = 80) -> Path:
        """Run complete games between every pair of registered agents. Returns CSV path."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        rows: List[Dict] = []
        game_index = 0

        for white_idx, white_agent in enumerate(self.agent_wrappers):
            for black_idx, black_agent in enumerate(self.agent_wrappers):
                if white_idx == black_idx:
                    continue
                for _ in range(games_per_matchup):
                    row = self._play_game(
                        white_agent=white_agent,
                        black_agent=black_agent,
                        game_index=game_index,
                        max_moves=max_moves,
                        timestamp=timestamp,
                    )
                    rows.append(row)
                    game_index += 1

        filename = f"{timestamp}_games.csv"
        return self._write_csv(self.benchmarks_dir, filename, rows)

    def run_tournament(self, games_per_matchup: int = 2, max_moves: int = 80) -> Path:
        """Alias for running a round-robin style tournament."""
        return self.run_games(games_per_matchup=games_per_matchup, max_moves=max_moves)

    def _play_game(
        self,
        *,
        white_agent: AgentWrapper,
        black_agent: AgentWrapper,
        game_index: int,
        max_moves: int,
        timestamp: str,
    ) -> Dict:
        starting_boards = self.scenarios.full_games()
        if not starting_boards:
            raise ValueError("No starting boards provided for game tests.")
        board = starting_boards[0]

        turn_order = cycle(board.players)
        move_counter = 0
        result = None
        termination_reason = None
        loser_name: Optional[str] = None

        white_agent.reset()
        black_agent.reset()

        while move_counter < max_moves:
            player = next(turn_order)
            board.current_player = player
            wrapper = white_agent if player.name.lower() == "white" else black_agent

            agent_view = board.clone() if hasattr(board, "clone") else board
            piece, move_opt = wrapper.make_move(agent_view, player)
            board, piece_on_board, mapped_move = copy_piece_move(board, piece, move_opt)

            if not piece_on_board or not mapped_move:
                termination_reason = f"{wrapper.agent_name} had no legal move"
                loser_name = player.name
                break

            try:
                piece_on_board.move(mapped_move)
            except Exception:
                termination_reason = f"{wrapper.agent_name} attempted illegal move"
                loser_name = player.name
                break

            move_counter += 1
            result = get_result(board)
            if result:
                termination_reason = result
                break

        if termination_reason is None:
            termination_reason = "max_moves_reached"

        winner = self._winner_from_result(termination_reason, loser_name)
        row: Dict = {
            "timestamp": timestamp,
            "test_type": "game",
            "game_index": game_index,
            "white_agent": white_agent.agent_name,
            "black_agent": black_agent.agent_name,
            "white_config": self._config_label(white_agent.config),
            "black_config": self._config_label(black_agent.config),
            "winner": winner,
            "result": termination_reason,
            "moves_played": move_counter,
        }
        row.update(self._prefixed_stats("white", white_agent.metrics.get_statistics()))
        row.update(self._prefixed_stats("black", black_agent.metrics.get_statistics()))
        return row

    def _write_csv(self, directory: Path, filename: str, rows: List[Dict]) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / filename
        if not rows:
            file_path.touch()
            return file_path

        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(file_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return file_path

    def _prefixed_stats(self, prefix: str, stats: Dict) -> Dict:
        return {f"{prefix}_{key}": value for key, value in stats.items()}

    def _config_label(self, config: Dict) -> str:
        if not config:
            return ""
        try:
            return json.dumps(config, sort_keys=True)
        except TypeError:
            return str(config)

    def _winner_from_result(self, result: str, loser_name: Optional[str] = None) -> str:
        if loser_name:
            name = loser_name.lower()
            if name == "white":
                return "black"
            if name == "black":
                return "white"
        if not result:
            return "undecided"
        lower = result.lower()
        if "white" in lower and "loses" in lower:
            return "black"
        if "black" in lower and "loses" in lower:
            return "white"
        if "draw" in lower:
            return "draw"
        if "max_moves" in lower:
            return "draw"
        return result

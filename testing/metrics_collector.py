from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional


def _safe_mean(data: Iterable[float]) -> float:
    data_list = list(data)
    return mean(data_list) if data_list else 0.0


def _safe_std(data: Iterable[float]) -> float:
    data_list = list(data)
    return pstdev(data_list) if len(data_list) > 1 else 0.0


def _safe_max(data: Iterable[float], default: float = 0.0) -> float:
    data_list = list(data)
    return max(data_list) if data_list else default


def _safe_min(data: Iterable[float], default: float = 0.0) -> float:
    data_list = list(data)
    return min(data_list) if data_list else default


class MetricsCollector:
    """Collects timing, memory, and search statistics for an agent."""

    def __init__(self) -> None:
        self.thinking_times: List[float] = []
        self.memory_usage: List[float] = []
        self.nodes_explored: List[int] = []
        self.depths_reached: List[int] = []
        self.extra_metrics: Dict[str, List[float]] = defaultdict(list)

    def reset(self) -> None:
        self.thinking_times.clear()
        self.memory_usage.clear()
        self.nodes_explored.clear()
        self.depths_reached.clear()
        self.extra_metrics.clear()

    def record_move(
        self,
        *,
        thinking_time: float,
        memory_delta: Optional[float] = None,
        nodes: Optional[int] = None,
        depth: Optional[int] = None,
        extras: Optional[Dict[str, float]] = None,
    ) -> None:
        self.thinking_times.append(thinking_time)
        if memory_delta is not None:
            self.memory_usage.append(memory_delta)
        if nodes is not None:
            self.nodes_explored.append(int(nodes))
        if depth is not None:
            self.depths_reached.append(int(depth))
        if extras:
            for key, value in extras.items():
                if value is None:
                    continue
                try:
                    self.extra_metrics[key].append(float(value))
                except (TypeError, ValueError):
                    # Ignore non-numeric extras.
                    continue

    def last_move_stats(self) -> Dict[str, float]:
        if not self.thinking_times:
            return {}
        last = {
            "thinking_time": self.thinking_times[-1],
        }
        if self.memory_usage:
            last["memory_delta"] = self.memory_usage[-1]
        if self.nodes_explored:
            last["nodes"] = self.nodes_explored[-1]
        if self.depths_reached:
            last["depth"] = self.depths_reached[-1]
        return last

    def get_statistics(self) -> Dict[str, float]:
        stats: Dict[str, float] = {
            "moves": len(self.thinking_times),
            "avg_thinking_time": _safe_mean(self.thinking_times),
            "std_thinking_time": _safe_std(self.thinking_times),
            "max_thinking_time": _safe_max(self.thinking_times),
            "min_thinking_time": _safe_min(self.thinking_times),
            "total_time": sum(self.thinking_times),
            "avg_memory": _safe_mean(self.memory_usage),
            "peak_memory": _safe_max(self.memory_usage),
            "avg_nodes": _safe_mean(self.nodes_explored),
            "total_nodes": sum(self.nodes_explored),
            "max_depth": _safe_max(self.depths_reached),
        }

        for key, values in self.extra_metrics.items():
            stats[f"avg_{key}"] = _safe_mean(values)
            stats[f"max_{key}"] = _safe_max(values)
            stats[f"total_{key}"] = sum(values) if values else 0.0

        return stats

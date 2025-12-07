from __future__ import annotations

from agents.agent_minimax_ab import AlphaBetaAgent
from agents.agent_one_ply import agent as one_ply_agent
from agent import agent as random_agent
from testing.efficiency_tester import EfficiencyTester
from testing.test_scenarios import TestScenarios


def _build_minimax(depth: int):
    """Return callable and metrics hook for a configured AlphaBetaAgent."""
    instance = AlphaBetaAgent(max_depth=depth)
    move_fn = lambda board, player, _cfg=None, inst=instance: inst.choose(board, player)
    metrics_fn = lambda inst=instance: inst.get_search_statistics()
    return move_fn, metrics_fn


def main():
    tester = EfficiencyTester(scenarios=TestScenarios())

    # Register parameterised minimax/alpha-beta agents.
    for depth in (3, 4):
        move_fn, metrics_fn = _build_minimax(depth)
        tester.register_agent(
            f"Minimax_AB_depth{depth}",
            move_fn,
            config={"depth": depth},
            metrics_hook=metrics_fn,
        )

    # Baseline greedy and random agents.
    tester.register_agent("OnePly_Greedy", one_ply_agent)
    tester.register_agent("Random_Baseline", random_agent)

    position_csv = tester.run_position_suite()
    games_csv = tester.run_games(games_per_matchup=1, max_moves=60)

    print(f"Wrote position metrics to {position_csv}")
    print(f"Wrote game metrics to {games_csv}")


if __name__ == "__main__":
    main()

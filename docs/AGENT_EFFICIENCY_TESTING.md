# Agent Efficiency Testing Framework

## Overview

This document describes a comprehensive framework for evaluating and comparing the efficiency and performance of different chess agents in the `/agents` folder. The testing framework measures multiple metrics to provide insights into algorithm performance, computational efficiency, and decision-making quality.

## Key Performance Metrics

### 1. **Time Metrics**
- **Average Thinking Time**: Mean time taken per move decision
- **Maximum Thinking Time**: Worst-case thinking time (critical for real-time constraints)
- **Minimum Thinking Time**: Best-case thinking time
- **Standard Deviation**: Consistency of thinking time across moves
- **Cumulative Time**: Total time spent for entire game/test

### 2. **Search Efficiency Metrics**
- **Nodes Explored**: Total number of game states evaluated
- **Nodes per Second**: Search speed efficiency
- **Effective Branching Factor**: Average number of children evaluated per node
- **Pruning Efficiency**: Percentage of branches pruned (for alpha-beta algorithms)
- **Cache Hit Rate**: Effectiveness of transposition tables/memoization

### 3. **Decision Quality Metrics**
- **Win Rate**: Percentage of games won against standard opponents
- **Move Quality Score**: Average evaluation score improvement per move
- **Depth Achieved**: Average search depth reached within time constraints
- **Blunder Rate**: Percentage of significantly suboptimal moves
- **Tactical Accuracy**: Success rate in finding forced wins/draws

### 4. **Resource Usage Metrics**
- **Memory Consumption**: Peak and average memory usage
- **CPU Usage**: Processor utilization percentage
- **Board Clones Created**: Number of game state copies (overhead indicator)

## Implementation Architecture

### Testing Framework Structure

```
chess-agent/
├── agents/
│   ├── agent_minimax_ab.py
│   ├── agent_one_ply.py
│   └── [future agents]
├── testing/
│   ├── __init__.py
│   ├── efficiency_tester.py      # Main testing orchestrator
│   ├── metrics_collector.py      # Metric collection utilities
│   ├── performance_analyzer.py   # Statistical analysis tools
│   ├── test_scenarios.py         # Standard test positions
│   └── visualization.py          # Result plotting/reporting
└── results/
    ├── benchmarks/
    │   └── [timestamp]_results.json
    └── reports/
        └── [timestamp]_comparison.html
```

## Core Components

### 1. Base Agent Wrapper

All agents should be wrapped in a standardized interface for testing:

```python
class AgentWrapper:
    """Wrapper to instrument any agent function with metrics collection"""
    
    def __init__(self, agent_func, agent_name):
        self.agent_func = agent_func
        self.agent_name = agent_name
        self.metrics = MetricsCollector()
    
    def make_move(self, board, player, var=None):
        """Execute agent move with timing and metrics collection"""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        # Call actual agent
        piece, move = self.agent_func(board, player, var)
        
        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()
        
        # Record metrics
        self.metrics.record_move(
            thinking_time=end_time - start_time,
            memory_delta=end_memory - start_memory,
            board_state=board.clone()
        )
        
        return piece, move
```

### 2. Metrics Collector

```python
class MetricsCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self):
        self.thinking_times = []
        self.memory_usage = []
        self.nodes_explored = []
        self.depths_reached = []
        self.move_evaluations = []
    
    def record_move(self, **kwargs):
        """Record metrics for a single move"""
        pass
    
    def get_statistics(self):
        """Calculate aggregate statistics"""
        return {
            'avg_thinking_time': np.mean(self.thinking_times),
            'std_thinking_time': np.std(self.thinking_times),
            'max_thinking_time': np.max(self.thinking_times),
            'min_thinking_time': np.min(self.thinking_times),
            'total_time': np.sum(self.thinking_times),
            'avg_memory': np.mean(self.memory_usage),
            'peak_memory': np.max(self.memory_usage),
            # ... additional metrics
        }
```

### 3. Test Scenarios

Standard positions and game scenarios for consistent testing:

```python
class TestScenarios:
    """Predefined chess positions for standardized testing"""
    
    @staticmethod
    def opening_positions():
        """Standard opening positions (3-5 moves into game)"""
        pass
    
    @staticmethod
    def midgame_positions():
        """Complex midgame positions with tactical opportunities"""
        pass
    
    @staticmethod
    def endgame_positions():
        """Endgame positions requiring precise calculation"""
        pass
    
    @staticmethod
    def tactical_puzzles():
        """Positions with forced winning sequences"""
        pass
    
    @staticmethod
    def full_games():
        """Complete games from start to finish"""
        pass
```

## Testing Protocols

### Protocol 1: Single Position Analysis
Test agent on specific board position to measure single-move performance.

```python
def test_single_position(agent_wrapper, board, player):
    """Test agent on a single position"""
    piece, move = agent_wrapper.make_move(board, player)
    return agent_wrapper.metrics.get_last_move_stats()
```

### Protocol 2: Complete Game Test
Run full game between two agents or agent vs. baseline.

```python
def test_complete_game(agent1_wrapper, agent2_wrapper, max_moves=100):
    """Run complete game and collect metrics for both agents"""
    board = initialize_board()
    move_count = 0
    
    while move_count < max_moves and not is_game_over(board):
        current_player = get_current_player(board, move_count)
        agent = agent1_wrapper if current_player.name == "white" else agent2_wrapper
        
        piece, move = agent.make_move(board, current_player)
        execute_move(board, piece, move)
        move_count += 1
    
    return {
        'agent1_metrics': agent1_wrapper.metrics.get_statistics(),
        'agent2_metrics': agent2_wrapper.metrics.get_statistics(),
        'game_result': get_result(board),
        'total_moves': move_count
    }
```

### Protocol 3: Position Suite Benchmark
Test agent on standardized suite of positions.

```python
def benchmark_position_suite(agent_wrapper, position_suite):
    """Test agent on multiple positions and aggregate results"""
    results = []
    
    for position in position_suite:
        board, player = position['board'], position['player']
        stats = test_single_position(agent_wrapper, board, player)
        results.append(stats)
    
    return aggregate_results(results)
```

### Protocol 4: Round-Robin Tournament
Compare multiple agents in head-to-head matches.

```python
def round_robin_tournament(agent_list, games_per_matchup=10):
    """Each agent plays against every other agent"""
    results_matrix = {}
    
    for agent1 in agent_list:
        for agent2 in agent_list:
            if agent1 != agent2:
                matchup_results = []
                for _ in range(games_per_matchup):
                    result = test_complete_game(agent1, agent2)
                    matchup_results.append(result)
                
                results_matrix[(agent1.name, agent2.name)] = matchup_results
    
    return results_matrix
```

## Performance Instrumentation

### For Minimax/Alpha-Beta Agents

Instrument the search algorithm to track:
- Nodes explored at each depth
- Alpha-beta cutoffs
- Search depth achieved
- Time spent at each depth level

```python
class InstrumentedMinimaxAgent:
    def __init__(self):
        self.nodes_explored = 0
        self.alpha_cutoffs = 0
        self.beta_cutoffs = 0
        self.max_depth_reached = 0
    
    def minimax(self, board, depth, alpha, beta, maximizing):
        self.nodes_explored += 1
        self.max_depth_reached = max(self.max_depth_reached, depth)
        
        # ... minimax logic with cutoff tracking
        if value >= beta:
            self.beta_cutoffs += 1
            return value
        if value <= alpha:
            self.alpha_cutoffs += 1
            return value
```

### For Simple/Greedy Agents

Track:
- Number of moves evaluated
- Evaluation function calls
- Best move selection process

## Test Execution Example

```python
# test_agents.py
from testing.efficiency_tester import EfficiencyTester
from agents.agent_minimax_ab import minimax_agent
from agents.agent_one_ply import one_ply_agent

def main():
    tester = EfficiencyTester()
    
    # Register agents
    tester.register_agent("Minimax_AB_Depth3", minimax_agent, config={'depth': 3})
    tester.register_agent("Minimax_AB_Depth4", minimax_agent, config={'depth': 4})
    tester.register_agent("OnePly_Greedy", one_ply_agent)
    
    # Run benchmarks
    print("Running position suite benchmark...")
    position_results = tester.run_position_suite()
    
    print("Running complete games...")
    game_results = tester.run_games(games_per_matchup=5)
    
    print("Running round-robin tournament...")
    tournament_results = tester.run_tournament()
    
    # Generate report
    tester.generate_report(
        output_file="results/reports/comparison_report.html",
        include_plots=True
    )
```

## Analysis and Reporting

### Statistical Analysis

Calculate and compare:
- **Mean and variance** for all time metrics
- **Confidence intervals** for performance differences
- **Statistical significance** testing (t-tests, ANOVA)
- **Efficiency ratios** (quality per second, nodes per second)

### Visualization

Generate plots for:
1. **Time Distribution**: Box plots showing thinking time distribution
2. **Performance vs. Depth**: Line graphs showing time/quality tradeoffs
3. **Win Rate Matrix**: Heatmap of head-to-head results
4. **Resource Usage**: Memory and CPU usage over time
5. **Search Efficiency**: Nodes explored vs. depth/time

### Report Format

Generate comprehensive HTML/PDF reports containing:
- Executive summary with key findings
- Detailed metrics tables
- Comparative visualizations
- Statistical analysis results
- Recommendations for algorithm selection

## Example Output

```
=====================================
Agent Efficiency Benchmark Results
=====================================

Agent: Minimax_AB_Depth3
------------------------
Avg Thinking Time: 0.125s (±0.032s)
Max Thinking Time: 0.243s
Avg Nodes Explored: 2,847
Nodes/Second: 22,776
Pruning Efficiency: 67.3%
Win Rate vs OnePly: 95.0%
Memory Usage: 12.4 MB (peak)

Agent: Minimax_AB_Depth4
------------------------
Avg Thinking Time: 0.487s (±0.156s)
Max Thinking Time: 1.124s
Avg Nodes Explored: 14,923
Nodes/Second: 30,639
Pruning Efficiency: 71.8%
Win Rate vs OnePly: 98.0%
Memory Usage: 23.7 MB (peak)

Agent: OnePly_Greedy
--------------------
Avg Thinking Time: 0.003s (±0.001s)
Max Thinking Time: 0.008s
Avg Nodes Explored: 47
Nodes/Second: 15,667
Win Rate vs Random: 87.5%
Memory Usage: 2.1 MB (peak)

Recommendations:
- For time-constrained scenarios (<0.2s): Use Minimax_AB_Depth3
- For maximum strength: Use Minimax_AB_Depth4
- For rapid prototyping: Use OnePly_Greedy
```

## Advanced Features

### Adaptive Time Management
Test agents with varying time controls:
- Fixed time per move
- Total time allocation with increment
- Depth-limited vs. time-limited search

### Parallel Performance Testing
Run multiple test games simultaneously to:
- Reduce total testing time
- Evaluate multi-core scalability
- Test thread safety

### Regression Testing
Maintain historical performance baselines:
- Track performance changes over development iterations
- Detect performance regressions automatically
- Validate optimization improvements

### Profiling Integration
Use Python profilers for deep analysis:
```python
import cProfile
import pstats

def profile_agent(agent_func, board, player):
    profiler = cProfile.Profile()
    profiler.enable()
    
    piece, move = agent_func(board, player, None)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    return stats
```

## Best Practices

1. **Consistent Test Environments**: Use fixed random seeds for reproducibility
2. **Warm-up Runs**: Discard first few runs to account for JIT compilation/caching
3. **Multiple Iterations**: Run each test multiple times for statistical validity
4. **Isolate Tests**: Ensure tests don't interfere with each other
5. **Document Results**: Keep detailed logs of all test runs with timestamps
6. **Version Control**: Tag agent versions tested for future comparison

## Dependencies

Required Python packages:
```
numpy          # Statistical calculations
matplotlib     # Plotting and visualization
pandas         # Data manipulation
psutil         # Memory and CPU monitoring
tabulate       # Table formatting
jinja2         # HTML report generation
```

## Future Enhancements

- Machine learning-based performance prediction
- Automated parameter tuning based on efficiency metrics
- Integration with continuous integration (CI) pipelines
- Web dashboard for real-time performance monitoring
- Database storage for historical performance data
- A/B testing framework for algorithm variants

## Conclusion

This testing framework provides comprehensive insights into agent performance, enabling data-driven decisions about algorithm selection, optimization priorities, and deployment configurations. By systematically measuring and comparing agents across multiple dimensions, you can ensure your chess AI achieves the optimal balance of strength, speed, and resource efficiency.

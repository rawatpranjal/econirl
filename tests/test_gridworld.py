"""Tests for the GridworldEnvironment."""

import numpy as np
import pytest
import torch

from econirl.environments.gridworld import GridworldEnvironment
from econirl.core.types import DDCProblem
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.simulation.synthetic import simulate_panel


class TestGridworldBasics:
    """Basic construction and property tests."""

    def test_default_construction(self):
        """Default GridworldEnvironment should construct without error."""
        env = GridworldEnvironment()
        assert env.grid_size == 5
        assert env.num_states == 25
        assert env.num_actions == 5

    def test_custom_grid_size(self):
        """Should support custom grid sizes."""
        for n in [2, 3, 7, 10]:
            env = GridworldEnvironment(grid_size=n)
            assert env.num_states == n * n
            assert env.grid_size == n

    def test_invalid_grid_size(self):
        """Grid size < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="grid_size must be >= 2"):
            GridworldEnvironment(grid_size=1)

    def test_terminal_state(self):
        """Terminal state should be at (N-1, N-1)."""
        env = GridworldEnvironment(grid_size=5)
        assert env.terminal_state == 24  # 4*5 + 4

        env3 = GridworldEnvironment(grid_size=3)
        assert env3.terminal_state == 8  # 2*3 + 2

    def test_problem_spec(self):
        """problem_spec should return correct DDCProblem."""
        env = GridworldEnvironment(
            grid_size=4, discount_factor=0.95, scale_parameter=0.5
        )
        spec = env.problem_spec
        assert isinstance(spec, DDCProblem)
        assert spec.num_states == 16
        assert spec.num_actions == 5
        assert spec.discount_factor == 0.95
        assert spec.scale_parameter == 0.5

    def test_parameter_names(self):
        """Should return the three parameter names."""
        env = GridworldEnvironment()
        names = env.parameter_names
        assert names == ["step_penalty", "terminal_reward", "distance_weight"]

    def test_true_parameters(self):
        """true_parameters should match constructor args."""
        env = GridworldEnvironment(
            step_penalty=-0.5, terminal_reward=20.0, distance_weight=0.3
        )
        params = env.true_parameters
        assert params["step_penalty"] == -0.5
        assert params["terminal_reward"] == 20.0
        assert params["distance_weight"] == 0.3

    def test_true_parameter_vector(self):
        """get_true_parameter_vector should match parameter order."""
        env = GridworldEnvironment(
            step_penalty=-0.1, terminal_reward=10.0, distance_weight=0.1
        )
        vec = env.get_true_parameter_vector()
        assert torch.allclose(vec, torch.tensor([-0.1, 10.0, 0.1]))


class TestTransitionMatrices:
    """Tests for the deterministic transition matrices."""

    def test_shape(self):
        """Transition matrices should have shape (5, N^2, N^2)."""
        env = GridworldEnvironment(grid_size=5)
        T = env.transition_matrices
        assert T.shape == (5, 25, 25)

    def test_deterministic_each_row_has_one_nonzero(self):
        """Each row of each action's transition matrix should have exactly one 1.0."""
        env = GridworldEnvironment(grid_size=5)
        T = env.transition_matrices
        for a in range(5):
            for s in range(25):
                row = T[a, s, :]
                assert torch.sum(row).item() == pytest.approx(1.0)
                assert torch.max(row).item() == pytest.approx(1.0)
                assert torch.count_nonzero(row).item() == 1

    def test_rows_sum_to_one(self):
        """All rows should sum to 1 (valid probability distributions)."""
        env = GridworldEnvironment(grid_size=4)
        T = env.transition_matrices
        for a in range(5):
            row_sums = T[a].sum(dim=1)
            assert torch.allclose(row_sums, torch.ones(16))

    def test_terminal_state_is_absorbing(self):
        """All actions at the terminal state should self-loop."""
        env = GridworldEnvironment(grid_size=5)
        T = env.transition_matrices
        terminal = env.terminal_state
        for a in range(5):
            # The only nonzero entry in the terminal row should be the diagonal
            assert T[a, terminal, terminal].item() == pytest.approx(1.0)
            # All other entries should be zero
            row = T[a, terminal, :].clone()
            row[terminal] = 0.0
            assert torch.all(row == 0.0)

    def test_wall_collisions_stay_in_place(self):
        """Hitting a wall should keep agent at same state."""
        env = GridworldEnvironment(grid_size=5)
        T = env.transition_matrices

        # Top-left corner (state 0): Left and Up should stay
        assert T[env.LEFT, 0, 0].item() == pytest.approx(1.0)
        assert T[env.UP, 0, 0].item() == pytest.approx(1.0)

        # Top-right corner (state 4): Right and Up should stay
        assert T[env.RIGHT, 4, 4].item() == pytest.approx(1.0)
        assert T[env.UP, 4, 4].item() == pytest.approx(1.0)

        # Bottom-left corner (state 20): Left and Down should stay
        assert T[env.LEFT, 20, 20].item() == pytest.approx(1.0)
        assert T[env.DOWN, 20, 20].item() == pytest.approx(1.0)

    def test_movement_right(self):
        """Right action should move col+1 within bounds."""
        env = GridworldEnvironment(grid_size=5)
        T = env.transition_matrices

        # State 0 (row=0, col=0) -> Right -> State 1 (row=0, col=1)
        assert T[env.RIGHT, 0, 1].item() == pytest.approx(1.0)

        # State 6 (row=1, col=1) -> Right -> State 7 (row=1, col=2)
        assert T[env.RIGHT, 6, 7].item() == pytest.approx(1.0)

    def test_movement_down(self):
        """Down action should move row+1 within bounds."""
        env = GridworldEnvironment(grid_size=5)
        T = env.transition_matrices

        # State 0 (row=0, col=0) -> Down -> State 5 (row=1, col=0)
        assert T[env.DOWN, 0, 5].item() == pytest.approx(1.0)

        # State 12 (row=2, col=2) -> Down -> State 17 (row=3, col=2)
        assert T[env.DOWN, 12, 17].item() == pytest.approx(1.0)

    def test_stay_action(self):
        """Stay action should keep agent at the same state."""
        env = GridworldEnvironment(grid_size=5)
        T = env.transition_matrices

        for s in range(25):
            assert T[env.STAY, s, s].item() == pytest.approx(1.0)

    def test_different_grid_sizes(self):
        """Transitions should work for various grid sizes."""
        for n in [2, 3, 6]:
            env = GridworldEnvironment(grid_size=n)
            T = env.transition_matrices
            assert T.shape == (5, n * n, n * n)
            # Every row must be a valid probability distribution
            for a in range(5):
                row_sums = T[a].sum(dim=1)
                assert torch.allclose(row_sums, torch.ones(n * n))


class TestFeatureMatrix:
    """Tests for the feature matrix."""

    def test_shape(self):
        """Feature matrix should have shape (N^2, 5, 3)."""
        env = GridworldEnvironment(grid_size=5)
        F = env.feature_matrix
        assert F.shape == (25, 5, 3)

    def test_step_penalty_feature_nonterminal(self):
        """Feature 0 should be 1.0 for all non-terminal state-action pairs."""
        env = GridworldEnvironment(grid_size=5)
        F = env.feature_matrix
        terminal = env.terminal_state

        for s in range(25):
            for a in range(5):
                if s != terminal:
                    assert F[s, a, 0].item() == pytest.approx(1.0)

    def test_step_penalty_feature_terminal(self):
        """Feature 0 should be 0.0 at the terminal state."""
        env = GridworldEnvironment(grid_size=5)
        F = env.feature_matrix
        terminal = env.terminal_state

        for a in range(5):
            assert F[terminal, a, 0].item() == pytest.approx(0.0)

    def test_terminal_reward_feature(self):
        """Feature 1 should be 1.0 only when action leads to terminal."""
        env = GridworldEnvironment(grid_size=5)
        F = env.feature_matrix
        T = env.transition_matrices
        terminal = env.terminal_state

        for s in range(25):
            for a in range(5):
                # Determine where the action leads
                next_s = T[a, s, :].argmax().item()
                if next_s == terminal:
                    assert F[s, a, 1].item() == pytest.approx(1.0)
                else:
                    assert F[s, a, 1].item() == pytest.approx(0.0)

    def test_distance_feature_terminal(self):
        """Feature 2 should be 0.0 at the terminal state."""
        env = GridworldEnvironment(grid_size=5)
        F = env.feature_matrix
        terminal = env.terminal_state

        for a in range(5):
            assert F[terminal, a, 2].item() == pytest.approx(0.0)

    def test_distance_feature_nonterminal(self):
        """Feature 2 should equal -manhattan_dist / (2*N) at non-terminal states."""
        env = GridworldEnvironment(grid_size=5)
        F = env.feature_matrix
        N = env.grid_size
        terminal = env.terminal_state

        for s in range(25):
            if s == terminal:
                continue
            row, col = env.state_to_grid_position(s)
            expected_dist = (N - 1 - row) + (N - 1 - col)
            expected_feature = -expected_dist / (2.0 * N)
            for a in range(5):
                assert F[s, a, 2].item() == pytest.approx(expected_feature)

    def test_utility_matrix_computation(self):
        """compute_utility_matrix should equal feature_matrix @ true_params."""
        env = GridworldEnvironment(grid_size=4)
        U = env.compute_utility_matrix()
        F = env.feature_matrix
        theta = env.get_true_parameter_vector()
        expected = torch.einsum("sak,k->sa", F, theta)
        assert torch.allclose(U, expected, atol=1e-6)


class TestValueIteration:
    """Tests that value iteration converges on the gridworld."""

    def test_value_iteration_converges(self):
        """Value iteration should converge for a small gridworld."""
        env = GridworldEnvironment(grid_size=5, discount_factor=0.95)
        problem = env.problem_spec
        transitions = env.transition_matrices
        utility = env.compute_utility_matrix()

        operator = SoftBellmanOperator(problem, transitions)
        result = value_iteration(operator, utility, tol=1e-8, max_iter=500)

        assert result.converged
        assert result.final_error < 1e-8

    def test_policy_shape(self):
        """Converged policy should have correct shape and be valid probabilities."""
        env = GridworldEnvironment(grid_size=4, discount_factor=0.9)
        problem = env.problem_spec
        transitions = env.transition_matrices
        utility = env.compute_utility_matrix()

        operator = SoftBellmanOperator(problem, transitions)
        result = value_iteration(operator, utility)

        assert result.policy.shape == (16, 5)
        # Each row should sum to 1
        row_sums = result.policy.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(16), atol=1e-5)
        # All probabilities non-negative
        assert (result.policy >= 0).all()

    def test_terminal_state_highest_value(self):
        """The terminal (absorbing + high reward) state should have the highest value."""
        env = GridworldEnvironment(
            grid_size=5, discount_factor=0.95, terminal_reward=10.0
        )
        problem = env.problem_spec
        transitions = env.transition_matrices
        utility = env.compute_utility_matrix()

        operator = SoftBellmanOperator(problem, transitions)
        result = value_iteration(operator, utility)

        terminal = env.terminal_state
        assert result.V[terminal].item() >= result.V.max().item() - 1e-6

    def test_value_decreases_with_distance(self):
        """States closer to terminal should generally have higher value."""
        env = GridworldEnvironment(
            grid_size=5, discount_factor=0.95, terminal_reward=10.0
        )
        problem = env.problem_spec
        transitions = env.transition_matrices
        utility = env.compute_utility_matrix()

        operator = SoftBellmanOperator(problem, transitions)
        result = value_iteration(operator, utility)

        # Compare a state adjacent to terminal vs one far away
        # State 23 (row=4, col=3) is adjacent, state 0 (row=0, col=0) is far
        assert result.V[23].item() > result.V[0].item()


class TestStepAndReset:
    """Tests for the Gymnasium step/reset interface."""

    def test_reset(self):
        """Reset should return state 0 (top-left corner)."""
        env = GridworldEnvironment(grid_size=5, seed=42)
        obs, info = env.reset()
        assert obs == 0
        assert "period" in info

    def test_step_right(self):
        """Step with RIGHT from (0,0) should go to (0,1)."""
        env = GridworldEnvironment(grid_size=5, seed=42)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(env.RIGHT)
        assert obs == 1  # (0, 1)
        assert not terminated
        assert not truncated

    def test_step_terminal_absorbing(self):
        """Stepping from terminal state should stay at terminal."""
        env = GridworldEnvironment(grid_size=3, seed=42)
        env.reset()
        # Navigate to terminal state (8) for a 3x3 grid
        # (0,0) -> Right -> (0,1) -> Right -> (0,2) -> Down -> (1,2)
        # -> Down -> (2,2) which is terminal (state 8)
        env.step(env.RIGHT)  # -> state 1
        env.step(env.RIGHT)  # -> state 2
        env.step(env.DOWN)   # -> state 5
        obs, _, _, _, _ = env.step(env.DOWN)   # -> state 8 (terminal)
        assert obs == 8

        # All actions should stay at terminal
        for a in range(5):
            obs, _, terminated, _, _ = env.step(a)
            assert obs == 8
            assert not terminated  # Infinite horizon DDC

    def test_invalid_action_raises(self):
        """Invalid action should raise ValueError."""
        env = GridworldEnvironment(grid_size=5, seed=42)
        env.reset()
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(5)

    def test_step_before_reset_raises(self):
        """Stepping before reset should raise RuntimeError."""
        env = GridworldEnvironment(grid_size=5)
        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(0)


class TestSimulatePanel:
    """Tests for simulate_panel with the gridworld environment."""

    def test_simulate_panel_produces_data(self):
        """simulate_panel should produce valid panel data."""
        env = GridworldEnvironment(
            grid_size=4, discount_factor=0.9, seed=42
        )
        panel = simulate_panel(env, n_individuals=5, n_periods=20, seed=42)

        assert panel.num_individuals == 5
        assert panel.num_observations == 100  # 5 * 20

    def test_simulated_states_in_range(self):
        """All simulated states should be valid state indices."""
        env = GridworldEnvironment(grid_size=4, discount_factor=0.9, seed=42)
        panel = simulate_panel(env, n_individuals=10, n_periods=50, seed=42)

        all_states = panel.get_all_states()
        all_next = panel.get_all_next_states()

        assert (all_states >= 0).all()
        assert (all_states < env.num_states).all()
        assert (all_next >= 0).all()
        assert (all_next < env.num_states).all()

    def test_simulated_actions_in_range(self):
        """All simulated actions should be valid action indices."""
        env = GridworldEnvironment(grid_size=4, discount_factor=0.9, seed=42)
        panel = simulate_panel(env, n_individuals=10, n_periods=50, seed=42)

        all_actions = panel.get_all_actions()
        assert (all_actions >= 0).all()
        assert (all_actions < env.num_actions).all()

    def test_simulated_transitions_are_deterministic(self):
        """Each (state, action) -> next_state should match transition matrices."""
        env = GridworldEnvironment(grid_size=4, discount_factor=0.9, seed=42)
        panel = simulate_panel(env, n_individuals=5, n_periods=30, seed=42)
        T = env.transition_matrices

        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                a = traj.actions[t].item()
                ns = traj.next_states[t].item()
                # In a deterministic environment, the transition should be 1.0
                assert T[a, s, ns].item() == pytest.approx(1.0)


class TestCoordinateConversion:
    """Tests for state-to-grid and grid-to-state conversions."""

    def test_roundtrip(self):
        """Converting state -> (row,col) -> state should be identity."""
        env = GridworldEnvironment(grid_size=5)
        for s in range(25):
            row, col = env.state_to_grid_position(s)
            assert env.grid_position_to_state(row, col) == s

    def test_known_positions(self):
        """Check specific state-position mappings."""
        env = GridworldEnvironment(grid_size=5)
        assert env.state_to_grid_position(0) == (0, 0)
        assert env.state_to_grid_position(4) == (0, 4)
        assert env.state_to_grid_position(20) == (4, 0)
        assert env.state_to_grid_position(24) == (4, 4)
        assert env.state_to_grid_position(12) == (2, 2)


class TestDescribe:
    """Tests for the describe method."""

    def test_describe_returns_string(self):
        """describe() should return a non-empty string."""
        env = GridworldEnvironment(grid_size=5)
        desc = env.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "Gridworld" in desc
        assert "5x5" in desc

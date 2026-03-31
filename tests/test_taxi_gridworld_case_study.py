"""Tests for the taxi gridworld case study.

Validates MCE-IRL and MCEIRLNeural on small grids (5x5) with
action-dependent features to ensure estimators converge and recover
the true reward parameters.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.environments.gridworld import GridworldEnvironment
from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator
from econirl.estimators.mceirl_neural import MCEIRLNeural
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel_from_policy

sys.path.insert(0, "examples")


# ---------------------------------------------------------------
# Shared helpers (same logic as examples/taxi_gridworld.py)
# ---------------------------------------------------------------

def _build_features(grid_size: int):
    """Build well-identified action-dependent features."""
    n_states = grid_size * grid_size
    goal_r, goal_c = grid_size - 1, grid_size - 1
    deltas = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]

    names = ["move_cost", "goal_approach", "northward", "eastward"]
    features = torch.zeros(n_states, 5, 4)

    for s in range(n_states):
        r, c = s // grid_size, s % grid_size
        d = abs(r - goal_r) + abs(c - goal_c)
        for a, (dr, dc) in enumerate(deltas):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                nr, nc = r, c
            ns = nr * grid_size + nc
            nd = abs(nr - goal_r) + abs(nc - goal_c)
            features[s, a, 0] = -1.0 if ns != s else 0.0
            features[s, a, 1] = (1.0 if nd < d else -1.0) if ns != s else 0.0
            features[s, a, 2] = 1.0 if a == 2 else (-1.0 if a == 3 else 0.0)
            features[s, a, 3] = 1.0 if a == 1 else (-1.0 if a == 0 else 0.0)

    true_params = torch.tensor([-0.5, 2.0, 0.1, 0.1])
    return features, names, true_params


def _generate_panel(grid_size, features, true_params, transitions, discount,
                    n_individuals, n_periods, seed):
    """Generate panel data from true model."""
    n_states = grid_size * grid_size
    problem = DDCProblem(num_states=n_states, num_actions=5, discount_factor=discount)
    utility = ActionDependentReward(
        feature_matrix=features,
        parameter_names=[""] * len(true_params),
    )
    reward_matrix = utility.compute(true_params)
    operator = SoftBellmanOperator(problem, transitions)
    result = hybrid_iteration(operator, reward_matrix, tol=1e-10)

    initial_dist = torch.zeros(n_states)
    initial_dist[0] = 1.0
    return simulate_panel_from_policy(
        problem=problem,
        transitions=transitions,
        policy=result.policy,
        initial_distribution=initial_dist,
        n_individuals=n_individuals,
        n_periods=n_periods,
        seed=seed,
    )


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def grid_size():
    return 5


@pytest.fixture
def discount():
    return 0.95


@pytest.fixture
def env(grid_size, discount):
    return GridworldEnvironment(grid_size=grid_size, discount_factor=discount)


@pytest.fixture
def features_and_params(grid_size):
    return _build_features(grid_size)


@pytest.fixture
def panel(grid_size, env, features_and_params, discount):
    features, _, true_params = features_and_params
    return _generate_panel(
        grid_size=grid_size,
        features=features,
        true_params=true_params,
        transitions=env.transition_matrices,
        discount=discount,
        n_individuals=200,
        n_periods=50,
        seed=42,
    )


# ---------------------------------------------------------------
# MCE-IRL tabular tests
# ---------------------------------------------------------------

class TestMCEIRLOnSmallGrid:
    """MCE-IRL tabular estimator on 5x5 grid."""

    def test_mce_irl_fits_and_recovers_parameters(
        self, env, panel, features_and_params, discount
    ):
        """MCE-IRL should fit and recover parameters with cosine sim > 0.9."""
        features, names, true_params = features_and_params
        utility = ActionDependentReward(feature_matrix=features, parameter_names=names)
        problem = DDCProblem(
            num_states=env.num_states, num_actions=5, discount_factor=discount
        )

        estimator = MCEIRLEstimator(
            config=MCEIRLConfig(
                learning_rate=0.05,
                outer_max_iter=500,
                outer_tol=1e-8,
                inner_solver="hybrid",
                inner_tol=1e-10,
                inner_max_iter=10000,
                use_adam=True,
                compute_se=False,
                verbose=False,
            )
        )
        result = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=env.transition_matrices,
        )

        cos = F.cosine_similarity(
            result.parameters.unsqueeze(0), true_params.unsqueeze(0)
        ).item()

        assert cos > 0.9, (
            f"MCE-IRL cosine similarity {cos:.4f} too low (expected > 0.9). "
            f"True: {true_params.tolist()}, "
            f"Estimated: {result.parameters.tolist()}"
        )

    def test_mce_irl_returns_valid_policy(
        self, env, panel, features_and_params, discount
    ):
        """MCE-IRL should return a valid probability distribution over actions."""
        features, names, _ = features_and_params
        utility = ActionDependentReward(feature_matrix=features, parameter_names=names)
        problem = DDCProblem(
            num_states=env.num_states, num_actions=5, discount_factor=discount
        )

        estimator = MCEIRLEstimator(
            config=MCEIRLConfig(
                outer_max_iter=100,
                compute_se=False,
                verbose=False,
            )
        )
        result = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=env.transition_matrices,
        )

        assert result.policy is not None
        assert result.policy.shape == (25, 5)
        # Policy should sum to 1 across actions
        row_sums = result.policy.sum(dim=1)
        assert torch.allclose(
            row_sums, torch.ones(25, dtype=row_sums.dtype), atol=1e-5
        )


# ---------------------------------------------------------------
# MCEIRLNeural tests
# ---------------------------------------------------------------

class TestMCEIRLNeuralOnSmallGrid:
    """MCEIRLNeural on the small 5x5 grid."""

    def _make_encoder(self, grid_size):
        def state_encoder(s, gs=grid_size):
            s_long = s.long()
            row = (s_long // gs).float() / max(gs - 1, 1)
            col = (s_long % gs).float() / max(gs - 1, 1)
            return torch.stack([row, col], dim=-1)
        return state_encoder

    def test_neural_fits_on_small_grid(self, env, panel, features_and_params):
        """MCEIRLNeural should fit without errors on a 5x5 grid."""
        features, names, _ = features_and_params

        model = MCEIRLNeural(
            n_states=env.num_states,
            n_actions=5,
            discount=0.95,
            reward_hidden_dim=32,
            reward_num_layers=1,
            max_epochs=50,
            lr=1e-3,
            state_encoder=self._make_encoder(env.grid_size),
            state_dim=2,
            feature_names=names,
            verbose=False,
        )
        model.fit(
            data=panel,
            features=features,
            transitions=env.transition_matrices,
        )

        assert model.policy_ is not None
        assert model.policy_.shape == (25, 5)
        assert model.reward_ is not None
        assert model.reward_.shape == (25,)
        assert model.n_epochs_ is not None
        assert model.n_epochs_ > 0

    def test_neural_projects_parameters(self, env, panel, features_and_params):
        """MCEIRLNeural should project rewards onto features for interpretable params."""
        features, names, _ = features_and_params

        model = MCEIRLNeural(
            n_states=env.num_states,
            n_actions=5,
            discount=0.95,
            reward_hidden_dim=32,
            reward_num_layers=2,
            max_epochs=100,
            lr=1e-3,
            state_encoder=self._make_encoder(env.grid_size),
            state_dim=2,
            feature_names=names,
            verbose=False,
        )
        model.fit(
            data=panel,
            features=features,
            transitions=env.transition_matrices,
        )

        # Should have projected parameters
        assert model.params_ is not None
        assert len(model.params_) == 4
        assert model.projection_r2_ is not None

        # Parameter names should match
        for name in names:
            assert name in model.params_

    @pytest.mark.slow
    def test_neural_learns_good_policy(self, env, features_and_params, discount):
        """MCEIRLNeural should learn a policy that matches the expert.

        Since MCEIRLNeural learns R(s) (state-only reward) but the true
        reward is R(s,a), parameter projection onto action-dependent
        features is inherently lossy. Instead, we evaluate policy quality:
        the learned policy should place high probability on the same actions
        as the expert policy.
        """
        import numpy as np

        features, names, true_params = features_and_params
        # More data for better recovery
        panel = _generate_panel(
            grid_size=env.grid_size,
            features=features,
            true_params=true_params,
            transitions=env.transition_matrices,
            discount=discount,
            n_individuals=500,
            n_periods=50,
            seed=123,
        )

        model = MCEIRLNeural(
            n_states=env.num_states,
            n_actions=5,
            discount=discount,
            reward_hidden_dim=64,
            reward_num_layers=2,
            max_epochs=200,
            lr=1e-3,
            state_encoder=self._make_encoder(env.grid_size),
            state_dim=2,
            feature_names=names,
            verbose=False,
        )
        model.fit(
            data=panel,
            features=features,
            transitions=env.transition_matrices,
        )

        # Compute expert policy
        utility = ActionDependentReward(
            feature_matrix=features, parameter_names=names
        )
        problem = DDCProblem(
            num_states=env.num_states, num_actions=5, discount_factor=discount
        )
        operator = SoftBellmanOperator(problem, env.transition_matrices)
        expert_result = hybrid_iteration(
            operator, utility.compute(true_params), tol=1e-10
        )
        expert_policy = expert_result.policy.numpy()

        # Compare: most-likely action should match for most states
        learned_policy = model.policy_
        expert_best = np.argmax(expert_policy, axis=1)
        learned_best = np.argmax(learned_policy, axis=1)
        agreement = np.mean(expert_best == learned_best)

        assert agreement > 0.5, (
            f"Policy agreement {agreement:.2%} too low (expected > 50%). "
            f"MCEIRLNeural should learn a reasonable policy even with "
            f"state-only reward."
        )

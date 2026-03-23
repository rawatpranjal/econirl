"""Unified benchmark harness for all 10 estimators on one MDP.

Runs every estimator on the same MultiComponentBusEnvironment DGP,
comparing parameter recovery, policy accuracy, and wall-clock time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import _policy_evaluation_matrix, value_iteration
from econirl.core.types import DDCProblem
from econirl.environments import MultiComponentBusEnvironment
from econirl.evaluation.adapters import build_utility_for_estimator
from econirl.evaluation.utils import compute_policy
from econirl.simulation.synthetic import simulate_panel


@dataclass
class BenchmarkDGP:
    """Data generating process specification.

    Attributes:
        n_states: Number of mileage bins (M for K=1 environment).
        replacement_cost: Fixed cost of engine replacement.
        operating_cost: Linear operating cost coefficient.
        quadratic_cost: Quadratic operating cost coefficient.
        discount_factor: Time discount factor beta.
    """

    n_states: int = 20
    replacement_cost: float = 2.0
    operating_cost: float = 1.0
    quadratic_cost: float = 0.5
    discount_factor: float = 0.99
    transfer_transition_probs: tuple[float, float, float] = (0.5, 0.4, 0.1)


@dataclass
class EstimatorSpec:
    """Specification for a single estimator in the benchmark.

    Attributes:
        estimator_class: The estimator class to instantiate.
        kwargs: Keyword arguments for the estimator constructor.
        name: Human-readable name (defaults to class name).
        can_recover_params: Whether parameter RMSE is meaningful.
    """

    estimator_class: type
    kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    can_recover_params: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.estimator_class.__name__


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.

    Attributes:
        estimator: Name of the estimator.
        n_states: State space size.
        n_agents: Number of simulated agents.
        seed: Random seed used.
        param_rmse: Parameter RMSE (None if not applicable).
        policy_rmse: Policy RMSE vs true optimal policy.
        time_seconds: Wall-clock estimation time.
        converged: Whether estimation converged.
        estimates: Dict of estimated parameter values.
        true_params: Dict of true parameter values.
    """

    estimator: str
    n_states: int
    n_agents: int
    seed: int
    param_rmse: float | None
    policy_rmse: float
    pct_optimal: float
    time_seconds: float
    converged: bool
    pct_optimal_transfer: float | None = None
    estimates: dict[str, float] = field(default_factory=dict)
    true_params: dict[str, float] = field(default_factory=dict)
    estimated_reward: torch.Tensor | None = None
    learned_policy: torch.Tensor | None = None
    true_reward: torch.Tensor | None = None


def _make_env(dgp: BenchmarkDGP) -> MultiComponentBusEnvironment:
    """Create a K=1 multi-component bus environment from DGP."""
    return MultiComponentBusEnvironment(
        K=1,
        M=dgp.n_states,
        replacement_cost=dgp.replacement_cost,
        operating_cost=dgp.operating_cost,
        quadratic_cost=dgp.quadratic_cost,
        discount_factor=dgp.discount_factor,
    )


def _evaluate_pct_optimal(
    learned_policy: torch.Tensor,
    true_utility: torch.Tensor,
    transitions: torch.Tensor,
    problem: DDCProblem,
) -> float:
    """Compute % of optimal value achieved by a learned policy.

    Uses baseline-normalized score: (V_learned - V_random) / (V_star - V_random) * 100.
    Returns 100% when learned matches optimal, 0% when learned matches random.
    Works correctly regardless of value sign (costs or rewards).
    """
    operator = SoftBellmanOperator(problem=problem, transitions=transitions)
    v_star = value_iteration(operator, true_utility).V

    v_learned = _policy_evaluation_matrix(
        true_utility, learned_policy, transitions,
        beta=problem.discount_factor,
        sigma=problem.scale_parameter,
    )

    # Baseline: uniform random policy
    uniform_policy = torch.ones_like(learned_policy) / learned_policy.shape[1]
    v_random = _policy_evaluation_matrix(
        true_utility, uniform_policy, transitions,
        beta=problem.discount_factor,
        sigma=problem.scale_parameter,
    )

    mean_v_star = v_star.mean().item()
    mean_v_learned = v_learned.mean().item()
    mean_v_random = v_random.mean().item()

    denom = mean_v_star - mean_v_random
    if abs(denom) < 1e-10:
        return 100.0

    return ((mean_v_learned - mean_v_random) / denom) * 100.0


def _compute_transfer_pct_optimal(
    est_utility: torch.Tensor,
    true_utility: torch.Tensor,
    problem: DDCProblem,
    dgp: BenchmarkDGP,
) -> float:
    """Compute % of optimal value under transferred transition dynamics.

    Creates a transfer environment with different mileage wear rates,
    re-solves the MDP using estimated rewards, and evaluates the resulting
    policy under true rewards + new transitions.

    Args:
        est_utility: Estimated reward matrix (n_states, n_actions).
        true_utility: True reward matrix (n_states, n_actions).
        problem: MDP specification.
        dgp: DGP with transfer_transition_probs.
    """
    transfer_env = MultiComponentBusEnvironment(
        K=1,
        M=dgp.n_states,
        replacement_cost=dgp.replacement_cost,
        operating_cost=dgp.operating_cost,
        quadratic_cost=dgp.quadratic_cost,
        discount_factor=dgp.discount_factor,
        mileage_transition_probs=dgp.transfer_transition_probs,
    )
    transfer_transitions = transfer_env.transition_matrices

    # True optimal under transfer
    operator = SoftBellmanOperator(problem=problem, transitions=transfer_transitions)
    v_star = value_iteration(operator, true_utility).V

    # Policy from estimated rewards under transfer transitions
    transfer_policy = value_iteration(operator, est_utility).policy

    # Evaluate transfer policy under true rewards
    v_transfer = _policy_evaluation_matrix(
        true_utility, transfer_policy, transfer_transitions,
        beta=problem.discount_factor,
        sigma=problem.scale_parameter,
    )

    # Baseline: uniform random policy under transfer transitions
    uniform_policy = torch.ones(problem.num_states, problem.num_actions) / problem.num_actions
    v_random = _policy_evaluation_matrix(
        true_utility, uniform_policy, transfer_transitions,
        beta=problem.discount_factor,
        sigma=problem.scale_parameter,
    )

    mean_v_star = v_star.mean().item()
    mean_v_transfer = v_transfer.mean().item()
    mean_v_random = v_random.mean().item()

    denom = mean_v_star - mean_v_random
    if abs(denom) < 1e-10:
        return 100.0

    return ((mean_v_transfer - mean_v_random) / denom) * 100.0


def _evaluate_policy_transfer(
    learned_policy: torch.Tensor,
    true_utility: torch.Tensor,
    problem: DDCProblem,
    dgp: BenchmarkDGP,
) -> float:
    """Evaluate a fixed policy on transfer dynamics.

    Unlike _compute_transfer_pct_optimal which re-solves the MDP with
    estimated rewards, this directly evaluates the learned policy under
    transfer transitions. Used for estimators that produce policies
    without recovering rewards (e.g. behavioral cloning).

    Args:
        learned_policy: Policy to evaluate (n_states, n_actions).
        true_utility: True reward matrix (n_states, n_actions).
        problem: MDP specification.
        dgp: DGP with transfer_transition_probs.
    """
    transfer_env = MultiComponentBusEnvironment(
        K=1,
        M=dgp.n_states,
        replacement_cost=dgp.replacement_cost,
        operating_cost=dgp.operating_cost,
        quadratic_cost=dgp.quadratic_cost,
        discount_factor=dgp.discount_factor,
        mileage_transition_probs=dgp.transfer_transition_probs,
    )
    transfer_transitions = transfer_env.transition_matrices

    operator = SoftBellmanOperator(problem=problem, transitions=transfer_transitions)
    v_star = value_iteration(operator, true_utility).V

    v_learned = _policy_evaluation_matrix(
        true_utility, learned_policy, transfer_transitions,
        beta=problem.discount_factor,
        sigma=problem.scale_parameter,
    )

    uniform_policy = torch.ones_like(learned_policy) / learned_policy.shape[1]
    v_random = _policy_evaluation_matrix(
        true_utility, uniform_policy, transfer_transitions,
        beta=problem.discount_factor,
        sigma=problem.scale_parameter,
    )

    mean_v_star = v_star.mean().item()
    mean_v_learned = v_learned.mean().item()
    mean_v_random = v_random.mean().item()

    denom = mean_v_star - mean_v_random
    if abs(denom) < 1e-10:
        return 100.0

    return ((mean_v_learned - mean_v_random) / denom) * 100.0


def get_default_estimator_specs() -> list[EstimatorSpec]:
    """Get benchmark-tuned specs for all estimators."""
    from econirl.estimation import (
        AIRLEstimator,
        BayesianIRLEstimator,
        BehavioralCloningEstimator,
        DeepMaxEntIRLEstimator,
        FIRLEstimator,
        NNESEstimator,
        SEESEstimator,
        CCPEstimator,
        GAILEstimator,
        GCLEstimator,
        GLADIUSEstimator,
        GLADIUSConfig,
        MaxEntIRLEstimator,
        MaxMarginIRLEstimator,
        MaxMarginPlanningEstimator,
        MCEIRLEstimator,
        NFXPEstimator,
        TDCCPEstimator,
        TDCCPConfig,
    )

    return [
        EstimatorSpec(
            BehavioralCloningEstimator,
            kwargs=dict(smoothing=1.0),
            name="BC",
            can_recover_params=False,
        ),
        EstimatorSpec(
            NFXPEstimator,
            kwargs=dict(
                inner_solver="hybrid",
                inner_max_iter=10000,
                compute_hessian=False,
            ),
            name="NFXP",
        ),
        EstimatorSpec(
            CCPEstimator,
            kwargs=dict(
                num_policy_iterations=5,
                compute_hessian=False,
            ),
            name="CCP",
        ),
        EstimatorSpec(
            MCEIRLEstimator,
            kwargs=dict(
                learning_rate=0.5,
                use_adam=False,
                outer_max_iter=1000,
                inner_max_iter=10000,
                gradient_clip=1.0,
                compute_se=False,
            ),
            name="MCE IRL",
        ),
        EstimatorSpec(
            MaxEntIRLEstimator,
            kwargs=dict(
                inner_solver="value",
                inner_tol=1e-8,
                inner_max_iter=5000,
                outer_max_iter=300,
                compute_hessian=False,
            ),
            name="MaxEnt IRL",
            can_recover_params=False,
        ),
        EstimatorSpec(
            MaxMarginPlanningEstimator,
            kwargs=dict(
                learning_rate=1.0,
                max_iterations=3000,
                compute_se=False,
                loss_type="trajectory_hamming",
                loss_scale=0.5,
                regularization_lambda=0.0,
                inner_max_iter=5000,
            ),
            name="Max Margin",
        ),
        EstimatorSpec(
            MaxMarginIRLEstimator,
            kwargs=dict(
                max_iterations=50,
                margin_tol=1e-4,
                compute_hessian=False,
            ),
            name="Max Margin IRL",
            can_recover_params=False,
        ),
        EstimatorSpec(
            TDCCPEstimator,
            kwargs=dict(
                config=TDCCPConfig(
                    hidden_dim=32,
                    avi_iterations=30,
                    epochs_per_avi=20,
                    learning_rate=5e-4,
                    batch_size=512,
                    compute_se=False,
                ),
            ),
            name="TD-CCP",
        ),
        EstimatorSpec(
            GLADIUSEstimator,
            kwargs=dict(
                max_epochs=500,
                q_hidden_dim=32,
                v_hidden_dim=32,
                q_num_layers=2,
                v_num_layers=2,
                compute_se=False,
                batch_size=256,
                bellman_penalty_weight=0.1,
                weight_decay=1e-3,
            ),
            name="GLADIUS",
        ),
        EstimatorSpec(
            GAILEstimator,
            kwargs=dict(
                discriminator_type="linear",
                max_rounds=500,
                discriminator_lr=0.02,
                discriminator_steps=10,
                reward_transform="logit",
                convergence_tol=0,
                compute_se=False,
            ),
            name="GAIL",
            can_recover_params=False,
        ),
        EstimatorSpec(
            AIRLEstimator,
            kwargs=dict(
                reward_type="linear",
                max_rounds=500,
                reward_lr=0.02,
                discriminator_steps=10,
                compute_se=False,
            ),
            name="AIRL",
            can_recover_params=False,
        ),
        EstimatorSpec(
            GCLEstimator,
            kwargs=dict(
                max_iterations=300,
                n_sample_trajectories=200,
                cost_lr=5e-4,
                embed_dim=16,
                hidden_dims=[32, 32],
                importance_clipping=5.0,
                normalize_reward=True,
            ),
            name="GCL",
            can_recover_params=False,
        ),
        EstimatorSpec(
            BayesianIRLEstimator,
            kwargs=dict(
                n_samples=2000,
                burnin=500,
                proposal_sigma=0.1,
                prior_sigma=5.0,
            ),
            name="BIRL",
            can_recover_params=False,
        ),
        EstimatorSpec(
            DeepMaxEntIRLEstimator,
            kwargs=dict(
                hidden_dims=[32, 32],
                lr=1e-3,
                max_epochs=300,
            ),
            name="Deep MaxEnt",
            can_recover_params=False,
        ),
        EstimatorSpec(
            NNESEstimator,
            kwargs=dict(
                hidden_dim=32,
                v_epochs=500,
                outer_max_iter=200,
                compute_se=False,
            ),
            name="NNES",
            can_recover_params=True,
        ),
        EstimatorSpec(
            FIRLEstimator,
            kwargs=dict(
                f_divergence="kl",
                lr=0.5,
                max_iter=500,
            ),
            name="f-IRL",
            can_recover_params=False,
        ),
        EstimatorSpec(
            SEESEstimator,
            kwargs=dict(
                basis_type="fourier",
                basis_dim=8,
                penalty_lambda=0.01,
                compute_se=False,
            ),
            name="SEES",
            can_recover_params=True,
        ),
    ]


def run_single(
    dgp: BenchmarkDGP,
    spec: EstimatorSpec,
    n_agents: int = 200,
    n_periods: int = 100,
    seed: int = 42,
) -> BenchmarkResult:
    """Run a single estimator on the benchmark DGP.

    Args:
        dgp: Data generating process specification.
        spec: Estimator specification with hyperparameters.
        n_agents: Number of simulated agents.
        n_periods: Number of time periods per agent.
        seed: Random seed for data simulation.

    Returns:
        BenchmarkResult with metrics.
    """
    env = _make_env(dgp)
    panel = simulate_panel(env, n_individuals=n_agents, n_periods=n_periods, seed=seed)
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = build_utility_for_estimator(env, spec.estimator_class)

    # Compute true optimal policy for policy RMSE
    true_params = env.get_true_parameter_vector()
    true_policy = compute_policy(true_params, problem, transitions, env.feature_matrix)
    true_dict = dict(zip(env.parameter_names, true_params.tolist()))
    true_utility = torch.einsum("sak,k->sa", env.feature_matrix, true_params)

    t0 = time.perf_counter()
    try:
        summary = spec.estimator_class(**spec.kwargs).estimate(
            panel, utility, problem, transitions
        )
        elapsed = time.perf_counter() - t0

        # Parameter RMSE (only for estimators with matching parameter space)
        param_rmse = None
        estimates: dict[str, float] = {}
        est_params = None
        if spec.can_recover_params and summary.parameters is not None:
            est_params = summary.parameters
            if len(est_params) == len(true_params):
                param_rmse = (
                    ((est_params - true_params) ** 2).float().mean().sqrt().item()
                )
                for i, name in enumerate(env.parameter_names):
                    estimates[name] = est_params[i].item()

        # Policy RMSE
        policy_rmse = float("nan")
        if summary.policy is not None:
            policy_rmse = (
                ((summary.policy - true_policy) ** 2).float().mean().sqrt().item()
            )

        # % of optimal value (works for all estimators with a policy)
        pct_optimal = float("nan")
        if summary.policy is not None:
            pct_optimal = _evaluate_pct_optimal(
                summary.policy, true_utility, transitions, problem,
            )

        # Reward matrices for visualization
        estimated_reward = None
        if summary.parameters is not None:
            n_s, n_a = problem.num_states, problem.num_actions
            if spec.name == "GCL":
                # GCL returns cost parameters c(s,a) — negate to get rewards
                estimated_reward = -summary.parameters.reshape(n_s, n_a)
            elif spec.name in ("Deep MaxEnt", "f-IRL"):
                # These return reward matrix R(s,a) directly
                estimated_reward = summary.parameters.reshape(n_s, n_a)
            elif len(summary.parameters) == len(true_params):
                estimated_reward = torch.einsum(
                    "sak,k->sa", env.feature_matrix, summary.parameters
                )

        # Transfer % of optimal
        pct_optimal_transfer = None
        if estimated_reward is not None:
            # Re-solve MDP with estimated rewards under transfer dynamics
            pct_optimal_transfer = _compute_transfer_pct_optimal(
                estimated_reward, true_utility, problem, dgp,
            )
        elif summary.policy is not None:
            # No reward recovery — evaluate learned policy directly on
            # transfer dynamics (e.g. behavioral cloning baseline)
            pct_optimal_transfer = _evaluate_policy_transfer(
                summary.policy, true_utility, problem, dgp,
            )

        return BenchmarkResult(
            estimator=spec.name,
            n_states=dgp.n_states,
            n_agents=n_agents,
            seed=seed,
            param_rmse=param_rmse,
            policy_rmse=policy_rmse,
            pct_optimal=pct_optimal,
            time_seconds=elapsed,
            converged=summary.converged,
            pct_optimal_transfer=pct_optimal_transfer,
            estimates=estimates,
            true_params=true_dict,
            estimated_reward=estimated_reward,
            learned_policy=summary.policy if summary.policy is not None else None,
            true_reward=true_utility,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return BenchmarkResult(
            estimator=spec.name,
            n_states=dgp.n_states,
            n_agents=n_agents,
            seed=seed,
            param_rmse=None,
            policy_rmse=float("nan"),
            pct_optimal=float("nan"),
            time_seconds=elapsed,
            converged=False,
            true_params=true_dict,
        )


def run_benchmark(
    dgp: BenchmarkDGP | None = None,
    n_states_list: list[int] | None = None,
    n_agents_list: list[int] | None = None,
    seeds: list[int] | None = None,
    estimators: list[EstimatorSpec] | None = None,
    n_periods: int = 100,
) -> pd.DataFrame:
    """Run the full benchmark across state sizes, sample sizes, and seeds.

    Args:
        dgp: Base DGP (defaults to BenchmarkDGP()).
        n_states_list: State space sizes to test (defaults to [20]).
        n_agents_list: Sample sizes to test (defaults to [200]).
        seeds: Random seeds (defaults to [42]).
        estimators: Estimator specs (defaults to all 10).
        n_periods: Periods per agent.

    Returns:
        DataFrame with one row per (estimator, n_states, n_agents, seed).
    """
    if dgp is None:
        dgp = BenchmarkDGP()
    if n_states_list is None:
        n_states_list = [dgp.n_states]
    if n_agents_list is None:
        n_agents_list = [200]
    if seeds is None:
        seeds = [42]
    if estimators is None:
        estimators = get_default_estimator_specs()

    results: list[BenchmarkResult] = []
    for n_states in n_states_list:
        for n_agents in n_agents_list:
            for seed in seeds:
                current_dgp = BenchmarkDGP(
                    n_states=n_states,
                    replacement_cost=dgp.replacement_cost,
                    operating_cost=dgp.operating_cost,
                    quadratic_cost=dgp.quadratic_cost,
                    discount_factor=dgp.discount_factor,
                    transfer_transition_probs=dgp.transfer_transition_probs,
                )
                for spec in estimators:
                    result = run_single(
                        current_dgp, spec, n_agents=n_agents,
                        n_periods=n_periods, seed=seed,
                    )
                    results.append(result)

    rows = []
    for r in results:
        rows.append({
            "estimator": r.estimator,
            "n_states": r.n_states,
            "n_agents": r.n_agents,
            "seed": r.seed,
            "param_rmse": r.param_rmse,
            "policy_rmse": r.policy_rmse,
            "pct_optimal": r.pct_optimal,
            "pct_optimal_transfer": r.pct_optimal_transfer,
            "time_seconds": r.time_seconds,
            "converged": r.converged,
            **{f"est_{k}": v for k, v in r.estimates.items()},
            **{f"true_{k}": v for k, v in r.true_params.items()},
        })
    return pd.DataFrame(rows)


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize benchmark results by estimator.

    Args:
        df: DataFrame from run_benchmark().

    Returns:
        Summary DataFrame with mean/std metrics per estimator.
    """
    agg = df.groupby("estimator").agg(
        param_rmse_mean=("param_rmse", "mean"),
        param_rmse_std=("param_rmse", "std"),
        policy_rmse_mean=("policy_rmse", "mean"),
        policy_rmse_std=("policy_rmse", "std"),
        pct_optimal_mean=("pct_optimal", "mean"),
        pct_optimal_std=("pct_optimal", "std"),
        pct_optimal_transfer_mean=("pct_optimal_transfer", "mean"),
        pct_optimal_transfer_std=("pct_optimal_transfer", "std"),
        time_mean=("time_seconds", "mean"),
        time_std=("time_seconds", "std"),
        converged_pct=("converged", "mean"),
        n_runs=("converged", "count"),
    )
    return agg.sort_values("policy_rmse_mean")

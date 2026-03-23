"""Convergence tracking for benchmark estimators.

Runs an estimator at increasing iteration checkpoints to profile
how quickly parameter/policy RMSE decreases over optimization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from econirl.environments import MultiComponentBusEnvironment
from econirl.evaluation.adapters import build_utility_for_estimator
from econirl.evaluation.benchmark import BenchmarkDGP, EstimatorSpec, _make_env
from econirl.evaluation.utils import compute_policy
from econirl.simulation.synthetic import simulate_panel

# Config parameter names for max iterations, by estimator class
_ITER_PARAM_MAP: dict[str, str] = {
    "NFXPEstimator": "outer_max_iter",
    "CCPEstimator": "num_policy_iterations",
    "MCEIRLEstimator": "outer_max_iter",
    "MaxEntIRLEstimator": "outer_max_iter",
    "MaxMarginPlanningEstimator": "max_iterations",
    "TDCCPEstimator": "outer_max_iter",
    "GLADIUSEstimator": "max_epochs",
    "GAILEstimator": "max_rounds",
    "AIRLEstimator": "max_rounds",
    "GCLEstimator": "max_iterations",
}


@dataclass
class ConvergenceProfile:
    """Convergence profile from running an estimator at checkpoints.

    Attributes:
        estimator: Name of the estimator.
        iterations: Iteration counts at each checkpoint.
        param_rmse: Parameter RMSE at each checkpoint (None if N/A).
        policy_rmse: Policy RMSE at each checkpoint.
        time_seconds: Cumulative wall-clock time at each checkpoint.
    """

    estimator: str
    iterations: list[int] = field(default_factory=list)
    param_rmse: list[float | None] = field(default_factory=list)
    policy_rmse: list[float] = field(default_factory=list)
    time_seconds: list[float] = field(default_factory=list)


def track_convergence(
    dgp: BenchmarkDGP,
    spec: EstimatorSpec,
    checkpoints: list[int] | None = None,
    n_agents: int = 200,
    n_periods: int = 100,
    seed: int = 42,
) -> ConvergenceProfile:
    """Track estimator convergence by running at iteration checkpoints.

    Re-runs the estimator from scratch at each checkpoint with
    increasing max iterations.

    Args:
        dgp: Data generating process.
        spec: Estimator specification.
        checkpoints: Iteration counts to evaluate at.
        n_agents: Number of simulated agents.
        n_periods: Periods per agent.
        seed: Random seed.

    Returns:
        ConvergenceProfile with metrics at each checkpoint.
    """
    if checkpoints is None:
        checkpoints = [10, 25, 50, 100, 200, 500]

    env = _make_env(dgp)
    panel = simulate_panel(env, n_individuals=n_agents, n_periods=n_periods, seed=seed)
    problem = env.problem_spec
    transitions = env.transition_matrices

    true_params = env.get_true_parameter_vector()
    true_policy = compute_policy(true_params, problem, transitions, env.feature_matrix)

    # Determine which kwarg controls max iterations
    class_name = spec.estimator_class.__name__
    iter_param = _ITER_PARAM_MAP.get(class_name, "outer_max_iter")

    profile = ConvergenceProfile(estimator=spec.name)

    for n_iter in checkpoints:
        kwargs = dict(spec.kwargs)
        kwargs[iter_param] = n_iter

        utility = build_utility_for_estimator(env, spec.estimator_class)

        t0 = time.perf_counter()
        try:
            summary = spec.estimator_class(**kwargs).estimate(
                panel, utility, problem, transitions
            )
            elapsed = time.perf_counter() - t0

            # Parameter RMSE
            p_rmse = None
            if spec.can_recover_params and summary.parameters is not None:
                est = summary.parameters
                if len(est) == len(true_params):
                    p_rmse = ((est - true_params) ** 2).float().mean().sqrt().item()

            # Policy RMSE
            pol_rmse = float("nan")
            if summary.policy is not None:
                pol_rmse = (
                    ((summary.policy - true_policy) ** 2).float().mean().sqrt().item()
                )

            profile.iterations.append(n_iter)
            profile.param_rmse.append(p_rmse)
            profile.policy_rmse.append(pol_rmse)
            profile.time_seconds.append(elapsed)

        except Exception:
            elapsed = time.perf_counter() - t0
            profile.iterations.append(n_iter)
            profile.param_rmse.append(None)
            profile.policy_rmse.append(float("nan"))
            profile.time_seconds.append(elapsed)

    return profile

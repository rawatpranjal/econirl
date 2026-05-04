"""Known-truth synthetic validation harness.

This module is experiment infrastructure, deliberately kept out of the
public package API. It defines one adaptable DGP, exact truth objects,
pre-estimation checks, estimator contracts, hard recovery gates, and a
small CLI for oracle and estimator runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.occupancy import (
    compute_state_action_visitation,
    compute_state_visitation,
)
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.inference.results import EstimationSummary
from econirl.preferences.action_reward import ActionDependentReward

# --- Configuration ---
StateMode = Literal["low_dim", "high_dim"]
RewardMode = Literal["state_only", "action_dependent"]
RewardDim = Literal["low", "high"]
HeterogeneityMode = Literal["none", "latent_segments"]
InitialStateMode = Literal["start", "uniform_regular"]
CounterfactualKind = Literal["type_a", "type_b", "type_c"]


@dataclass(frozen=True)
class KnownTruthDGPConfig:
    """Specification for the configurable known-truth DGP.

    The state space is represented by discrete indices so exact Bellman
    solutions remain available. High-dimensional modes expose richer state
    encodings and reward features on top of that common grid.
    """

    state_mode: StateMode = "low_dim"
    reward_mode: RewardMode = "action_dependent"
    reward_dim: RewardDim = "low"
    heterogeneity: HeterogeneityMode = "none"
    num_regular_states: int = 20
    num_actions: int = 3
    high_state_dim: int = 12
    high_reward_features: int = 24
    num_segments: int = 2
    discount_factor: float = 0.95
    scale_parameter: float = 1.0
    seed: int = 42
    initial_state_mode: InitialStateMode = "uniform_regular"
    exit_action: int = 2
    transition_noise: float = 0.05
    feature_scale: float = 1.0

    @property
    def num_states(self) -> int:
        return self.num_regular_states + 1

    @property
    def absorbing_state(self) -> int:
        return self.num_regular_states

    @property
    def uses_exit_anchor(self) -> bool:
        return 0 <= self.exit_action < self.num_actions

    def validate(self) -> None:
        if self.num_regular_states < 3:
            raise ValueError("num_regular_states must be at least 3")
        if self.num_actions < 2:
            raise ValueError("num_actions must be at least 2")
        if not 0 <= self.discount_factor < 1:
            raise ValueError("discount_factor must be in [0, 1)")
        if self.scale_parameter <= 0:
            raise ValueError("scale_parameter must be positive")
        if self.heterogeneity == "latent_segments" and self.num_segments < 2:
            raise ValueError("latent_segments requires num_segments >= 2")
        if self.reward_dim == "high" and self.high_reward_features < 8:
            raise ValueError("high_reward_features must be at least 8")
        if self.state_mode == "high_dim" and self.high_state_dim < 4:
            raise ValueError("high_state_dim must be at least 4")
        if self.uses_exit_anchor and self.exit_action >= self.num_actions:
            raise ValueError("exit_action must be a valid action index")
        if not 0 <= self.transition_noise < 1:
            raise ValueError("transition_noise must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationConfig:
    """Panel simulation controls."""

    n_individuals: int = 500
    n_periods: int = 80
    seed: int = 42
    show_progress: bool = False


@dataclass(frozen=True)
class CounterfactualConfig:
    """Default oracle counterfactual controls."""

    type_a_shift: float = -0.25
    type_b_skip: int = 2
    type_c_action: int = 1
    type_c_penalty: float = -1_000.0


# --- DGP and Simulation ---
@dataclass(frozen=True)
class KnownTruthDGP:
    """A fully specified synthetic DGP with all truth objects exposed."""

    config: KnownTruthDGPConfig
    problem: DDCProblem
    transitions: jnp.ndarray
    feature_matrix: jnp.ndarray
    state_features: jnp.ndarray
    parameter_names: list[str]
    true_parameters: jnp.ndarray
    reward_matrix: jnp.ndarray
    initial_distribution: jnp.ndarray
    segment_probabilities: jnp.ndarray | None = None

    @property
    def num_segments(self) -> int:
        if self.true_parameters.ndim == 1:
            return 1
        return int(self.true_parameters.shape[0])

    @property
    def homogeneous_parameters(self) -> jnp.ndarray:
        if self.true_parameters.ndim == 1:
            return self.true_parameters
        weights = self.segment_probabilities
        if weights is None:
            weights = jnp.ones(self.num_segments) / self.num_segments
        return jnp.einsum("g,gk->k", weights, self.true_parameters)

    @property
    def homogeneous_reward(self) -> jnp.ndarray:
        if self.reward_matrix.ndim == 2:
            return self.reward_matrix
        weights = self.segment_probabilities
        if weights is None:
            weights = jnp.ones(self.num_segments) / self.num_segments
        return jnp.einsum("g,gsa->sa", weights, self.reward_matrix)

    def utility(self) -> ActionDependentReward:
        return ActionDependentReward(self.feature_matrix, self.parameter_names)

    def metadata(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "parameter_names": self.parameter_names,
            "true_parameters": np.asarray(self.true_parameters).tolist(),
            "segment_probabilities": (
                None
                if self.segment_probabilities is None
                else np.asarray(self.segment_probabilities).tolist()
            ),
        }


def build_known_truth_dgp(config: KnownTruthDGPConfig | None = None) -> KnownTruthDGP:
    """Build a known-truth DGP from a config."""

    if config is None:
        config = KnownTruthDGPConfig()
    config.validate()

    transitions = _build_transitions(config)
    state_features = _build_state_features(config)
    feature_matrix, parameter_names = _build_reward_features(config, state_features)
    true_parameters = _build_parameters(config, len(parameter_names))
    reward_matrix = _compute_rewards(feature_matrix, true_parameters)
    initial_distribution = _build_initial_distribution(config)

    problem = DDCProblem(
        num_states=config.num_states,
        num_actions=config.num_actions,
        discount_factor=config.discount_factor,
        scale_parameter=config.scale_parameter,
        state_dim=int(state_features.shape[1]),
        state_encoder=lambda states: state_features[states],
    )

    segment_probabilities = None
    if config.heterogeneity == "latent_segments":
        segment_probabilities = jnp.ones(config.num_segments, dtype=jnp.float32)
        segment_probabilities = segment_probabilities / segment_probabilities.sum()

    return KnownTruthDGP(
        config=config,
        problem=problem,
        transitions=transitions,
        feature_matrix=feature_matrix,
        state_features=state_features,
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        reward_matrix=reward_matrix,
        initial_distribution=initial_distribution,
        segment_probabilities=segment_probabilities,
    )


def simulate_known_truth_panel(
    dgp: KnownTruthDGP,
    config: SimulationConfig | None = None,
) -> Panel:
    """Simulate panel data from the known optimal policy."""


    if config is None:
        config = SimulationConfig()

    rng = np.random.default_rng(config.seed)
    solutions = [
        solve_known_truth(dgp, segment_index=g)
        for g in range(dgp.num_segments)
    ]

    iterator = range(config.n_individuals)
    if config.show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="simulate known-truth panel")

    trajectories: list[Trajectory] = []
    segment_labels: list[int] = []
    segment_probs = (
        np.asarray(dgp.segment_probabilities)
        if dgp.segment_probabilities is not None
        else np.ones(1)
    )

    for i in iterator:
        segment = int(rng.choice(dgp.num_segments, p=segment_probs))
        segment_labels.append(segment)
        policy = np.asarray(solutions[segment].policy)

        state = int(rng.choice(dgp.problem.num_states, p=np.asarray(dgp.initial_distribution)))
        states = np.empty(config.n_periods, dtype=np.int32)
        actions = np.empty(config.n_periods, dtype=np.int32)
        next_states = np.empty(config.n_periods, dtype=np.int32)

        for t in range(config.n_periods):
            action_probs = policy[state].astype(np.float64)
            action_probs = action_probs / action_probs.sum()
            action = int(rng.choice(dgp.problem.num_actions, p=action_probs))
            transition_probs = np.asarray(dgp.transitions[action, state], dtype=np.float64)
            transition_probs = transition_probs / transition_probs.sum()
            next_state = int(rng.choice(dgp.problem.num_states, p=transition_probs))

            states[t] = state
            actions[t] = action
            next_states[t] = next_state
            state = next_state

        trajectories.append(
            Trajectory(
                states=jnp.array(states, dtype=jnp.int32),
                actions=jnp.array(actions, dtype=jnp.int32),
                next_states=jnp.array(next_states, dtype=jnp.int32),
                individual_id=i,
                metadata={"segment": segment},
            )
        )

    metadata = dgp.metadata()
    metadata.update(
        {
            "simulation": {
                "n_individuals": config.n_individuals,
                "n_periods": config.n_periods,
                "seed": config.seed,
            },
            "segment_labels": segment_labels,
        }
    )
    return Panel(trajectories=trajectories, metadata=metadata)


def _build_initial_distribution(config: KnownTruthDGPConfig) -> jnp.ndarray:
    dist = np.zeros(config.num_states, dtype=np.float64)
    if config.initial_state_mode == "start":
        dist[0] = 1.0
    else:
        dist[: config.num_regular_states] = 1.0 / config.num_regular_states
    return jnp.array(dist, dtype=jnp.float32)


def _build_transitions(config: KnownTruthDGPConfig) -> jnp.ndarray:
    transitions = np.zeros(
        (config.num_actions, config.num_states, config.num_states), dtype=np.float64
    )
    absorbing = config.absorbing_state
    non_exit_actions = [a for a in range(config.num_actions) if a != config.exit_action]

    for state in range(config.num_regular_states):
        for action in range(config.num_actions):
            if action == config.exit_action:
                transitions[action, state, absorbing] = 1.0
                continue

            action_position = non_exit_actions.index(action)
            if action_position == 0:
                target = min(state + 1, config.num_regular_states - 1)
            elif action_position == 1:
                target = state
            elif action_position % 2 == 0:
                target = min(state + 2, config.num_regular_states - 1)
            else:
                target = max(state - 1, 0)

            noise = config.transition_noise
            transitions[action, state, target] += 1.0 - noise
            if noise > 0:
                left = max(target - 1, 0)
                right = min(target + 1, config.num_regular_states - 1)
                if left == right:
                    transitions[action, state, target] += noise
                else:
                    transitions[action, state, left] += noise / 2.0
                    transitions[action, state, right] += noise / 2.0

    transitions[:, absorbing, absorbing] = 1.0
    transitions = transitions / transitions.sum(axis=2, keepdims=True)
    return jnp.array(transitions, dtype=jnp.float32)


def _build_state_features(config: KnownTruthDGPConfig) -> jnp.ndarray:
    regular = config.num_regular_states
    progress = np.linspace(0.0, 1.0, regular, dtype=np.float64)

    if config.state_mode == "low_dim":
        features = np.column_stack(
            [
                progress,
                np.sin(2.0 * np.pi * progress),
            ]
        )
    else:
        rng = np.random.default_rng(config.seed + 17)
        cols = [
            progress,
            progress**2,
            progress**3,
            np.sin(2.0 * np.pi * progress),
            np.cos(2.0 * np.pi * progress),
        ]
        while len(cols) < config.high_state_dim:
            freq = rng.uniform(0.5, 4.0)
            phase = rng.uniform(-np.pi, np.pi)
            cols.append(np.sin(freq * np.pi * progress + phase))
        features = np.column_stack(cols[: config.high_state_dim])

    absorbing = np.zeros((1, features.shape[1]), dtype=np.float64)
    features = np.vstack([features, absorbing])
    return jnp.array(features * config.feature_scale, dtype=jnp.float32)


def _build_reward_features(
    config: KnownTruthDGPConfig,
    state_features: jnp.ndarray,
) -> tuple[jnp.ndarray, list[str]]:
    S, A = config.num_states, config.num_actions
    non_exit_actions = [a for a in range(A) if a != config.exit_action]
    progress = np.asarray(state_features[:, 0])
    wave = np.asarray(state_features[:, 1]) if state_features.shape[1] > 1 else progress

    if config.reward_dim == "low":
        if config.reward_mode == "state_only":
            names = ["state_intercept", "state_progress", "state_wave"]
            features = np.zeros((S, A, len(names)), dtype=np.float64)
            base = np.column_stack([np.ones(S), progress, wave])
            for action in non_exit_actions:
                features[:, action, :] = base
        else:
            names = []
            features = np.zeros((S, A, 2 * len(non_exit_actions)), dtype=np.float64)
            col = 0
            for action in non_exit_actions:
                names.extend([f"action_{action}_intercept", f"action_{action}_progress"])
                features[:, action, col] = 1.0
                features[:, action, col + 1] = progress
                col += 2
            features[:, :, :] += 0.05 * wave[:, None, None]
            features[:, config.exit_action, :] = 0.0
    else:
        K = config.high_reward_features
        names = [f"theta_{k}" for k in range(K)]
        features = np.zeros((S, A, K), dtype=np.float64)
        basis = _expand_state_basis(np.asarray(state_features), K)
        if config.reward_mode == "state_only":
            for action in non_exit_actions:
                features[:, action, :] = basis
        else:
            rng = np.random.default_rng(config.seed + 31)
            action_embeddings = rng.normal(size=(A, K))
            action_embeddings = action_embeddings / np.maximum(
                np.linalg.norm(action_embeddings, axis=1, keepdims=True), 1e-8
            )
            for action in non_exit_actions:
                features[:, action, :] = basis * (1.0 + 0.5 * action_embeddings[action])
        features[:, config.exit_action, :] = 0.0

    features[config.absorbing_state, :, :] = 0.0
    return jnp.array(features, dtype=jnp.float32), names


def _expand_state_basis(state_features: np.ndarray, n_features: int) -> np.ndarray:
    cols = [np.ones(state_features.shape[0])]
    for j in range(state_features.shape[1]):
        cols.append(state_features[:, j])
        if len(cols) >= n_features:
            break
        cols.append(state_features[:, j] ** 2)
        if len(cols) >= n_features:
            break
    j = 0
    while len(cols) < n_features:
        cols.append(np.sin((j + 1) * state_features[:, j % state_features.shape[1]]))
        j += 1
    basis = np.column_stack(cols[:n_features])
    scale = np.maximum(np.std(basis, axis=0, keepdims=True), 1e-8)
    basis = (basis - basis.mean(axis=0, keepdims=True)) / scale
    basis[:, 0] = 1.0
    return basis


def _build_parameters(config: KnownTruthDGPConfig, n_params: int) -> jnp.ndarray:
    rng = np.random.default_rng(config.seed + 53)
    if config.reward_dim == "low":
        if config.reward_mode == "action_dependent":
            non_exit_actions = [
                action
                for action in range(config.num_actions)
                if action != config.exit_action
            ]
            intercepts = np.linspace(0.10, 0.00, len(non_exit_actions))
            slopes = np.linspace(0.50, -0.20, len(non_exit_actions))
            base = np.empty(n_params, dtype=np.float64)
            for idx in range(len(non_exit_actions)):
                base[2 * idx] = intercepts[idx]
                base[2 * idx + 1] = slopes[idx]
        else:
            base = np.array([0.15, 0.40, -0.20], dtype=np.float64)[:n_params]
    else:
        base = rng.normal(size=n_params) / np.sqrt(n_params)
        keep = max(4, n_params // 3)
        base[keep:] *= 0.25

    if config.heterogeneity == "none":
        return jnp.array(base, dtype=jnp.float32)

    segments = []
    for g in range(config.num_segments):
        direction = -1.0 if g % 2 else 1.0
        perturb = direction * 0.35 * np.roll(base, g + 1)
        segments.append(base + perturb)
    return jnp.array(np.vstack(segments), dtype=jnp.float32)


def _compute_rewards(feature_matrix: jnp.ndarray, parameters: jnp.ndarray) -> jnp.ndarray:
    if parameters.ndim == 1:
        return jnp.einsum("sak,k->sa", feature_matrix, parameters)
    return jnp.einsum("sak,gk->gsa", feature_matrix, parameters)


# --- Truth Solver ---
@dataclass(frozen=True)
class KnownTruthSolution:
    """Exact Bellman solution and occupancy objects for one segment."""

    segment_index: int
    reward_matrix: jnp.ndarray
    Q: jnp.ndarray
    V: jnp.ndarray
    policy: jnp.ndarray
    state_occupancy: jnp.ndarray
    state_action_occupancy: jnp.ndarray
    converged: bool
    num_iterations: int
    final_error: float


def get_segment_reward(dgp: KnownTruthDGP, segment_index: int = 0) -> jnp.ndarray:
    if dgp.reward_matrix.ndim == 2:
        if segment_index != 0:
            raise IndexError("homogeneous DGP has only segment 0")
        return dgp.reward_matrix
    if not 0 <= segment_index < dgp.reward_matrix.shape[0]:
        raise IndexError(f"segment_index {segment_index} is out of range")
    return dgp.reward_matrix[segment_index]


def solve_known_truth(
    dgp: KnownTruthDGP,
    segment_index: int = 0,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> KnownTruthSolution:
    """Solve the DGP exactly under the true reward."""

    reward = get_segment_reward(dgp, segment_index)
    operator = SoftBellmanOperator(dgp.problem, dgp.transitions)
    result = value_iteration(operator, reward, tol=tol, max_iter=max_iter)
    state_occ = compute_state_visitation(
        result.policy,
        dgp.transitions,
        dgp.problem,
        dgp.initial_distribution,
    )
    state_action_occ = compute_state_action_visitation(
        result.policy,
        dgp.transitions,
        dgp.problem,
        dgp.initial_distribution,
    )
    return KnownTruthSolution(
        segment_index=segment_index,
        reward_matrix=reward,
        Q=result.Q,
        V=result.V,
        policy=result.policy,
        state_occupancy=state_occ,
        state_action_occupancy=state_action_occ,
        converged=result.converged,
        num_iterations=result.num_iterations,
        final_error=result.final_error,
    )


# --- Counterfactual Oracles ---
@dataclass(frozen=True)
class CounterfactualDGP:
    """A counterfactual environment derived from a baseline DGP."""

    kind: CounterfactualKind
    description: str
    baseline: KnownTruthDGP
    reward_matrix: jnp.ndarray
    transitions: jnp.ndarray
    disabled_action: int | None = None


@dataclass(frozen=True)
class CounterfactualOracle:
    """Baseline and counterfactual oracle solutions for one segment."""

    counterfactual: CounterfactualDGP
    segment_index: int
    baseline_solution: KnownTruthSolution
    counterfactual_solution: KnownTruthSolution


def build_counterfactual(
    dgp: KnownTruthDGP,
    kind: CounterfactualKind,
    config: CounterfactualConfig | None = None,
) -> CounterfactualDGP:
    """Build a Type A, Type B, or Type C counterfactual DGP."""

    if config is None:
        config = CounterfactualConfig()

    reward = dgp.reward_matrix
    transitions = dgp.transitions
    disabled_action = None

    if kind == "type_a":
        shift = _state_shift(dgp, config.type_a_shift)
        reward = reward + shift
        description = "Type A reward feature shift with baseline transitions"
    elif kind == "type_b":
        transitions = _skip_transitions(dgp, config.type_b_skip)
        description = "Type B transition change with baseline reward"
    elif kind == "type_c":
        disabled_action = config.type_c_action
        reward = _penalize_action(dgp, reward, disabled_action, config.type_c_penalty)
        description = "Type C action design intervention by disabling one action"
    else:
        raise ValueError(f"unknown counterfactual kind {kind!r}")

    return CounterfactualDGP(
        kind=kind,
        description=description,
        baseline=dgp,
        reward_matrix=reward,
        transitions=transitions,
        disabled_action=disabled_action,
    )


def solve_counterfactual_oracle(
    dgp: KnownTruthDGP,
    kind: CounterfactualKind,
    segment_index: int = 0,
    config: CounterfactualConfig | None = None,
) -> CounterfactualOracle:
    """Solve baseline and counterfactual policies for one segment."""

    counterfactual = build_counterfactual(dgp, kind, config)
    cf_dgp = _replace_truth_objects(
        dgp,
        reward_matrix=counterfactual.reward_matrix,
        transitions=counterfactual.transitions,
    )
    return CounterfactualOracle(
        counterfactual=counterfactual,
        segment_index=segment_index,
        baseline_solution=solve_known_truth(dgp, segment_index),
        counterfactual_solution=solve_known_truth(cf_dgp, segment_index),
    )


def _replace_truth_objects(
    dgp: KnownTruthDGP,
    reward_matrix: jnp.ndarray,
    transitions: jnp.ndarray,
) -> KnownTruthDGP:
    return KnownTruthDGP(
        config=dgp.config,
        problem=dgp.problem,
        transitions=transitions,
        feature_matrix=dgp.feature_matrix,
        state_features=dgp.state_features,
        parameter_names=dgp.parameter_names,
        true_parameters=dgp.true_parameters,
        reward_matrix=reward_matrix,
        initial_distribution=dgp.initial_distribution,
        segment_probabilities=dgp.segment_probabilities,
    )


def _state_shift(dgp: KnownTruthDGP, amount: float) -> jnp.ndarray:
    progress = dgp.state_features[:, 0]
    regular_mask = jnp.arange(dgp.problem.num_states) != dgp.config.absorbing_state
    action_mask = jnp.ones(dgp.problem.num_actions, dtype=jnp.float32)
    if dgp.config.uses_exit_anchor:
        action_mask = action_mask.at[dgp.config.exit_action].set(0.0)
    shift = amount * progress[:, None] * action_mask[None, :]
    shift = jnp.where(regular_mask[:, None], shift, 0.0)
    if dgp.reward_matrix.ndim == 3:
        return shift[None, :, :]
    return shift


def _skip_transitions(dgp: KnownTruthDGP, skip: int) -> jnp.ndarray:
    transitions = np.asarray(dgp.transitions).copy()
    advance_action = 0
    if advance_action == dgp.config.exit_action:
        advance_action = 1
    transitions[advance_action, :, :] = 0.0
    absorbing = dgp.config.absorbing_state
    for state in range(dgp.config.num_regular_states):
        target = min(state + skip, dgp.config.num_regular_states - 1)
        transitions[advance_action, state, target] = 1.0
    transitions[advance_action, absorbing, absorbing] = 1.0
    transitions = transitions / transitions.sum(axis=2, keepdims=True)
    return jnp.array(transitions, dtype=jnp.float32)


def _penalize_action(
    dgp: KnownTruthDGP,
    reward: jnp.ndarray,
    action: int,
    penalty: float,
) -> jnp.ndarray:
    if not 0 <= action < dgp.problem.num_actions:
        raise ValueError(f"action {action} is out of range")
    if action == dgp.config.exit_action:
        raise ValueError("Type C should not disable the anchor exit action")
    if reward.ndim == 2:
        return reward.at[: dgp.config.num_regular_states, action].add(penalty)
    return reward.at[:, : dgp.config.num_regular_states, action].add(penalty)


# --- Pre-Estimation Diagnostics ---
@dataclass(frozen=True)
class PreEstimationDiagnostics:
    """Diagnostics that should be checked before estimator execution."""

    feature_rank: int
    num_features: int
    condition_number: float
    is_action_dependent: bool
    max_transition_row_error: float
    observed_states: int | None = None
    num_states: int | None = None
    single_action_states: int | None = None
    state_action_coverage: float | None = None
    action_shares: list[float] | None = None
    min_action_share: float | None = None
    min_positive_ccp: float | None = None
    anchor_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


def run_pre_estimation_diagnostics(
    dgp: KnownTruthDGP,
    panel: Panel | None = None,
    condition_threshold: float = 1e6,
) -> PreEstimationDiagnostics:
    """Run structural and sample diagnostics before fitting an estimator."""

    features = np.asarray(dgp.feature_matrix, dtype=np.float64)
    flat_features = features.reshape(-1, features.shape[-1])
    nonzero_rows = flat_features[np.linalg.norm(flat_features, axis=1) > 1e-12]
    if nonzero_rows.size == 0:
        feature_rank = 0
        condition_number = float("inf")
    else:
        feature_rank = int(np.linalg.matrix_rank(nonzero_rows))
        condition_number = _safe_condition_number(nonzero_rows)

    transitions = np.asarray(dgp.transitions, dtype=np.float64)
    row_sums = transitions.sum(axis=2)
    max_transition_row_error = float(np.max(np.abs(row_sums - 1.0)))

    non_exit_actions = [a for a in range(dgp.problem.num_actions) if a != dgp.config.exit_action]
    action_features = features[:, non_exit_actions, :]
    action_reference = action_features[:, :1, :]
    action_diff = np.max(np.abs(action_features - action_reference))
    is_action_dependent = bool(action_diff > 1e-8)

    anchor_valid = True
    if dgp.config.uses_exit_anchor:
        exit_reward = np.asarray(dgp.homogeneous_reward[:, dgp.config.exit_action])
        absorbing_reward = np.asarray(dgp.homogeneous_reward[dgp.config.absorbing_state, :])
        exit_transitions = transitions[dgp.config.exit_action, : dgp.config.num_regular_states]
        anchor_target = exit_transitions[:, dgp.config.absorbing_state]
        anchor_valid = bool(
            np.max(np.abs(exit_reward)) < 1e-6
            and np.max(np.abs(absorbing_reward)) < 1e-6
            and np.min(anchor_target) > 1.0 - 1e-6
        )

    errors: list[str] = []
    warnings: list[str] = []
    if feature_rank < features.shape[-1]:
        errors.append(
            f"feature rank {feature_rank} is less than {features.shape[-1]} features"
        )
    if condition_number > condition_threshold:
        warnings.append(f"feature condition number {condition_number:.3g} is high")
    if max_transition_row_error > 1e-6:
        errors.append(
            f"transition rows are not stochastic, max error {max_transition_row_error:.3g}"
        )
    if not anchor_valid:
        errors.append("exit or absorbing-state anchor is invalid")

    observed_states = None
    single_action_states = None
    coverage = None
    action_shares = None
    min_action_share = None
    min_positive_ccp = None
    if panel is not None:
        states = np.asarray(panel.get_all_states(), dtype=np.int64)
        actions = np.asarray(panel.get_all_actions(), dtype=np.int64)
        counts = np.zeros((dgp.problem.num_states, dgp.problem.num_actions), dtype=np.float64)
        for state, action in zip(states, actions):
            counts[state, action] += 1.0
        action_counts = counts.sum(axis=0)
        action_shares = (action_counts / max(action_counts.sum(), 1.0)).tolist()
        min_action_share = float(np.min(action_shares))
        observed_state_mask = counts.sum(axis=1) > 0
        observed_states = int(observed_state_mask.sum())
        single_action_states = int(
            np.logical_and((counts > 0).sum(axis=1) == 1, observed_state_mask).sum()
        )
        coverage = float((counts > 0).sum() / counts.size)
        positive = counts[counts > 0]
        min_positive_ccp = None
        if positive.size:
            row_sums = counts.sum(axis=1, keepdims=True)
            ccps = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
            min_positive_ccp = float(ccps[ccps > 0].min())
        if observed_states < dgp.problem.num_states:
            warnings.append(
                f"{observed_states} of {dgp.problem.num_states} states are observed"
            )
        if single_action_states > 0:
            warnings.append(f"{single_action_states} observed states have one action")
        if min_action_share < 0.02:
            warnings.append(f"minimum action share {min_action_share:.3g} is very low")

    return PreEstimationDiagnostics(
        feature_rank=feature_rank,
        num_features=features.shape[-1],
        condition_number=condition_number,
        is_action_dependent=is_action_dependent,
        max_transition_row_error=max_transition_row_error,
        observed_states=observed_states,
        num_states=dgp.problem.num_states if panel is not None else None,
        single_action_states=single_action_states,
        state_action_coverage=coverage,
        action_shares=action_shares,
        min_action_share=min_action_share,
        min_positive_ccp=min_positive_ccp,
        anchor_valid=anchor_valid,
        errors=errors,
        warnings=warnings,
    )


def _safe_condition_number(x: np.ndarray) -> float:
    try:
        value = float(np.linalg.cond(x))
    except np.linalg.LinAlgError:
        value = float("inf")
    if not np.isfinite(value):
        return float("inf")
    return value


# --- Recovery Metrics ---
@dataclass(frozen=True)
class PolicyMetrics:
    l1: float
    linf: float
    tv: float
    kl: float


@dataclass(frozen=True)
class CounterfactualMetrics:
    policy: PolicyMetrics
    value_rmse: float
    regret: float


@dataclass(frozen=True)
class ParameterMetrics:
    rmse: float
    relative_rmse: float
    max_abs_error: float
    cosine_similarity: float


def rmse(estimated: jnp.ndarray, truth: jnp.ndarray) -> float:
    estimated = jnp.asarray(estimated)
    truth = jnp.asarray(truth)
    return float(jnp.sqrt(jnp.mean((estimated - truth) ** 2)))


def parameter_metrics(
    truth: jnp.ndarray,
    estimated: jnp.ndarray,
    eps: float = 1e-12,
) -> ParameterMetrics:
    """Compare estimated structural parameters to known truth."""

    truth = jnp.asarray(truth)
    estimated = jnp.asarray(estimated)
    error = estimated - truth
    error_rmse = jnp.sqrt(jnp.mean(error**2))
    truth_rms = jnp.sqrt(jnp.mean(truth**2))
    max_abs = jnp.max(jnp.abs(error))
    denom = jnp.linalg.norm(truth) * jnp.linalg.norm(estimated)
    cosine = jnp.where(
        denom > eps,
        jnp.dot(truth, estimated) / denom,
        jnp.nan,
    )
    return ParameterMetrics(
        rmse=float(error_rmse),
        relative_rmse=float(error_rmse / jnp.maximum(truth_rms, eps)),
        max_abs_error=float(max_abs),
        cosine_similarity=float(cosine),
    )


def policy_divergence(
    truth: jnp.ndarray,
    estimated: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    eps: float = 1e-12,
) -> PolicyMetrics:
    truth = jnp.asarray(truth)
    estimated = jnp.asarray(estimated)
    if weights is None:
        weights = jnp.ones(truth.shape[0]) / truth.shape[0]
    weights = weights / weights.sum()
    diff = jnp.abs(truth - estimated)
    l1_by_state = diff.sum(axis=1)
    l1 = jnp.sum(weights * l1_by_state)
    linf = jnp.max(l1_by_state)
    tv = 0.5 * l1
    p = jnp.clip(truth, eps, 1.0)
    q = jnp.clip(estimated, eps, 1.0)
    kl = jnp.sum(weights * jnp.sum(p * (jnp.log(p) - jnp.log(q)), axis=1))
    return PolicyMetrics(l1=float(l1), linf=float(linf), tv=float(tv), kl=float(kl))


def evaluate_policy_value(
    reward: jnp.ndarray,
    transitions: jnp.ndarray,
    policy: jnp.ndarray,
    discount_factor: float,
    scale_parameter: float = 1.0,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Evaluate a stochastic policy under the soft-logit Bellman objective."""

    clipped_policy = jnp.clip(policy, eps, 1.0)
    entropy_flow = -scale_parameter * jnp.sum(policy * jnp.log(clipped_policy), axis=1)
    reward_pi = jnp.sum(policy * reward, axis=1) + entropy_flow
    transition_pi = jnp.einsum("sa,ast->st", policy, transitions)
    lhs = jnp.eye(reward.shape[0]) - discount_factor * transition_pi
    return jnp.linalg.solve(lhs, reward_pi)


def q_from_value(
    reward: jnp.ndarray,
    value: jnp.ndarray,
    transitions: jnp.ndarray,
    discount_factor: float,
) -> jnp.ndarray:
    """Compute Q(s,a) from a reward matrix and continuation value."""

    continuation = jnp.einsum("ast,t->as", transitions, value).T
    return reward + discount_factor * continuation


def counterfactual_metrics(
    oracle_policy: jnp.ndarray,
    oracle_value: jnp.ndarray,
    estimated_policy: jnp.ndarray,
    reward: jnp.ndarray,
    transitions: jnp.ndarray,
    discount_factor: float,
    initial_distribution: jnp.ndarray,
    scale_parameter: float = 1.0,
) -> CounterfactualMetrics:
    estimated_value = evaluate_policy_value(
        reward=reward,
        transitions=transitions,
        policy=estimated_policy,
        discount_factor=discount_factor,
        scale_parameter=scale_parameter,
    )
    policy_metrics = policy_divergence(oracle_policy, estimated_policy)
    value_error = rmse(estimated_value, oracle_value)
    regret_by_state = oracle_value - estimated_value
    regret = float(jnp.dot(initial_distribution, regret_by_state))
    return CounterfactualMetrics(
        policy=policy_metrics,
        value_rmse=value_error,
        regret=regret,
    )


def evaluate_estimator_against_truth(
    dgp: Any,
    summary: Any,
    *,
    segment_index: int = 0,
    counterfactual_kinds: tuple[str, ...] = ("type_a", "type_b", "type_c"),
) -> dict[str, Any]:
    """Compute estimator-independent known-truth recovery metrics.

    The estimator policy for counterfactuals is obtained by solving the
    intervention under the estimator's recovered reward model, then evaluating
    that policy in the true counterfactual environment.
    """



    truth = solve_known_truth(dgp, segment_index=segment_index)
    estimated_params = jnp.asarray(summary.parameters)
    true_params = jnp.asarray(dgp.homogeneous_parameters)

    metrics: dict[str, Any] = {
        "parameters": None,
        "reward_rmse": None,
        "value_rmse": None,
        "q_rmse": None,
        "policy": None,
        "counterfactuals": {},
    }

    if estimated_params.shape == true_params.shape:
        metrics["parameters"] = parameter_metrics(true_params, estimated_params)

    estimated_reward = _extract_estimated_reward(dgp, summary, estimated_params)
    true_reward = get_segment_reward(dgp, segment_index)
    if estimated_reward is not None:
        metrics["reward_rmse"] = rmse(estimated_reward, true_reward)

    if summary.value_function is not None:
        estimated_value = jnp.asarray(summary.value_function)
        metrics["value_rmse"] = rmse(estimated_value, truth.V)
        if estimated_reward is not None:
            estimated_q = q_from_value(
                estimated_reward,
                estimated_value,
                dgp.transitions,
                dgp.problem.discount_factor,
            )
            metrics["q_rmse"] = rmse(estimated_q, truth.Q)

    if summary.policy is not None:
        estimated_policy = jnp.asarray(summary.policy)
        metrics["policy"] = policy_divergence(truth.policy, estimated_policy)

        if estimated_reward is not None:
            for kind in counterfactual_kinds:
                oracle = solve_counterfactual_oracle(dgp, kind, segment_index=segment_index)
                cf_reward = oracle.counterfactual_solution.reward_matrix
                reward_delta = cf_reward - true_reward
                estimated_cf_reward = estimated_reward + reward_delta
                operator = SoftBellmanOperator(dgp.problem, oracle.counterfactual.transitions)
                estimated_cf = value_iteration(
                    operator,
                    estimated_cf_reward,
                    tol=1e-8,
                    max_iter=10_000,
                )
                metrics["counterfactuals"][kind] = counterfactual_metrics(
                    oracle_policy=oracle.counterfactual_solution.policy,
                    oracle_value=oracle.counterfactual_solution.V,
                    estimated_policy=estimated_cf.policy,
                    reward=cf_reward,
                transitions=oracle.counterfactual.transitions,
                discount_factor=dgp.problem.discount_factor,
                initial_distribution=dgp.initial_distribution,
                scale_parameter=dgp.problem.scale_parameter,
            )

    return metrics


def _extract_estimated_reward(
    dgp: Any,
    summary: Any,
    estimated_params: jnp.ndarray,
) -> jnp.ndarray | None:
    """Recover an estimator reward matrix when its output supports one."""

    true_params = jnp.asarray(dgp.homogeneous_parameters)
    if estimated_params.shape == true_params.shape:
        return dgp.utility().compute(estimated_params)

    metadata_reward = summary.metadata.get("reward_matrix")
    if metadata_reward is not None:
        reward = jnp.asarray(metadata_reward)
        if reward.shape == dgp.homogeneous_reward.shape:
            return reward

    if estimated_params.size == dgp.problem.num_states * dgp.problem.num_actions:
        return estimated_params.reshape((dgp.problem.num_states, dgp.problem.num_actions))

    return None


# --- Estimator Contracts ---
Support = Literal["valid", "valid_with_normalization", "diagnostic_only", "unsupported"]


@dataclass(frozen=True)
class EstimatorContract:
    """Estimator requirements and validation targets."""

    name: str
    code_path: str
    paper_paths: tuple[str, ...]
    required_reward_modes: tuple[str, ...]
    required_state_modes: tuple[str, ...]
    requires_transitions: bool
    recovers: tuple[str, ...]
    type_a_support: Support
    type_b_support: Support
    type_c_support: Support
    gpu_recommended: bool = False
    notes: str = ""


ESTIMATOR_CONTRACTS: dict[str, EstimatorContract] = {
    "NFXP": EstimatorContract(
        name="NFXP",
        code_path="src/econirl/estimation/nfxp.py",
        paper_paths=(
            "papers/foundational/1987_rust_optimal_replacement.md",
            "papers/foundational/iskhakov_rust_schjerning_2016_mpec_comment.md",
        ),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim",),
        requires_transitions=True,
        recovers=("theta", "reward", "policy", "value", "Q"),
        type_a_support="valid",
        type_b_support="valid",
        type_c_support="valid",
        notes="Exact structural reference for manageable tabular state spaces.",
    ),
    "CCP": EstimatorContract(
        name="CCP",
        code_path="src/econirl/estimation/ccp.py",
        paper_paths=(
            "papers/foundational/hotz_miller_1993_ccp.md",
            "papers/foundational/AguirregabiriaMira_ECMA2002.md",
        ),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim",),
        requires_transitions=True,
        recovers=("theta", "reward", "policy", "value"),
        type_a_support="valid",
        type_b_support="valid",
        type_c_support="valid",
        notes="Use enough NPL iterations before treating this as MLE-like.",
    ),
    "MPEC": EstimatorContract(
        name="MPEC",
        code_path="src/econirl/estimation/mpec.py",
        paper_paths=(
            "papers/foundational/su_judd_2012_mpec.md",
            "papers/foundational/iskhakov_rust_schjerning_2016_mpec_comment.md",
        ),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim",),
        requires_transitions=True,
        recovers=("theta", "reward", "policy", "value"),
        type_a_support="valid",
        type_b_support="valid",
        type_c_support="valid",
    ),
    "MCE-IRL": EstimatorContract(
        name="MCE-IRL",
        code_path="src/econirl/estimation/mce_irl.py",
        paper_paths=("papers/foundational/ziebart_2010_mce_irl.md",),
        required_reward_modes=("action_dependent", "state_only"),
        required_state_modes=("low_dim",),
        requires_transitions=True,
        recovers=("reward", "policy", "occupancy"),
        type_a_support="valid_with_normalization",
        type_b_support="valid_with_normalization",
        type_c_support="valid_with_normalization",
        notes="Reward comparisons require the accepted IRL normalization.",
    ),
    "TD-CCP": EstimatorContract(
        name="TD-CCP",
        code_path="src/econirl/estimation/td_ccp.py",
        paper_paths=("papers/foundational/adusumilli_eckardt_2025_td_ccp.md",),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim", "high_dim"),
        requires_transitions=False,
        recovers=("theta", "policy", "value"),
        type_a_support="valid",
        type_b_support="valid",
        type_c_support="valid",
        gpu_recommended=True,
    ),
    "NNES": EstimatorContract(
        name="NNES",
        code_path="src/econirl/estimation/nnes.py",
        paper_paths=("papers/foundational/nguyen_2025_nnes.md",),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim", "high_dim"),
        requires_transitions=True,
        recovers=("theta", "policy", "value"),
        type_a_support="valid",
        type_b_support="valid",
        type_c_support="valid",
        gpu_recommended=True,
    ),
    "SEES": EstimatorContract(
        name="SEES",
        code_path="src/econirl/estimation/sees.py",
        paper_paths=("papers/foundational/luo_sang_2024_sees.md",),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim", "high_dim"),
        requires_transitions=True,
        recovers=("theta", "policy", "value"),
        type_a_support="valid",
        type_b_support="valid",
        type_c_support="valid",
    ),
    "GLADIUS": EstimatorContract(
        name="GLADIUS",
        code_path="src/econirl/estimation/gladius.py",
        paper_paths=("papers/econirl_package_jss/plans/alignment/11_gladius.md",),
        required_reward_modes=("action_dependent", "state_only"),
        required_state_modes=("high_dim",),
        requires_transitions=True,
        recovers=("Q", "reward_projection", "policy"),
        type_a_support="valid_with_normalization",
        type_b_support="diagnostic_only",
        type_c_support="diagnostic_only",
        gpu_recommended=True,
        notes="Without observed rewards, Type B and Type C expose structural bias.",
    ),
    "IQ-Learn": EstimatorContract(
        name="IQ-Learn",
        code_path="src/econirl/estimation/iq_learn.py",
        paper_paths=("papers/foundational/2022_iq_learn.md",),
        required_reward_modes=("action_dependent", "state_only"),
        required_state_modes=("low_dim", "high_dim"),
        requires_transitions=True,
        recovers=("Q", "reward", "policy"),
        type_a_support="valid_with_normalization",
        type_b_support="diagnostic_only",
        type_c_support="diagnostic_only",
        gpu_recommended=True,
    ),
    "AIRL": EstimatorContract(
        name="AIRL",
        code_path="src/econirl/estimation/adversarial/airl.py",
        paper_paths=("papers/priority/fu_2018_airl.pdf",),
        required_reward_modes=("state_only",),
        required_state_modes=("low_dim", "high_dim"),
        requires_transitions=False,
        recovers=("reward", "policy"),
        type_a_support="valid_with_normalization",
        type_b_support="valid_with_normalization",
        type_c_support="valid_with_normalization",
        gpu_recommended=True,
    ),
    "AIRL-Het": EstimatorContract(
        name="AIRL-Het",
        code_path="src/econirl/estimation/adversarial/airl_het.py",
        paper_paths=(
            "papers/priority/fu_2018_airl.pdf",
            "papers/foundational/arcidiacono_miller_2011_ccp_unobserved.md",
        ),
        required_reward_modes=("state_only", "action_dependent"),
        required_state_modes=("low_dim", "high_dim"),
        requires_transitions=False,
        recovers=("segment_reward", "segment_policy", "segment_membership"),
        type_a_support="valid_with_normalization",
        type_b_support="valid_with_normalization",
        type_c_support="valid_with_normalization",
        gpu_recommended=True,
        notes="Must be run on a latent-segment DGP for the main validation.",
    ),
    "f-IRL": EstimatorContract(
        name="f-IRL",
        code_path="src/econirl/estimation/f_irl.py",
        paper_paths=("src/econirl/estimation/f_irl.py",),
        required_reward_modes=("action_dependent", "state_only"),
        required_state_modes=("low_dim",),
        requires_transitions=True,
        recovers=("occupancy", "reward", "policy"),
        type_a_support="valid_with_normalization",
        type_b_support="diagnostic_only",
        type_c_support="diagnostic_only",
    ),
}


REQUIRED_ESTIMATORS: tuple[str, ...] = tuple(ESTIMATOR_CONTRACTS)


def get_estimator_contract(name: str) -> EstimatorContract:
    try:
        return ESTIMATOR_CONTRACTS[name]
    except KeyError as exc:
        raise KeyError(f"unknown known-truth estimator {name!r}") from exc


# --- Estimator Adapters and Gates ---
@dataclass(frozen=True)
class CompatibilityReport:
    estimator: str
    compatible: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecoveryGate:
    name: str
    value: float | bool
    operator: str
    threshold: float | bool
    passed: bool


@dataclass(frozen=True)
class EstimatorRun:
    estimator: str
    summary: EstimationSummary
    diagnostics: PreEstimationDiagnostics
    compatibility: CompatibilityReport
    metrics: dict[str, Any] = field(default_factory=dict)
    gates: list[RecoveryGate] = field(default_factory=list)


class RecoveryGateFailure(AssertionError):
    """Raised when a non-smoke known-truth run fails hard recovery gates."""


def check_estimator_compatibility(
    estimator_name: str,
    dgp: KnownTruthDGP,
    diagnostics: PreEstimationDiagnostics | None = None,
) -> CompatibilityReport:
    """Check whether an estimator should run on a DGP cell."""

    contract = get_estimator_contract(estimator_name)
    if diagnostics is None:
        diagnostics = run_pre_estimation_diagnostics(dgp)

    errors = list(diagnostics.errors)
    warnings = list(diagnostics.warnings)

    if dgp.config.reward_mode not in contract.required_reward_modes:
        errors.append(
            f"{estimator_name} does not support reward mode {dgp.config.reward_mode}"
        )
    if dgp.config.state_mode not in contract.required_state_modes:
        errors.append(
            f"{estimator_name} does not support state mode {dgp.config.state_mode}"
        )
    if "theta" in contract.recovers and not diagnostics.is_action_dependent:
        errors.append(f"{estimator_name} needs action-dependent features for theta recovery")
    if estimator_name == "NFXP" and dgp.config.heterogeneity != "none":
        errors.append("NFXP main validation requires a homogeneous DGP")
    if estimator_name == "MPEC" and dgp.config.heterogeneity != "none":
        errors.append("MPEC main validation requires a homogeneous DGP")
    if estimator_name == "SEES" and dgp.config.heterogeneity != "none":
        errors.append("SEES main validation requires a homogeneous DGP")
    if (
        estimator_name in {"NFXP", "MPEC", "SEES"}
        and diagnostics.min_action_share is not None
        and diagnostics.min_action_share < 0.05
    ):
        errors.append(
            f"{estimator_name} requires empirical action support; minimum action share is "
            f"{diagnostics.min_action_share:.3g}"
        )
    if estimator_name == "AIRL-Het" and dgp.config.heterogeneity != "latent_segments":
        errors.append("AIRL-Het main validation requires a latent-segment DGP")
    if estimator_name != "AIRL-Het" and dgp.config.heterogeneity == "latent_segments":
        warnings.append(
            f"{estimator_name} will be evaluated on mixture-average behavior unless segmented"
        )
    if contract.requires_transitions:
        row_error = diagnostics.max_transition_row_error
        if row_error > 1e-6:
            errors.append(f"{estimator_name} requires stochastic transitions")

    return CompatibilityReport(
        estimator=estimator_name,
        compatible=not errors,
        errors=errors,
        warnings=warnings,
    )


def make_estimator(
    estimator_name: str,
    dgp: KnownTruthDGP,
    *,
    smoke: bool = False,
    verbose: bool = False,
) -> Any:
    """Instantiate an estimator with known-truth defaults.

    Smoke settings are intentionally small. Medium-scale runs should use the
    same factory with smoke disabled and estimator-specific config overrides
    added in the run matrix.
    """

    if estimator_name == "NFXP":
        from econirl.estimation.nfxp import NFXPEstimator

        return NFXPEstimator(
            optimizer="BHHH",
            inner_solver="hybrid",
            inner_tol=1e-9 if smoke else 1e-12,
            outer_max_iter=30 if smoke else 500,
            inner_max_iter=2_000 if smoke else 100_000,
            compute_hessian=not smoke,
            verbose=verbose,
        )
    if estimator_name == "CCP":
        from econirl.estimation.ccp import CCPEstimator

        return CCPEstimator(
            num_policy_iterations=3 if smoke else 10,
            outer_max_iter=50 if smoke else 500,
            se_method="asymptotic" if smoke else "robust",
            compute_hessian=not smoke,
            verbose=verbose,
        )
    if estimator_name == "MPEC":
        from econirl.estimation.mpec import MPECConfig, MPECEstimator

        return MPECEstimator(
            config=MPECConfig(
                solver="sqp",
                outer_max_iter=30 if smoke else 200,
                tol=1e-6 if smoke else 1e-8,
                constraint_tol=1e-5 if smoke else 1e-6,
            ),
            se_method="asymptotic" if smoke else "robust",
            compute_hessian=not smoke,
            verbose=verbose,
        )
    if estimator_name == "MCE-IRL":
        from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator

        return MCEIRLEstimator(
            config=MCEIRLConfig(
                outer_max_iter=30 if smoke else 300,
                inner_max_iter=1_000 if smoke else 10_000,
                compute_se=False,
                verbose=verbose,
            )
        )
    if estimator_name == "TD-CCP":
        from econirl.estimation.td_ccp import TDCCPConfig, TDCCPEstimator

        return TDCCPEstimator(
            config=TDCCPConfig(
                method="semigradient",
                basis_dim=min(8, dgp.feature_matrix.shape[-1]),
                cross_fitting=False if smoke else True,
                robust_se=False,
                outer_max_iter=50 if smoke else 200,
                compute_se=False,
                verbose=verbose,
            )
        )
    if estimator_name == "NNES":
        from econirl.estimation.nnes import NNESEstimator

        return NNESEstimator(
            hidden_dim=16 if smoke else 32,
            v_epochs=20 if smoke else 500,
            outer_max_iter=30 if smoke else 200,
            n_outer_iterations=1 if smoke else 3,
            compute_se=False,
            verbose=verbose,
        )
    if estimator_name == "SEES":
        from econirl.estimation.sees import SEESEstimator

        return SEESEstimator(
            basis_type="bspline",
            basis_dim=min(dgp.problem.num_states, 6 if smoke else 21),
            penalty_weight=100.0,
            max_iter=40 if smoke else 1_000,
            tol=1e-7,
            compute_se=not smoke,
            verbose=verbose,
        )
    if estimator_name == "GLADIUS":
        from econirl.estimation.gladius import GLADIUSConfig, GLADIUSEstimator

        return GLADIUSEstimator(
            config=GLADIUSConfig(
                q_hidden_dim=16 if smoke else 128,
                v_hidden_dim=16 if smoke else 128,
                q_num_layers=1 if smoke else 3,
                v_num_layers=1 if smoke else 3,
                max_epochs=10 if smoke else 500,
                batch_size=128 if smoke else 512,
                compute_se=False,
                verbose=verbose,
            )
        )
    if estimator_name == "IQ-Learn":
        from econirl.estimation.iq_learn import IQLearnConfig, IQLearnEstimator

        return IQLearnEstimator(
            config=IQLearnConfig(
                max_iter=25 if smoke else 500,
                optimizer="adam" if smoke else "L-BFGS-B",
                verbose=verbose,
            )
        )
    if estimator_name == "AIRL":
        from econirl.estimation.adversarial.airl import AIRLConfig, AIRLEstimator

        return AIRLEstimator(
            config=AIRLConfig(
                reward_type="tabular",
                max_rounds=10 if smoke else 200,
                generator_max_iter=500 if smoke else 5_000,
                compute_se=False,
                verbose=verbose,
            )
        )
    if estimator_name == "AIRL-Het":
        from econirl.estimation.adversarial.airl_het import AIRLHetConfig, AIRLHetEstimator

        return AIRLHetEstimator(
            config=AIRLHetConfig(
                num_segments=dgp.config.num_segments,
                exit_action=dgp.config.exit_action,
                absorbing_state=dgp.config.absorbing_state,
                reward_type="tabular",
                max_airl_rounds=5 if smoke else 100,
                max_em_iterations=3 if smoke else 50,
                generator_max_iter=500 if smoke else 5_000,
                verbose=verbose,
            )
        )
    if estimator_name == "f-IRL":
        from econirl.estimation.f_irl import FIRLEstimator

        return FIRLEstimator(
            max_iter=20 if smoke else 500,
            inner_max_iter=500 if smoke else 5_000,
            compute_se=False,
            verbose=verbose,
        )
    raise KeyError(f"unknown known-truth estimator {estimator_name!r}")


def run_estimator(
    estimator_name: str,
    dgp: KnownTruthDGP,
    panel: Panel,
    *,
    smoke: bool = False,
    verbose: bool = False,
    initial_params: jnp.ndarray | None = None,
    enforce_gates: bool | None = None,
) -> EstimatorRun:
    """Run one estimator after compatibility and pre-estimation checks."""

    if enforce_gates is None:
        enforce_gates = not smoke

    diagnostics = run_pre_estimation_diagnostics(dgp, panel)
    compatibility = check_estimator_compatibility(estimator_name, dgp, diagnostics)
    if not compatibility.compatible:
        joined = "; ".join(compatibility.errors)
        raise ValueError(f"{estimator_name} is incompatible with this DGP: {joined}")

    estimator = make_estimator(estimator_name, dgp, smoke=smoke, verbose=verbose)
    if initial_params is None and estimator_name == "NFXP":
        initial_params = known_truth_initial_params(dgp)
    summary = estimator.estimate(
        panel=panel,
        utility=dgp.utility(),
        problem=dgp.problem,
        transitions=dgp.transitions,
        initial_params=initial_params,
    )
    metrics = evaluate_estimator_against_truth(dgp, summary)
    gates = recovery_gates(estimator_name, summary, metrics, smoke=smoke)
    if enforce_gates:
        failed = [gate for gate in gates if not gate.passed]
        if failed:
            details = "; ".join(
                f"{gate.name}={gate.value} {gate.operator} {gate.threshold}"
                for gate in failed
            )
            raise RecoveryGateFailure(
                f"{estimator_name} failed known-truth recovery gates: {details}"
            )
    return EstimatorRun(
        estimator=estimator_name,
        summary=summary,
        diagnostics=diagnostics,
        compatibility=compatibility,
        metrics=metrics,
        gates=gates,
    )


def contract_for(estimator_name: str) -> EstimatorContract:
    return get_estimator_contract(estimator_name)


def known_truth_initial_params(
    dgp: KnownTruthDGP,
    *,
    perturbation_scale: float = 0.02,
) -> jnp.ndarray:
    """Deterministic known-truth starting point for structural validation runs."""

    truth = np.asarray(dgp.homogeneous_parameters, dtype=np.float64)
    rng = np.random.default_rng(dgp.config.seed + 7_919)
    scale = np.maximum(np.abs(truth), 1.0)
    perturbation = perturbation_scale * scale * rng.normal(size=truth.shape)
    return jnp.array(truth + perturbation, dtype=jnp.float32)


def recovery_gates(
    estimator_name: str,
    summary: EstimationSummary,
    metrics: dict[str, Any],
    *,
    smoke: bool,
) -> list[RecoveryGate]:
    """Return hard known-truth recovery gates for non-smoke validation."""

    if smoke:
        return []
    if estimator_name == "CCP":
        se_available = summary.standard_errors is not None and bool(
            jnp.all(jnp.isfinite(jnp.asarray(summary.standard_errors)))
        )
        checks = [
            _numeric_gate(
                "npl_iterations",
                float(summary.num_iterations),
                ">=",
                5.0,
            ),
            _bool_gate("standard_errors_finite", se_available, True),
            _numeric_gate(
                "parameter_cosine",
                metrics["parameters"].cosine_similarity,
                ">=",
                0.98,
            ),
            _numeric_gate(
                "parameter_relative_rmse",
                metrics["parameters"].relative_rmse,
                "<=",
                0.15,
            ),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.03),
            _numeric_gate("value_rmse", metrics["value_rmse"], "<=", 0.10),
            _numeric_gate("q_rmse", metrics["q_rmse"], "<=", 0.10),
        ]
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.05)
            )
        return checks

    if estimator_name == "MPEC":
        se_available = summary.standard_errors is not None and bool(
            jnp.all(jnp.isfinite(jnp.asarray(summary.standard_errors)))
        )
        constraint_violation = float(
            summary.metadata.get("final_constraint_violation", float("inf"))
        )
        checks = [
            _bool_gate("converged", bool(summary.converged), True),
            _numeric_gate(
                "constraint_violation",
                constraint_violation,
                "<=",
                1e-6,
            ),
            _bool_gate("standard_errors_finite", se_available, True),
            _numeric_gate(
                "parameter_cosine",
                metrics["parameters"].cosine_similarity,
                ">=",
                0.98,
            ),
            _numeric_gate(
                "parameter_relative_rmse",
                metrics["parameters"].relative_rmse,
                "<=",
                0.15,
            ),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.03),
            _numeric_gate("value_rmse", metrics["value_rmse"], "<=", 0.10),
            _numeric_gate("q_rmse", metrics["q_rmse"], "<=", 0.10),
        ]
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.05)
            )
        return checks

    if estimator_name == "SEES":
        se_available = summary.standard_errors is not None and bool(
            jnp.all(jnp.isfinite(jnp.asarray(summary.standard_errors)))
        )
        bellman_violation = float(summary.metadata.get("bellman_violation", float("inf")))
        checks = [
            _numeric_gate("bellman_violation", bellman_violation, "<=", 0.05),
            _bool_gate("standard_errors_finite", se_available, True),
            _numeric_gate(
                "parameter_cosine",
                metrics["parameters"].cosine_similarity,
                ">=",
                0.99,
            ),
            _numeric_gate(
                "parameter_relative_rmse",
                metrics["parameters"].relative_rmse,
                "<=",
                0.15,
            ),
            _numeric_gate("reward_rmse", metrics["reward_rmse"], "<=", 0.03),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.02),
            _numeric_gate("value_rmse", metrics["value_rmse"], "<=", 0.10),
            _numeric_gate("q_rmse", metrics["q_rmse"], "<=", 0.10),
        ]
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.01)
            )
        return checks

    if estimator_name != "NFXP":
        raise NotImplementedError(
            f"No hard non-smoke recovery gates are implemented for {estimator_name}"
        )

    checks = [
        _bool_gate("converged", bool(summary.converged), True),
        _numeric_gate(
            "parameter_cosine",
            metrics["parameters"].cosine_similarity,
            ">=",
            0.98,
        ),
        _numeric_gate(
            "parameter_relative_rmse",
            metrics["parameters"].relative_rmse,
            "<=",
            0.15,
        ),
        _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.03),
        _numeric_gate("value_rmse", metrics["value_rmse"], "<=", 0.10),
    ]
    for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
        checks.append(
            _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.05)
        )
    return checks


def _numeric_gate(
    name: str,
    value: float,
    operator: str,
    threshold: float,
) -> RecoveryGate:
    if operator == "<=":
        passed = value <= threshold
    elif operator == ">=":
        passed = value >= threshold
    else:
        raise ValueError(f"unknown gate operator {operator!r}")
    return RecoveryGate(
        name=name,
        value=float(value),
        operator=operator,
        threshold=float(threshold),
        passed=bool(passed),
    )


def _bool_gate(name: str, value: bool, threshold: bool) -> RecoveryGate:
    return RecoveryGate(
        name=name,
        value=bool(value),
        operator="is",
        threshold=bool(threshold),
        passed=bool(value) == bool(threshold),
    )


# --- Artifacts ---
def stable_hash(payload: Any, length: int = 12) -> str:
    encoded = json.dumps(to_jsonable(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return value


# --- Cell Matrix ---
@dataclass(frozen=True)
class KnownTruthCell:
    cell_id: str
    dgp_config: KnownTruthDGPConfig
    simulation_config: SimulationConfig = field(default_factory=SimulationConfig)
    description: str = ""


DEFAULT_CELLS: tuple[KnownTruthCell, ...] = (
    KnownTruthCell(
        cell_id="canonical_low_action",
        dgp_config=KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=20,
            seed=42,
        ),
        simulation_config=SimulationConfig(n_individuals=2_000, n_periods=80, seed=42),
        description="Universal DGP preset: low-dimensional action-dependent structural benchmark.",
    ),
    KnownTruthCell(
        cell_id="canonical_low_state_only",
        dgp_config=KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="state_only",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=20,
            seed=43,
        ),
        description="Universal DGP preset: state-only reward benchmark for AIRL-style assumptions.",
    ),
    KnownTruthCell(
        cell_id="canonical_high_action",
        dgp_config=KnownTruthDGPConfig(
            state_mode="high_dim",
            reward_mode="action_dependent",
            reward_dim="high",
            heterogeneity="none",
            num_regular_states=80,
            high_state_dim=16,
            high_reward_features=32,
            seed=44,
        ),
        description="Universal DGP preset: high-dimensional state and reward stress benchmark.",
    ),
    KnownTruthCell(
        cell_id="canonical_latent_segments",
        dgp_config=KnownTruthDGPConfig(
            state_mode="high_dim",
            reward_mode="action_dependent",
            reward_dim="high",
            heterogeneity="latent_segments",
            num_regular_states=60,
            high_state_dim=12,
            high_reward_features=24,
            num_segments=2,
            seed=45,
        ),
        description="Universal DGP preset: latent-segment benchmark for heterogeneous estimators.",
    ),
)


CELL_ALIASES: dict[str, str] = {
    "low_state_action_reward": "canonical_low_action",
    "low_state_state_only_reward": "canonical_low_state_only",
    "high_state_high_reward": "canonical_high_action",
    "latent_segments": "canonical_latent_segments",
}


def get_cell(cell_id: str) -> KnownTruthCell:
    resolved = CELL_ALIASES.get(cell_id, cell_id)
    for cell in DEFAULT_CELLS:
        if cell.cell_id == resolved:
            if resolved == cell_id:
                return cell
            return replace(cell, cell_id=cell_id)
    raise KeyError(f"unknown known-truth cell {cell_id!r}")


# --- CLI entrypoints ---

def run_cell_estimator(
    cell_id: str,
    estimator: str,
    output_dir: Path,
    *,
    smoke: bool = False,
    show_progress: bool = False,
    verbose: bool = False,
) -> Path:
    """Run one estimator on one known-truth cell and write result.json."""

    cell = get_cell(cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    simulation_config = _simulation_config(cell.simulation_config, smoke, show_progress)
    panel = simulate_known_truth_panel(dgp, simulation_config)
    config_hash = stable_hash(
        {
            "cell": cell.dgp_config.to_dict(),
            "simulation": simulation_config,
            "estimator": estimator,
            "smoke": smoke,
        }
    )
    run_dir = output_dir / f"{cell_id}_{estimator.lower().replace('-', '')}_{config_hash}"
    try:
        result = run_estimator(estimator, dgp, panel, smoke=smoke, verbose=verbose)
        payload = {
            "cell": cell,
            "simulation": simulation_config,
            "estimator": estimator,
            "diagnostics": result.diagnostics,
            "compatibility": result.compatibility,
            "summary": _summary_payload(result.summary),
            "metrics": result.metrics,
            "gates": result.gates,
            "exception": None,
        }
    except Exception:
        payload = {
            "cell": cell,
            "simulation": simulation_config,
            "estimator": estimator,
            "exception": traceback.format_exc(),
        }
        write_json(run_dir / "result.json", payload)
        raise
    write_json(run_dir / "result.json", payload)
    return run_dir


def run_oracle_cell(cell_id: str, output_dir: Path) -> Path:
    """Build one known-truth cell and write its oracle artifacts."""

    cell = get_cell(cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    panel = simulate_known_truth_panel(dgp, cell.simulation_config)
    diagnostics = run_pre_estimation_diagnostics(dgp, panel)
    solutions = [solve_known_truth(dgp, segment_index=g) for g in range(dgp.num_segments)]
    counterfactuals = {
        kind: [
            solve_counterfactual_oracle(dgp, kind, segment_index=g)
            for g in range(dgp.num_segments)
        ]
        for kind in ("type_a", "type_b", "type_c")
    }

    config_hash = stable_hash(cell.dgp_config.to_dict())
    cell_dir = output_dir / f"{cell.cell_id}_{config_hash}"
    write_json(
        cell_dir / "oracle.json",
        {
            "cell": cell,
            "diagnostics": diagnostics,
            "solutions": solutions,
            "counterfactuals": counterfactuals,
            "panel_metadata": panel.metadata,
        },
    )
    return cell_dir


def _summary_payload(summary: EstimationSummary) -> dict[str, Any]:
    return {
        "method": summary.method,
        "converged": bool(summary.converged),
        "num_iterations": int(summary.num_iterations),
        "log_likelihood": (
            None if summary.log_likelihood is None else float(summary.log_likelihood)
        ),
        "parameters": np.asarray(summary.parameters).tolist(),
        "parameter_names": list(summary.parameter_names),
        "standard_errors": np.asarray(summary.standard_errors).tolist(),
        "num_observations": int(summary.num_observations),
        "num_individuals": int(summary.num_individuals),
        "estimation_time": float(summary.estimation_time),
        "convergence_message": summary.convergence_message,
        "goodness_of_fit": summary.goodness_of_fit,
        "metadata": summary.metadata,
        "value_function": (
            None
            if summary.value_function is None
            else np.asarray(summary.value_function).tolist()
        ),
        "policy": None if summary.policy is None else np.asarray(summary.policy).tolist(),
    }


def _simulation_config(
    base: SimulationConfig,
    smoke: bool,
    show_progress: bool,
) -> SimulationConfig:
    if smoke:
        return replace(
            base,
            n_individuals=min(base.n_individuals, 40),
            n_periods=min(base.n_periods, 20),
            show_progress=show_progress,
        )
    return replace(base, show_progress=show_progress)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", default="canonical_low_action")
    parser.add_argument("--estimator")
    parser.add_argument("--output-dir", default="outputs/known_truth")
    parser.add_argument("--oracles", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.estimator:
        path = run_cell_estimator(
            args.cell_id,
            args.estimator,
            output_dir,
            smoke=args.smoke,
            show_progress=args.show_progress,
            verbose=args.verbose,
        )
        print(path)
        return

    cell_ids = [cell.cell_id for cell in DEFAULT_CELLS] if args.cell_id == "all" else [args.cell_id]
    for cell_id in cell_ids:
        path = run_oracle_cell(cell_id, output_dir)
        print(path)


if __name__ == "__main__":
    main()

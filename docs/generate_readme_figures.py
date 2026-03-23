#!/usr/bin/env python3
"""Generate README figures: reward heatmaps and animated GIFs.

Usage:
    python docs/generate_readme_figures.py                          # uses cache if available
    python docs/generate_readme_figures.py --no-cache               # force re-run all
    python docs/generate_readme_figures.py --estimators BC,"Max Margin IRL"  # target specific

Outputs:
    docs/reward_heatmaps.png       — heatmap comparing true vs estimated rewards
    docs/mdp_data_generation.gif   — agent following optimal policy through MDP
    docs/internal_validity.gif     — algorithms executing on training dynamics
    docs/external_validity.gif     — algorithms executing on transfer dynamics
    docs/benchmark_cache.pt        — cached results for fast regeneration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

# Ensure project is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.environments import MultiComponentBusEnvironment
from econirl.evaluation.benchmark import (
    BenchmarkDGP,
    _evaluate_pct_optimal,
    get_default_estimator_specs,
    run_single,
)

# ── Configuration ──────────────────────────────────────────────────
DGP = BenchmarkDGP(n_states=5, discount_factor=0.95)
N_AGENTS = 100
N_PERIODS = 50
SEED = 42

OUT_DIR = Path(__file__).resolve().parent
CACHE_PATH = OUT_DIR / "benchmark_cache.pt"

# GIF layout constants
_XS = np.linspace(1.0, 9.0, 5)
_Y = 2.0
_R = 0.38
_ACTION_LABELS = ["Keep", "Replace"]
_ACTION_COLORS = ["#1f77b4", "#d62728"]
_GIF_DPI = 100
_GIF_FPS = 5


# ── Cache ──────────────────────────────────────────────────────────

def _dgp_fingerprint() -> str:
    """Hash DGP + simulation params for cache invalidation."""
    d = {
        "n_states": DGP.n_states,
        "replacement_cost": DGP.replacement_cost,
        "operating_cost": DGP.operating_cost,
        "quadratic_cost": DGP.quadratic_cost,
        "discount_factor": DGP.discount_factor,
        "n_agents": N_AGENTS,
        "n_periods": N_PERIODS,
        "seed": SEED,
    }
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


def _load_cache() -> dict[str, dict] | None:
    """Load cached estimator results if fingerprint matches.

    Returns dict keyed by estimator name, each containing:
        reward: ndarray | None, policy: ndarray | None, pct_optimal: float
    """
    if not CACHE_PATH.exists():
        return None
    cache = torch.load(CACHE_PATH, weights_only=False)
    if cache.get("fingerprint") != _dgp_fingerprint():
        print("  Cache fingerprint mismatch — re-running estimators")
        return None
    # Migrate old list-based cache to dict format
    if isinstance(cache.get("panels"), list):
        print("  Migrating old cache format — re-running estimators")
        return None
    print(f"  Loaded cached results from {CACHE_PATH.name}")
    return cache["panels"]


def _save_cache(panels: dict[str, dict]) -> None:
    """Save estimator results to cache."""
    torch.save({"fingerprint": _dgp_fingerprint(), "panels": panels}, CACHE_PATH)
    print(f"  Saved cache to {CACHE_PATH.name}")


# ── Utilities ──────────────────────────────────────────────────────

def _normalize(r: np.ndarray) -> np.ndarray:
    """Normalize to zero-mean, unit-variance (handles IRL scale ambiguity)."""
    std = r.std()
    if std < 1e-10:
        return r - r.mean()
    return (r - r.mean()) / std


def _make_env() -> MultiComponentBusEnvironment:
    """Create the 5-state bus environment matching README example."""
    return MultiComponentBusEnvironment(
        K=1, M=5, discount_factor=0.95,
        replacement_cost=2.0, operating_cost=1.0, quadratic_cost=0.5,
    )


def _sample_trajectory(
    policy: np.ndarray,
    transitions: np.ndarray,
    n_steps: int,
    rng: np.random.RandomState,
    start_state: int = 0,
) -> tuple[list[int], list[int]]:
    """Sample trajectory: returns (states[n_steps+1], actions[n_steps])."""
    states = [start_state]
    actions = []
    state = start_state
    for _ in range(n_steps):
        action = rng.choice(policy.shape[1], p=policy[state])
        next_state = rng.choice(policy.shape[0], p=transitions[action, state])
        actions.append(action)
        states.append(next_state)
        state = next_state
    return states, actions


# ── GIF Frame Helpers ──────────────────────────────────────────────

def _setup_axes(ax: plt.Axes) -> None:
    """Configure axes for MDP frame."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0.3, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_nodes(ax: plt.Axes, current_state: int, V: np.ndarray) -> None:
    """Draw 5 MDP nodes, highlighting current state in gold."""
    cmap = plt.cm.YlOrRd_r
    V_norm = (V - V.min()) / (V.max() - V.min() + 1e-10)
    for s in range(5):
        if s == current_state:
            glow = plt.Circle(
                (_XS[s], _Y), _R + 0.08, facecolor="#FFD700", alpha=0.4, zorder=3,
            )
            ax.add_patch(glow)
            face, lw = "#FFD700", 2.5
        else:
            face, lw = cmap(V_norm[s]), 1.5
        circle = plt.Circle(
            (_XS[s], _Y), _R, facecolor=face, edgecolor="black", lw=lw, zorder=5,
        )
        ax.add_patch(circle)
        ax.text(
            _XS[s], _Y + 0.06, f"$s_{s}$",
            fontsize=11, ha="center", va="center", fontweight="bold", zorder=6,
        )
        ax.text(
            _XS[s], _Y - 0.18, f"V={V[s]:.1f}",
            fontsize=6, ha="center", va="center", color="0.2", zorder=6,
        )


def _draw_arrow(
    ax: plt.Axes, prev_state: int | None, curr_state: int, action: int | None,
) -> None:
    """Draw transition arrow between states."""
    if prev_state is None or action is None or prev_state == curr_state:
        return
    color = _ACTION_COLORS[action]
    if action == 1:  # Replace → curved arc to s0
        rad = 0.2 + 0.08 * prev_state
        ax.annotate(
            "", xy=(_XS[curr_state], _Y + _R + 0.05),
            xytext=(_XS[prev_state], _Y + _R + 0.05),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=2.5, alpha=0.8,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )
    else:  # Keep → straight right
        ax.annotate(
            "", xy=(_XS[curr_state] - _R - 0.02, _Y),
            xytext=(_XS[prev_state] + _R + 0.02, _Y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, alpha=0.8),
        )


# ── Figure Generators ──────────────────────────────────────────────

def generate_reward_heatmaps(
    use_cache: bool = True,
    only_estimators: list[str] | None = None,
) -> None:
    """Run estimators and generate reward comparison heatmap.

    Args:
        use_cache: Whether to use cached results.
        only_estimators: If set, only (re-)run these estimators; merge into cache.
    """
    cached: dict[str, dict] | None = None
    if use_cache:
        cached = _load_cache()

    specs = get_default_estimator_specs()

    if cached is not None:
        panels = dict(cached)
        if only_estimators is not None:
            # Selective re-run: only run specified estimators
            specs_to_run = [s for s in specs if s.name in only_estimators]
        else:
            # Run any estimators missing from cache
            specs_to_run = [s for s in specs if s.name not in cached]
    else:
        # No cache — run all (or filtered subset)
        panels = {}
        if only_estimators is not None:
            specs_to_run = [s for s in specs if s.name in only_estimators]
        else:
            specs_to_run = list(specs)

    # Run required estimators
    for spec in specs_to_run:
        print(f"  {spec.name}...", end=" ", flush=True)
        result = run_single(DGP, spec, n_agents=N_AGENTS, n_periods=N_PERIODS, seed=SEED)
        status = f"{result.pct_optimal:.1f}%" if not np.isnan(result.pct_optimal) else "FAIL"
        print(f"{status} ({result.time_seconds:.1f}s)")

        reward = (result.estimated_reward.detach().numpy()
                  if result.estimated_reward is not None
                  else None)
        policy = (result.learned_policy.detach().numpy()
                  if result.learned_policy is not None
                  else None)
        panels[spec.name] = {
            "reward": reward,
            "pct_optimal": result.pct_optimal,
            "policy": policy,
        }

    # Store true reward from a quick run if not yet in panels
    if "__true_reward__" not in panels:
        # Get true reward from any result or compute directly
        env = _make_env()
        true_params = env.get_true_parameter_vector()
        true_reward = torch.einsum("sak,k->sa", env.feature_matrix, true_params).numpy()
        panels["__true_reward__"] = {"reward": true_reward, "pct_optimal": 100.0, "policy": None}

    if specs_to_run:
        _save_cache(panels)

    # Build ordered (name, reward) list for plotting: True first, then estimators
    true_reward = panels["__true_reward__"]["reward"]
    plot_panels: list[tuple[str, np.ndarray]] = [("True\nReward", true_reward)]
    for spec in specs:
        if spec.name in panels:
            entry = panels[spec.name]
            reward = entry["reward"] if entry["reward"] is not None else np.full_like(true_reward, np.nan)
            plot_panels.append((spec.name, reward))

    # Normalize all for fair comparison
    panels_norm = [(name, _normalize(r)) for name, r in plot_panels]

    # Global color range
    valid = np.concatenate([r.ravel() for _, r in panels_norm if not np.isnan(r).all()])
    vmax = max(abs(valid.min()), abs(valid.max()))

    n = len(panels_norm)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.1 + 0.8, 3.2))

    for i, (ax, (name, reward)) in enumerate(zip(axes, panels_norm)):
        im = ax.imshow(
            reward, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(name, fontsize=7.5, fontweight="bold" if i == 0 else "normal")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Keep", "Repl."], fontsize=5.5)
        if i == 0:
            ax.set_yticks(range(5))
            ax.set_yticklabels([f"s{j}" for j in range(5)], fontsize=6)
        else:
            ax.set_yticks([])
        ax.tick_params(length=2, pad=1)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.75, pad=0.02)
    cbar.set_label("Normalized reward", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    n_estimators = len(specs)
    fig.suptitle(
        f"Estimated vs True Rewards — 5-state Bus MDP ({n_estimators} estimators, "
        f"{N_AGENTS} agents × {N_PERIODS} periods)",
        fontsize=10,
    )
    plt.savefig(OUT_DIR / "reward_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved {OUT_DIR / 'reward_heatmaps.png'}")


def generate_data_gif() -> None:
    """Generate animated GIF of agent following optimal policy through 5-state MDP."""
    env = _make_env()
    true_params = env.get_true_parameter_vector()
    true_utility = torch.einsum("sak,k->sa", env.feature_matrix, true_params)
    operator = SoftBellmanOperator(
        problem=env.problem_spec, transitions=env.transition_matrices,
    )
    sol = value_iteration(operator, true_utility)
    policy = sol.policy.numpy()
    V = sol.V.numpy()
    transitions = env.transition_matrices.numpy()

    rng = np.random.RandomState(SEED)
    n_steps = 30
    states, actions = _sample_trajectory(policy, transitions, n_steps, rng)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    def update(frame):
        ax.clear()
        _setup_axes(ax)
        state = states[frame]
        action = actions[frame - 1] if frame > 0 else None
        prev = states[frame - 1] if frame > 0 else None

        _draw_arrow(ax, prev, state, action)
        _draw_nodes(ax, state, V)

        if action is not None:
            ax.text(
                5, 3.3, f"Action: {_ACTION_LABELS[action]}",
                fontsize=14, ha="center", fontweight="bold",
                color=_ACTION_COLORS[action],
            )

        ax.set_title(
            f"Data Generation — Optimal Policy (step {frame}/{n_steps})",
            fontsize=12, pad=10,
        )

    anim = FuncAnimation(fig, update, frames=n_steps + 1, interval=200)
    anim.save(
        OUT_DIR / "mdp_data_generation.gif",
        writer=PillowWriter(fps=_GIF_FPS), dpi=_GIF_DPI,
    )
    plt.close()
    print(f"  Saved {OUT_DIR / 'mdp_data_generation.gif'}")


def generate_policy_gif(mode: str = "internal", use_cache: bool = True) -> None:
    """Generate animated GIF cycling through each algorithm's policy execution.

    Args:
        mode: "internal" for training dynamics, "external" for transfer dynamics.
        use_cache: Whether to use cached results.
    """
    env = _make_env()
    true_params = env.get_true_parameter_vector()
    true_utility = torch.einsum("sak,k->sa", env.feature_matrix, true_params)
    problem = env.problem_spec

    # Load estimated results from cache
    cached = _load_cache() if use_cache else None
    if cached is None:
        print("  No cache — running estimators first")
        generate_reward_heatmaps(use_cache=False)
        cached = _load_cache()

    # Select transitions based on mode
    if mode == "external":
        transfer_env = MultiComponentBusEnvironment(
            K=1, M=5, discount_factor=0.95,
            replacement_cost=2.0, operating_cost=1.0, quadratic_cost=0.5,
            mileage_transition_probs=DGP.transfer_transition_probs,
        )
        transitions = transfer_env.transition_matrices
        metric_label = "% Transfer"
        filename = "external_validity.gif"
    else:
        transitions = env.transition_matrices
        metric_label = "% Optimal"
        filename = "internal_validity.gif"

    transitions_np = transitions.numpy()
    operator = SoftBellmanOperator(problem=problem, transitions=transitions)

    # True value function under these transitions (for node display)
    V_display = value_iteration(operator, true_utility).V.numpy()

    # Build per-estimator data: solve policy, compute metric, sample trajectory
    specs = get_default_estimator_specs()
    n_traj_steps = 15
    title_frames = 3
    traj_frames = n_traj_steps + 1  # initial state + steps
    estimator_data = []

    for spec in specs:
        name = spec.name
        if name not in cached:
            continue
        entry = cached[name]
        reward = entry.get("reward")
        policy = entry.get("policy")

        # Determine policy: use cached policy if available, else solve from reward
        if reward is not None and not np.isnan(reward).all():
            est_reward = torch.tensor(reward, dtype=torch.float32)
            try:
                learned_policy = value_iteration(operator, est_reward).policy
                pct = _evaluate_pct_optimal(learned_policy, true_utility, transitions, problem)
            except Exception:
                learned_policy = torch.ones(5, 2) / 2
                pct = 0.0
        elif policy is not None:
            # BC-style: no reward, but has a cached policy
            learned_policy = torch.tensor(policy, dtype=torch.float32)
            pct = _evaluate_pct_optimal(learned_policy, true_utility, transitions, problem)
        else:
            continue

        rng = np.random.RandomState(SEED)
        states, actions = _sample_trajectory(
            learned_policy.numpy(), transitions_np, n_traj_steps, rng,
        )
        estimator_data.append((name, pct, states, actions))

    # Build animation
    fig, ax = plt.subplots(figsize=(10, 3.5))
    frames_per_est = title_frames + traj_frames
    total_frames = len(estimator_data) * frames_per_est

    def update(frame):
        ax.clear()
        est_idx = frame // frames_per_est
        local_frame = frame % frames_per_est
        if est_idx >= len(estimator_data):
            return

        name, pct, states, actions = estimator_data[est_idx]

        if local_frame < title_frames:
            # Title card
            ax.text(
                5, 2.2, name,
                fontsize=24, ha="center", va="center", fontweight="bold",
            )
            ax.text(
                5, 1.4, f"{metric_label}: {pct:.1f}%",
                fontsize=16, ha="center", va="center", color="#333333",
            )
            ax.set_xlim(0, 10)
            ax.set_ylim(0.3, 3.8)
            ax.set_aspect("equal")
            ax.axis("off")
        else:
            # Trajectory frame
            step = local_frame - title_frames
            _setup_axes(ax)
            state = states[step]
            action = actions[step - 1] if step > 0 else None
            prev = states[step - 1] if step > 0 else None

            _draw_arrow(ax, prev, state, action)
            _draw_nodes(ax, state, V_display)

            if action is not None:
                ax.text(
                    5, 3.3, f"Action: {_ACTION_LABELS[action]}",
                    fontsize=14, ha="center", fontweight="bold",
                    color=_ACTION_COLORS[action],
                )

            ax.set_title(
                f"{name} — {metric_label}: {pct:.1f}% (step {step}/{n_traj_steps})",
                fontsize=11, pad=10,
            )

    anim = FuncAnimation(fig, update, frames=total_frames, interval=200)
    anim.save(OUT_DIR / filename, writer=PillowWriter(fps=_GIF_FPS), dpi=_GIF_DPI)
    plt.close()
    print(f"  Saved {OUT_DIR / filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate README figures")
    parser.add_argument("--no-cache", action="store_true", help="Force re-run all estimators")
    parser.add_argument(
        "--estimators",
        type=str,
        default=None,
        help='Comma-separated estimator names to (re-)run, e.g. \'BC,"Max Margin IRL"\'',
    )
    args = parser.parse_args()

    use_cache = not args.no_cache
    only_estimators: list[str] | None = None
    if args.estimators is not None:
        # Parse comma-separated, strip quotes and whitespace
        only_estimators = [
            name.strip().strip('"').strip("'")
            for name in args.estimators.split(",")
            if name.strip()
        ]
        print(f"Targeting estimators: {only_estimators}")

    n_estimators = len(get_default_estimator_specs())

    print("Generating data generation GIF...")
    generate_data_gif()

    print(f"\nRunning {n_estimators} estimators for reward heatmaps...")
    generate_reward_heatmaps(use_cache=use_cache, only_estimators=only_estimators)

    print("\nGenerating internal validity GIF...")
    generate_policy_gif(mode="internal", use_cache=True)

    print("\nGenerating external validity GIF...")
    generate_policy_gif(mode="external", use_cache=True)

    print("\nDone!")

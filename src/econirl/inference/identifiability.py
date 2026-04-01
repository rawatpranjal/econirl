"""Reward identifiability diagnostics for inverse reinforcement learning.

Implements the graph-theoretic identifiability test from Kim, Garg,
Shiragur, and Ermon (2021, "Reward Identification in Inverse
Reinforcement Learning", ICML). The test checks whether the MDP
transition structure allows unique reward recovery under MaxEnt IRL.

The key condition is aperiodicity of the domain graph derived from
the transition support. If the domain graph has period > 1, the reward
is not strongly identifiable from demonstrations alone.
"""

from __future__ import annotations

from collections import deque
from math import gcd

import numpy as np


def check_reward_identifiability(
    transitions: np.ndarray,
    num_actions: int | None = None,
) -> dict[str, object]:
    """Test reward identifiability for MaxEnt IRL via graph aperiodicity.

    Constructs the domain graph from the MDP transition support and checks
    whether the graph is aperiodic. An aperiodic and strongly connected
    domain graph is necessary and sufficient for strong reward
    identifiability under the MaxEnt IRL objective (Kim et al. 2021,
    Theorem 3).

    Args:
        transitions: Transition probabilities. Shape (A, S, S) where
            transitions[a, s, s'] = P(s' | s, a), or shape (S, S) for
            a single-action aggregated transition.
        num_actions: Number of actions. Inferred from transitions shape
            if not provided.

    Returns:
        Dict with keys:
            is_identifiable (bool): True if the domain graph is aperiodic.
            period (int): Period of the domain graph. 1 means aperiodic.
            num_states (int): Number of states.
            num_edges (int): Number of edges in the domain graph.
            is_strongly_connected (bool): Whether all states communicate.
            status (str): Human-readable summary.
    """
    transitions = np.asarray(transitions)

    if transitions.ndim == 2:
        # Single transition matrix (S, S) -- treat as one action
        adj = transitions > 0
        num_states = transitions.shape[0]
    elif transitions.ndim == 3:
        # (A, S, S) -- union of supports across actions
        num_states = transitions.shape[1]
        adj = np.any(transitions > 0, axis=0)  # (S, S)
    else:
        raise ValueError(f"transitions must be 2D or 3D, got {transitions.ndim}D")

    num_edges = int(np.sum(adj))

    # Build adjacency list
    neighbors = [[] for _ in range(num_states)]
    for s in range(num_states):
        for sp in range(num_states):
            if adj[s, sp]:
                neighbors[s].append(sp)

    # Check strong connectivity via BFS from state 0 and reverse BFS
    strongly_connected = (
        _bfs_reachable(neighbors, 0, num_states) == num_states
        and _bfs_reachable(_reverse_adj(neighbors, num_states), 0, num_states) == num_states
    )

    if not strongly_connected:
        return {
            "is_identifiable": False,
            "period": -1,
            "num_states": num_states,
            "num_edges": num_edges,
            "is_strongly_connected": False,
            "status": "Not identifiable: domain graph is not strongly connected",
        }

    # Compute period via GCD of cycle lengths from state 0
    period = _compute_period(neighbors, num_states)

    is_identifiable = period == 1

    if is_identifiable:
        status = "Strongly identifiable: domain graph is aperiodic (period 1)"
    else:
        status = f"Not identifiable: domain graph has period {period}"

    return {
        "is_identifiable": is_identifiable,
        "period": period,
        "num_states": num_states,
        "num_edges": num_edges,
        "is_strongly_connected": True,
        "status": status,
    }


def _bfs_reachable(neighbors: list[list[int]], start: int, n: int) -> int:
    """Count states reachable from start via BFS."""
    visited = [False] * n
    visited[start] = True
    queue = deque([start])
    count = 1
    while queue:
        s = queue.popleft()
        for sp in neighbors[s]:
            if not visited[sp]:
                visited[sp] = True
                count += 1
                queue.append(sp)
    return count


def _reverse_adj(neighbors: list[list[int]], n: int) -> list[list[int]]:
    """Build reverse adjacency list."""
    rev = [[] for _ in range(n)]
    for s in range(n):
        for sp in neighbors[s]:
            rev[sp].append(s)
    return rev


def _compute_period(neighbors: list[list[int]], n: int) -> int:
    """Compute graph period via BFS from state 0.

    The period is the GCD of lengths of all cycles through state 0.
    Equivalently, BFS from state 0 and take GCD of (depth[s'] - depth[s] - 1)
    for all back/cross edges (s, s') where s' is already visited.
    """
    dist = [-1] * n
    dist[0] = 0
    queue = deque([0])
    g = 0  # running GCD

    while queue:
        s = queue.popleft()
        for sp in neighbors[s]:
            if dist[sp] == -1:
                dist[sp] = dist[s] + 1
                queue.append(sp)
            else:
                # Cycle length = dist[s] + 1 - dist[sp]
                cycle_len = dist[s] + 1 - dist[sp]
                if cycle_len > 0:
                    g = gcd(g, cycle_len)

    return g if g > 0 else 1

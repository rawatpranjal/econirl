"""Robinson Crusoe Production/Leisure Dataset.

A simple pedagogical DDC model where Robinson Crusoe must choose between
fishing (accumulating food inventory) and leisure (consuming inventory).

This model is useful for:
- Teaching DDC estimation basics
- Testing estimator implementations
- Benchmarking performance

The model has:
- 1 state variable: inventory level (discrete)
- 2-3 actions: fish, leisure, (optionally: hunt)
- Simple transition dynamics
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_robinson_crusoe(
    n_individuals: int = 200,
    n_periods: int = 50,
    n_inventory_bins: int = 20,
    include_hunt: bool = False,
    as_panel: bool = False,
    seed: Optional[int] = 1719,
) -> pd.DataFrame:
    """Load or generate Robinson Crusoe production/leisure data.

    Model structure:
    - State: inventory (0 to n_inventory_bins-1)
    - Actions:
        - 0: Fish (increases inventory)
        - 1: Leisure (decreases inventory, increases utility)
        - 2: Hunt (optional, higher risk/reward than fishing)
    - Transition: inventory += catch - consumption

    Args:
        n_individuals: Number of individuals to simulate
        n_periods: Decision periods per individual
        n_inventory_bins: Number of discrete inventory states
        include_hunt: If True, add hunting as third action
        as_panel: If True, return as Panel object
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns:
            - id: Individual identifier
            - period: Decision period
            - inventory: Current inventory state
            - choice: Chosen action
            - catch: Units caught this period
            - consumption: Units consumed this period

    Example:
        >>> from econirl.datasets import load_robinson_crusoe
        >>> df = load_robinson_crusoe(n_individuals=100, seed=42)
        >>> print(df['choice'].value_counts())
    """
    if seed is not None:
        np.random.seed(seed)

    # Model parameters
    utility_leisure = 2.0
    utility_consumption = 1.5
    disutility_fish = -0.5
    disutility_hunt = -1.0
    discount = 0.95

    # Fishing/hunting outcomes
    fish_success_prob = 0.8
    fish_catch_mean = 2
    hunt_success_prob = 0.4
    hunt_catch_mean = 5
    consumption_per_period = 1

    records = []

    for i in range(1, n_individuals + 1):
        inventory = np.random.randint(3, 8)  # Start with some inventory

        for t in range(1, n_periods + 1):
            # Current state
            current_inventory = min(inventory, n_inventory_bins - 1)

            # Choice probabilities based on simple utility comparison
            # V(fish) = disutility_fish + E[future value]
            # V(leisure) = utility_leisure + utility_consumption - cost of low inventory

            if inventory <= 2:
                # Low inventory: strongly prefer fishing
                p_fish = 0.9
                p_leisure = 0.1
            elif inventory >= n_inventory_bins - 3:
                # High inventory: prefer leisure
                p_fish = 0.2
                p_leisure = 0.8
            else:
                # Medium inventory: balanced
                p_fish = 0.5
                p_leisure = 0.5

            n_actions = 3 if include_hunt else 2

            if include_hunt:
                p_hunt = 0.1
                total = p_fish + p_leisure + p_hunt
                probs = [p_fish/total, p_leisure/total, p_hunt/total]
            else:
                probs = [p_fish, p_leisure]

            choice = np.random.choice(range(n_actions), p=probs)

            # Outcomes
            if choice == 0:  # Fish
                success = np.random.random() < fish_success_prob
                catch = np.random.poisson(fish_catch_mean) if success else 0
                consumption = 0
            elif choice == 1:  # Leisure
                catch = 0
                consumption = min(consumption_per_period, inventory)
            else:  # Hunt
                success = np.random.random() < hunt_success_prob
                catch = np.random.poisson(hunt_catch_mean) if success else 0
                consumption = 0

            records.append({
                'id': i,
                'period': t,
                'inventory': current_inventory,
                'choice': choice,
                'catch': catch,
                'consumption': consumption,
            })

            # State transition
            inventory = max(0, inventory + catch - consumption)
            inventory = min(inventory, n_inventory_bins - 1)

    df = pd.DataFrame(records)

    if as_panel:
        return _to_panel(df)

    return df


def _to_panel(df: pd.DataFrame):
    """Convert to Panel format."""
    from econirl.core.types import Panel, Trajectory
    import torch

    trajectories = []
    for ind_id in df['id'].unique():
        ind_data = df[df['id'] == ind_id].sort_values('period')

        states = torch.tensor(ind_data['inventory'].values, dtype=torch.long)
        actions = torch.tensor(ind_data['choice'].values, dtype=torch.long)
        next_states = torch.cat([states[1:], states[-1:]])

        traj = Trajectory(
            states=states,
            actions=actions,
            next_states=next_states,
            individual_id=int(ind_id),
        )
        trajectories.append(traj)

    return Panel(trajectories=trajectories)


def get_robinson_crusoe_info() -> dict:
    """Get metadata about Robinson Crusoe dataset."""
    return {
        "name": "Robinson Crusoe Production/Leisure",
        "type": "synthetic",
        "state_variables": ["inventory"],
        "choices": {
            0: "Fish (production)",
            1: "Leisure (consumption)",
            2: "Hunt (high risk/reward, optional)",
        },
        "purpose": "Pedagogical DDC model for teaching and testing",
        "reference": "Common textbook example, used in respy tutorials",
    }

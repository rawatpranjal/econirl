"""
Built-in datasets for econirl.

This module provides access to classic datasets used in the dynamic discrete
choice and inverse reinforcement learning literature.

DDC Datasets (Structural Econometrics):
- Rust (1987): Bus engine replacement (simple, 1 state, 2 actions)
- Keane & Wolpin (1994): Career decisions (complex, 3+ states, 4 actions)
- Robinson Crusoe: Production/leisure (pedagogical, synthetic)

IRL Datasets (Inverse Reinforcement Learning):
- T-Drive: Beijing taxi GPS trajectories (MaxEnt IRL on road networks)
- GeoLife: Human mobility GPS trajectories (182 users)
- Stanford Drone: Campus pedestrian/cyclist trajectories
- ETH/UCY: Classic pedestrian trajectory benchmark (5 scenes)

Search/Click Datasets (Sequential Choice):
- Trivago (2019): Hotel search sessions (browse/refine/clickout/abandon)
"""

# DDC datasets
from econirl.datasets.rust_bus import load_rust_bus
from econirl.datasets.occupational_choice import load_occupational_choice
from econirl.datasets.keane_wolpin import load_keane_wolpin, get_keane_wolpin_info
from econirl.datasets.robinson_crusoe import load_robinson_crusoe, get_robinson_crusoe_info
from econirl.datasets.equipment_replacement import load_equipment_replacement

# IRL datasets
from econirl.datasets.tdrive import load_tdrive, get_tdrive_info
from econirl.datasets.geolife import load_geolife, get_geolife_info
from econirl.datasets.stanford_drone import load_stanford_drone, get_stanford_drone_info
from econirl.datasets.eth_ucy import load_eth_ucy, get_eth_ucy_info

# Real-world DDC/IRL datasets
from econirl.datasets.foursquare import load_foursquare, get_foursquare_info
from econirl.datasets.ngsim import load_ngsim, get_ngsim_info
from econirl.datasets.taxi_gridworld import load_taxi_gridworld, get_taxi_gridworld_info
from econirl.datasets.shanghai_route import (
    load_shanghai_network,
    load_shanghai_route,
    load_shanghai_trajectories,
    build_transition_matrix,
    build_edge_features,
    build_state_action_features,
)
from econirl.datasets.trivago_search import (
    load_trivago_search,
    load_trivago_sessions,
    build_trivago_mdp,
    build_trivago_panel,
    build_trivago_features,
    build_trivago_transitions,
    get_trivago_info,
)

__all__ = [
    # DDC Datasets
    "load_rust_bus",
    "load_occupational_choice",
    "load_keane_wolpin",
    "get_keane_wolpin_info",
    "load_robinson_crusoe",
    "get_robinson_crusoe_info",
    "load_equipment_replacement",
    # IRL Datasets
    "load_tdrive",
    "get_tdrive_info",
    "load_geolife",
    "get_geolife_info",
    "load_stanford_drone",
    "get_stanford_drone_info",
    "load_eth_ucy",
    "get_eth_ucy_info",
    # Real-world DDC/IRL Datasets
    "load_foursquare",
    "get_foursquare_info",
    "load_ngsim",
    "get_ngsim_info",
    # Benchmark Datasets
    "load_taxi_gridworld",
    "get_taxi_gridworld_info",
    # Shanghai route-choice
    "load_shanghai_network",
    "load_shanghai_route",
    "load_shanghai_trajectories",
    "build_transition_matrix",
    "build_edge_features",
    "build_state_action_features",
    # Trivago hotel search
    "load_trivago_search",
    "load_trivago_sessions",
    "build_trivago_mdp",
    "build_trivago_panel",
    "build_trivago_features",
    "build_trivago_transitions",
    "get_trivago_info",
]

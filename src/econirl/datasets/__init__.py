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
]

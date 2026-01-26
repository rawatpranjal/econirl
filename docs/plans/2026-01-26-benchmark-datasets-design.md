# Benchmark Datasets Design

**Date:** 2026-01-26
**Status:** Approved

## Overview

Add comprehensive benchmark datasets to econirl for testing IRL estimators. Covers synthetic environments (Gridworld, Objectworld, Pursuit-evasion) and real-world data (Pittsburgh Taxi, NGSIM, HighD, D4RL).

**Approach:** Download-on-demand with caching to `~/.econirl/datasets/`.

## API Design

All datasets (synthetic and real) share the same interface:

```python
from econirl.datasets import load_dataset, list_datasets

# List available datasets
list_datasets()
# ['gridworld', 'objectworld', 'pursuit_evasion', 'pittsburgh_taxi',
#  'ngsim', 'highd', 'd4rl_halfcheetah', 'd4rl_hopper', 'd4rl_ant', 'd4rl_walker2d']

# Load any dataset - downloads automatically on first use
dataset = load_dataset('pittsburgh_taxi')
dataset = load_dataset('gridworld', size=10, n_objects=5)  # synthetic accepts params

# Consistent return structure
dataset.trajectories   # List[Trajectory] - existing econirl type
dataset.transitions    # Transition matrix (if discrete MDP)
dataset.features       # Feature matrix (if available)
dataset.metadata       # Dict with source info, citation, etc.
```

**Cache location:** `~/.econirl/datasets/` (respects `ECONIRL_CACHE_DIR` env var)

## Dataset Catalog

| Dataset | Source URL | Size | License |
|---------|-----------|------|---------|
| **Synthetic (generated locally)** |
| `gridworld` | N/A | ~KB | - |
| `objectworld` | N/A | ~KB | - |
| `pursuit_evasion` | N/A | ~KB | - |
| **Real Trajectory Data** |
| `pittsburgh_taxi` | CMU archives / Ziebart's page | ~50MB | Academic |
| `ngsim` | FHWA data.transportation.gov | ~500MB | Public domain |
| `highd` | highd-dataset.com | ~2GB | Requires registration |
| **D4RL (MuJoCo experts)** |
| `d4rl_halfcheetah` | rail.eecs.berkeley.edu/datasets | ~50MB | MIT |
| `d4rl_hopper` | rail.eecs.berkeley.edu/datasets | ~50MB | MIT |
| `d4rl_ant` | rail.eecs.berkeley.edu/datasets | ~50MB | MIT |
| `d4rl_walker2d` | rail.eecs.berkeley.edu/datasets | ~50MB | MIT |

**Handling restricted datasets (HighD):** Raise helpful error with registration instructions.

## Module Structure

```
src/econirl/datasets/
├── __init__.py          # load_dataset(), list_datasets()
├── base.py              # Dataset class, download utilities
├── cache.py             # Cache management (~/.econirl/datasets/)
│
├── # Synthetic generators
├── gridworld.py         # Configurable NxN grid
├── objectworld.py       # Feature-based gridworld (Levine et al.)
├── pursuit_evasion.py   # Multi-agent game (Ziebart 2010)
│
├── # Real trajectory loaders
├── pittsburgh_taxi.py   # GPS traces → discrete MDP
├── ngsim.py             # Highway vehicle trajectories
├── highd.py             # Drone highway data
│
└── # D4RL continuous control
    └── d4rl.py          # All D4RL loaders (shared HDF5 parsing)
```

## Dataset Class & Download Logic

```python
@dataclass
class Dataset:
    name: str
    trajectories: List[Trajectory]
    transitions: Optional[np.ndarray] = None    # (n_actions, n_states, n_states)
    features: Optional[np.ndarray] = None       # (n_states, n_actions, n_features)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Convenience properties
    @property
    def n_states(self) -> int: ...
    @property
    def n_actions(self) -> int: ...
    @property
    def is_continuous(self) -> bool: ...

def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Download with progress bar, retries, checksum verification."""
    # Uses tqdm for progress
    # Retries 3x with exponential backoff
    # Verifies SHA256 if provided in dataset metadata

def get_cache_dir() -> Path:
    """Returns ~/.econirl/datasets/, creates if needed."""
    # Respects ECONIRL_CACHE_DIR env var if set
```

**Download flow:**
1. Check cache → return immediately if exists
2. Download to temp file with progress bar
3. Verify checksum (if available)
4. Move to cache location
5. Extract if archive

## Synthetic Dataset Generators

**Gridworld:**
```python
load_dataset('gridworld',
    size=10,              # 10x10 grid
    n_obstacles=5,        # Random obstacle cells
    goal_reward=10.0,     # Reward at goal
    step_cost=-0.1,       # Per-step penalty
    n_trajectories=100,   # Expert demonstrations
    trajectory_length=50, # Max steps per trajectory
    noise=0.1)            # Action noise (stochasticity)
```

**Objectworld** (Levine et al.):
```python
load_dataset('objectworld',
    size=32,              # 32x32 grid
    n_objects=50,         # Colored objects placed randomly
    n_colors=2,           # Inner/outer colors
    n_trajectories=100,
    discount=0.9)
# Features: distance to nearest object of each color
```

**Pursuit-Evasion** (Ziebart 2010):
```python
load_dataset('pursuit_evasion',
    grid_size=4,          # 4x4 grid
    n_agents=3,           # 1 evader + 2 pursuers
    n_trajectories=5,
    trajectory_length=40)
# Joint state space, tests multi-agent IRL
```

All generators use `np.random.default_rng(seed)` for reproducibility.

## Real Trajectory Loaders

**Pittsburgh Taxi (Ziebart 2008):**
```python
load_dataset('pittsburgh_taxi',
    split='train')        # 'train' (20%) or 'test' (80%)
# Downloads GPS traces, discretizes to road network MDP
# ~300k states (road segments), ~900k actions (intersection turns)
# Returns: trajectories as state-action sequences on road graph
```

**NGSIM (Highway driving):**
```python
load_dataset('ngsim',
    location='i80',       # 'i80' or 'us101'
    discretize=True)      # Discretize to lane/position grid
# Raw: continuous (x, y, velocity) per vehicle per frame
# Discretized: (lane, longitudinal_bin, speed_bin) states
# Actions: lane_change_left, lane_change_right, accelerate, decelerate, maintain
```

**HighD (Drone highway data):**
```python
load_dataset('highd')
# Raises InstructionError with registration link:
# "HighD requires registration at https://www.highd-dataset.com/
#  After downloading, place files in ~/.econirl/datasets/highd/"
# Once files present, parses same format as NGSIM
```

**Discretization approach:** All real datasets include a `discretize_trajectory()` method to convert continuous observations to tabular MDPs compatible with existing estimators.

## D4RL Continuous Control Datasets

**Shared structure for all D4RL datasets:**
```python
load_dataset('d4rl_halfcheetah', quality='expert')  # 'expert', 'medium', 'random'
load_dataset('d4rl_hopper', quality='expert')
load_dataset('d4rl_ant', quality='expert')
load_dataset('d4rl_walker2d', quality='expert')
```

**Return format:**
```python
dataset.trajectories     # List[Trajectory] with continuous states/actions
dataset.is_continuous    # True
dataset.state_dim        # e.g., 17 for HalfCheetah
dataset.action_dim       # e.g., 6 for HalfCheetah
dataset.metadata         # {'env': 'HalfCheetah-v2', 'quality': 'expert', ...}

# Each trajectory contains:
traj.states   # np.ndarray, shape (T, state_dim)
traj.actions  # np.ndarray, shape (T, action_dim)
traj.rewards  # np.ndarray, shape (T,) - original rewards from environment
```

**No transitions/features:** Continuous MDPs don't have discrete transition matrices. Users need continuous-state IRL methods (GAIL, AIRL with function approximation) for these.

**Source:** Downloads HDF5 files directly from `rail.eecs.berkeley.edu/datasets/offline_rl/`

## Error Handling

**Network failures:**
```python
class DownloadError(Exception):
    """Raised when download fails after retries."""

# Retry logic: 3 attempts with exponential backoff (1s, 2s, 4s)
# Clear error message with URL and suggestion to check connectivity
```

**Corrupted cache:**
```python
# Checksum verification on download
# If cached file fails checksum, delete and re-download
load_dataset('ngsim', force_download=True)  # Force fresh download
```

**Missing dependencies:**
```python
# D4RL needs h5py
load_dataset('d4rl_halfcheetah')
# Raises: "D4RL datasets require h5py. Install with: pip install h5py"

# NGSIM/HighD need pandas for CSV parsing
load_dataset('ngsim')
# Raises: "NGSIM requires pandas. Install with: pip install pandas"
```

**Registration-gated datasets:**
```python
load_dataset('highd')
# Raises DatasetRegistrationRequired with clear instructions
```

**Cache management:**
```python
from econirl.datasets import clear_cache, cache_info
clear_cache('ngsim')      # Remove specific dataset
clear_cache()             # Remove all cached data
cache_info()              # Show cache size and contents
```

## Files to Create

| File | Purpose |
|------|---------|
| `src/econirl/datasets/__init__.py` | Public API exports |
| `src/econirl/datasets/base.py` | Dataset class, download utils |
| `src/econirl/datasets/cache.py` | Cache management |
| `src/econirl/datasets/gridworld.py` | Gridworld generator |
| `src/econirl/datasets/objectworld.py` | Objectworld generator |
| `src/econirl/datasets/pursuit_evasion.py` | Pursuit-evasion generator |
| `src/econirl/datasets/pittsburgh_taxi.py` | Taxi GPS loader |
| `src/econirl/datasets/ngsim.py` | NGSIM highway loader |
| `src/econirl/datasets/highd.py` | HighD loader (registration-gated) |
| `src/econirl/datasets/d4rl.py` | All D4RL loaders |
| `tests/test_datasets.py` | Unit tests |

## References

- Ziebart et al. (2008) - Maximum Entropy Inverse Reinforcement Learning
- Ziebart et al. (2010) - Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy
- Levine et al. - Objectworld environment
- Fu et al. (2020) - D4RL: Datasets for Deep Data-Driven Reinforcement Learning
- NGSIM - Next Generation Simulation (FHWA)
- HighD - The highD Dataset

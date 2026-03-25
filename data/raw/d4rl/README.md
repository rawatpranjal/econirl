# D4RL Expert Trajectory Data

## Contents

This directory contains expert demonstration datasets from D4RL (Datasets for Deep Data-Driven Reinforcement Learning) in HDF5 format, downloaded from the Berkeley RAIL lab hosting.

### Files

| File | Task | Obs Dim | Act Dim | Steps | Episodes | Size |
|------|------|---------|---------|-------|----------|------|
| `halfcheetah-expert-v2.hdf5` | HalfCheetah-v2 | 17 | 6 | 1,000,000 | 1,000 | 226 MB |
| `hopper-expert-v2.hdf5` | Hopper-v2 | 11 | 3 | 1,000,000 | 1,027 | 145 MB |
| `walker2d-expert-v2.hdf5` | Walker2d-v2 | 17 | 6 | 1,000,000 | 1,000 | 220 MB |

### HDF5 Structure

Each file contains the following datasets:

- `observations` - float32, shape `(N, obs_dim)` - state observations
- `next_observations` - float32, shape `(N, obs_dim)` - next state observations
- `actions` - float32, shape `(N, act_dim)` - continuous actions
- `rewards` - float32, shape `(N,)` - scalar rewards
- `terminals` - bool, shape `(N,)` - True when episode ended due to terminal state
- `timeouts` - bool, shape `(N,)` - True when episode ended due to time limit
- `infos` - group containing additional info (e.g., `qpos`, `qvel`)
- `metadata` - group containing dataset metadata

Episodes can be reconstructed by splitting at indices where `terminals | timeouts` is True.

## Source

Downloaded from the D4RL dataset hosting:

```
http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/
```

Specific URLs:
- `http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_expert-v2.hdf5`
- `http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_expert-v2.hdf5`
- `http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_expert-v2.hdf5`

## How to Re-download

```bash
mkdir -p data/raw/d4rl
cd data/raw/d4rl

curl -L -O "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_expert-v2.hdf5"
curl -L -O "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_expert-v2.hdf5"
curl -L -O "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_expert-v2.hdf5"

# Rename to match D4RL naming convention
mv halfcheetah_expert-v2.hdf5 halfcheetah-expert-v2.hdf5
mv hopper_expert-v2.hdf5 hopper-expert-v2.hdf5
mv walker2d_expert-v2.hdf5 walker2d-expert-v2.hdf5
```

## Alternative: Using Minari (D4RL Successor)

Minari (pip install minari) also hosts these datasets. Equivalent datasets:

- `mujoco/halfcheetah/expert-v0`
- `mujoco/hopper/expert-v0`
- `mujoco/walker2d/expert-v0`

```python
import minari
dataset = minari.load_dataset("mujoco/halfcheetah/expert-v0", download=True)
```

## Loading the Data

```python
import h5py
import numpy as np

with h5py.File("data/raw/d4rl/halfcheetah-expert-v2.hdf5", "r") as f:
    observations = f["observations"][:]      # (1000000, 17)
    actions = f["actions"][:]                # (1000000, 6)
    rewards = f["rewards"][:]                # (1000000,)
    terminals = f["terminals"][:]            # (1000000,)
    timeouts = f["timeouts"][:]              # (1000000,)

    # Split into episodes
    done_indices = np.where(terminals | timeouts)[0]
    episodes = []
    start = 0
    for end in done_indices:
        episodes.append({
            "observations": observations[start:end+1],
            "actions": actions[start:end+1],
            "rewards": rewards[start:end+1],
        })
        start = end + 1
```

## References

- Fu et al. (2020). "D4RL: Datasets for Deep Data-Driven Reinforcement Learning." arXiv:2004.06750.
- Minari documentation: https://minari.farama.org/

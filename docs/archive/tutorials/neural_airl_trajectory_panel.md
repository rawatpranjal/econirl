# NeuralAIRL with TrajectoryPanel

`TrajectoryPanel` stores its arrays in JAX. `NeuralAIRL` trains its networks in Torch. You can still pass a `TrajectoryPanel` directly to `fit()`. The estimator handles the framework boundary during minibatching, so your panel workflow can stay on the JAX side.

## Build a small panel

This toy dataset uses a small discrete state space with two actions. We start from a DataFrame and convert it to a `TrajectoryPanel`.

```python
import jax.numpy as jnp
import numpy as np
import pandas as pd

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import TrajectoryPanel
from econirl.estimators import NeuralAIRL

records = []
for uid in range(20):
    state = 0
    for t in range(10):
        action = int(state >= 3)
        next_state = min(state + 1 + action, 5)
        records.append(
            {
                "id": uid,
                "state": state,
                "action": action,
                "next_state": next_state,
            }
        )
        state = next_state

df = pd.DataFrame(records)
panel = TrajectoryPanel.from_dataframe(
    df,
    state="state",
    action="action",
    id="id",
)
```

## Fit NeuralAIRL from the panel

The feature projection below uses a `RewardSpec` backed by JAX arrays. This keeps the full data path in JAX until the neural training step.

```python
n_states = int(panel.all_states.max()) + 1
n_actions = 2

features = jnp.zeros((n_states, n_actions, 2))
features = features.at[:, :, 0].set(
    -jnp.arange(n_states)[:, None] / max(n_states - 1, 1)
)
features = features.at[:, 1, 1].set(-1.0)

reward_spec = RewardSpec(
    features,
    names=["state_cost", "action_1_cost"],
)

model = NeuralAIRL(
    n_actions=n_actions,
    discount=0.95,
    max_epochs=50,
    patience=10,
    reward_hidden_dim=32,
    reward_num_layers=2,
    shaping_hidden_dim=32,
    shaping_num_layers=2,
    policy_hidden_dim=32,
    policy_num_layers=2,
    batch_size=64,
    disc_steps=1,
)

model.fit(panel, features=reward_spec)

proba = model.predict_proba(np.array([0, 1, 2, 3, 4, 5]))
print(np.round(proba, 3))
print(model.params_)
```

## Fit from a DataFrame when you want column names

If your pipeline is still in pandas, you can fit from the DataFrame directly. The `TrajectoryPanel` route is useful when the rest of your workflow already uses econirl panel objects.

```python
model_df = NeuralAIRL(
    n_actions=n_actions,
    discount=0.95,
    max_epochs=50,
    patience=10,
    disc_steps=1,
)

model_df.fit(
    df,
    state="state",
    action="action",
    id="id",
    features=reward_spec,
)
```

## Why this pattern matters

Many econirl workflows build `TrajectoryPanel` objects early because they are the common container for simulation and estimation. `NeuralAIRL` now accepts that object directly. The panel stays in JAX form until a minibatch is drawn for Torch training. This removes the need for manual array conversion in user code and keeps the estimator interface consistent across structural and neural methods.

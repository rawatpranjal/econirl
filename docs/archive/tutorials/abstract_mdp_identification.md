# Identification and Counterfactuals in Abstract MDPs

This example strips away application detail and keeps only the logic of the appendix. We build a tiny abstract MDP and compare three objects that all match observed behavior in the original environment. The true reward does so structurally. A shaped reward does so through potential based shaping. The advantage scores do so because their softmax reproduces the observed choice probabilities. The lesson is simple. Agreement in the observed environment does not guarantee agreement after the transition law changes.

## Setup

We use three states and two actions. State `2` is absorbing. Action `0` means continue and action `1` means exit. Under the baseline dynamics, continue moves the agent forward and exit jumps straight to the absorbing state. The true reward makes continue attractive in the first state and less attractive in the second state.

```python
import jax.numpy as jnp

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem

problem = DDCProblem(num_states=3, num_actions=2, discount_factor=0.95)

# action 0 is continue
# action 1 is exit
P_continue = jnp.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
)
P_exit = jnp.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
)
transitions = jnp.stack([P_continue, P_exit])

reward_true = jnp.array(
    [
        [1.0, 0.0],
        [-0.2, 0.0],
        [0.0, 0.0],
    ]
)

operator = SoftBellmanOperator(problem, transitions)
oracle = value_iteration(operator, reward_true)

oracle.policy
```

The oracle policy is approximately

```python
Array(
    [
        [0.713, 0.287],
        [0.450, 0.550],
        [0.500, 0.500],
    ]
)
```

## Three observationally equivalent objects

The appendix studies three different objects. The first is the true reward. The second is the advantage function, which is identified from observed choice probabilities. The third is a shaped reward that differs from the true reward by a potential term. In this toy example we construct the shaped reward directly from a hand chosen potential.

$$
g(s,a) = r(s,a) - \Phi(s) + \beta \mathbb{E}[\Phi(s') \mid s,a].
$$

```python
advantage = oracle.Q - oracle.V[:, None]

phi = jnp.array([0.8, 0.3, 0.0])
expected_phi = jnp.einsum("ast,t->as", transitions, phi).T
reward_shaped = reward_true - phi[:, None] + problem.discount_factor * expected_phi

policy_from_advantage = jnp.exp(advantage)
policy_from_advantage = policy_from_advantage / policy_from_advantage.sum(
    axis=1, keepdims=True
)

shaped = value_iteration(operator, reward_shaped)

float(jnp.max(jnp.abs(policy_from_advantage - oracle.policy)))
float(jnp.max(jnp.abs(shaped.policy - oracle.policy)))
```

Both differences are zero up to numerical tolerance. This is the identification point in its most compact form. In the baseline environment, observed behavior alone does not tell us whether we have recovered the structural reward, a shaped reward, or only the advantage scores.

## Type A and Type B counterfactuals

For a Type A exercise, nothing structural changes. We only evaluate behavior at new state realizations under the same transition law. In that case the advantage scores are enough because only within state action differences matter. For a Type B exercise, the transition law changes and the model must be solved again. That is where the distinction between structural reward and observationally equivalent objects becomes important.

We now change the continue action in state `0`. Instead of moving to state `1`, it keeps the agent in state `0`. This makes continuation much more persistent than before.

```python
from econirl.simulation.counterfactual import neural_transition_counterfactual

P_continue_cf = jnp.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
)
transitions_cf = jnp.stack([P_continue_cf, P_exit])

cf_true = neural_transition_counterfactual(
    reward_true,
    transitions_cf,
    problem,
    transitions,
)
cf_shaped = neural_transition_counterfactual(
    reward_shaped,
    transitions_cf,
    problem,
    transitions,
)
cf_advantage = neural_transition_counterfactual(
    advantage,
    transitions_cf,
    problem,
    transitions,
)

cf_true.counterfactual_policy[0]
cf_shaped.counterfactual_policy[0]
cf_advantage.counterfactual_policy[0]
```

The first state now shows a sharp split. The true reward implies a counterfactual policy close to `[0.999, 0.001]`. The shaped reward implies a noticeably different policy close to `[0.806, 0.194]`. The advantage scores differ even more and stay near `[0.713, 0.287]`. All three objects matched the original data. Only one of them supports the changed environment.

## What this means for `econirl`

This is the package level lesson of the appendix. `econirl` can represent many reward like objects under one interface, but not all of them support the same scientific claim. Reduced form scores can be useful for fit and Type A state extrapolation. Structural reward recovery matters when the user wants Type B counterfactuals under changed transitions or changed action sets. That is why the package keeps reward recovery and counterfactual validity separate from in sample fit, and why estimators such as anchored AIRL, IQ Learn, GLADIUS, and NFXP are valuable even when a reduced form model already matches observed behavior.

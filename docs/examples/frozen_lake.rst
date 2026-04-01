FrozenLake Navigation
=====================

.. image:: /_static/frozen_lake_overview.png
   :alt: FrozenLake 4x4 grid layout showing start, frozen, hole, and goal cells alongside slippery transition probability diagram.
   :width: 100%

This example applies IRL estimators to the classic FrozenLake environment from reinforcement learning. An agent navigates a 4x4 frozen lake to reach the goal cell without falling into holes. On the slippery surface, each action moves the agent in the intended direction with probability one third and in each perpendicular direction with probability one third. The environment has 16 states and 4 actions.

FrozenLake is the simplest stochastic DDC testbed. It has known ground-truth parameters, making it ideal for verifying parameter recovery and comparing estimators on a problem small enough to inspect by hand.

Quick start
-----------

.. code-block:: python

   from econirl.environments.frozen_lake import FrozenLakeEnvironment
   from econirl.datasets.frozen_lake import load_frozen_lake

   env = FrozenLakeEnvironment(discount_factor=0.99)
   panel = load_frozen_lake(n_individuals=500, n_periods=100, as_panel=True)

Estimation
----------

All estimators recover the three utility parameters from trajectory data.

.. code-block:: python

   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.estimation.ccp import CCPEstimator
   from econirl.estimation.mce_irl import MCEIRLEstimator
   from econirl.preferences.linear import LinearUtility
   import jax.numpy as jnp

   utility = LinearUtility(features=env.feature_matrix, parameter_names=env.parameter_names)
   init = jnp.zeros(3)

   nfxp = NFXPEstimator(env.problem_spec, env.transition_matrices)
   result = nfxp.fit(panel, utility, init_params=init)

   for name, val in zip(env.parameter_names, result.parameters):
       print(f"{name}: {float(val):.4f}")

The true parameters are step penalty of negative 0.04, goal reward of 1.0, and hole penalty of negative 1.0. With 500 trajectories of 100 steps, all three estimators should recover these values closely.

Environment details
-------------------

The default 4x4 map uses the same layout as Gymnasium FrozenLake-v1. The econirl implementation does not depend on Gymnasium at runtime. The transition logic is hardcoded directly, keeping the dependency footprint minimal.

Holes at states 5, 7, 11, and 12 are absorbing. The goal at state 15 is also absorbing. The start state is 0. On the slippery surface, the agent's intended action executes with probability one third, and each of the two perpendicular directions executes with probability one third. Setting ``is_slippery=False`` makes transitions deterministic.

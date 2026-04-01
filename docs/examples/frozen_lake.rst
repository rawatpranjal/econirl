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

Three estimators recover the utility parameters from 500 trajectories of 100 steps each.

.. code-block:: python

   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.estimation.mce_irl import MCEIRLEstimator
   from econirl.preferences.linear import LinearUtility

   utility = LinearUtility(feature_matrix=env.feature_matrix, parameter_names=env.parameter_names)
   nfxp = NFXPEstimator()
   result = nfxp.estimate(panel, utility, env.problem_spec, env.transition_matrices)

MCE-IRL recovers the true parameters most accurately on this environment. The absorbing states (holes and goal) concentrate the data distribution, which makes NFXP identification harder in this small state space.

.. list-table:: Parameter Recovery (500 trajectories, 100 steps)
   :header-rows: 1

   * - Parameter
     - True
     - NFXP
     - CCP
     - MCE-IRL
   * - step_penalty
     - -0.0400
     - -1.2365
     - 0.1220
     - -0.0340
   * - goal_reward
     - 1.0000
     - -0.1790
     - 1.0026
     - 1.0115
   * - hole_penalty
     - -1.0000
     - -2.1677
     - -0.9040
     - -0.9772

MCE-IRL recovers the step penalty within 0.006, the goal reward within 0.012, and the hole penalty within 0.023 of the true values. MCE-IRL standard errors are 0.014 for the step penalty, 0.019 for the goal reward, and 0.009 for the hole penalty.

Counterfactual analysis
-----------------------

Doubling the hole penalty from negative 1.0 to negative 2.0 makes the agent more cautious around holes. The welfare change is negative 75.6, reflecting the harsher environment.

.. code-block:: python

   from econirl.simulation.counterfactual import elasticity_analysis

   ea = elasticity_analysis(
       result, utility, env.problem_spec, env.transition_matrices,
       parameter_name="hole_penalty",
       pct_changes=[-0.50, -0.25, 0.25, 0.50],
   )

.. list-table:: Hole Penalty Elasticity
   :header-rows: 1

   * - % Change
     - Welfare Change
     - Avg Policy Change
   * - -50%
     - +46.855
     - 0.065
   * - -25%
     - +21.487
     - 0.026
   * - +25%
     - -18.997
     - 0.002
   * - +50%
     - -37.888
     - 0.003

The response is asymmetric. Reducing the hole penalty (making holes less dangerous) produces large welfare gains and substantial policy changes. Increasing the penalty has diminishing marginal effect because the agent is already avoiding holes under the baseline parameters.

Environment details
-------------------

The default 4x4 map uses the same layout as Gymnasium FrozenLake-v1. The econirl implementation does not depend on Gymnasium at runtime. The transition logic is hardcoded directly, keeping the dependency footprint minimal.

Holes at states 5, 7, 11, and 12 are absorbing. The goal at state 15 is also absorbing. The start state is 0. On the slippery surface, the agent's intended action executes with probability one third, and each of the two perpendicular directions executes with probability one third. Setting ``is_slippery=False`` makes transitions deterministic.

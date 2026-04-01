FrozenLake Navigation
=====================

.. image:: /_static/frozen_lake_overview.png
   :alt: FrozenLake 4x4 grid layout showing start, frozen, hole, and goal cells alongside slippery transition probability diagram.
   :width: 100%

This example applies three estimators to the classic FrozenLake environment from reinforcement learning. An agent navigates a 4x4 frozen lake to reach the goal cell without falling into holes. On the slippery surface, each action moves the agent in the intended direction with probability one third and in each perpendicular direction with probability one third. The environment has 16 states and 4 actions.

FrozenLake is the simplest stochastic DDC testbed. It has known ground-truth parameters, making it ideal for verifying parameter recovery and comparing estimators on a problem small enough to inspect by hand.

Quick start
-----------

.. code-block:: python

   from econirl.environments.frozen_lake import FrozenLakeEnvironment

   env = FrozenLakeEnvironment(discount_factor=0.99)
   panel = env.generate_panel(n_individuals=500, n_periods=100)

Estimation
----------

Three estimators recover the utility parameters from 500 trajectories of 100 steps each. The true parameters are a step penalty of negative 0.04, a goal reward of 1.0, and a hole penalty of negative 1.0.

.. code-block:: python

   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.estimation.ccp import CCPEstimator
   from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
   from econirl.preferences.linear import LinearUtility

   utility = LinearUtility(feature_matrix=env.feature_matrix, parameter_names=env.parameter_names)
   transitions = env.transition_matrices

   nfxp_result = NFXPEstimator(se_method="robust").estimate(panel, utility, env.problem_spec, transitions)
   ccp_result = CCPEstimator(num_policy_iterations=20, se_method="robust").estimate(panel, utility, env.problem_spec, transitions)
   mce_result = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.1, outer_max_iter=500)).estimate(panel, utility, env.problem_spec, transitions)

.. list-table:: Parameter Recovery (500 individuals, 100 periods)
   :header-rows: 1

   * - Parameter
     - True
     - NFXP
     - CCP (K=20)
     - MCE-IRL
   * - step_penalty
     - -0.0400
     - -0.1761
     - 0.1220
     - -0.0340
   * - goal_reward
     - 1.0000
     - 0.8709
     - 1.0026
     - 1.0115
   * - hole_penalty
     - -1.0000
     - -1.1179
     - -0.9040
     - -0.9772

MCE-IRL recovers all three parameters within 0.03 of the true values, with standard errors of 0.016 for the step penalty, 0.021 for the goal reward, and 0.018 for the hole penalty. NFXP finds the correct signs but produces standard errors above 4.0 for every parameter. The Hessian condition number is 391,106, indicating weak identification. FrozenLake has absorbing states at the four holes and the goal, which concentrates the data distribution and collapses the likelihood surface for NFXP. CCP recovers goal_reward and hole_penalty accurately but misses the sign on step_penalty. Its standard errors are numerically zero, an artifact of the NPL Hessian at absorbing-state boundaries.

Post-estimation diagnostics
---------------------------

The ``etable`` function places all three models side by side with significance stars.

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result, mce_result))

.. code-block:: text

   ==============================================================
                       NFXP (Nested Fixed Point)    NPL (K=20)MCE IRL (Ziebart 2010)
   ==============================================================
           step_penalty       -0.1761     0.1220***     -0.0340**
                             (4.4964)      (0.0000)      (0.0162)
            goal_reward        0.8709     1.0026***     1.0115***
                             (4.4458)      (0.0000)      (0.0211)
           hole_penalty       -1.1179    -0.9040***    -0.9772***
                             (4.4422)      (0.0000)      (0.0184)
   --------------------------------------------------------------
           Observations        50,000        50,000        50,000
         Log-Likelihood    -40,701.35    -40,716.69    -40,865.90
                    AIC      81,408.7      81,439.4      81,737.8
   ==============================================================

The Vuong test comparing NFXP and MCE-IRL yields a z-statistic of 11.07 with a p-value below 0.0001, favoring NFXP on log-likelihood. This reflects the absorbing-state advantage that NFXP exploits in its likelihood computation rather than genuine parameter accuracy.

.. code-block:: python

   from econirl.inference import vuong_test, brier_score, kl_divergence

   obs_states = jnp.array(panel.get_all_states())
   obs_actions = jnp.array(panel.get_all_actions())
   vt = vuong_test(nfxp_result.policy, mce_result.policy, obs_states, obs_actions)

.. list-table:: Fit Metrics
   :header-rows: 1

   * - Metric
     - NFXP
     - CCP
     - MCE-IRL
   * - Brier Score
     - 0.4354
     - 0.4355
     - 0.4370
   * - KL Divergence
     - 0.0004
     - 0.0007
     - 0.0125
   * - Log-Likelihood
     - -40,701
     - -40,717
     - -40,866

Brier scores are nearly identical across all three estimators at 0.435, confirming that all three produce similar predictive accuracy. The KL divergence is lowest for NFXP at 0.0004, showing its model-implied CCPs closely match the empirical frequencies. MCE-IRL has a slightly higher KL of 0.013 because it optimizes feature matching rather than the likelihood directly.

The reward identifiability test reports that the domain graph is not strongly connected because absorbing states (holes and goal) have no outgoing transitions. This means reward parameters at absorbing states are formally unidentified in the Kim et al. (2021) sense. MCE-IRL still recovers the parameters well because its occupancy-weighted feature matching is robust to absorbing states, while NFXP struggles precisely because its likelihood depends on transitions that never occur from absorbing states.

Counterfactual analysis
-----------------------

Doubling the hole penalty from negative 1.0 to negative 2.0 makes the agent more cautious around holes.

.. code-block:: python

   from econirl.simulation.counterfactual import elasticity_analysis

   ea = elasticity_analysis(
       mce_result, utility, env.problem_spec, transitions,
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

Running the example
-------------------

.. code-block:: bash

   python examples/frozen-lake/run_estimation.py

Environment details
-------------------

The default 4x4 map uses the same layout as Gymnasium FrozenLake-v1. The econirl implementation does not depend on Gymnasium at runtime. The transition logic is hardcoded directly, keeping the dependency footprint minimal.

Holes at states 5, 7, 11, and 12 are absorbing. The goal at state 15 is also absorbing. The start state is 0. On the slippery surface, the agent's intended action executes with probability one third, and each of the two perpendicular directions executes with probability one third. Setting ``is_slippery=False`` makes transitions deterministic.

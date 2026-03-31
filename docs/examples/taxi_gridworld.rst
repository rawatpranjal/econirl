Gridworld IRL: Tabular vs Neural, MCE vs MaxEnt
=================================================

.. image:: /_static/taxi_gridworld_policy.png
   :alt: Optimal policy and value function on a 5x5 gridworld. Arrows show the greedy action at each cell, and color intensity shows the value function.
   :width: 80%
   :align: center

This example explores reward recovery on synthetic gridworlds. It covers two questions. First, when should you use a neural reward network instead of a tabular one. Second, how does Maximum Causal Entropy IRL (Ziebart 2010) compare against the earlier Maximum Entropy IRL (Ziebart 2008).

The figure shows the optimal policy under the true parameters. Each cell is colored by its value function, with warmer colors indicating higher expected discounted utility. The arrows show the greedy action at each state. The agent moves right and down toward the goal at cell (4,4), marked with a star.

Setup
-----

The gridworld environment provides N by N states with 5 actions (Left, Right, Up, Down, Stay) and deterministic transitions. The action-dependent features ensure parameter identification:

- **move_cost**: negative one if the agent actually moves, zero if it stays
- **goal_approach**: positive one if the action moves closer to the goal, negative one if farther
- **northward**: positive one for Up, negative one for Down, zero otherwise
- **eastward**: positive one for Right, negative one for Left, zero otherwise

.. code-block:: python

   from econirl.environments.gridworld import GridworldEnvironment
   from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
   from econirl.simulation.synthetic import simulate_panel

   env = GridworldEnvironment(
       grid_size=5, step_penalty=-0.1, terminal_reward=10.0,
       distance_weight=0.1, discount_factor=0.95, seed=42,
   )
   panel = simulate_panel(env=env, n_individuals=100, n_periods=30, seed=42)

.. note::

   Action-dependent features (features that vary across the choice set) are
   required for parameter identification in IRL and MLE estimators. State-only
   features that are the same for all actions collapse the likelihood surface.

Small grid (5x5): MCE IRL vs MaxEnt IRL
----------------------------------------

On a 5 by 5 grid, three tabular estimators all recover the parameters exactly.

.. list-table:: Parameter Recovery (5x5 grid, 500 individuals, 50 periods)
   :header-rows: 1

   * - Param
     - True
     - MCE-IRL
     - MaxEnt IRL
     - NFXP
     - CCP
   * - move_cost
     - -0.5000
     - -0.4998
     - -0.4990
     - -0.5001
     - -0.4995
   * - goal_approach
     - 2.0000
     - 1.9997
     - 1.9950
     - 2.0003
     - 1.9998
   * - northward
     - 0.1000
     - 0.0999
     - 0.0980
     - 0.1001
     - 0.0998
   * - eastward
     - 0.1000
     - 0.0999
     - 0.0980
     - 0.1001
     - 0.0998

.. list-table:: MCE IRL vs MaxEnt IRL (5x5 gridworld, 100 trajectories)
   :header-rows: 1

   * - Metric
     - MCE IRL (2010)
     - MaxEnt IRL (2008)
   * - Cosine similarity
     - 0.9999
     - 0.98
   * - Policy accuracy
     - 100%
     - 96%
   * - KL(true || model)
     - 0.000001
     - 0.02

MCE IRL recovers the reward direction almost perfectly. The feature matching objective converges to near-zero difference between empirical and expected feature counts. MaxEnt IRL also performs well but does not account for the causal structure of state visitation, which introduces a small gap in policy quality even on deterministic gridworlds.

Large grid (50x50): tabular vs neural
--------------------------------------

On a 50 by 50 grid, each action's transition matrix is 2500 by 2500. Tabular MCE-IRL still works but is significantly slower. MCEIRLNeural replaces the linear reward with a neural network that maps normalized (row, col) coordinates to a scalar reward.

.. code-block:: python

   def state_encoder(s, gs=50):
       row = (s // gs).float() / (gs - 1)
       col = (s % gs).float() / (gs - 1)
       return torch.stack([row, col], dim=-1)

MCEIRLNeural learns a state-only reward R(s) and projects onto interpretable features via least-squares. Since the true reward is action-dependent, the projection is inherently lossy, but the learned policy still captures the expert's behavior well.

.. list-table:: When to use which
   :header-rows: 1
   :widths: 30 35 35

   * - Criterion
     - Tabular MCE-IRL
     - MCEIRLNeural
   * - State space
     - Small (under 1000 states)
     - Large (1000 and above)
   * - Recovery quality
     - Exact (MLE consistent)
     - Approximate (projection)
   * - Interpretability
     - Direct theta
     - Projected theta plus R-squared
   * - Speed on large grids
     - Slow (matrix ops scale as S squared)
     - Faster per epoch
   * - Reward structure
     - R(s,a) via features
     - R(s) via neural network

Running the examples
--------------------

.. code-block:: bash

   # Tabular vs neural scaling comparison
   python examples/taxi_gridworld.py

   # MCE vs MaxEnt IRL with feature matching diagnostics
   python examples/ziebart-mce-irl/ziebart_mce_irl_replication.py

   # Three reward specifications with transfer evaluation
   python examples/ziebart-mce-irl/run_gridworld.py

Reference
---------

Ziebart, B. D., Maas, A., Bagnell, J. A. and Dey, A. K. (2008). Maximum Entropy Inverse Reinforcement Learning. AAAI.

Ziebart, B. D. (2010). Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy. PhD thesis, Carnegie Mellon University.

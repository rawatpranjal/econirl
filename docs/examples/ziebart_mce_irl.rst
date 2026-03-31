Ziebart MCE-IRL Gridworld
=========================

This example replicates the gridworld experiment from Ziebart (2008, 2010) using both Maximum Causal Entropy IRL and the earlier Maximum Entropy IRL. An agent navigates a 5 by 5 grid toward a terminal state in the bottom-right corner. The reward has three components: a step penalty, a terminal reward, and a distance weight that encourages movement toward the goal.

The example generates 100 expert trajectories of 30 steps each from the optimal soft policy, then recovers the reward parameters using MCE IRL and MaxEnt IRL. Because IRL rewards are identified only up to additive constants and multiplicative scale (Kim et al. 2021, Cao and Cohen 2021), the comparison focuses on cosine similarity and policy quality rather than raw parameter values.

.. code-block:: python

   from econirl.environments.gridworld import GridworldEnvironment
   from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
   from econirl.simulation.synthetic import simulate_panel

   env = GridworldEnvironment(
       grid_size=5, step_penalty=-0.1, terminal_reward=10.0,
       distance_weight=0.1, discount_factor=0.95, seed=42,
   )
   panel = simulate_panel(env=env, n_individuals=100, n_periods=30, seed=42)

   config = MCEIRLConfig(
       learning_rate=0.1, outer_max_iter=500, use_adam=True, verbose=True,
   )
   estimator = MCEIRLEstimator(config=config)
   result = estimator.estimate(
       panel=panel, utility=reward_fn, problem=env.problem_spec,
       transitions=env.transition_matrices, true_params=env.get_true_parameter_vector(),
   )

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

The script also compares policies at key states, showing that MCE IRL matches the true soft policy at every grid cell while MaxEnt IRL occasionally assigns slightly different action probabilities near the boundary.

.. code-block:: bash

   python examples/ziebart-mce-irl/ziebart_mce_irl_replication.py

A second script in the same directory runs a comparative benchmark across three reward specifications (full state-action features, Rust-style features, and pure state-only features) with in-sample, out-of-sample, and transfer evaluation.

.. code-block:: bash

   python examples/ziebart-mce-irl/run_gridworld.py

Reference
---------

Ziebart, B. D., Maas, A., Bagnell, J. A. and Dey, A. K. (2008). Maximum Entropy Inverse Reinforcement Learning. AAAI.

Ziebart, B. D. (2010). Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy. PhD thesis, Carnegie Mellon University.

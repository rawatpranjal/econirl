Quickstart
==========

This page shows how to fit a structural model in three steps. It assumes you know what dynamic discrete choice models and inverse reinforcement learning are.

Fitting a model
---------------

Load the Rust bus engine replacement dataset and fit two estimators.

.. code-block:: python

   from econirl import NFXP, CCP
   from econirl.datasets import load_rust_bus

   df = load_rust_bus()

   nfxp = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   ccp  = CCP(discount=0.9999, num_policy_iterations=5).fit(df, state="mileage_bin", action="replaced", id="bus_id")

Both estimators recover the same parameters. NFXP uses nested fixed-point maximum likelihood. CCP uses Hotz-Miller inversion with nested pseudo-likelihood iterations. CCP is faster on this problem because it avoids the inner Bellman loop.

.. code-block:: python

   print(nfxp.summary())
   print(ccp.summary())
   print(nfxp.params_)   # {'theta_c': 0.001, 'RC': 3.07}
   print(nfxp.se_)        # {'theta_c': 0.00003, 'RC': 0.12}
   print(nfxp.conf_int()) # {'theta_c': (0.00094, 0.00106), 'RC': (2.83, 3.31)}

Predicting choices
------------------

Every fitted estimator can predict choice probabilities for new states.

.. code-block:: python

   import numpy as np
   proba = nfxp.predict_proba(np.array([0, 30, 60, 89]))
   # proba[i, 0] = P(keep | mileage=i), proba[i, 1] = P(replace | mileage=i)

Counterfactual analysis
-----------------------

After fitting a model, you can ask what happens under different parameters. This is the main advantage over a predictive model. A gradient-boosted classifier can predict replacement probabilities but cannot answer what happens if the replacement cost doubles.

.. code-block:: python

   cf = nfxp.counterfactual(RC=6.0)
   print(cf.policy[50, :])  # P(keep), P(replace) at mileage 50

   # The manager delays replacement when the cost is higher
   print(f"Baseline P(replace|m=50): {nfxp.policy_[50, 1]:.3f}")
   print(f"Doubled RC P(replace|m=50): {cf.policy[50, 1]:.3f}")

Simulating data
---------------

You can generate synthetic data from the estimated model. This is useful for Monte Carlo studies and for validating that the estimator recovers the true parameters from its own data-generating process.

.. code-block:: python

   sim_df = nfxp.simulate(n_agents=500, n_periods=100, seed=42)
   print(sim_df.head())
   # Columns: agent_id, period, state, action

   # Re-estimate on simulated data to verify recovery
   nfxp2 = NFXP(discount=0.9999).fit(sim_df, state="state", action="action", id="agent_id")
   print(nfxp2.params_)  # Should be close to nfxp.params_

Neural estimators
-----------------

When the state space is too large for exact methods, use a neural estimator. NeuralGLADIUS learns a nonlinear reward function via Q-networks and does not need a transition matrix.

.. code-block:: python

   from econirl import NeuralGLADIUS

   model = NeuralGLADIUS(
       n_actions=8, discount=0.95,
       q_hidden_dim=128, q_num_layers=3,
       max_epochs=300, batch_size=1024,
       verbose=True,
   )
   model.fit(data=panel, context=destinations, features=feature_tensor)

   print(model.params_)          # projected theta
   print(model.projection_r2_)   # how linear is the learned reward?
   print(model.predict_reward(states, actions, contexts))

The projection R-squared tells you how much of the neural reward is explained by your linear features. If R-squared is 0.95, the reward is nearly linear and the projected theta is trustworthy. If R-squared is 0.06, there are strong nonlinear effects that the linear model misses.

Custom features
---------------

To use your own data with custom features, create a RewardSpec.

.. code-block:: python

   import torch
   from econirl import NFXP, RewardSpec

   features = torch.randn(50, 3, 4)  # 50 states, 3 actions, 4 features
   spec = RewardSpec(features, names=["price", "quality", "distance", "brand"])

   model = NFXP(n_states=50, n_actions=3, discount=0.95)
   model.fit(panel, reward=spec, transitions=transitions)
   print(model.params_)

RewardSpec also supports state-only features that are broadcast to all actions.

.. code-block:: python

   state_features = torch.randn(50, 3)  # 50 states, 3 features
   spec = RewardSpec.state_dependent(state_features, names=["x", "y", "z"], n_actions=4)

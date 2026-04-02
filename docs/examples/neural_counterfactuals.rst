Neural Reward Counterfactuals
==============================

This example compares counterfactual predictions from a neural estimator (NeuralGLADIUS) and a structural estimator (NFXP) on two simulated bus engine datasets. The first dataset has a linear reward (the structural model is correctly specified). The second has a nonlinear reward with a cost kink at 300,000 miles (the structural model is misspecified). We simulate 1000 buses over 100 periods for each dataset, giving 100,000 observations per DGP.

The linear DGP uses the Rust (1987) specification with theta_c equal to 0.001 and RC equal to 3.0. The nonlinear DGP adds a penalty of 2.0 utility units to the keep action at states above bin 60, representing a sudden jump in operating costs at high mileage.

Estimation
----------

.. code-block:: python

   from econirl import NFXP
   from econirl.estimators.neural_gladius import NeuralGLADIUS
   from econirl.environments.rust_bus import RustBusEnvironment
   from econirl.simulation.synthetic import simulate_panel

   env = RustBusEnvironment(operating_cost=0.001, replacement_cost=3.0, discount_factor=0.9999)
   panel = simulate_panel(env, n_individuals=1000, n_periods=100, seed=42)
   df = panel.to_dataframe().rename(columns={"state": "mileage_bin", "action": "replaced"})

   nfxp = NFXP(n_states=90, discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="id")
   gladius = NeuralGLADIUS(n_actions=2, discount=0.9999, max_epochs=500).fit(
       df, state="mileage_bin", action="replaced", id="id")

NFXP recovers theta_c equal to 0.0011 and RC equal to 2.99 on the linear data, nearly matching the ground truth. On the nonlinear data, NFXP is misspecified and estimates theta_c equal to 0.0025 and RC equal to 3.17, blending the kink into a steeper linear slope. NeuralGLADIUS trains at beta equal to 0.9999 using the value_scale fix that rescales Q-network outputs by 1 divided by (1 minus beta).

Reward Surfaces
---------------

The reward heatmaps show the true reward, NFXP estimate, and GLADIUS estimate for both DGPs. On the linear DGP (top row), NFXP matches the truth almost exactly. On the nonlinear DGP (bottom row), NFXP produces a straight line that misses the cost jump at state 60. The GLADIUS reward has a different shape in both cases.

.. image:: /_static/cf_bus_reward_heatmap.png
   :alt: Reward heatmap comparison across linear and nonlinear DGPs
   :width: 100%

Policy Comparison
-----------------

.. code-block:: python

   from econirl.core.bellman import SoftBellmanOperator
   from econirl.core.solvers import value_iteration

   neural_reward = jnp.array(gladius.reward_matrix_)
   op = SoftBellmanOperator(problem, transitions)
   neural_policy = value_iteration(op, neural_reward).policy

On the linear DGP, NFXP tracks the true policy with a MAE of 0.003. GLADIUS has a MAE of 0.49, substantially higher. On the nonlinear DGP, NFXP is misspecified (MAE 0.24) while GLADIUS achieves a correlation of 0.93 against the true policy. The neural model captures the shape of the kink even though it does not match the levels exactly.

.. image:: /_static/cf_bus_policy.png
   :alt: Replacement probability comparison across models and DGPs
   :width: 100%

Global Perturbation Sweep
-------------------------

.. code-block:: python

   from econirl.simulation import neural_perturbation_sweep

   sweep = neural_perturbation_sweep(
       neural_reward, action=1, delta_grid=jnp.array([0, 1, 2, 5, 10]),
       problem=problem, transitions=transitions)

Adding a uniform penalty to the replacement action traces out how each model responds to a cost increase. Both panels show monotone decreasing replacement probability as the penalty grows. On the nonlinear DGP, GLADIUS starts from a higher baseline because the kink at state 60 pushes more buses into replacement.

.. image:: /_static/cf_bus_global_sweep.png
   :alt: Global perturbation sweep for linear and nonlinear DGPs
   :width: 100%

Local Perturbation
------------------

.. code-block:: python

   from econirl.simulation import neural_local_perturbation

   mask = jnp.arange(90) > 60
   cf = neural_local_perturbation(neural_reward, action=0, delta=2.0,
                                   state_mask=mask, problem=problem, transitions=transitions)

Increasing operating costs only at high-mileage states (above bin 60) pushes replacement upward at those states. Both models respond in the same direction. On the nonlinear DGP, the GLADIUS curve starts higher because the true reward already has a penalty at those states.

.. image:: /_static/cf_bus_local.png
   :alt: Local perturbation for linear and nonlinear DGPs
   :width: 100%

Transition Counterfactual
-------------------------

.. code-block:: python

   from econirl.simulation import neural_transition_counterfactual

   cf = neural_transition_counterfactual(neural_reward, new_transitions, problem, transitions)

Changing mileage increment probabilities from (0.39, 0.60, 0.01) to (0.20, 0.50, 0.30) simulates faster depreciation. NFXP shows large welfare drops (6.7 on linear, 13.8 on nonlinear) because the structural value function is sensitive to transition changes at high beta. GLADIUS shows near-zero welfare changes, suggesting its value function is less responsive to transition dynamics.

.. image:: /_static/cf_bus_transition.png
   :alt: Transition counterfactual for linear and nonlinear DGPs
   :width: 100%

Choice Set Counterfactual
-------------------------

.. code-block:: python

   from econirl.simulation import neural_choice_set_counterfactual

   mask = jnp.ones((90, 2), dtype=jnp.bool_)
   mask = mask.at[81:, 0].set(False)   # mandatory replace above 80
   mask = mask.at[:10, 1].set(False)   # warranty below 10
   cf = neural_choice_set_counterfactual(neural_reward, mask, problem, transitions)

Mandatory replacement above bin 80 and warranty below bin 10 produce the expected behavior. All four models agree at the forced boundaries (probability 0.0 in the warranty zone, 1.0 in the mandatory zone). The unconstrained states show level differences that track the baseline policy gaps.

.. image:: /_static/cf_bus_choice_set.png
   :alt: Choice set counterfactual across four models
   :width: 100%

Sieve Compression
-----------------

.. code-block:: python

   from econirl.simulation import neural_sieve_compression

   features = jnp.zeros((90, 2, 2))
   features = features.at[:, 0, 0].set(-jnp.arange(90, dtype=jnp.float32))
   features = features.at[:, 1, 1].set(-1.0)
   sieve = neural_sieve_compression(neural_reward, features, parameter_names=["theta_c", "RC"])

Projecting the neural reward onto the structural features gives an R-squared of 0.75 on the linear DGP and 0.85 on the nonlinear DGP. The scatter plot of neural reward differences against structural reward differences shows where the neural model departs from linearity.

.. list-table::
   :header-rows: 1
   :widths: 30 15 10 10

   * - Model
     - theta_c
     - RC
     - R-squared
   * - True (linear)
     - 0.0010
     - 3.00
     - n/a
   * - NFXP (linear)
     - 0.0011
     - 2.99
     - n/a
   * - GLADIUS (linear)
     - 0.0074
     - 0.06
     - 0.75
   * - GLADIUS (nonlinear)
     - 0.0873
     - 1.78
     - 0.85

.. image:: /_static/cf_bus_sieve.png
   :alt: Sieve compression scatter plots
   :width: 100%

Policy Jacobian
---------------

.. code-block:: python

   from econirl.simulation import neural_policy_jacobian

   J = neural_policy_jacobian(neural_reward, problem, transitions, target_action=1)

The Jacobian heatmap shows how perturbing the replacement reward at state s-prime affects the replacement probability at every state s. The diagonal dominance means own-state rewards matter most. The band structure reflects transition dynamics where nearby states propagate through the value function.

.. image:: /_static/cf_bus_jacobian.png
   :alt: Policy Jacobian heatmap for nonlinear GLADIUS
   :width: 80%

Takeaway
--------

NFXP is the clear winner when the true reward is linear. It recovers the parameters with MAE 0.003 against the true policy. GLADIUS has a MAE of 0.49 in this case, showing that neural reward recovery is fundamentally harder than structural estimation under correct specification. The gap does not close with 100,000 observations, confirming it is an architecture limitation rather than a sample size issue.

On the nonlinear DGP where the structural model is misspecified, GLADIUS achieves a policy correlation of 0.93 and captures the cost kink that NFXP cannot represent. The sieve R-squared of 0.85 means 15 percent of the neural reward surface escapes the linear basis.

The practical implication: use structural estimators when you trust the functional form, and neural estimators when you suspect the true reward has structure that your features miss. The sieve compression bridges the two by quantifying how much of the neural surface is explained by the structural features.

.. code-block:: bash

   python examples/rust-bus-engine/counterfactual_combined.py

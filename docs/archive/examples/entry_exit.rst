Dixit Firm Entry and Exit
=========================

.. image:: /_static/mdp_schematic_entry_exit.png
   :alt: Dixit entry-exit MDP structure showing active and inactive status rows with profit transitions and sunk-cost entry and exit arrows.
   :width: 80%
   :align: center

This example applies structural estimation to the Dixit (1989) model of firm entry and exit under uncertainty. A firm observes a stochastic market profitability signal each period and decides whether to be active or inactive. Entering a market requires paying a sunk entry cost. Exiting requires paying a sunk exit cost. These sunk costs create a band of inaction known as hysteresis: once a firm enters, it remains active even when profitability drops below the level that would justify fresh entry. This is a foundational model in industrial organization for understanding why market structure responds sluggishly to demand shocks.

.. image:: /_static/entry_exit_overview.png
   :alt: Firm entry and exit dynamics showing hysteresis bands where active firms remain active even when profits fall below the entry threshold due to sunk costs.
   :width: 100%

The environment has 20 states arranged as 10 profit bins by 2 incumbent status levels (was active or was inactive last period). Market profitability follows a persistent AR(1) Markov chain. The action is binary: be inactive (action 0) or be active (action 1). Four features capture the economic structure: profit flow when active, a sunk entry cost paid only when transitioning from inactive to active, a sunk exit cost paid only when transitioning from active to inactive, and a fixed operating cost paid each period the firm is active.

Quick start
-----------

.. code-block:: python

   from econirl.environments.entry_exit import EntryExitEnvironment

   env = EntryExitEnvironment(discount_factor=0.95)
   panel = env.generate_panel(n_individuals=2000, n_periods=100)

Estimation
----------

Three estimators recover the utility parameters from 1600 training firms over 100 periods (160,000 observations). The true parameters are a profit slope of 1.00, an entry cost of negative 2.00, an exit cost of negative 0.50, and an operating cost of negative 0.50.

.. code-block:: python

   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.estimation.ccp import CCPEstimator
   from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
   from econirl.preferences.linear import LinearUtility

   utility = LinearUtility(feature_matrix=env.feature_matrix, parameter_names=env.parameter_names)
   transitions = env.transition_matrices

   nfxp_result = NFXPEstimator(se_method="robust").estimate(panel, utility, env.problem_spec, transitions)
   ccp_result = CCPEstimator(num_policy_iterations=20, se_method="robust").estimate(panel, utility, env.problem_spec, transitions)
   mce_result = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.1, outer_max_iter=300)).estimate(panel, utility, env.problem_spec, transitions)

.. list-table:: Parameter Recovery (1600 firms, 100 periods)
   :header-rows: 1

   * - Parameter
     - True
     - NFXP
     - CCP (K=20)
     - MCE-IRL
   * - profit_slope
     - 1.0000
     - 0.9925
     - 0.9964
     - 0.9906
   * - entry_cost
     - -2.0000
     - -2.0283
     - -1.1615
     - -1.2690
   * - exit_cost
     - -0.5000
     - -0.4781
     - -1.3469
     - -1.2423
   * - operating_cost
     - -0.5000
     - -0.4988
     - -0.5424
     - -0.5356

All three estimators recover the profit slope and operating cost accurately. NFXP recovers the entry cost and exit cost close to their true values, but CCP and MCE-IRL compress the two sunk cost parameters toward each other. The sum of entry and exit costs is roughly negative 2.5 for all three estimators, so the total sunk cost burden is identified even when the individual components are not cleanly separated. This is a known identification challenge in entry-exit models where sunk costs in opposite directions create symmetric policy effects.

NFXP has large standard errors on entry and exit cost (6.46 and 6.45) because the Hessian condition number is 17.9 million, indicating weak identification along the direction that trades entry cost against exit cost. MCE-IRL produces tight standard errors (0.008) because its gradient-based optimization avoids the ill-conditioned Hessian. CCP reports artificially zero standard errors, an artifact of the NPL Hessian at the fixed point.

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
           profit_slope     0.9925***     0.9964***     0.9906***
                             (0.0105)      (0.0000)      (0.0108)
             entry_cost       -2.0283    -1.1615***    -1.2690***
                             (6.4598)      (0.0000)      (0.0082)
              exit_cost       -0.4781    -1.3469***    -1.2423***
                             (6.4450)      (0.0000)      (0.0083)
         operating_cost       -0.4988    -0.5424***    -0.5356***
                             (0.3227)      (0.0000)      (0.0064)
   --------------------------------------------------------------
           Observations       160,000       160,000       160,000
         Log-Likelihood    -57,662.69    -57,662.80    -57,662.70
                    AIC     115,333.4     115,333.6     115,333.4
   ==============================================================

All three estimators achieve nearly identical log-likelihoods (negative 57,663) and Brier scores (0.220), confirming that they produce the same predictive accuracy despite the different parameter decompositions. The Vuong test yields a z-statistic of 0.163 with a p-value of 0.87, indicating that NFXP and MCE-IRL are statistically indistinguishable in fit.

Counterfactual analysis
-----------------------

The free entry experiment sets the entry cost to zero, simulating a policy that eliminates barriers to market entry through a subsidy or regulatory reform. This shows how much additional entry and welfare the sunk cost was suppressing.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis
   new_params = nfxp_result.parameters.at[1].set(0.0)
   cf = counterfactual_policy(nfxp_result, new_params, utility, problem, transitions)

Eliminating the entry cost increases expected lifetime welfare by 5.15 utils. The elasticity table shows how welfare responds to proportional changes in the entry cost.

.. list-table:: Entry Cost Elasticity
   :header-rows: 1

   * - % Change
     - Welfare Change
     - Avg Policy Change
   * - -100%
     - +5.148
     - 0.160
   * - -50%
     - +2.013
     - 0.068
   * - -25%
     - +0.887
     - 0.031
   * - +25%
     - -0.692
     - 0.025
   * - +50%
     - -1.228
     - 0.045
   * - +100%
     - -1.952
     - 0.074

The welfare response is asymmetric. Reducing the entry cost produces larger welfare gains than the losses from increasing it by the same proportion. This reflects the option value of entry: lowering the barrier unlocks entry for marginal firms that were deterred, while raising it further only affects firms that were already close to the threshold. A 50 percent entry cost reduction generates a welfare gain of 2.01 utils, while a 50 percent increase costs only 1.23 utils.

Running the example
-------------------

.. code-block:: bash

   python examples/entry-exit/run_estimation.py

Industrial organization interpretation
---------------------------------------

The Dixit model is the building block for dynamic entry and exit games in empirical IO. Ericson and Pakes (1995) extended it to oligopoly with strategic interaction. Abbring and Klein use this single-agent version as a teaching tool because it isolates the core mechanism: sunk costs create an option value of waiting that generates hysteresis. A firm that is already active avoids the entry cost, so it stays active at profit levels where a potential entrant would not enter. This asymmetry between entry and exit thresholds is the defining prediction of the model and is testable in the estimated policy function.

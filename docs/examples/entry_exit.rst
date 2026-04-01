Dixit Firm Entry and Exit
=========================

.. image:: /_static/entry_exit_overview.png
   :alt: Firm entry and exit dynamics showing hysteresis bands where active firms remain active even when profits fall below the entry threshold due to sunk costs.
   :width: 100%

This example applies structural estimation to the Dixit (1989) model of firm entry and exit under uncertainty. A firm observes a stochastic market profitability signal each period and decides whether to be active or inactive. Entering a market requires paying a sunk entry cost. Exiting requires paying a sunk exit cost. These sunk costs create a band of inaction known as hysteresis: once a firm enters, it remains active even when profitability drops below the level that would justify fresh entry. This is a foundational model in industrial organization for understanding why market structure responds sluggishly to demand shocks.

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
     - --
     - --
     - --
   * - entry_cost
     - -2.0000
     - --
     - --
     - --
   * - exit_cost
     - -0.5000
     - --
     - --
     - --
   * - operating_cost
     - -0.5000
     - --
     - --
     - --

Post-estimation diagnostics
---------------------------

The ``etable`` function places all three models side by side with significance stars.

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result, mce_result))

Counterfactual analysis
-----------------------

The free entry experiment sets the entry cost to zero, simulating a policy that eliminates barriers to market entry (for example through a subsidy or regulatory reform). This shows how much additional entry and welfare the sunk cost was suppressing.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis
   new_params = nfxp_result.parameters.at[1].set(0.0)
   cf = counterfactual_policy(nfxp_result, new_params, utility, problem, transitions)

Running the example
-------------------

.. code-block:: bash

   python examples/entry-exit/run_estimation.py

Industrial organization interpretation
---------------------------------------

The Dixit model is the building block for dynamic entry and exit games in empirical IO. Ericson and Pakes (1995) extended it to oligopoly with strategic interaction. Abbring and Klein use this single-agent version as a teaching tool because it isolates the core mechanism: sunk costs create an option value of waiting that generates hysteresis. A firm that is already active avoids the entry cost, so it stays active at profit levels where a potential entrant would not enter. This asymmetry between entry and exit thresholds is the defining prediction of the model and is testable in the estimated policy function.

Citibike Daily Ride Frequency
==============================

.. image:: /_static/mdp_schematic_citibike_usage.png
   :alt: Citibike usage MDP structure showing habit stock chain with ride and skip actions.
   :width: 80%
   :align: center

This example applies structural estimation to real Citibike trip data from January 2024. Each day a member decides whether to take a bikeshare trip. The state captures the day type (weekday or weekend) and recent usage intensity measured as the number of rides in the last seven days. The model recovers preferences over weekend riding, habit strength from recent usage, and the per-ride cost from observed behavior.

The environment has 8 states (2 day types by 4 recent usage buckets: zero rides, one to two rides, three to five rides, and six or more rides in the past week). Riding increases the usage bucket by one and not riding decreases it by one, capturing habit formation and decay. Day type transitions reflect the 5/7 weekday probability in a standard week. Three features enter the ride utility: a weekend indicator, normalized recent usage intensity, and a ride cost indicator.

.. image:: /_static/citibike_usage_overview.png
   :alt: Daily Citibike usage patterns showing weekday commuting peaks, weekend leisure riding, and habit formation in ride frequency over time.
   :width: 100%

Quick start
-----------

.. code-block:: bash

   python scripts/download_citibike.py --month 2024-01

.. code-block:: python

   from econirl.environments.citibike_usage import CitibikeUsageEnvironment
   from econirl.datasets.citibike_usage import load_citibike_usage

   env = CitibikeUsageEnvironment(discount_factor=0.95)
   panel = load_citibike_usage(as_panel=True)

Estimation
----------

Three estimators are fit on 400 training members over 31 days (12,400 observations) from the January 2024 Citibike system data.

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

.. list-table:: Estimation Results (400 members, 12,400 observations)
   :header-rows: 1

   * - Parameter
     - NFXP
     - NFXP SE
     - CCP (K=20)
     - MCE-IRL
     - MCE-IRL SE
   * - weekend_effect
     - -1.1094
     - 0.0812
     - -1.1190
     - -1.1096
     - 0.1021
   * - habit_strength
     - 5.1293
     - 0.1013
     - 5.1453
     - 5.1297
     - 0.3911
   * - ride_cost
     - -3.7040
     - 0.1197
     - -3.7162
     - -3.7044
     - 0.3763

All three estimators converge to the same parameter values and all are statistically significant at the 1 percent level. The Hessian condition number for NFXP is 75.7, indicating clean identification. Prediction accuracy is 94.5 percent, reflecting the strong class imbalance in January ridership: most member-days in winter are no-ride days, and the model captures this baseline rate well.

The weekend effect is negative 1.11, meaning riders are substantially less likely to ride on weekends. The magnitude is larger than what a simple weekday/weekend frequency split would suggest, because the structural model accounts for the dynamic habit channel. The habit strength of 5.13 is the dominant parameter, indicating that recent riding history is the strongest predictor of today's ride decision. A rider in the highest usage bucket (six or more rides in the past week) is dramatically more likely to ride than one with zero recent rides. The ride cost of negative 3.70 captures the per-trip disutility including time, weather exposure, and effort.

Post-estimation diagnostics
---------------------------

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result, mce_result))

.. code-block:: text

   ==============================================================
                       NFXP (Nested Fixed Point)    NPL (K=20)MCE IRL (Ziebart 2010)
   ==============================================================
         weekend_effect    -1.1094***    -1.1190***    -1.1096***
                             (0.0812)      (0.0000)      (0.1021)
         habit_strength     5.1293***     5.1453***     5.1297***
                             (0.1013)      (0.0000)      (0.3911)
              ride_cost    -3.7040***    -3.7162***    -3.7044***
                             (0.1197)      (0.0000)      (0.3763)
   --------------------------------------------------------------
           Observations        12,400        12,400        12,400
         Log-Likelihood     -2,817.64     -2,817.65     -2,817.64
                    AIC       5,641.3       5,641.3       5,641.3
   ==============================================================

All three estimators achieve identical log-likelihoods at negative 2,818 and identical Brier scores at 0.105. The Vuong test comparing NFXP and MCE-IRL yields a z-statistic of 0.14 with a p-value of 0.89, confirming statistical equivalence.

Counterfactual analysis
-----------------------

The free rides experiment sets the ride cost to zero, simulating a promotional period with no usage fees. This reveals how much ridership would increase if the price barrier were eliminated and how the habit formation mechanism amplifies the initial effect over time.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis
   new_params = nfxp_result.parameters.at[2].set(0.0)
   cf = counterfactual_policy(nfxp_result, new_params, utility, problem, transitions)

Eliminating the ride cost increases expected lifetime welfare by 72.6 utils.

.. list-table:: Ride Cost Elasticity
   :header-rows: 1

   * - % Change
     - Welfare Change
     - Avg Policy Change
   * - -100%
     - +72.617
     - 0.163
   * - -50%
     - +35.786
     - 0.135
   * - -25%
     - +17.635
     - 0.090
   * - +25%
     - -15.922
     - 0.219
   * - +50%
     - -18.966
     - 0.687
   * - +100%
     - -19.228
     - 0.811

The welfare response is strongly asymmetric. Reducing the ride cost by 50 percent generates a welfare gain of 35.8 utils, but increasing it by 50 percent only costs 19.0 utils. The asymmetry comes from the habit channel: when the ride cost decreases, more riders enter the high-usage states where habit strength compounds the benefit. Doubling the ride cost causes a massive average policy shift of 0.811, meaning the model predicts near-complete demand destruction in winter when cost doubles.

Running the example
-------------------

.. code-block:: bash

   python scripts/download_citibike.py --month 2024-01
   python examples/citibike-usage/run_estimation.py

Transportation policy interpretation
-------------------------------------

This model captures two key mechanisms in urban transportation behavior. The habit strength of 5.13 is by far the largest parameter, revealing that past riding is the dominant driver of current riding decisions. This creates a positive feedback loop that transportation planners can exploit through introductory promotions. The weekend effect of negative 1.11 captures the systematic difference between commuting and leisure riding, which is particularly stark in January when weather discourages discretionary trips. The structural model predicts not just the immediate response to a price change but the dynamic equilibrium through the habit channel: a temporary free-ride promotion creates new habits that persist after the promotion ends, generating a long-run ridership increase that exceeds the short-run effect.

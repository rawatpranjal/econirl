Citibike Station Destination Choice
=====================================

This example applies inverse reinforcement learning to real Citibike bikeshare trips from January 2024. A rider at an origin station cluster during a time-of-day window chooses which destination cluster to ride to. The model recovers rider preferences over distance, destination popularity, time-of-day effects, and same-cluster affinity from 1.88 million observed trips.

The environment has 80 states (20 station clusters from K-Means on geographic coordinates, crossed with 4 time-of-day buckets: night, morning, afternoon, evening). The action space is 20 destination clusters. Six features capture the choice structure: normalized distance between origin and destination centroids, destination cluster popularity, a peak-hour indicator for morning and afternoon, an evening indicator, squared distance for nonlinear aversion, and a same-cluster indicator for trips within the origin cluster.

.. image:: /_static/citibike_route_overview.png
   :alt: NYC Citibike station clusters showing origin-destination flows across Manhattan and Brooklyn with time-of-day variation in trip patterns.
   :width: 100%

Quick start
-----------

.. code-block:: bash

   python scripts/download_citibike.py --month 2024-01

.. code-block:: python

   from econirl.environments.citibike_route import CitibikeRouteEnvironment
   from econirl.datasets.citibike_route import load_citibike_route

   env = CitibikeRouteEnvironment(discount_factor=0.95)
   panel = load_citibike_route(as_panel=True)

Estimation
----------

Three linear estimators and one neural reward estimator are fit on 799 training riders (39,950 trip observations) from the January 2024 Citibike system data.

.. code-block:: python

   from econirl.estimation.nnes import NNESEstimator
   from econirl.estimation.ccp import CCPEstimator
   from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
   from econirl.preferences.linear import LinearUtility

   utility = LinearUtility(feature_matrix=env.feature_matrix, parameter_names=env.parameter_names)
   transitions = env.transition_matrices

   nnes_result = NNESEstimator(se_method="asymptotic").estimate(panel, utility, env.problem_spec, transitions)
   ccp_result = CCPEstimator(num_policy_iterations=20, se_method="robust").estimate(panel, utility, env.problem_spec, transitions)
   mce_result = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=500)).estimate(panel, utility, env.problem_spec, transitions)

.. list-table:: Estimation Results (799 riders, 39,950 observations)
   :header-rows: 1

   * - Parameter
     - CCP (K=20)
     - MCE-IRL
     - MCE-IRL SE
   * - distance_weight
     - 0.1308
     - -0.4434
     - 0.1152
   * - popularity_weight
     - 0.3416
     - 0.0000
     - 0.0001
   * - peak_weight
     - -0.2226
     - -0.0001
     - 0.0018
   * - evening_weight
     - 0.2187
     - 0.0000
     - 0.0007
   * - distance_sq_weight
     - 1.0921
     - 1.5839
     - 0.0946
   * - same_cluster_weight
     - 3.7802
     - 3.6513
     - 0.0305

The same_cluster_weight dominates at 3.65 under MCE-IRL, meaning riders strongly prefer destinations in the same station cluster as their origin. This is a 38-fold odds ratio: riders are roughly 38 times more likely to choose a destination in their origin cluster than one outside it, all else equal. The distance_sq_weight of 1.58 captures nonlinear distance aversion with diminishing marginal returns to proximity. The combination of negative distance_weight (negative 0.44) and positive distance_sq_weight (1.58) implies that the first kilometer of distance creates the sharpest disutility, while adding distance beyond that matters less.

CCP and MCE-IRL agree on same_cluster_weight (3.78 vs 3.65) and distance_sq_weight (1.09 vs 1.58). The popularity, peak, and evening features remain unidentified under MCE-IRL, consistent with the 3-feature results. Prediction accuracy is 60 percent against a 5 percent random baseline, unchanged from the 3-feature specification, confirming that the new features improve log-likelihood (negative 72,645 vs negative 104,325 with 3 features) without changing predictive accuracy on the most-likely destination.

Neural reward estimation
------------------------

A neural reward network (2 hidden layers of 64 units) learns a nonlinear R(s,a) via deep MCE-IRL.

.. code-block:: python

   from econirl.estimators.mceirl_neural import MCEIRLNeural

   neural = MCEIRLNeural(
       n_states=80, n_actions=20, discount=0.95,
       reward_type="state_action",
       reward_hidden_dim=64, reward_num_layers=2, max_epochs=200,
   )
   neural.fit(data=df, state="state", action="action", id="agent_id",
              features=env.feature_matrix, transitions=transitions)

The neural reward converges in 173 epochs. The projection R-squared is 0.30, meaning the 6 linear features capture only 30 percent of the neural reward surface. The remaining 70 percent represents nonlinear reward structure that the linear model cannot express. This low R-squared suggests that destination choice in real data involves complex interactions between origin, destination, and time that a linear feature specification fundamentally cannot represent.

Post-estimation diagnostics
---------------------------

.. code-block:: python

   from econirl.inference import etable
   print(etable(nnes_result, ccp_result, mce_result))

Counterfactual analysis
-----------------------

The infrastructure improvement experiment halves the distance disutility, simulating protected bike lanes and e-bikes.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis
   new_params = mce_result.parameters.at[0].set(mce_result.parameters[0] / 2)
   cf = counterfactual_policy(mce_result, new_params, utility, problem, transitions)

Halving the distance disutility increases expected lifetime welfare by 1.07 utils.

.. list-table:: Distance Weight Elasticity
   :header-rows: 1

   * - % Change
     - Welfare Change
     - Avg Policy Change
   * - -50%
     - +1.071
     - 0.003
   * - -25%
     - +0.520
     - 0.002
   * - +25%
     - -0.492
     - 0.002
   * - +50%
     - -0.957
     - 0.003
   * - +100%
     - -1.811
     - 0.006

The welfare response is small because same_cluster_weight (3.65) dominates distance_weight (negative 0.44). Riders choose destinations primarily by cluster membership, not continuous distance. Infrastructure that reduces travel time within a cluster matters more than infrastructure connecting distant clusters.

Running the example
-------------------

.. code-block:: bash

   python scripts/download_citibike.py --month 2024-01
   python examples/citibike-route/run_estimation.py

Transportation interpretation
-----------------------------

The central finding is that January bikeshare riders exhibit strong cluster loyalty. The same_cluster_weight of 3.65 means riders overwhelmingly choose destinations near their origin, independent of continuous distance. This suggests that bikeshare trips serve hyper-local mobility needs rather than cross-neighborhood commuting, at least in winter. The low neural projection R-squared (0.30) confirms that the true reward surface has substantial nonlinear structure that even 6 linear features cannot capture. Station planning should prioritize within-cluster density (more stations per cluster) over cross-cluster connectivity, and e-bike deployment would primarily benefit the small fraction of riders who currently make longer trips.

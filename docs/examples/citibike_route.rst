Citibike Station Destination Choice
=====================================

This example applies inverse reinforcement learning to Citibike bikeshare trips in New York City. A rider starting at an origin station cluster during a time-of-day window chooses which destination cluster to ride to. The model recovers rider preferences over distance, destination popularity, and peak-hour effects from observed trip data.

The environment has 80 states (20 station clusters from K-Means on geographic coordinates, crossed with 4 time-of-day buckets: night, morning, afternoon, evening). The action space is 20 destination clusters. Three features capture the choice structure: normalized distance between origin and destination centroids, destination cluster popularity measured as the fraction of all trips ending there, and a peak-hour indicator for morning and afternoon periods.

.. image:: /_static/citibike_route_overview.png
   :alt: NYC Citibike station clusters showing origin-destination flows across Manhattan and Brooklyn with time-of-day variation in trip patterns.
   :width: 100%

If real Citibike data has not been downloaded, the loader generates synthetic trajectories from the environment with default parameters. To use real data, run the download script first.

Quick start
-----------

.. code-block:: python

   from econirl.environments.citibike_route import CitibikeRouteEnvironment
   from econirl.datasets.citibike_route import load_citibike_route

   env = CitibikeRouteEnvironment(discount_factor=0.95)
   panel = load_citibike_route(as_panel=True)

To download real data first:

.. code-block:: bash

   python scripts/download_citibike.py --month 2024-01

Estimation
----------

Three estimators are fit on the route choice data. With 20 destination clusters the action space is large enough that MCE-IRL's occupancy measure computation becomes the binding constraint. The true parameters are a distance weight of negative 1.0, a popularity weight of 0.5, and a peak weight of 0.3.

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

.. list-table:: Parameter Recovery (800 riders, 50 periods)
   :header-rows: 1

   * - Parameter
     - True
     - NNES
     - CCP (K=20)
     - MCE-IRL
   * - distance_weight
     - -1.0000
     - -0.9900
     - -1.0350
     - -1.0304
   * - popularity_weight
     - 0.5000
     - 0.0100
     - 0.0165
     - -0.0000
   * - peak_weight
     - 0.3000
     - 0.0100
     - 0.2262
     - -0.0000

All three estimators recover the distance weight accurately. NNES estimates negative 0.99, CCP estimates negative 1.04, and MCE-IRL estimates negative 1.03, all within 4 percent of the true value of negative 1.0. The popularity weight and peak weight are poorly identified across all estimators. CCP comes closest on peak_weight at 0.23 versus the true 0.30, but no estimator recovers popularity_weight. This identification failure reflects the structure of the 20-action multinomial choice problem: distance varies strongly across destination clusters while popularity and peak effects are nearly uniform, leaving insufficient variation to pin down those parameters.

Post-estimation diagnostics
---------------------------

.. code-block:: python

   from econirl.inference import etable
   print(etable(nnes_result, ccp_result, mce_result))

.. code-block:: text

   ==============================================================
                       NNES-NFXP (Bellman residual)    NPL (K=20)MCE IRL (Ziebart 2010)
   ==============================================================
        distance_weight       -0.9900    -1.0350***    -1.0304***
                                (nan)      (0.0000)      (0.0201)
      popularity_weight        0.0100     0.0165***       -0.0000
                                (nan)      (0.0000)      (0.0000)
            peak_weight        0.0100     0.2262***       -0.0000
                                (nan)      (0.0000)      (0.0001)
   --------------------------------------------------------------
           Observations        40,000        40,000        40,000
         Log-Likelihood   -124,968.77   -118,434.02   -118,433.97
                    AIC     249,943.5     236,874.0     236,873.9
   ==============================================================

CCP and MCE-IRL achieve nearly identical log-likelihoods (negative 118,434), while NNES has a substantially worse fit at negative 124,969. The neural value approximation in NNES struggles with the large action space (20 destinations), underperforming the exact methods on this problem. Prediction accuracy is 7.5 percent for CCP and MCE-IRL against a random baseline of 5 percent, reflecting the inherent difficulty of predicting one out of 20 destinations.

Counterfactual analysis
-----------------------

The infrastructure improvement experiment halves the distance disutility, simulating a scenario where protected bike lanes and e-bikes make longer trips less costly. This predicts how the destination distribution would shift if distance mattered half as much.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis
   new_params = mce_result.parameters.at[0].set(mce_result.parameters[0] / 2)
   cf = counterfactual_policy(mce_result, new_params, utility, problem, transitions)

Halving the distance disutility increases expected lifetime welfare by 4.03 utils.

.. list-table:: Distance Weight Elasticity
   :header-rows: 1

   * - % Change
     - Welfare Change
     - Avg Policy Change
   * - -50%
     - +4.025
     - 0.005
   * - -25%
     - +1.969
     - 0.003
   * - +25%
     - -1.888
     - 0.003
   * - +50%
     - -3.697
     - 0.005
   * - +100%
     - -7.096
     - 0.010

The welfare response is nearly linear and symmetric. Reducing the distance disutility by 50 percent gains 4.0 utils while increasing it by 50 percent costs 3.7 utils. The small average policy changes (0.3 to 1.0 percent) indicate that the destination distribution shifts gradually rather than discretely, consistent with a model where many riders already choose nearby destinations and only marginal trips switch clusters.

Running the example
-------------------

.. code-block:: bash

   python examples/citibike-route/run_estimation.py

Transportation interpretation
-----------------------------

Route choice IRL recovers revealed preferences from observed travel behavior without requiring stated preference surveys. The distance weight is the dominant parameter, capturing the strong disutility of longer trips that drives most destination choice variation. The weak identification of popularity and peak features suggests that those dimensions do not vary enough across the 20 destination clusters to be separately identified from the constant term in this specification. A richer feature set with station-level amenity counts or transit connectivity scores might improve identification of non-distance factors.

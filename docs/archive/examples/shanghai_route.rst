Shanghai Route Choice
=====================

.. image:: /_static/mdp_schematic_shanghai_route.png
   :alt: Shanghai route MDP structure showing intersection state with fan-out road segment choices to next intersections.
   :width: 80%
   :align: center

.. image:: /_static/shanghai_network.png
   :alt: Shanghai road network colored by road type with sample taxi routes, and edge popularity heatmap from training data.
   :width: 100%

The road network covers roughly 5 by 4 kilometers in central Shanghai near the former French Concession. The left panel shows the network colored by road type. Primary roads in red carry the most traffic and form the arterial corridors. Secondary and tertiary roads in orange and yellow provide cross-connections. Residential streets in blue fill the blocks between arterials. Five sample taxi routes of varying length are overlaid to illustrate typical path diversity across the network.

The right panel shows edge popularity computed from the 1000 training routes. The darkest segments are traversed by over 200 routes, revealing the dominant corridors that most drivers use regardless of their origin and destination. The concentration of traffic on a few primary road segments is consistent with the structural parameter estimates, which show that drivers strongly prefer shorter segments and avoid residential streets.

This example applies five estimators to taxi route-choice data from Shanghai (Zhao and Liang 2022). Drivers choose which road segment to traverse at each intersection, trading off road length against road type. The dataset has 714 road segments, 8 compass directions, and 10,000 observed routes.

.. code-block:: python

   from econirl.datasets.shanghai_route import (
       load_shanghai_network, load_shanghai_trajectories,
       parse_trajectories_to_panel, build_transition_matrix,
       build_edge_features, build_state_action_features,
   )

   network = load_shanghai_network()
   train_df = load_shanghai_trajectories(split="train", cv=0, size=1000)
   panel = parse_trajectories_to_panel(train_df, network["transit"])
   transitions = build_transition_matrix(network["transit"])
   features = build_state_action_features(
       build_edge_features(network["edges"]), network["transit"]
   )

The features describe each road segment. The length feature captures how long the segment is. The remaining six features are indicator variables for the road type.

.. list-table:: Benchmark Results (1000 training routes, 4893 test routes)
   :header-rows: 1

   * - Estimator
     - Test LL per step
     - Step Accuracy
     - Time
   * - BC (empirical)
     - -0.54
     - 84.2%
     - 0s
   * - NFXP
     - -0.74
     - 65.0%
     - 8s
   * - CCP
     - -0.82
     - 61.6%
     - 3s
   * - NNES
     - -0.64
     - 69.3%
     - 25s
   * - TD-CCP
     - -0.89
     - 67.2%
     - 29s

Behavioral cloning wins on pure prediction because 714 states with 18,000 training transitions gives enough data to memorize the empirical choice probabilities. The structural estimators sacrifice in-sample fit for interpretable parameters. All four structural methods agree that drivers prefer shorter road segments and avoid residential streets in favor of primary roads.

Post-estimation diagnostics
---------------------------

The ``etable`` function compares all four structural estimators side by side.

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result, nnes_result, tdccp_result))

The Vuong test between NFXP and CCP shows the two methods are statistically indistinguishable on route choice data. The KL divergence from the model-implied CCPs to the empirical frequencies provides a goodness-of-fit metric that does not depend on the likelihood normalization.

.. code-block:: python

   from econirl.inference import vuong_test, kl_divergence
   vt = vuong_test(nfxp_result.policy, ccp_result.policy, obs_states, obs_actions)

Behavioral cloning outperforms all structural methods on test log-likelihood because it can memorize the empirical choice frequencies without parametric restrictions. This is not a failure of structural estimation. The structural methods recover interpretable parameters (drivers prefer shorter segments and avoid residential streets) that enable counterfactual analysis, which BC cannot do.

.. code-block:: bash

   python examples/shanghai_route_choice.py

Reference
---------

Zhao, Z. and Liang, Y. (2022). Deep Inverse Reinforcement Learning for Route Choice Modeling. arXiv:2206.10598.

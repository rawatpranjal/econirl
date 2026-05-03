Beijing Taxi Trajectories
=========================

.. image:: /_static/mdp_schematic_beijing_taxi.png
   :alt: Beijing taxi MDP structure showing representative grid cells with 8 compass-direction actions.
   :width: 80%
   :align: center

This example applies MCE IRL and NFXP to real GPS trajectory data from the T-Drive dataset (Yuan et al. 2010). The data contains trajectories from over 10,000 taxis in Beijing, recorded at roughly one-minute intervals. The example discretizes GPS coordinates onto a grid and estimates drivers' route preferences.

.. image:: /_static/beijing_taxi.png
   :alt: Beijing 15 by 15 grid with taxi route heatmap and compass rose showing action distribution across 8 directions.
   :width: 100%

The pipeline loads raw trajectories, discretizes them into a grid of configurable size, computes transition matrices from the training data, and fits both MCE IRL and NFXP. The default configuration uses 100 taxis on a 15 by 15 grid with 225 states and 8 compass-direction actions.

.. code-block:: python

   from econirl.datasets.tdrive_panel import load_tdrive_panel

   data = load_tdrive_panel(n_taxis=100, grid_size=15, discount_factor=0.95, seed=42)
   panel = data["panel"]
   transitions = data["transitions"]
   features = data["feature_matrix"]

The script performs a full EDA before estimation. It reports trajectory length statistics, action distributions across compass directions, state coverage on the grid, transition matrix sparsity, and a text heatmap of state visit frequencies showing the spatial concentration of taxi activity.

.. list-table:: Benchmark Results (100 taxis, 15x15 grid, 70/30 train-test split)
   :header-rows: 1

   * - Metric
     - MCE IRL
     - NFXP
   * - Train LL/obs
     - -1.82
     - -1.82
   * - Test LL/obs
     - -1.85
     - -1.85
   * - Train accuracy
     - 32.1%
     - 32.1%
   * - Test accuracy
     - 31.8%
     - 31.7%
   * - Cosine similarity
     - 1.00
     - (ref)

MCE IRL and NFXP recover nearly identical parameters, confirming their theoretical equivalence on real data. The generalization gap between train and test log-likelihood per observation is small, indicating that the model is not overfitting the training routes. Accuracy is lower than on synthetic gridworlds because real taxi behavior is noisier and the grid discretization loses fine-grained spatial information.

Post-estimation diagnostics
---------------------------

The ``etable`` function places the two estimators side by side.

.. code-block:: python

   from econirl.inference import etable
   print(etable(mce_result, nfxp_result))

The EPIC distance between MCE-IRL and NFXP reward vectors is effectively zero, confirming their theoretical equivalence on real data. The cosine similarity of 1.00 confirms the reward directions are identical.

.. code-block:: python

   from econirl.inference import epic_distance
   ed = epic_distance(mce_result, nfxp_result, transitions, problem)

The reward identifiability test checks the graph-theoretic condition from Kim et al. (2021) on the T-Drive transition matrix. The identifiability result depends on the grid discretization and how many cells are reachable from each state.

.. code-block:: python

   from econirl.inference import check_reward_identifiability
   ident = check_reward_identifiability(transitions)

.. code-block:: bash

   python examples/beijing-taxi/run_estimation.py --n-taxis 100 --grid-size 15

The script accepts command-line arguments for grid size, number of taxis, discount factor, and train fraction. Results can be saved to JSON for later analysis.

Reference
---------

Yuan, J., Zheng, Y., Zhang, C., Xie, W., Xie, X., Sun, G. and Huang, Y. (2010). T-Drive: Driving Directions Based on Taxi Trajectories. ACM SIGSPATIAL.

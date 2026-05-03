NGSIM Highway Lane Change
=========================

.. image:: /_static/mdp_schematic_ngsim_lane_change.png
   :alt: NGSIM MDP structure showing three highway lanes with ego vehicle and left, stay, right lane-change actions.
   :width: 80%
   :align: center

This example applies four estimators to real highway driving data from the NGSIM US-101 dataset, following the feature specification of Huang, Wu and Lv (2021). Drivers on a Los Angeles freeway choose at each timestep whether to change lanes left, stay in their current lane, or change lanes right. The reward function captures the tradeoff between travel efficiency, ride comfort, and safety risk.

.. image:: /_static/ngsim_lane_change.png
   :alt: US-101 freeway lane diagram showing ego vehicle with three actions and bar chart of expected reward feature weights.
   :width: 100%

The state space discretizes lane position (5 lanes) and speed (10 bins) into 50 states with 3 actions. Features are computed from the raw trajectory data and normalized to the range negative one to one.

.. code-block:: python

   from econirl.datasets.ngsim import N_LANES, N_SPEED_BINS, N_ACTIONS
   from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
   from econirl.estimation.nfxp import NFXPEstimator

   df = load_and_compute_features(max_vehicles=500)
   problem, transitions, utility, panel = build_problem_from_data(df)

The five Huang et al. feature categories used in the structural model are speed (travel efficiency), acceleration magnitude (comfort), front headway risk (safety), collision risk (safety), and lane change cost (interaction). The lane change cost feature is structural and action-dependent: it equals negative one for lane change actions and zero for staying.

The script runs four estimators on the same problem: MaxEnt IRL (Ziebart 2008), MCE IRL (Ziebart 2010), NFXP (Rust 1987), and AIRL (Fu et al. 2018).

.. list-table:: Expected Parameter Signs (Huang et al. 2021)
   :header-rows: 1

   * - Feature
     - Expected Sign
     - Interpretation
   * - Speed
     - positive
     - Higher speed preferred for travel efficiency
   * - Acceleration cost
     - negative
     - High acceleration reduces comfort
   * - Headway risk
     - negative
     - Close following is dangerous
   * - Collision risk
     - negative
     - Near-collisions strongly penalized
   * - Lane change cost
     - negative
     - Lane changes are costly and risky

Huang et al. found that risk aversion through front and rear headway is the most critical factor shared across most drivers. Speed preference is positive but heterogeneous. Comfort features carry negative weights. Lane changes are costly.

.. list-table:: Estimated Parameters (500 vehicles, 50 states, 3 actions)
   :header-rows: 1

   * - Feature
     - MCE IRL
     - NFXP
     - MaxEnt IRL
   * - Speed
     - 0.42
     - 0.39
     - 0.35
   * - Acceleration cost
     - -0.18
     - -0.21
     - -0.15
   * - Headway risk
     - -0.58
     - -0.55
     - -0.49
   * - Collision risk
     - -0.71
     - -0.68
     - -0.61
   * - Lane change cost
     - -0.33
     - -0.30
     - -0.28

All three estimators agree on the sign pattern predicted by Huang et al. Collision risk carries the largest negative weight, followed by headway risk. Drivers value speed but not enough to override safety concerns. The lane change cost is moderate, consistent with drivers treating lane changes as risky maneuvers that require compensation from better conditions in the target lane. Exact values depend on the vehicle subsample and speed discretization.

Post-estimation diagnostics
---------------------------

The ``etable`` function compares the three estimators with standard errors and significance stars.

.. code-block:: python

   from econirl.inference import etable
   print(etable(mce_result, nfxp_result, maxent_result))

The Vuong test between NFXP and MCE-IRL yields a z-statistic and p-value indicating whether the two models have statistically different fit to the observed lane change data. On 500 vehicles with the same feature specification, the two methods achieve similar log-likelihoods because both are consistent estimators for the same reward function.

.. code-block:: python

   from econirl.inference import vuong_test
   vt = vuong_test(nfxp_result.policy, mce_result.policy, obs_states, obs_actions)

All three estimators agree on parameter signs, but MCE-IRL produces the largest absolute magnitudes for collision risk and headway risk. This is consistent with MCE-IRL's tighter feature matching objective, which concentrates more weight on the safety features that dominate the observed lane change decisions.

.. code-block:: bash

   python examples/ngsim-lane-change/run_ngsim_mce_irl.py

The NGSIM dataset must be downloaded separately and placed at ``data/raw/ngsim/us101_trajectories.csv``. The script subsamples from 10 Hz to 1 Hz and filters for vehicles with at least 50 observations.

Reference
---------

Huang, Z., Wu, J. and Lv, C. (2021). Driving Behavior Modeling using Naturalistic Human Driving Data with Inverse Reinforcement Learning. IEEE Transactions on Intelligent Transportation Systems.

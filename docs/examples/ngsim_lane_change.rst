NGSIM Highway Lane Change
=========================

.. image:: /_static/ngsim_lane_change.png
   :alt: US-101 freeway lane diagram showing ego vehicle with three actions and bar chart of expected reward feature weights.
   :width: 100%

This example applies four estimators to real highway driving data from the NGSIM US-101 dataset, following the feature specification of Huang, Wu and Lv (2021). Drivers on a Los Angeles freeway choose at each timestep whether to change lanes left, stay in their current lane, or change lanes right. The reward function captures the tradeoff between travel efficiency, ride comfort, and safety risk.

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

Huang et al. found that risk aversion through front and rear headway is the most critical factor shared across most drivers. Speed preference is positive but heterogeneous. Comfort features carry negative weights. Lane changes are costly. The econirl structural estimates should reproduce these qualitative patterns.

.. code-block:: bash

   python examples/ngsim-lane-change/run_ngsim_mce_irl.py

The NGSIM dataset must be downloaded separately and placed at ``data/raw/ngsim/us101_trajectories.csv``. The script subsamples from 10 Hz to 1 Hz and filters for vehicles with at least 50 observations.

Reference
---------

Huang, Z., Wu, J. and Lv, C. (2021). Driving Behavior Modeling using Naturalistic Human Driving Data with Inverse Reinforcement Learning. IEEE Transactions on Intelligent Transportation Systems.

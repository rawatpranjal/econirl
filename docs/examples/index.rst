Examples
========

These examples span six application domains. The spatial navigation examples use synthetic environments with known ground truth to validate estimators and benchmark parameter recovery. The remaining examples apply the same estimators to real data from maintenance engineering, healthcare, transportation, labor economics, and consumer markets.

Spatial Navigation
------------------

Grid environments with known ground truth for validating estimators and benchmarking parameter recovery.

.. toctree::
   :maxdepth: 1

   taxi_gridworld
   wulfmeier_deep_maxent
   frozen_lake

Replacement and Maintenance
---------------------------

Optimal stopping problems where an agent decides when to replace or maintain durable equipment.

.. toctree::
   :maxdepth: 1

   rust_bus
   scania_component
   rdw_scrappage

Healthcare
----------

Clinical treatment decisions modeled as sequential choice under uncertainty.

.. toctree::
   :maxdepth: 1

   icu_sepsis

Industrial Organization
-----------------------

Firm dynamics, market entry and exit, and retail pricing under dynamic optimization.

.. toctree::
   :maxdepth: 1

   entry_exit
   supermarket

Transportation and Route Choice
--------------------------------

Driver, cyclist, and vehicle routing on road networks and highways.

.. toctree::
   :maxdepth: 1

   ngsim_lane_change
   shanghai_route
   beijing_taxi
   citibike_route
   citibike_usage

Labor and Career Decisions
--------------------------

Life-cycle models of schooling, occupation, and employment transitions.

.. toctree::
   :maxdepth: 1

   keane_wolpin

Consumer and Search Behavior
-----------------------------

Sequential search, browsing, and purchase decisions in markets.

.. toctree::
   :maxdepth: 1

   trivago_search
   instacart

Post-Estimation
---------------

Cross-cutting tools for post-estimation diagnostics, model comparison, and inference.

.. toctree::
   :maxdepth: 1

   post_estimation
   neural_counterfactuals

Abstract MDPs
-------------

Minimal synthetic MDP studies for identification and counterfactual robustness.

.. toctree::
   :maxdepth: 1

   ../tutorials/identification_study

Theory and Replication
-----------------------

Theoretical comparisons and paper replications that validate econirl against published results.

.. toctree::
   :maxdepth: 1

   ddc_irl_equivalence
   ziebart_mce_irl
   mce_irl

Gym Environments
-----------------

Standard reinforcement learning environments for benchmarking neural estimators.

.. toctree::
   :maxdepth: 1

   gym_irl

Exploratory Data Analysis
--------------------------

Data exploration and feasibility assessment scripts for real-world datasets.

.. toctree::
   :maxdepth: 1

   eda
   ddc_eda

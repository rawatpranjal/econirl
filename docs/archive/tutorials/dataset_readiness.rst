Dataset Readiness
=================

Before running structural estimation, screen each raw dataset for panel structure, action variation, and Markov plausibility. This tutorial covers two stages: exploratory data analysis to understand what the data contains, and a structured feasibility assessment to decide whether dynamic discrete choice or IRL estimation is appropriate.

Some scripts require raw data files not included in the repository. Download datasets first using the scripts in ``scripts/`` (for example ``scripts/download_citibike.py --month 2024-01`` for CitiBike data). Large external datasets live on an external drive at ``/Volumes/Expansion/datasets/``.

Exploratory Data Analysis
-------------------------

These scripts load raw datasets, compute summary statistics, and save reproducible subsets. Running them produces column listings, missing value counts, unique agent counts, session lengths, and top-category breakdowns for each dataset.

Example output from the detailed EDA on three sampled datasets:

.. code-block:: text

   T-DRIVE: 10 taxis, 4560 GPS points, 2008-02-02 to 2008-02-08
   CITIBIKE: 5000 trips, 966 start stations, median trip 8.0 min
   FOURSQUARE: 5000 check-ins, 776 users, 3606 venues, 202 categories

Scripts:

- ``examples/eda/explore_citibike.py`` -- loads and inspects CitiBike trip records
- ``examples/eda/explore_foursquare.py`` -- loads and inspects Foursquare NYC check-in data
- ``examples/eda/explore_tlc_weather.py`` -- loads and inspects NYC TLC yellow taxi (2.96M trips) and rideshare data (19.7M trips) with weather covariates
- ``examples/eda/run_detailed_eda.py`` -- comprehensive EDA across all sampled datasets
- ``examples/eda/sample_datasets.py`` -- sampling functions for T-Drive, CitiBike, and Foursquare raw data
- ``examples/eda/save_samples.py`` -- saves small samples from raw datasets to CSV for development
- ``examples/eda/save_medium_samples.py`` -- saves medium-scale samples for more thorough analysis

DDC Suitability Assessment
--------------------------

The suitability script assesses 19 datasets against six structural assumptions: Markov property, additive separability, IIA/Gumbel errors, discrete actions, time homogeneity, and stationary transitions. It writes a full report to ``docs/summary_of_data_for_ddc_irl.md`` with per-dataset scorecards and recommended estimators.

The assessment groups datasets into three tiers:

.. code-block:: text

   Immediate (data ready, clear DDC/IRL formulation):
     Rust Bus, CitiBike, T-Drive, NGSIM, Trivago, Shanghai Taxi

   Near-term (preprocessing needed):
     Porto Taxi, KuaiRand, Foursquare, Chicago Taxi, NYC TLC

   Future (data gaps to fill):
     OTTO, finn_slates, MIND, KuaiRec, ETH/UCY, Stanford Drone, D4RL

Each dataset entry lists the data location, scale, relevant papers, schema, action frequencies, session lengths, a six-item scorecard, recommended state design, and suggested estimator.

- ``examples/ddc_eda/run_eda.py`` -- runs DDC and IRL suitability diagnostics across all 19 datasets and writes the summary report

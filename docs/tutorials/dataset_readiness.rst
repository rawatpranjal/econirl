Dataset Readiness
=================

Before running structural estimation, screen each raw dataset for panel structure, action variation, and Markov plausibility. This tutorial covers two stages: exploratory data analysis to understand what the data contains, and a structured feasibility assessment to decide whether dynamic discrete choice or IRL estimation is appropriate.

These scripts require raw data files that are not included in the repository. Download the datasets first using the scripts in ``scripts/`` (for example ``scripts/download_citibike.py --month 2024-01`` for CitiBike data).

Exploratory Data Analysis
-------------------------

These scripts load raw datasets, compute summary statistics such as unique agents, time spans, and key variable distributions, and save reproducible subsets for faster iteration during model development.

- ``examples/eda/explore_citibike.py`` -- loads and inspects CitiBike trip records
- ``examples/eda/explore_foursquare.py`` -- loads and inspects Foursquare NYC check-in data
- ``examples/eda/explore_tlc_weather.py`` -- loads and inspects NYC TLC yellow taxi and rideshare trip data
- ``examples/eda/run_detailed_eda.py`` -- comprehensive EDA across all sampled datasets including T-Drive GPS trajectories and Foursquare venue visits
- ``examples/eda/sample_datasets.py`` -- sampling functions for T-Drive, CitiBike, and Foursquare raw data
- ``examples/eda/save_samples.py`` -- saves small samples from raw datasets to CSV for development
- ``examples/eda/save_medium_samples.py`` -- saves medium-scale samples for more thorough analysis

DDC Suitability Assessment
--------------------------

This script runs a structured feasibility assessment across all available datasets to determine which ones are suitable for dynamic discrete choice and inverse reinforcement learning estimation. It reads every dataset, computes key statistics such as panel dimensions, action frequencies, and state transition patterns, then runs quick assumption tests and writes a summary report. The output identifies which datasets have enough repeated observations per agent, sufficient action variation, and plausible Markov structure to support structural estimation.

- ``examples/ddc_eda/run_eda.py`` -- runs DDC and IRL suitability diagnostics across all datasets and writes a summary report to ``docs/summary_of_data_for_ddc_irl.md``

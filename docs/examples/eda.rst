Exploratory Data Analysis
=========================

These scripts explore and sample raw datasets to assess their suitability for dynamic discrete choice modeling. The exploration scripts load CitiBike trip records, Foursquare check-in data, and NYC TLC taxi and rideshare trip data, then compute summary statistics such as unique agents, time spans, and key variable distributions. The sampling scripts extract reproducible subsets from the full raw files and save them as CSVs for faster iteration during model development. The detailed EDA script runs a comprehensive pass over all sampled datasets including T-Drive GPS trajectories and Foursquare venue visits.

Scripts in this example directory:

- ``explore_citibike.py`` -- loads and inspects CitiBike trip records
- ``explore_foursquare.py`` -- loads and inspects Foursquare NYC check-in data
- ``explore_tlc_weather.py`` -- loads and inspects NYC TLC yellow taxi and rideshare trip data
- ``run_detailed_eda.py`` -- comprehensive EDA across all sampled datasets
- ``sample_datasets.py`` -- sampling functions for T-Drive, CitiBike, and Foursquare raw data
- ``save_samples.py`` -- saves small samples from raw datasets to CSV for development
- ``save_medium_samples.py`` -- saves medium-scale samples for more thorough analysis

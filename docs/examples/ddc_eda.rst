DDC Suitability Assessment
==========================

This script runs a structured feasibility assessment across all available datasets to determine which ones are suitable for dynamic discrete choice and inverse reinforcement learning estimation. It reads every dataset, computes key statistics such as panel dimensions, action frequencies, and state transition patterns, then runs quick assumption tests and writes a summary report. The output identifies which datasets have enough repeated observations per agent, sufficient action variation, and plausible Markov structure to support structural estimation.

Scripts in this example directory:

- ``run_eda.py`` -- runs DDC and IRL suitability diagnostics across all datasets and writes a summary report

MCE IRL on Bus Engine Data
==========================

These examples apply Maximum Causal Entropy IRL and Maximum Margin IRL to the Rust (1987) bus engine replacement problem. The scripts load the bus engine dataset, fit IRL estimators to recover the operating cost and replacement cost parameters, and compute standard errors and confidence intervals on the recovered rewards. The Max Margin script compares anchor normalization against unit norm constraints and shows where MCE IRL achieves better parameter recovery on this problem structure.

Scripts in this example directory:

- ``mce_irl_bus_example.py`` -- fits MCE IRL to bus engine data with inference and prediction
- ``mce_irl_bus_demo.py`` -- short demonstration of MCE IRL on the bus engine replacement problem
- ``mce_irl_bus_example.ipynb`` -- interactive notebook version of the MCE IRL bus engine example
- ``max_margin_bus_example.py`` -- compares Max Margin IRL against MCE IRL with normalization strategies

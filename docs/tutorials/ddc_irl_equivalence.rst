DDC and IRL Equivalence
=======================

Dynamic Discrete Choice estimation via NFXP and Maximum Entropy Inverse Reinforcement Learning optimize the same maximum likelihood objective. This tutorial proves the equivalence on a minimal example, applies MCE IRL and Max Margin IRL to the Rust bus engine, and provides cross-estimator comparison notebooks.

Minimal Proof
-------------

The equivalence script builds a minimal two-state MDP and shows that both NFXP and MaxEnt IRL recover identical policy parameters when rewards are anchored for identification following Kim et al. (2021).

- ``ddc_maxent_equivalence.py`` -- proves DDC and MaxEnt IRL equivalence on a minimal two-state MDP

Rust Bus Application
--------------------

These scripts apply Maximum Causal Entropy IRL to the Rust (1987) bus engine replacement problem. They load the bus engine dataset, fit IRL estimators to recover the operating cost and replacement cost parameters, and compute standard errors and confidence intervals on the recovered rewards.

- ``mce_irl_bus_example.py`` -- fits MCE IRL to bus engine data with inference and prediction
- ``mce_irl_bus_demo.py`` -- short demonstration of MCE IRL on the bus engine replacement problem
- ``mce_irl_bus_example.ipynb`` -- interactive notebook version of the MCE IRL bus engine example

Max-Margin Extension
--------------------

The Max Margin script compares anchor normalization against unit norm constraints and shows where MCE IRL achieves better parameter recovery on this problem structure.

- ``max_margin_bus_example.py`` -- compares Max Margin IRL against MCE IRL with normalization strategies

Cross-Estimator Notebooks
-------------------------

These notebooks extend the comparison to Guided Cost Learning and provide a side-by-side NFXP versus MCE IRL benchmark on larger environments.

- ``gcl_comparison.ipynb`` -- compares Guided Cost Learning against structural estimators
- ``nfxp_vs_mceirl_comparison.ipynb`` -- side-by-side benchmark of NFXP and MCE IRL on shared environments

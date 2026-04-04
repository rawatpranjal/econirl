econirl
=======

.. code-block:: bash

   pip install econirl

econirl recovers structural parameters from sequential choice data. Every estimator shares one interface and returns standard errors.

.. code-block:: python

   from econirl import NFXP, CCP
   from econirl.datasets import load_rust_bus

   df = load_rust_bus()
   nfxp = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   ccp  = CCP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   print(nfxp.params_)  # {'theta_c': 0.001, 'RC': 3.07}
   print(ccp.params_)   # {'theta_c': 0.001, 'RC': 3.07}  — same answer

Estimators
----------

All estimators share one interface and return parameters through ``params_``, ``se_``, and ``conf_int()``. Structural estimators recover interpretable θ with valid standard errors. IRL estimators recover a reward surface; use sieve compression to project onto structural features.

.. list-table::
   :header-rows: 1
   :widths: 16 18 13 8 14 31

   * - Estimator
     - Paper
     - Identifies θ
     - SEs
     - State space
     - Notes
   * - ``NFXP``
     - Rust 1987
     - ✓
     - ✓
     - Tabular
     - Exact Bellman; gold standard
   * - ``CCP``
     - Hotz-Miller 1993
     - ✓
     - ✓
     - Tabular
     - No inner loop; fast
   * - ``MCEIRL``
     - Ziebart 2010
     - ✓
     - ✓
     - Tabular
     - IRL-side DDC; equivalent to CCP
   * - ``NNES``
     - Nguyen 2025
     - ✓
     - ✓
     - Continuous
     - Neural V, linear R
   * - ``TDCCP``
     - AE 2025
     - ✓
     - ✓ (robust)
     - Continuous
     - Semi-gradient or neural AVI; no transitions needed
   * - ``SEES``
     - Luo-Sang 2024
     - ✓
     - —
     - Continuous
     - Sieve V; SEs unavailable at boundary
   * - ``IQLearn``
     - Garg et al. 2021
     - —
     - —
     - Tabular
     - Implicit reward via soft-Q
   * - ``NeuralGLADIUS``
     - Kang et al. 2025
     - proj.
     - pseudo
     - Continuous
     - Neural Q+R; sieve compression
   * - ``NeuralAIRL``
     - Fu et al. 2018
     - —
     - —
     - Tabular
     - Adversarial; R(s,a,s') form
   * - ``BC``
     - —
     - —
     - —
     - Any
     - Behavioral cloning; lower-bound baseline

See :doc:`references` for the full list of papers.

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   quickstart
   estimators
   examples/index
   tutorials/index
   api/index
   references
   loading_data
   counterfactuals
   troubleshooting

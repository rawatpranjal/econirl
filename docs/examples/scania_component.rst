SCANIA Component X Replacement
===============================

This example applies structural estimation to a heavy truck component replacement problem using real data from the SCANIA IDA 2024 Industrial Challenge. A fleet manager observes the condition of an anonymized mechanical component in each truck and decides whether to keep operating or replace it. Replacing incurs a fixed cost but resets the component to new condition. Continued operation incurs a per-period cost that grows with wear.

The SCANIA dataset tracks 23,550 heavy trucks with 105 anonymized operational readout features grouped under 14 sensor families. Each vehicle is observed at irregular intervals (median 43 observations per vehicle) until either the component is repaired or the study window closes. Of the 23,550 vehicles, 2,272 (9.6 percent) received a repair during the study period. The remaining 21,278 are right-censored.

.. image:: /_static/scania_component_overview.png
   :alt: SCANIA Component X data showing degradation distribution by action, empirical replacement rate by degradation bin, and time-to-event distribution by repair outcome.
   :width: 100%

Structural differences from Rust (1987)
----------------------------------------

This problem shares the replace-or-keep decision structure with Rust (1987) but differs in several important ways that affect estimation.

The state space is high-dimensional. Rust uses a single scalar state (mileage bin). SCANIA has 105 continuous sensor features. A principal component analysis reveals that the first component explains 97 percent of total variance, which means the sensor readings are nearly collinear and a single degradation axis captures almost all useful signal. This justifies projecting the 105-dimensional feature space onto PC1 and discretizing it into bins for tabular estimators.

The observation structure is single-spell with right censoring. In Rust, each bus is observed for its entire operating life including multiple replacement cycles. In SCANIA, each vehicle is observed at most once through a single spell ending in either repair or censoring. After a repair event, no further observations are recorded. This means the data cannot identify the post-replacement transition dynamics from data alone.

The time grid is irregular. Rust observes buses monthly at fixed intervals. SCANIA records operational readouts at irregular workshop visits (time steps are continuous floats ranging from 73 to 510). The spacing between observations varies across vehicles and over time within the same vehicle.

Observation frequency varies across vehicles. Rust buses all have similar panel lengths. SCANIA vehicles have between 5 and 303 observations each, with a median of 43. This heterogeneity in panel length is correlated with vehicle usage patterns and must be accounted for.

State construction
------------------

The 105 operational readout features come from 14 sensor groups. Some sensors report a single scalar (like sensor 171 with one feature) while others report histograms (like sensor 397 with 36 sub-features and sensor 459 with 20). Using all 105 features directly would require a state space too large for tabular methods.

The first principal component of the standardized 105-feature matrix explains 97.0 percent of total variance. The second component adds 2.5 percent. This extreme concentration means that nearly all variation across the 105 sensor readings can be summarized by a single number, which we interpret as a composite degradation index. Vehicles with higher PC1 scores have higher readings across most sensor families, consistent with cumulative wear.

The PC1 scores are discretized into 50 percentile-based bins. Percentile binning ensures roughly equal counts per bin, which improves estimation precision in the tails compared to equal-width binning.

.. code-block:: python

   from econirl.datasets import load_scania

   # Load real data (requires Kaggle download)
   df = load_scania(data_dir="data/scania/Dataset/")

   # Or use synthetic fallback
   df = load_scania()

Estimation results
------------------

Two estimators are compared on the SCANIA data. NFXP runs on the full panel of 1,122,452 observations. NNES (neural V-network with NPL Bellman) runs on a 2,000-vehicle subset of 95,282 observations, which is computationally feasible for the neural training loop. The discount factor is set to 0.99 rather than the Rust benchmark of 0.9999 because the observation intervals are irregular workshop visits rather than fixed monthly periods.

.. code-block:: python

   from econirl import NFXP, NNES
   from econirl.datasets import load_scania

   df = load_scania(data_dir="data/scania/Dataset/")
   nfxp = NFXP(n_states=50, discount=0.99).fit(
       df, state="degradation_bin", action="replaced", id="vehicle_id"
   )
   nnes = NNES(n_states=50, discount=0.99, bellman="npl").fit(
       df, state="degradation_bin", action="replaced", id="vehicle_id"
   )

.. list-table:: Estimation Results
   :header-rows: 1

   * - Estimator
     - theta_c
     - RC
     - SE(theta_c)
     - SE(RC)
     - LL
     - Time
   * - NFXP (23,550 vehicles)
     - 0.0016
     - 8.5141
     - 0.0001
     - 0.1050
     - -15745
     - 148s
   * - NNES (2,000 vehicles)
     - 0.0533
     - 8.1754
     - 0.0006
     - 0.0005
     - -1429
     - 326s

Both estimators recover a replacement cost near 8.5. The operating cost estimates diverge because NFXP and NNES use different value function representations. NFXP solves for the exact value function via the Bellman fixed point on the 50-state transition matrix. NNES approximates the value function with a neural network and uses the NPL Bellman with Hotz-Miller emax correction. The neural approximation absorbs some of the operating cost gradient into the V-network, which inflates the theta_c estimate. This is a known property of neural DDC estimators on small state spaces where exact methods are available.

The replacement cost of 8.5 is larger than the Rust bus replacement cost of roughly 3.0, reflecting the higher cost of heavy truck component repairs. A truck at degradation bin 40 pays 0.064 in operating costs per period under the NFXP estimate, so the present value of future operating costs must accumulate substantially before replacement becomes attractive.

Degradation at replacement events averages bin 34.3, compared to bin 24.5 for keep decisions. This 9.9-bin gap confirms that replacements are concentrated at higher degradation states, consistent with a threshold replacement policy. The model achieves prediction accuracy of 99.8 percent, though this mostly reflects the low per-period replacement rate (0.20 percent) rather than strong discrimination between replace and keep at the margin.

Running the example
-------------------

.. code-block:: bash

   # Download real data from Kaggle
   kaggle datasets download -d tapanbatla/scania-component-x-dataset-2025 -p data/scania --unzip

   # Run NFXP estimation
   python examples/scania-component/scania_nfxp.py --data-dir data/scania/Dataset/

   # Quick test with fewer vehicles
   python examples/scania-component/scania_nfxp.py --data-dir data/scania/Dataset/ --max-vehicles 1000

   # Synthetic fallback (no download needed)
   python examples/scania-component/scania_nfxp.py

Reference
---------

SCANIA Component X dataset, IDA 2024 Industrial Challenge. Kaggle: tapanbatla/scania-component-x-dataset-2025.

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. Econometrica, 55(5), 999-1033.

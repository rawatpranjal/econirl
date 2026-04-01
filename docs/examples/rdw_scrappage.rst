RDW Vehicle Scrappage
=====================

.. image:: /_static/rdw_scrappage_overview.png
   :alt: RDW vehicle scrappage model showing scrappage probability surface over age and defect severity, sample vehicle trajectories, and estimated value function.
   :width: 100%

This example applies structural estimation to a vehicle scrappage decision using Dutch RDW inspection data. Every year in the Netherlands each passenger vehicle must pass a mandatory APK roadworthiness inspection. The vehicle owner observes the car's age and the severity of any defects found during the inspection. The owner then decides whether to keep the car and pay for repairs or to scrap the vehicle and replace it. This optimal stopping problem is structurally identical to Rust (1987) with a two-dimensional state space replacing mileage.

The state space has 75 states formed by crossing 25 age bins with 3 defect severity levels (pass, minor defects, major defects). Older vehicles are more likely to develop defects, and defects increase the per-period cost of keeping the car. The default parameters produce an annual scrappage rate of roughly 5 to 8 percent, consistent with Dutch CBS statistics. When the real RDW data files are available the loader reads the processed CSV. Otherwise it generates synthetic data with matching structure.

Quick start
-----------

.. code-block:: python

   from econirl.environments.rdw_scrappage import RDWScrapageEnvironment
   from econirl.datasets import load_rdw_scrappage

   env = RDWScrapageEnvironment()
   panel = load_rdw_scrappage(as_panel=True)

   # Or with real RDW data
   panel = load_rdw_scrappage(data_dir="path/to/rdw_data/", as_panel=True)

Estimation
----------

NFXP and CCP estimators recover the four structural parameters from the panel. The age cost and replacement cost are well identified because they govern the rising hazard rate and the frequency of scrappage events. The defect cost parameters are identified through the differential scrappage rates across defect levels conditional on age.

.. code-block:: python

   from econirl import NFXP, CCP
   from econirl.datasets import load_rdw_scrappage

   df = load_rdw_scrappage()

   nfxp = NFXP(discount=0.95).fit(
       df, state="state", action="scrapped", id="vehicle_id"
   )
   ccp = CCP(discount=0.95, num_policy_iterations=20).fit(
       df, state="state", action="scrapped", id="vehicle_id"
   )

Low-level estimation pipeline
------------------------------

The environment and estimator classes give full control over transitions, features, and optimizer settings. Because the state space is two-dimensional (age times defect level), transition estimation counts state-to-state frequencies in the flattened space rather than using the one-dimensional increment estimator.

.. code-block:: python

   from econirl.environments.rdw_scrappage import RDWScrapageEnvironment
   from econirl.datasets import load_rdw_scrappage
   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.preferences.linear import LinearUtility

   env = RDWScrapageEnvironment(discount_factor=0.95)
   panel = load_rdw_scrappage(as_panel=True)
   utility = LinearUtility.from_environment(env)

   nfxp = NFXPEstimator(optimizer="BHHH", inner_solver="policy", compute_hessian=True)
   result = nfxp.estimate(panel, utility, env.problem_spec, env.transition_matrices)
   print(result.summary())

Using real RDW data
-------------------

The RDW makes vehicle registration and inspection data freely available under a CC-0 license at opendata.rdw.nl. The download script queries the SODA API for a specific vehicle cohort, merges the vehicle and inspection tables, and builds the car-year panel. By default it downloads Volkswagen Golf models registered between 2005 and 2015.

.. code-block:: bash

   # Download default cohort (VW Golf 2005-2015)
   python scripts/download_rdw.py

   # Download a different cohort
   python scripts/download_rdw.py --brand TOYOTA --model COROLLA --min-year 2008 --max-year 2018

   # Run estimation on downloaded data
   python examples/rdw-scrappage/rdw_nfxp.py --data-dir src/econirl/datasets/

Running the example
-------------------

.. code-block:: bash

   # Synthetic data (no download needed)
   python examples/rdw-scrappage/rdw_nfxp.py

   # Limit to 500 vehicles for quick testing
   python examples/rdw-scrappage/rdw_nfxp.py --max-vehicles 500

   # With real RDW data
   python examples/rdw-scrappage/rdw_nfxp.py --data-dir /path/to/rdw/

Reference
---------

RDW Open Data. Rijksdienst voor het Wegverkeer. https://opendata.rdw.nl

El Boubsi, M. (2023). Predicting Vehicle Inspection Outcomes Using Machine Learning. MSc Thesis, Delft University of Technology.

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. Econometrica, 55(5), 999-1033.

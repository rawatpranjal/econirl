RDW Vehicle Scrappage
=====================

.. image:: /_static/mdp_schematic_rdw_scrappage.png
   :alt: RDW scrappage MDP structure showing age chain with defect severity dimension and keep or scrap actions.
   :width: 80%
   :align: center

Every year in the Netherlands each passenger vehicle over three years old must pass a mandatory APK roadworthiness inspection. The vehicle owner observes the car's age and the severity of any defects found during the inspection. The owner then decides whether to keep the car and pay for repairs or to scrap the vehicle and exit the fleet. This example formulates the scrappage decision as a dynamic discrete choice model in the tradition of Rust (1987) and estimates it on real administrative data from the Dutch RDW open data registry.

The dataset contains 170,852 Volkswagen Golf vehicles registered between 2005 and 2015, observed annually from age 3 through either scrappage or right-censoring at the current date. The 2,009,144 vehicle-year observations include 17,106 exit events (scrappage or export). Defect severity comes from 447,296 individual defect records downloaded from the RDW geconstateerde gebreken dataset.

The data
--------

The scrappage hazard rises from near zero at age 3 to 8.2 percent at age 20. The observation count declines at older ages because later cohorts (registered 2010 to 2015) have not yet reached age 20.

.. image:: /_static/rdw_scrappage_by_age.png
   :alt: Bar chart showing annual scrappage rate rising from near zero at age 3 to 8.2 percent at age 20, with a line showing the number of vehicles observed declining at older ages.
   :width: 100%

Defect severity increases with age. At age 5 virtually all vehicles pass the APK inspection. By age 20 only 51 percent pass, 30 percent have minor defects, and 19 percent have major defects. Major defects like structural rust and braking failures roughly double the exit probability conditional on age.

.. image:: /_static/rdw_defect_by_age.png
   :alt: Grouped bar chart showing APK inspection defect distribution by vehicle age. At age 5 all vehicles pass. By age 20 only 51 percent pass, 30 percent have minor defects, and 19 percent have major defects.
   :width: 100%

The structural model
--------------------

The owner of vehicle :math:`i` in period :math:`t` observes the state :math:`s_{it} = (\text{age}_{it}, d_{it})` where :math:`\text{age}_{it} \in \{0, 1, \ldots, 24\}` is the vehicle age in years and :math:`d_{it} \in \{0, 1, 2\}` is the APK defect severity (0 for pass, 1 for minor, 2 for major). The owner chooses action :math:`a_{it} \in \{0, 1\}` where 0 means keep and 1 means scrap. The per-period flow utility is

.. math::

   u(s, a; \theta) = \begin{cases}
   -\theta_1 \cdot \text{age} - \theta_2 \cdot \mathbf{1}[d = 1] - \theta_3 \cdot \mathbf{1}[d = 2] + \varepsilon_{0} & \text{if } a = 0 \\
   -\theta_4 + \varepsilon_{1} & \text{if } a = 1
   \end{cases}

where :math:`\varepsilon_a` are i.i.d. Type I Extreme Value shocks. The parameter :math:`\theta_1` is the per-year operating cost that captures the rising maintenance burden of aging vehicles. The parameters :math:`\theta_2` and :math:`\theta_3` are the cost penalties for minor and major APK defects respectively. The parameter :math:`\theta_4` is the net replacement cost of scrapping the current vehicle and purchasing a new one.

The state transitions have two components. Age increments deterministically by one year when the owner keeps the car and resets to zero upon scrappage. Defect severity transitions stochastically with probabilities that depend on age. Older vehicles are more likely to develop defects in the next period. Formally, the transition probability under the keep action is

.. math::

   P\big((\text{age}+1, d') \mid (\text{age}, d), a=0\big) = \pi_{d \to d'}(\text{age})

where :math:`\pi_{d \to d'}(\text{age})` is the age-dependent defect transition matrix. Under the scrap action the state resets to :math:`(0, 0)` with probability one.

The owner solves the Bellman equation

.. math::

   V(s) = \sigma \log \sum_{a \in \{0,1\}} \exp\!\Big(\frac{u(s,a;\theta) + \beta \sum_{s'} P(s' \mid s,a) V(s')}{\sigma}\Big)

where :math:`\beta = 0.95` is the annual discount factor and :math:`\sigma = 1` is the logit scale parameter. The state space has :math:`25 \times 3 = 75` states and the model has 4 structural parameters.

Estimation results
------------------

NFXP maximizes the log-likelihood directly by solving the Bellman equation in an inner loop at each parameter evaluation. CCP inverts the Hotz-Miller mapping and iterates via nested pseudo-likelihood. Both estimators converge and agree closely on all four parameters.

.. list-table:: Structural Parameter Estimates (10,000 VW Golfs, RDW Real Data)
   :header-rows: 1

   * - Parameter
     - NFXP
     - CCP
   * - Age cost (:math:`\theta_1`)
     - 0.1099 (0.0054)
     - 0.1049
   * - Minor defect cost (:math:`\theta_2`)
     - -0.0979 (0.0959)
     - 0.0871
   * - Major defect cost (:math:`\theta_3`)
     - 0.3997 (0.1175)
     - 0.3219
   * - Replacement cost (:math:`\theta_4`)
     - 8.3857 (0.1986)
     - 8.2524
   * - Log-likelihood
     - -4478.7
     - -4481.4

NFXP standard errors are in parentheses. The age cost is highly significant with a t-statistic of 20.3 and the replacement cost has a t-statistic of 42.2. Minor defect cost is not statistically significant, indicating that minor APK defects like worn tires or dim bulbs do not materially influence scrappage decisions. Major defect cost is significant at the 0.1 percent level with a t-statistic of 3.4, confirming that structural rust, braking failures, and steering issues do push owners toward scrapping. NFXP and CCP produce closely agreeing estimates, with the log-likelihood difference of 2.7 points.

.. list-table:: Model Diagnostics
   :header-rows: 1

   * - Metric
     - NFXP
     - CCP
   * - Converged
     - Yes
     - Yes
   * - Prediction accuracy
     - 99.2%
     - 99.2%
   * - Hessian condition number
     - 57,488
     - N/A
   * - Identification status
     - Potentially weak
     - N/A
   * - Time (seconds)
     - 24
     - 248

The high correlation between age cost and replacement cost (0.976) is expected in Rust-style models because both parameters affect the optimal stopping threshold. Despite this, both parameters are individually significant and the Wald test rejects the null hypothesis that the replacement cost equals 5 at any conventional significance level.

Counterfactual analysis
-----------------------

The estimated model supports four types of counterfactual analysis. Each counterfactual solves a new Bellman equation under the modified primitives and compares the implied policy and welfare to the baseline.

Scrappage subsidy
^^^^^^^^^^^^^^^^^

A government policy that subsidizes 30 percent of the replacement cost reduces the effective :math:`\theta_4` from 8.39 to 5.87. The subsidy increases scrappage rates at every age, with the largest effect on vehicles aged 15 where the owner was previously on the margin.

.. image:: /_static/rdw_subsidy_counterfactual.png
   :alt: Paired bar charts comparing baseline and subsidized scrappage probabilities by age, separately for vehicles with no defects and vehicles with major defects.
   :width: 100%

.. list-table:: Scrappage Probability Under 30% Subsidy
   :header-rows: 1

   * - State (age, defect)
     - Baseline
     - With subsidy
     - Change
   * - (5, pass)
     - 0.01%
     - 0.14%
     - +0.13pp
   * - (10, pass)
     - 0.42%
     - 3.99%
     - +3.57pp
   * - (15, pass)
     - 2.47%
     - 18.44%
     - +15.97pp
   * - (20, pass)
     - 0.95%
     - 9.62%
     - +8.66pp
   * - (15, major)
     - 3.91%
     - 26.28%
     - +22.37pp
   * - (20, major)
     - 1.41%
     - 13.69%
     - +12.28pp

The subsidy has the largest impact on 15-year-old vehicles, where baseline scrappage is already elevated. The effect is similar across defect levels conditional on age, suggesting that the replacement cost rather than defect severity is the binding constraint.

Elasticity and welfare
^^^^^^^^^^^^^^^^^^^^^^

Varying the replacement cost by plus or minus 10 to 50 percent traces out the policy response surface and the associated welfare changes.

.. image:: /_static/rdw_elasticity.png
   :alt: Two-panel figure showing welfare change and average policy change as functions of the percentage change in replacement cost.
   :width: 100%

.. list-table:: Elasticity of Scrappage to Replacement Cost
   :header-rows: 1

   * - RC change
     - Avg policy change
     - Welfare change
   * - -50%
     - 24.13%
     - +7.90
   * - -30%
     - 8.54%
     - +2.56
   * - -10%
     - 1.35%
     - +0.38
   * - +10%
     - 0.63%
     - -0.18
   * - +30%
     - 1.04%
     - -0.29
   * - +50%
     - 1.12%
     - -0.31

The welfare response is strongly asymmetric. Reducing the replacement cost produces large welfare gains because it moves marginal owners past the scrappage threshold. Increasing the replacement cost produces small welfare losses because few owners are on the margin of keeping when the cost is already high.

The total welfare change from the 30 percent subsidy decomposes into a direct effect of 2.63 (value improvement at every state holding the fleet distribution fixed) and a distribution effect of 0.37 (the fleet shifts toward younger vehicles with fewer defects). The direct effect accounts for 88 percent of the total.

Defect deterioration
^^^^^^^^^^^^^^^^^^^^

If road conditions worsen and the defect-age sensitivity doubles from 0.02 to 0.04, vehicles accumulate defects faster. This raises scrappage probability at age 15 from 2.5 percent to 41 percent and at age 20 from 1.0 percent to 62 percent. The deterioration scenario accelerates fleet turnover dramatically.

Using real RDW data
-------------------

The RDW makes vehicle registration and inspection data freely available under a CC-0 license at opendata.rdw.nl. The download script queries the Socrata SODA API for a specific vehicle cohort, downloads defect records, and builds the car-year panel. By default it downloads Volkswagen Golf models registered between 2005 and 2015.

.. code-block:: bash

   # Download default cohort (VW Golf 2005-2015, ~170K vehicles)
   python scripts/download_rdw.py

   # Download a different cohort
   python scripts/download_rdw.py --brand TOYOTA --model COROLLA

   # Run the full showcase (estimation + counterfactuals)
   python examples/rdw-scrappage/rdw_showcase.py --data-dir src/econirl/datasets/

Running the example
-------------------

.. code-block:: bash

   # Full showcase with synthetic fallback (no download needed)
   python examples/rdw-scrappage/rdw_showcase.py

   # Quick test with 500 vehicles
   python examples/rdw-scrappage/rdw_showcase.py --max-vehicles 500

   # NFXP only (faster)
   python examples/rdw-scrappage/rdw_nfxp.py

   # With real RDW data
   python examples/rdw-scrappage/rdw_showcase.py --data-dir src/econirl/datasets/ --max-vehicles 5000

Reference
---------

RDW Open Data. Rijksdienst voor het Wegverkeer. https://opendata.rdw.nl

El Boubsi, M. (2023). Predicting Vehicle Inspection Outcomes Using Machine Learning. MSc Thesis, Delft University of Technology.

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. Econometrica, 55(5), 999-1033.

Kang, M., Yoganarasimhan, H., and Jain, V. (2025). Efficient Estimation of Random Coefficients Demand Models using Machine Learning. Working Paper.

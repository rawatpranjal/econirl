Keane-Wolpin Career Decisions
=============================

.. image:: /_static/mdp_schematic_keane_wolpin.png
   :alt: Keane-Wolpin MDP structure showing state with schooling and experience dimensions and four career-action branches.
   :width: 80%
   :align: center

This example replicates a simplified version of Keane and Wolpin (1997) on occupational choice data. Young men choose each year between four options: attend school, work white-collar, work blue-collar, or stay home. The decision is forward-looking because schooling and work experience accumulate as human capital that raises future wages.

.. image:: /_static/keane_wolpin_careers.png
   :alt: Career choice tree showing four options at each period and heatmap of schooling probability by state.
   :width: 100%

The state space encodes schooling level (10 to 20 years), white-collar experience (0 to 7 years), and blue-collar experience (0 to 7 years), giving 704 discrete states. Transitions are deterministic: choosing school increments the schooling counter, choosing white-collar work increments the white-collar experience counter, and so on. The model runs for 10 periods corresponding to ages 17 through 26.

.. code-block:: python

   from econirl.datasets import load_keane_wolpin
   from econirl.estimation.nfxp import NFXPEstimator

   df = load_keane_wolpin()
   # 704 states = 11 schooling x 8 exp_wc x 8 exp_bc
   # 4 actions = school, white-collar, blue-collar, home

The simplification relative to the original paper is in the shock distribution. Keane and Wolpin use multivariate normal shocks estimated via GHK simulation, which requires specialized likelihood machinery. This example uses logit (Type I extreme value) shocks, which gives the standard softmax choice probabilities that the econirl NFXP estimator handles natively. The finite-horizon backward induction algorithm and the state space structure match the original paper exactly.

.. list-table:: Estimated Parameters (logit simplification, 500 individuals, 10 periods)
   :header-rows: 1

   * - Parameter
     - Description
     - NFXP
   * - school_return
     - Returns to schooling
     - 2.065
   * - wc_exp_return
     - Returns to white-collar experience
     - 0.383
   * - bc_exp_return
     - Returns to blue-collar experience
     - 0.670
   * - school_cost
     - Direct cost of attending school
     - 0.231
   * - wc_intercept
     - Base white-collar wage
     - -0.104
   * - bc_intercept
     - Base blue-collar wage
     - 0.488
   * - home_value
     - Value of home production
     - 0.709

The school_return coefficient dominates at 2.065, reflecting the strong incentive for human capital accumulation. Both experience return coefficients are positive, confirming that work experience raises future earnings. The school_cost feature is coded as negative one in the feature matrix, so the positive coefficient translates to a net cost of attending school relative to other options. The model uses logit shocks rather than the multivariate normal shocks in the original paper, so point estimates differ from published values. Standard errors are unavailable because the Hessian is near-singular at this solution, a known challenge with high-dimensional finite-horizon models where many state cells have few observations.

Post-estimation diagnostics
---------------------------

Standard errors are unavailable for this model because the Hessian is near-singular. The ``check_identification`` function confirms the identification difficulty by reporting the Hessian condition number and minimum eigenvalue.

.. code-block:: python

   from econirl.inference.standard_errors import check_identification
   diag = check_identification(result.hessian)

The near-singular Hessian indicates that some parameters are weakly identified with 500 individuals and 10 periods. This is a known challenge with high-dimensional finite-horizon models. Many state cells in the 704-state space have few or zero observations, which leaves the likelihood surface flat in those directions. Increasing the sample size or reducing the state space dimensionality would improve identification.

The ``etable`` function still displays the point estimates even when standard errors are not available. The table shows NaN for SE and p-values, making the identification problem visible to the reader.

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result))

.. code-block:: bash

   python examples/keane-wolpin-careers/replicate.py

This example demonstrates that econirl handles finite-horizon problems with large multi-dimensional state spaces and more than two actions. The backward induction solver correctly propagates value through all 10 periods.

Reference
---------

Keane, M. P. and Wolpin, K. I. (1997). The Career Decisions of Young Men. Journal of Political Economy, 105(3), 473-522.

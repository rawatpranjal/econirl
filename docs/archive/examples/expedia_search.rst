Expedia Hotel Search
====================

This example models hotel search sessions on Expedia as a sequential discrete choice problem. The data comes from the Expedia ICDM 2013 Hotel Recommendation Competition and contains 9.9 million search result rows. Within each search session a user scrolls through hotel listings and decides at each position to scroll past, click for details, or book. Booking terminates the session.

The state space has 30 states arranged as 5 position bins by 3 price tertiles by 2 quality bins. Position captures how far the user has scrolled through the results. Price tertiles are computed within each session to normalize across searches with different budgets. Quality combines star rating and review score into an above-median or below-median indicator. The three actions are scroll (continue browsing), click (evaluate a listing in detail), and book (complete a reservation).

Estimation
----------

The CCP estimator with NPL refinement recovers the structural parameters from 8,625 search sessions over 120,527 total observations.

.. code-block:: python

   from econirl.estimation.ccp import CCPEstimator
   from econirl.preferences.linear import LinearUtility
   from econirl.core.types import DDCProblem

   utility = LinearUtility(feature_matrix=features, parameter_names=param_names)
   problem = DDCProblem(num_states=30, num_actions=3, discount_factor=0.95)

   npl = CCPEstimator(num_policy_iterations=10, compute_hessian=True)
   result = npl.estimate(panel, utility, problem, transitions)

.. list-table:: Structural Parameters (NPL K=10, 8625 sessions)
   :header-rows: 1

   * - Parameter
     - Estimate
     - SE
   * - theta_position
     - 0.2342
     - 0.0067
   * - theta_price
     - 0.7560
     - 0.0298
   * - theta_quality
     - 0.4260
     - 0.0256
   * - click_cost
     - -3.1888
     - 0.0236
   * - book_value
     - -2.4406
     - 0.0238

All five parameters are significant at the 0.1 percent level. The position coefficient is positive on the scroll action, meaning the utility of scrolling increases with position. This captures search fatigue: deeper in the results, the user is more willing to keep scrolling rather than click or book on a mediocre listing. The price coefficient is positive, and the price feature is coded as the negative of the price midpoint, so a positive coefficient means users are price-sensitive and prefer cheaper hotels. The quality coefficient is positive, confirming that higher quality increases the attractiveness of clicking or booking.

The click cost is negative 3.19, reflecting the attention and evaluation effort required to examine a listing in detail. The book value is negative 2.44 relative to the scroll baseline. Booking has positive net value through the quality and price match, but the large negative intercept means that users need a sufficiently attractive listing before they commit. The structural model rationalizes the low booking rate in the data (most sessions end without a booking) through these cost and value thresholds.

Diagnostics
-----------

The feature matrix has full rank (5 out of 5) with all 30 states observed. The three-action structure means that the CCP estimates have richer variation across states than a binary choice model. Position varies across states but quality and price are also informative, giving the model enough identifying variation to separate all five parameters.

Running the example
-------------------

.. code-block:: bash

   python examples/expedia-search/run_ccp.py

The robustness script varies the discount factor, session window sizes, subsampling seeds, action framing, and interaction terms.

.. code-block:: bash

   python examples/expedia-search/robustness.py

Reference
---------

Expedia ICDM 2013 Hotel Recommendation Competition.

Honka, E., Hortacsu, A., Vitorino, M.A. (2017). Advertising, Consumer Awareness, and Choice. RAND Journal of Economics 48(3): 611-646.

Trivago Hotel Search
====================

.. image:: /_static/mdp_schematic_trivago_search.png
   :alt: Trivago search MDP structure showing session state with browse and refine self-loops and clickout and abandon terminal actions.
   :width: 80%
   :align: center

This example models hotel search sessions on Trivago as a sequential discrete choice problem. At each step in a search session, the user decides to browse hotel details, refine the search with filters, click out to book a hotel, or abandon the session. The state space has 37 states encoding session depth, items viewed, and device type.

.. image:: /_static/trivago_search.png
   :alt: Hotel search session flow diagram showing browse, refine, clickout, and abandon actions and bar chart of structural parameters.
   :width: 100%

.. code-block:: python

   from econirl.datasets.trivago_search import (
       load_trivago_sessions, build_trivago_mdp, build_trivago_panel,
       build_trivago_features, build_trivago_transitions,
   )

   sessions = load_trivago_sessions(n_sessions=10000)
   mdp = build_trivago_mdp(sessions)
   panel = build_trivago_panel(mdp)

The reward function depends on both the state and the action. Browsing costs time. Refining the search with filters actually helps the user find better matches. Clicking out captures the booking value. Abandoning is the outside option.

.. list-table:: Structural Parameters
   :header-rows: 1

   * - Parameter
     - NFXP
     - CCP (K=3)
   * - Step cost
     - -0.71
     - -0.72
   * - Browse cost
     - -0.64
     - -0.64
   * - Refine value
     - 1.31
     - 1.31
   * - Clickout value
     - 1.80
     - 1.79

The refine parameter is positive. This means filtering and sorting reduces effective search cost rather than adding to it. A platform design team would read this as evidence that investing in better filter tools pays off more than adding more hotel images.

Post-estimation diagnostics
---------------------------

The ``etable`` function compares NFXP and CCP side by side with standard errors and significance stars. Both estimators produce nearly identical parameters, confirming that the CCP approximation is accurate for this 37-state problem.

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result))

The Vuong test between NFXP and CCP yields an indistinguishable result, as expected when both methods converge to the same MLE. The Brier score measures prediction quality against observed actions. On hotel search data, the Brier scores are moderate because search behavior has high inherent stochasticity.

.. code-block:: python

   from econirl.inference import vuong_test, brier_score
   vt = vuong_test(nfxp_result.policy, ccp_result.policy, obs_states, obs_actions)

The positive refine parameter is the most important structural finding. It means that adding filter and sort features to a hotel search platform reduces effective search cost rather than adding friction. This has a direct platform design implication: investing in better filter tools pays off more than adding more hotel images or review content.

The script also runs behavioral cloning as a non-structural baseline and compares test log-likelihood across all three estimators.

.. code-block:: bash

   python examples/trivago_search_ddc.py

Reference
---------

RecSys Challenge 2019. Session-Based Hotel Recommendations. https://recsys.trivago.cloud/

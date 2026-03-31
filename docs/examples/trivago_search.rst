Trivago Hotel Search
====================

.. image:: /_static/trivago_search.png
   :alt: Hotel search session flow diagram showing browse, refine, clickout, and abandon actions and bar chart of structural parameters.
   :width: 100%

This example models hotel search sessions on Trivago as a sequential discrete choice problem. At each step in a search session, the user decides to browse hotel details, refine the search with filters, click out to book a hotel, or abandon the session. The state space has 37 states encoding session depth, items viewed, and device type.

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

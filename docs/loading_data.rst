Loading Data
============

econirl expects panel data where each row is one decision by one individual in one period. This page explains how to get your data into the right shape, whether you are using a built-in dataset or bringing your own.

Built-in datasets
-----------------

econirl ships with datasets from the structural econometrics and inverse reinforcement learning literatures. Each loader returns a pandas DataFrame ready for estimation.

.. code-block:: python

   from econirl.datasets import load_rust_bus, load_keane_wolpin

   df = load_rust_bus()
   print(df.columns.tolist())

.. code-block:: text

   ['bus_id', 'period', 'mileage', 'mileage_bin', 'replaced', 'group']

.. list-table:: Built-in Datasets
   :header-rows: 1
   :widths: 25 15 15 45

   * - Loader
     - States
     - Actions
     - Domain
   * - ``load_rust_bus()``
     - 90
     - 2
     - Bus engine replacement (Rust 1987)
   * - ``load_keane_wolpin()``
     - 3+
     - 4
     - Career decisions (Keane and Wolpin 1994)
   * - ``load_robinson_crusoe()``
     - varies
     - 2
     - Production and leisure (pedagogical)
   * - ``load_equipment_replacement()``
     - varies
     - 2
     - Equipment replacement (generic)
   * - ``load_shanghai_trajectories()``
     - varies
     - varies
     - Taxi route choice on road network
   * - ``load_trivago_search()``
     - varies
     - 4
     - Hotel search sessions (Trivago 2019)
   * - ``load_taxi_gridworld()``
     - 25
     - 4
     - Gridworld navigation (benchmark)
   * - ``load_ngsim()``
     - varies
     - varies
     - Highway lane change (NGSIM)

Every loader accepts an ``as_panel=True`` flag that returns a Panel object instead of a DataFrame. The DataFrame form is better for exploration. The Panel form is what estimators use internally.

.. code-block:: python

   panel = load_rust_bus(as_panel=True)
   print(f"{panel.num_individuals} buses, {panel.num_observations} observations")

Your own data
-------------

If your data is in a pandas DataFrame, tell econirl which columns hold the state, action, and individual identifier.

.. code-block:: python

   from econirl import NFXP

   model = NFXP(discount=0.95)
   model.fit(df, state="state_col", action="action_col", id="id_col")

The ``state`` column must contain integer state indices starting from 0. The ``action`` column must contain integer action indices starting from 0. The ``id`` column identifies individuals so that transitions are computed within each individual's sequence.

If your state variable is continuous, discretize it first.

.. code-block:: python

   from econirl.preprocessing import discretize_mileage, discretize_state

   df["mileage_bin"] = discretize_mileage(df["mileage"], bin_width=5000, max_bins=90)

   df["income_bin"] = discretize_state(df["income"], method="quantile", n_bins=20)

``discretize_mileage`` follows the Rust (1987) convention where each bin is 5,000 miles wide. ``discretize_state`` supports both uniform (equal-width) and quantile (equal-count) binning for arbitrary continuous variables.

The Panel object
----------------

Under the hood, every estimator converts your DataFrame into a Panel. A Panel is a list of Trajectory objects, one per individual.

.. code-block:: python

   from econirl.core.types import Panel, Trajectory
   import jax.numpy as jnp

   traj = Trajectory(
       states=jnp.array([0, 5, 12, 18]),
       actions=jnp.array([0, 0, 0, 1]),
       next_states=jnp.array([5, 12, 18, 0]),
       individual_id="bus_001",
   )
   panel = Panel(trajectories=[traj])

You rarely need to construct these by hand. Use ``Panel.from_dataframe`` to convert from a DataFrame or ``Panel.from_numpy`` to convert from arrays.

.. code-block:: python

   panel = Panel.from_dataframe(
       df,
       state="mileage_bin",
       action="replaced",
       id="bus_id",
   )
   print(f"{panel.num_individuals} individuals")
   print(f"{panel.num_observations} total observations")

If you omit the ``next_state`` column, ``from_dataframe`` infers next states from sequential rows within each individual. If you have explicit next-state data, pass it as the ``next_state`` argument.

.. code-block:: python

   panel = Panel.from_dataframe(
       df,
       state="state",
       action="action",
       id="agent_id",
       next_state="next_state",
   )

Building from arrays
~~~~~~~~~~~~~~~~~~~~

If your data is already in NumPy arrays, use ``from_numpy``.

.. code-block:: python

   import numpy as np
   from econirl.core.types import Panel

   states = np.array([0, 1, 2, 0, 1, 3])
   actions = np.array([0, 0, 1, 0, 0, 1])
   next_states = np.array([1, 2, 0, 1, 3, 0])
   ids = np.array([0, 0, 0, 1, 1, 1])

   panel = Panel.from_numpy(states, actions, next_states, individual_ids=ids)

Transition matrices
-------------------

Most estimators need a transition matrix in addition to the panel data. The transition matrix has shape ``(n_actions, n_states, n_states)`` where ``transitions[a, s, s']`` is the probability of moving from state s to state s' when action a is taken.

You can estimate transition probabilities directly from the data.

.. code-block:: python

   from econirl.estimation import estimate_transition_probs
   from econirl.core.types import DDCProblem

   problem = DDCProblem(num_states=90, num_actions=2, discount_factor=0.9999)
   transitions = estimate_transition_probs(panel, problem)
   print(transitions.shape)

.. code-block:: text

   (2, 90, 90)

The rows of each ``transitions[a]`` matrix sum to 1. If some state-action pairs have zero observations, the corresponding row falls back to a uniform distribution with epsilon smoothing.

Validating your data
--------------------

Before estimation, check that your data has the expected structure. Common issues include missing values, gaps in the period sequence, and unbalanced panels.

.. code-block:: python

   from econirl.preprocessing import check_panel_structure

   result = check_panel_structure(
       df,
       id_col="bus_id",
       period_col="period",
       state_col="mileage_bin",
       action_col="replaced",
   )
   print(f"Valid: {result.valid}")
   print(f"Balanced: {result.is_balanced}")
   if result.warnings:
       print("Warnings:", result.warnings)

Feature normalization
---------------------

IRL estimators are sensitive to feature scale. If your features span different orders of magnitude, normalize them before estimation. The ``RunningNorm`` utility computes running mean and variance using a numerically stable algorithm (Chan, Golub, and LeVeque 1979) and can normalize new batches on the fly.

.. code-block:: python

   from econirl.preprocessing import RunningNorm

   norm = RunningNorm(size=n_features)
   norm.update(feature_matrix.reshape(-1, n_features))
   normalized_features = norm.normalize(feature_matrix)

For simpler cases, scaling features to the range negative 1 to 1 is usually sufficient.

Saving and loading panels
-------------------------

After constructing a panel, you can save it to disk as a compressed NumPy archive and reload it later. This is useful for reproducibility and for avoiding repeated preprocessing.

.. code-block:: python

   panel.save_npz("my_panel.npz")
   loaded = Panel.load_npz("my_panel.npz")
   assert loaded.num_observations == panel.num_observations

The ``.npz`` format stores states, actions, next states, trajectory lengths, and individual IDs. It is compact and does not require pickle.

Converting back to DataFrame
-----------------------------

If you need a DataFrame from a Panel for plotting or further analysis, use ``to_dataframe``.

.. code-block:: python

   df_out = panel.to_dataframe()
   print(df_out.columns.tolist())

.. code-block:: text

   ['id', 'period', 'state', 'action', 'next_state']

Sufficient statistics
---------------------

For tabular estimators that operate on counts rather than raw observations, Panel can precompute state-action counts, empirical CCPs, transition matrices, and the initial state distribution in a single pass.

.. code-block:: python

   stats = panel.sufficient_stats(n_states=90, n_actions=2)
   print(stats.state_action_counts.shape)
   print(stats.empirical_ccps.shape)
   print(stats.transitions.shape)
   print(stats.initial_distribution.shape)

.. code-block:: text

   (90, 2)
   (90, 2)
   (2, 90, 90)
   (90,)

This avoids redundant computation when multiple estimators are run on the same data.

Mini-batch iteration
--------------------

For neural estimators that train with SGD, Panel supports shuffled mini-batch iteration over transitions.

.. code-block:: python

   for states, actions, next_states in panel.iter_transitions(batch_size=512, seed=0):
       # Each batch is a tuple of JAX arrays with shape (B,)
       pass

Bootstrap resampling
--------------------

For bootstrap standard errors, Panel can resample individuals (trajectories) with replacement.

.. code-block:: python

   boot_panel = panel.resample_individuals(seed=42)
   print(f"Resampled: {boot_panel.num_individuals} individuals")

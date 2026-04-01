Loading Data
============

econirl expects panel data where each row is one decision by one individual in one period. This page explains how to get your data into the right shape, whether you are using a built-in dataset or bringing your own.

Built-in datasets
-----------------

econirl ships with datasets from the structural econometrics and inverse reinforcement learning literatures. Each loader returns a pandas DataFrame ready for estimation.

.. code-block:: python

   from econirl.datasets import load_rust_bus

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

.. code-block:: text

   90 buses, 9410 observations

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
   print(f"{panel.num_individuals} individual, {panel.num_observations} observations")

.. code-block:: text

   1 individual, 4 observations

You rarely need to construct these by hand. Use ``Panel.from_dataframe`` to convert from a DataFrame or ``Panel.from_numpy`` to convert from arrays.

.. code-block:: python

   from econirl.datasets import load_rust_bus
   from econirl.core.types import Panel

   df = load_rust_bus()
   panel = Panel.from_dataframe(
       df,
       state="mileage_bin",
       action="replaced",
       id="bus_id",
   )
   print(f"{panel.num_individuals} individuals")
   print(f"{panel.num_observations} total observations")

.. code-block:: text

   90 individuals
   9410 total observations

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
   print(f"{panel.num_individuals} individuals, {panel.num_observations} observations")

.. code-block:: text

   2 individuals, 6 observations

Transition matrices
-------------------

Most estimators need a transition matrix. For the Rust bus model, ``estimate_transition_probs`` estimates the mileage increment distribution from the data. It returns the probabilities of incrementing by 0, 1, or 2 bins per period.

.. code-block:: python

   from econirl.datasets import load_rust_bus
   from econirl.estimation import estimate_transition_probs

   df = load_rust_bus()
   theta = estimate_transition_probs(df)
   print(f"theta = ({theta[0]:.4f}, {theta[1]:.4f}, {theta[2]:.4f})")

.. code-block:: text

   theta = (0.3888, 0.5973, 0.0139)

For the general case, ``sufficient_stats`` computes the full transition matrix with shape ``(n_actions, n_states, n_states)`` where ``transitions[a, s, s']`` is the probability of moving from state s to state s' when action a is taken. The rows of each action slice sum to 1.

.. code-block:: python

   panel = Panel.from_dataframe(df, state="mileage_bin", action="replaced", id="bus_id")
   stats = panel.sufficient_stats(n_states=90, n_actions=2)
   print(f"Transition matrix shape: {stats.transitions.shape}")

.. code-block:: text

   Transition matrix shape: (2, 90, 90)

If some state-action pairs have zero observations, the corresponding row falls back to a uniform distribution with epsilon smoothing.

Validating your data
--------------------

Before estimation, check that your data has the expected structure. Common issues include missing values, gaps in the period sequence, and unbalanced panels.

.. code-block:: python

   from econirl.preprocessing import check_panel_structure

   df = load_rust_bus()
   result = check_panel_structure(
       df,
       id_col="bus_id",
       period_col="period",
       state_col="mileage_bin",
       action_col="replaced",
   )
   print(f"Valid: {result.valid}")
   print(f"Balanced: {result.is_balanced}")
   print(f"Individuals: {result.n_individuals}")
   print(f"Observations: {result.n_observations}")
   if result.warnings:
       print("Warnings:", result.warnings)

.. code-block:: text

   Valid: True
   Balanced: False
   Individuals: 90
   Observations: 9410
   Warnings: ['Unbalanced panel: 80-120 periods per individual']

Feature normalization
---------------------

IRL estimators are sensitive to feature scale. If your features span different orders of magnitude, normalize them before estimation. The ``RunningNorm`` utility computes running mean and variance using a numerically stable algorithm (Chan, Golub, and LeVeque 1979) and can normalize new batches on the fly.

.. code-block:: python

   from econirl.preprocessing import RunningNorm
   import numpy as np

   features = np.array([
       [0.001, 3.0, 100.0],
       [0.002, 5.0, 200.0],
       [0.003, 7.0, 300.0],
       [0.001, 4.0, 150.0],
   ])
   norm = RunningNorm(size=3)
   norm.update(features)
   print(f"Mean: {norm.mean}")
   print(f"Std:  {norm.std}")
   normalized = norm.normalize(features)
   print(f"Normalized mean: {[round(float(x), 1) for x in np.asarray(normalized).mean(axis=0)]}")
   print(f"Normalized std:  {[round(float(x), 1) for x in np.asarray(normalized).std(axis=0)]}")

.. code-block:: text

   Mean: [1.750e-03 4.750e+00 1.875e+02]
   Std:  [8.29e-04 1.48e+00 7.40e+01]
   Normalized mean: [0.0, 0.0, 0.0]
   Normalized std:  [1.0, 1.0, 1.0]

For simpler cases, scaling features to the range negative 1 to 1 is usually sufficient.

Saving and loading panels
-------------------------

After constructing a panel, you can save it to disk as a compressed NumPy archive and reload it later. This is useful for reproducibility and for avoiding repeated preprocessing.

.. code-block:: python

   panel.save_npz("rust_bus_panel.npz")
   loaded = Panel.load_npz("rust_bus_panel.npz")
   print(f"Saved: {panel.num_observations} observations")
   print(f"Loaded: {loaded.num_observations} observations, {loaded.num_individuals} individuals")

.. code-block:: text

   Saved: 9410 observations
   Loaded: 9410 observations, 90 individuals

The ``.npz`` format stores states, actions, next states, trajectory lengths, and individual IDs. The Rust bus panel compresses to 9.7 KB.

Converting back to DataFrame
-----------------------------

If you need a DataFrame from a Panel for plotting or further analysis, use ``to_dataframe``.

.. code-block:: python

   df_out = panel.to_dataframe()
   print(df_out.columns.tolist())
   print(df_out.head())

.. code-block:: text

   ['id', 'period', 'state', 'action', 'next_state']
      id  period  state  action  next_state
   0   1       0      0       0           0
   1   1       1      0       0           1
   2   1       2      1       0           1
   3   1       3      1       0           1
   4   1       4      1       0           1

Sufficient statistics
---------------------

For tabular estimators that operate on counts rather than raw observations, Panel can precompute state-action counts, empirical CCPs, transition matrices, and the initial state distribution in a single pass.

.. code-block:: python

   stats = panel.sufficient_stats(n_states=90, n_actions=2)
   print(f"state_action_counts: {stats.state_action_counts.shape}")
   print(f"empirical_ccps:      {stats.empirical_ccps.shape}")
   print(f"transitions:         {stats.transitions.shape}")
   print(f"initial_distribution:{stats.initial_distribution.shape}")

.. code-block:: text

   state_action_counts: (90, 2)
   empirical_ccps:      (90, 2)
   transitions:         (2, 90, 90)
   initial_distribution:(90,)

This avoids redundant computation when multiple estimators are run on the same data.

Mini-batch iteration
--------------------

For neural estimators that train with SGD, Panel supports shuffled mini-batch iteration over transitions.

.. code-block:: python

   count = 0
   for states, actions, next_states in panel.iter_transitions(batch_size=512, seed=0):
       count += 1
       if count == 1:
           print(f"First batch: {states.shape[0]} transitions")
   print(f"Total batches: {count}")

.. code-block:: text

   First batch: 512 transitions
   Total batches: 19

Bootstrap resampling
--------------------

For bootstrap standard errors, Panel can resample individuals (trajectories) with replacement.

.. code-block:: python

   boot_panel = panel.resample_individuals(seed=42)
   print(f"Original:  {panel.num_individuals} individuals, {panel.num_observations} observations")
   print(f"Resampled: {boot_panel.num_individuals} individuals, {boot_panel.num_observations} observations")

.. code-block:: text

   Original:  90 individuals, 9410 observations
   Resampled: 90 individuals, 9355 observations

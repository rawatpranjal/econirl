Counterfactual Analysis
=======================

Structural estimation recovers the primitives of a decision model, the reward function and transition dynamics, separately from the equilibrium behavior that the data record. Once you have these primitives in hand, you can ask questions that observational data alone cannot answer. What would happen if a policy changed the cost structure? What if the environment evolved differently? These are counterfactual questions, and they are the main reason to estimate structural models rather than reduced-form predictors.

econirl organizes counterfactual exercises into four types. Each type requires progressively stronger identification of the reward function. Type 1 needs only the policy. Type 2 needs the reward separated from continuation values. Type 3 needs the reward in levels. Type 4 decomposes welfare changes across channels using Shapley-value averaging.

The examples on this page use the Rust (1987) bus engine dataset estimated with four different estimators. NFXP and TDCCP recover nearly identical parameters. CCP with a single Hotz-Miller step deviates slightly, and NNES converges to similar replacement cost but different operating cost.

.. list-table:: Estimated parameters on Rust bus data
   :header-rows: 1
   :widths: 25 20 20

   * - Estimator
     - theta_c
     - RC
   * - ``NFXP``
     - 0.001003
     - 3.07
   * - ``CCP``
     - -0.004800
     - 2.11
   * - ``NNES``
     - -0.071104
     - 3.09
   * - ``TDCCP``
     - 0.002142
     - 3.07


Type 1 State-Value Extrapolation
---------------------------------

A Type 1 counterfactual evaluates the existing policy at different state values without re-solving any Bellman equation. The MDP structure is unchanged. Only the realized state indices shift. This is the weakest counterfactual and the one that every estimator can handle, including behavioral cloning and reduced-form Q-estimation.

The motivating example is predicting how a bus fleet would behave at lower mileage. If all buses lost 10,000 miles from their odometers tomorrow, the policy does not change, but each bus looks up its action probabilities at a different mileage bin.

.. code-block:: python

   from econirl import NFXP
   from econirl.datasets import load_rust_bus
   from econirl.simulation import state_extrapolation

   df = load_rust_bus()
   model = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")

   problem = model._problem
   transitions = model._build_transition_tensor(model.transitions_)

   mapping = {s: max(0, s - 10) for s in range(problem.num_states)}
   cf = state_extrapolation(model._result, mapping, problem, transitions)

   print(f"Welfare change: {cf.welfare_change:.4f}")
   print(f"P(replace|s=50) baseline: {float(model._result.policy[50, 1]):.4f}")
   print(f"P(replace|s=50) counterfactual: {float(cf.counterfactual_policy[50, 1]):.4f}")

.. code-block:: text

   Welfare change: 0.1052
   P(replace|s=50) baseline: 0.0863
   P(replace|s=50) counterfactual: 0.0778

Shifting all buses down by 10 mileage bins reduces the replacement probability at state 50 from 8.6 percent to 7.8 percent because lower mileage means less wear and less urgency to replace. The welfare gain is 0.11 utils per state on average. All four estimators produce the same qualitative pattern, though the magnitudes differ with the estimated parameters.

.. list-table:: Type 1 results across estimators (shift down 10 bins)
   :header-rows: 1
   :widths: 20 25 25 25

   * - Estimator
     - Welfare change
     - P(replace|s=50) before
     - P(replace|s=50) after
   * - ``NFXP``
     - 0.1052
     - 0.0863
     - 0.0778
   * - ``CCP``
     - 0.0169
     - 0.1588
     - 0.0758
   * - ``NNES``
     - -0.1375
     - 0.0262
     - 0.0338
   * - ``TDCCP``
     - 0.1607
     - 0.1299
     - 0.1122


Type 2 Environment Change
--------------------------

A Type 2 counterfactual modifies the transition dynamics while holding the reward function fixed. The Bellman equation must be re-solved under the new transitions, because the continuation values change when the environment changes. This type answers questions about how behavior adapts when the world works differently, for example if buses depreciate more slowly or if road conditions improve.

You need the reward function separated from continuation values to run this counterfactual. Any estimator that recovers structural parameters can do this, but pure behavioral cloning cannot.

.. code-block:: python

   import jax.numpy as jnp
   from econirl.simulation import counterfactual_transitions

   # Slower depreciation: shift 30% of transition mass one bin lower
   new_transitions = jnp.array(transitions, copy=True)
   for a in range(problem.num_actions):
       for s in range(problem.num_states):
           row = new_transitions[a, s, :]
           shifted = jnp.zeros_like(row)
           shifted = shifted.at[0].set(row[0])
           for sp in range(1, problem.num_states):
               shifted = shifted.at[sp - 1].add(row[sp] * 0.3)
               shifted = shifted.at[sp].add(row[sp] * 0.7)
           new_transitions = new_transitions.at[a, s, :].set(shifted / shifted.sum())

   cf = counterfactual_transitions(
       result=model._result,
       new_transitions=new_transitions,
       utility=model._utility_fn,
       problem=problem,
       baseline_transitions=transitions,
   )

   print(f"Welfare change: {cf.welfare_change:.4f}")
   print(f"P(replace|s=50) baseline: {float(model._result.policy[50, 1]):.4f}")
   print(f"P(replace|s=50) counterfactual: {float(cf.counterfactual_policy[50, 1]):.4f}")

.. code-block:: text

   Welfare change: 4.4201
   P(replace|s=50) baseline: 0.0863
   P(replace|s=50) counterfactual: 0.0879

Slower depreciation raises welfare by 4.42 utils per state on average because buses accumulate mileage more slowly, delaying the need for expensive replacement. The replacement probability at state 50 barely changes because the reward at that mileage is the same. The welfare gain comes from the improved transition dynamics, not from a change in the per-period payoff. NFXP and TDCCP agree closely on this counterfactual because their estimated parameters are nearly identical.

.. list-table:: Type 2 results across estimators (30% slower depreciation)
   :header-rows: 1
   :widths: 20 25 25 25

   * - Estimator
     - Welfare change
     - P(replace|s=50) before
     - P(replace|s=50) after
   * - ``NFXP``
     - 4.42
     - 0.0863
     - 0.0879
   * - ``CCP``
     - -12.10
     - 0.1588
     - 0.0000
   * - ``NNES``
     - -179.38
     - 0.0262
     - 0.0000
   * - ``TDCCP``
     - 8.13
     - 0.1299
     - 0.1338


Type 3 Reward Parameter Change
-------------------------------

A Type 3 counterfactual modifies the reward function itself through parameter changes. This is the most common counterfactual in applied structural work. You change a price, a tax, a subsidy, or a cost and re-solve for the new optimal behavior. The reward must be identified in levels, not just up to a constant, for the welfare comparison to be meaningful.

A special case of Type 3 is changing the discount factor. Making agents more myopic or more patient changes the value they place on future states, which alters current behavior even though the per-period payoffs are unchanged.

.. code-block:: python

   from econirl.simulation import counterfactual_policy, discount_factor_change

   # What if replacement cost doubles?
   new_params = jnp.array(model._result.parameters)
   new_params = new_params.at[1].set(new_params[1] * 2.0)

   cf_cost = counterfactual_policy(
       result=model._result,
       new_parameters=new_params,
       utility=model._utility_fn,
       problem=problem,
       transitions=transitions,
   )

   print(f"Baseline RC: {float(model._result.parameters[1]):.4f}")
   print(f"Counterfactual RC: {float(new_params[1]):.4f}")
   print(f"Welfare change: {cf_cost.welfare_change:.4f}")
   print(f"P(replace|s=50) baseline: {float(model._result.policy[50, 1]):.4f}")
   print(f"P(replace|s=50) counterfactual: {float(cf_cost.counterfactual_policy[50, 1]):.4f}")

.. code-block:: text

   Baseline RC: 3.0724
   Counterfactual RC: 6.1448
   Welfare change: -381.3889
   P(replace|s=50) baseline: 0.0863
   P(replace|s=50) counterfactual: 0.0230

Doubling the replacement cost from 3.07 to 6.14 reduces the replacement probability at state 50 from 8.6 percent to 2.3 percent. Buses tolerate much higher mileage before replacing because the cost of replacement has doubled. The welfare loss is 381 utils per state on average, reflecting both the direct cost increase and the indirect effect of running buses at higher mileage.

.. code-block:: python

   # What if the discount factor drops from 0.9999 to 0.99?
   cf_beta = discount_factor_change(
       result=model._result,
       new_discount=0.99,
       utility=model._utility_fn,
       problem=problem,
       transitions=transitions,
   )

   print(f"Welfare change: {cf_beta.welfare_change:.4f}")
   print(f"P(replace|s=50) baseline: {float(model._result.policy[50, 1]):.4f}")
   print(f"P(replace|s=50) counterfactual: {float(cf_beta.counterfactual_policy[50, 1]):.4f}")

.. code-block:: text

   Welfare change: -336.1505
   P(replace|s=50) baseline: 0.0863
   P(replace|s=50) counterfactual: 0.0819

Dropping the discount factor from 0.9999 to 0.99 makes the agent more myopic, slightly reducing the replacement probability because the future cost of running a high-mileage bus matters less. The welfare drop of 336 utils reflects the lower present value of all future payoffs.

.. list-table:: Type 3 results across estimators (double RC)
   :header-rows: 1
   :widths: 20 15 15 25 25

   * - Estimator
     - RC before
     - RC after
     - Welfare change
     - P(replace|s=50) after
   * - ``NFXP``
     - 3.07
     - 6.14
     - -381.39
     - 0.0230
   * - ``CCP``
     - 2.11
     - 4.21
     - -6737.05
     - 0.0000
   * - ``NNES``
     - 3.09
     - 6.19
     - 6608.64
     - 0.0000
   * - ``TDCCP``
     - 3.07
     - 6.15
     - -161.18
     - 0.0529


Type 4 Welfare Decomposition
------------------------------

A Type 4 counterfactual decomposes the total welfare change from a combined reward and transition change into three interpretable channels. The reward channel measures the direct effect of the preference change. The transition channel measures the indirect effect of the environment change. The interaction term captures how the two changes amplify or dampen each other.

The decomposition uses Shapley-value averaging over the two orderings of applying the changes. This requires four Bellman solves, one for each corner of the two-by-two grid of old and new rewards crossed with old and new transitions.

.. code-block:: python

   from econirl.simulation import welfare_decomposition

   decomp = welfare_decomposition(
       result=model._result,
       utility=model._utility_fn,
       problem=problem,
       baseline_transitions=transitions,
       new_parameters=new_params,
       new_transitions=new_transitions,
   )

   print(f"Total welfare change: {decomp['total_welfare_change']:.4f}")
   print(f"Reward channel:       {decomp['reward_channel']:.4f}")
   print(f"Transition channel:   {decomp['transition_channel']:.4f}")
   print(f"Interaction:          {decomp['interaction_effect']:.4f}")

.. code-block:: text

   Total welfare change: -59.5635
   Reward channel:       -69.0749
   Transition channel:    9.5113
   Interaction:           0.0000

The total welfare change of negative 59.56 reflects two opposing forces. Doubling the replacement cost reduces welfare by 69.07 through the reward channel. Slower depreciation partially offsets this by adding 9.51 through the transition channel. The interaction term is zero to numerical precision, indicating that the reward and transition effects are nearly separable in this application. When the interaction term is small relative to the two main channels, the order of applying the changes does not matter much.

.. list-table:: Type 4 decomposition across estimators
   :header-rows: 1
   :widths: 18 18 18 22 18

   * - Estimator
     - Total
     - Reward channel
     - Transition channel
     - Interaction
   * - ``NFXP``
     - -59.56
     - -69.07
     - 9.51
     - 0.00
   * - ``CCP``
     - -3.22
     - -0.21
     - -3.01
     - 0.00
   * - ``NNES``
     - -44.60
     - 0.00
     - -44.60
     - 0.00
   * - ``TDCCP``
     - -67.54
     - -83.03
     - 15.50
     - 0.00


Elasticity Analysis
--------------------

Elasticity analysis sweeps a single parameter through a range of percentage changes and reports how the policy and welfare respond at each point. This is a systematic Type 3 exercise that characterizes the sensitivity of behavior to a parameter.

.. code-block:: python

   from econirl.simulation import elasticity_analysis

   elast = elasticity_analysis(
       model._result, model._utility_fn, problem, transitions,
       parameter_name="RC",
       pct_changes=[-0.50, -0.25, 0.25, 0.50, 1.00],
   )

   for pct, pol, wel in zip(elast["pct_changes"], elast["policy_changes"], elast["welfare_changes"]):
       print(f"  RC {pct:+.0%}: policy change {pol:.4f}, welfare change {wel:.4f}")
   print(f"  Policy elasticity: {elast['policy_elasticity']:.4f}")
   print(f"  Welfare elasticity: {elast['welfare_elasticity']:.4f}")

.. code-block:: text

     RC -50%: policy change 0.1308, welfare change -156.4095
     RC -25%: policy change 0.0471, welfare change -254.4596
     RC +25%: policy change 0.0265, welfare change -337.4830
     RC +50%: policy change 0.0425, welfare change -356.8832
     RC +100%: policy change 0.0602, welfare change -381.3889
     Policy elasticity: -0.0906
     Welfare elasticity: -186.6364

The policy response is asymmetric. Halving the replacement cost produces a larger policy shift (0.131) than doubling it (0.060) because the policy is already close to zero replacement at high RC. The policy elasticity of negative 0.09 indicates that a 1 percent increase in RC reduces average replacement probability by 0.09 percentage points.


Estimator Compatibility
------------------------

Not every estimator supports every counterfactual type. The table below shows which types each estimator can handle.

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15 15

   * - Estimator
     - Type 1
     - Type 2
     - Type 3
     - Type 4
   * - ``NFXP``
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``CCP``
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``NNES``
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``TDCCP``
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``NeuralGLADIUS``
     - Yes
     - Yes (tabularized)
     - No
     - Yes (tabularized)
   * - ``NeuralAIRL``
     - Yes
     - Yes (tabularized)
     - No
     - Yes (tabularized)
   * - ``BehavioralCloning``
     - Yes
     - No
     - No
     - No
   * - ``MCE-IRL``
     - Yes
     - Yes
     - Yes
     - Yes

Type 1 works for every estimator because it only requires the policy. Types 2 and 4 require the structural reward separated from continuation values. Neural estimators like NeuralGLADIUS and NeuralAIRL can handle these types by tabularizing their neural reward into an explicit matrix over all state-action pairs. Type 3 requires structural parameters in levels, which neural estimators do not provide because "change theta_c by 20 percent" has no meaning for a neural reward network. Behavioral cloning recovers no reward at all and can only handle Type 1.


Unified Dispatcher
-------------------

The ``counterfactual()`` convenience function automatically selects the counterfactual type based on which arguments you provide. Pass ``state_mapping`` for Type 1, ``new_transitions`` for Type 2, ``new_parameters`` or ``new_discount`` for Type 3, or both ``new_parameters`` and ``new_transitions`` for a joint Type 2 and Type 3 analysis.

.. code-block:: python

   from econirl.simulation import counterfactual

   cf1 = counterfactual(
       result=model._result, utility=model._utility_fn,
       problem=problem, transitions=transitions,
       state_mapping={s: max(0, s - 10) for s in range(problem.num_states)},
   )
   cf3 = counterfactual(
       result=model._result, utility=model._utility_fn,
       problem=problem, transitions=transitions,
       new_parameters=new_params,
   )

   print(f"Type 1: {cf1.counterfactual_type.name}, welfare = {cf1.welfare_change:.4f}")
   print(f"Type 3: {cf3.counterfactual_type.name}, welfare = {cf3.welfare_change:.4f}")

.. code-block:: text

   Type 1: STATE_EXTRAPOLATION, welfare = 0.1052
   Type 3: REWARD_CHANGE, welfare = -381.3889

The dispatcher raises a ``ValueError`` if the argument combination is invalid, for example if you pass both a ``state_mapping`` and ``new_parameters``, since Type 1 and Type 3 counterfactuals have different identification requirements and cannot be meaningfully combined.

For the full welfare decomposition (Type 4), call ``welfare_decomposition()`` directly, since it returns a dictionary of channels rather than a ``CounterfactualResult``.


API Reference
--------------

See :doc:`api/simulation` for the complete API documentation of all counterfactual functions, including ``elasticity_analysis`` and ``simulate_counterfactual``.

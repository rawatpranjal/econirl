Counterfactual Analysis
=======================

Structural estimation recovers the primitives of a decision model, the reward function and transition dynamics, separately from the equilibrium behavior that the data record. Once you have these primitives in hand, you can ask questions that observational data alone cannot answer. What would happen if a policy changed the cost structure? What if the environment evolved differently? These are counterfactual questions, and they are the main reason to estimate structural models rather than reduced-form predictors.

econirl organizes counterfactual exercises into four types. Each type requires progressively stronger identification of the reward function. Type 1 needs only the policy. Type 2 needs the reward separated from continuation values. Type 3 needs the reward in levels. Type 4 decomposes welfare changes across channels using Shapley-value averaging.


Type 1 State-Value Extrapolation
---------------------------------

A Type 1 counterfactual evaluates the existing policy at different state values without re-solving any Bellman equation. The MDP structure is unchanged. Only the realized state indices shift. This is the weakest counterfactual and the one that every estimator can handle, including behavioral cloning and reduced-form Q-estimation.

The motivating example is predicting how a bus fleet would behave at lower mileage. If all buses lost 10,000 miles from their odometers tomorrow, the policy does not change, but each bus looks up its action probabilities at a different mileage bin.

.. code-block:: python

   from econirl import NFXP
   from econirl.datasets import load_rust_bus
   from econirl.simulation import state_extrapolation

   df = load_rust_bus()
   model = NFXP(discount=0.9999)
   result = model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

   problem = model.problem_
   transitions = model.transitions_

   # Every bus shifts down by 10 mileage bins
   mapping = {s: max(0, s - 10) for s in range(problem.num_states)}
   cf = state_extrapolation(result, mapping, problem, transitions)

   print(f"Average welfare change: {cf.welfare_change:.4f}")
   print(f"Max policy shift: {float(cf.policy_change.max()):.4f}")


Type 2 Environment Change
--------------------------

A Type 2 counterfactual modifies the transition dynamics while holding the reward function fixed. The Bellman equation must be re-solved under the new transitions, because the continuation values change when the environment changes. This type answers questions about how behavior adapts when the world works differently, for example if buses depreciate more slowly or if road conditions improve.

You need the reward function separated from continuation values to run this counterfactual. Any estimator that recovers structural parameters can do this, but pure behavioral cloning cannot.

.. code-block:: python

   import jax.numpy as jnp
   from econirl.simulation import counterfactual_transitions
   from econirl.preferences.linear import LinearUtility

   utility = LinearUtility.from_environment(model.env_)

   # Slower depreciation: shift transition mass toward lower mileage increments
   new_transitions = transitions.copy()
   # (In practice, you would construct the new transition matrix from
   # the alternative depreciation model.)

   cf = counterfactual_transitions(
       result=result,
       new_transitions=new_transitions,
       utility=utility,
       problem=problem,
       baseline_transitions=transitions,
   )

   print(f"Welfare change from slower depreciation: {cf.welfare_change:.4f}")


Type 3 Reward Parameter Change
-------------------------------

A Type 3 counterfactual modifies the reward function itself through parameter changes. This is the most common counterfactual in applied structural work. You change a price, a tax, a subsidy, or a cost and re-solve for the new optimal behavior. The reward must be identified in levels, not just up to a constant, for the welfare comparison to be meaningful.

A special case of Type 3 is changing the discount factor. Making agents more myopic or more patient changes the value they place on future states, which alters current behavior even though the per-period payoffs are unchanged.

.. code-block:: python

   from econirl.simulation import counterfactual_policy, discount_factor_change

   # What if replacement cost doubles?
   new_params = result.parameters.copy()
   new_params = new_params.at[1].set(new_params[1] * 2.0)

   cf_cost = counterfactual_policy(
       result=result,
       new_parameters=new_params,
       utility=utility,
       problem=problem,
       transitions=transitions,
   )

   print(f"Welfare change from doubling RC: {cf_cost.welfare_change:.4f}")

   # What if the discount factor drops to 0.99?
   cf_beta = discount_factor_change(
       result=result,
       new_discount=0.99,
       utility=utility,
       problem=problem,
       transitions=transitions,
   )

   print(f"Welfare change from lower patience: {cf_beta.welfare_change:.4f}")


Type 4 Welfare Decomposition
------------------------------

A Type 4 counterfactual decomposes the total welfare change from a combined reward and transition change into three interpretable channels. The reward channel measures the direct effect of the preference change. The transition channel measures the indirect effect of the environment change. The interaction term captures how the two changes amplify or dampen each other.

The decomposition uses Shapley-value averaging over the two orderings of applying the changes. This requires four Bellman solves, one for each corner of the two-by-two grid of old and new rewards crossed with old and new transitions.

.. code-block:: python

   from econirl.simulation import welfare_decomposition

   decomp = welfare_decomposition(
       result=result,
       utility=utility,
       problem=problem,
       baseline_transitions=transitions,
       new_parameters=new_params,
       new_transitions=new_transitions,
   )

   print(f"Total welfare change: {decomp['total_welfare_change']:.4f}")
   print(f"Reward channel: {decomp['reward_channel']:.4f}")
   print(f"Transition channel: {decomp['transition_channel']:.4f}")
   print(f"Interaction: {decomp['interaction_effect']:.4f}")

The three components sum to the total welfare change by construction. When the interaction term is small relative to the two main channels, the effects are approximately separable and the order of applying the changes does not matter much.


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

   # Type 1 dispatch
   cf1 = counterfactual(
       result=result,
       utility=utility,
       problem=problem,
       transitions=transitions,
       state_mapping={s: max(0, s - 10) for s in range(problem.num_states)},
   )

   # Type 3 dispatch
   cf3 = counterfactual(
       result=result,
       utility=utility,
       problem=problem,
       transitions=transitions,
       new_parameters=new_params,
   )

   print(f"Type 1 welfare change: {cf1.welfare_change:.4f}")
   print(f"Type 3 welfare change: {cf3.welfare_change:.4f}")

The dispatcher raises a ``ValueError`` if the argument combination is invalid, for example if you pass both a ``state_mapping`` and ``new_parameters``, since Type 1 and Type 3 counterfactuals have different identification requirements and cannot be meaningfully combined.

For the full welfare decomposition (Type 4), call ``welfare_decomposition()`` directly, since it returns a dictionary of channels rather than a ``CounterfactualResult``.


API Reference
--------------

See :doc:`api/simulation` for the complete API documentation of all counterfactual functions, including ``elasticity_analysis`` and ``simulate_counterfactual``.

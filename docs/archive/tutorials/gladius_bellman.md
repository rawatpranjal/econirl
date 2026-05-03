# GLADIUS Bellman Decomposition and Global Convergence

| | |
|---|---|
| **Paper** | Kang, Yoganarasimhan, and Jain (2025). ERM for Offline Inverse Reinforcement Learning in Dynamic Discrete Choice |
| **Estimators** | GLADIUS (dual neural Q + EV networks) vs NFXP |
| **Environment** | Rust bus engine, 90 mileage bins, beta=0.95 |
| **Key finding** | GLADIUS provides global convergence guarantees via the PL condition on the minimax objective, and an R-squared diagnostic for whether linear features suffice. On tabular problems NFXP is more accurate due to a structural identification limitation. |

## Background

Structural estimators like NFXP solve the Bellman equation exactly at each parameter guess. This requires enumerating every state and storing a transition matrix of shape (n_actions, n_states, n_states). When the state space is continuous or high-dimensional, this enumeration is impossible.

GLADIUS replaces the Bellman solve with two neural networks. The Q-network learns the action-value function Q(s,a) from the negative log-likelihood of observed actions. The EV-network (called zeta in the paper) learns the expected next-period value EV(s,a) by minimizing the mean squared Bellman error against V(s') computed from the Q-network. The implied reward is the difference r(s,a) = Q(s,a) minus beta times EV(s,a). This decomposition is called the bi-conjugate Bellman decomposition because it separates the estimation into two complementary objectives that together identify the reward.

The paper proves that the composite objective (NLL plus Bellman MSE) satisfies the Polyak-Lojasiewicz condition, which is weaker than strong convexity but sufficient for global convergence of stochastic gradient descent-ascent at rate O(T^{-1/4}). This is the main theoretical result: no other neural DDC method has a comparable global convergence guarantee.

After training, the neural reward surface is projected onto linear features via least-squares. The R-squared of this projection tells the researcher whether the linear features capture the reward structure. If R-squared is high (above 0.95), the structural parameters from the projection are reliable. If R-squared is low, the true reward function has nonlinear structure that the linear features miss.

## Setup

The full script is at ``examples/rust-bus-engine/gladius_bellman_showcase.py``.

```python
from econirl.estimation.gladius import GLADIUSEstimator, GLADIUSConfig

gladius = GLADIUSEstimator(config=GLADIUSConfig(
    q_hidden_dim=64,
    v_hidden_dim=64,
    q_num_layers=2,
    v_num_layers=2,
    max_epochs=500,
    batch_size=256,
    alternating_updates=True,
    lr_decay_rate=0.001,
    bellman_penalty_weight=1.0,
))
result = gladius.estimate(panel, utility, problem, transitions)
```

The ``alternating_updates=True`` flag activates Algorithm 1 from the paper. On even mini-batches, only the EV-network parameters are updated (target: V(s') from the frozen Q-network). On odd mini-batches, only the Q-network parameters are updated (target: maximize NLL of observed actions). This alternation prevents Q-value explosion, which occurs when Bellman gradients flow through both networks simultaneously in the IRL setting where rewards are not observed.

## Parameter Recovery

GLADIUS recovers the replacement cost within about 9 percent but overestimates the operating cost by a factor of 3.3. NFXP recovers both parameters within 5 percent.

| Parameter | True | NFXP | GLADIUS |
|---|---|---|---|
| operating_cost | 0.010000 | 0.009504 | 0.033435 |
| replacement_cost | 3.0000 | 2.9825 | 2.7433 |

NFXP required 15.7 seconds. GLADIUS required 210.7 seconds. On tabular problems the computational overhead of neural training does not pay for itself.

The projection R-squared is 0.983, confirming that the two linear features (operating cost and replacement cost indicators) capture nearly all the structure in the neural reward surface. The issue is not feature misspecification but the structural identification bias.

The operating cost bias is a known structural limitation, not a tuning problem. Without observed rewards, Q is trained via NLL alone (behavioral cloning). This identifies Q up to a state-dependent constant c(s). Because the Rust bus transition structure is asymmetric (the maintain action stays near state s while the replace action jumps to state 0), the constant propagates differently per action, producing a systematic bias in the implied reward difference.

## The Bi-Conjugate Decomposition

The GLADIUS result contains three tables in its metadata: Q-values, EV-values, and implied rewards. The implied reward at each state-action pair is r(s,a) = Q(s,a) minus 0.95 times EV(s,a).

The Q-values represent the total discounted expected utility of taking action a in state s and then following the optimal policy. The EV-values represent the expected total utility in the next period. The difference isolates the current-period reward from the continuation value.

## Projection R-squared

After recovering the neural reward surface, GLADIUS projects it onto the linear features using action-difference regression. For each state s, the action difference in implied rewards dr(s) = r(s, replace) minus r(s, keep) is regressed on the action difference in features dphi(s) = phi(s, replace) minus phi(s, keep). The R-squared of this regression measures how well the two-parameter linear model explains the neural reward function.

On the Rust bus where the true reward IS linear in the features, the projection R-squared is typically above 0.95, confirming that the linear model is a good approximation. On environments with genuinely nonlinear rewards, the R-squared would be lower, signaling that the linear projection parameters should be interpreted with caution.

## When to Use GLADIUS

GLADIUS is designed for continuous-state environments where NFXP cannot operate. The neural networks map raw state features to Q-values and EV-values without building a transition matrix or enumerating states. The global convergence guarantee (PL condition) provides theoretical assurance that gradient-based training will find the true reward.

On tabular problems with known transitions, NFXP remains the superior choice because it solves the Bellman equation exactly, avoids the identification bias, and produces analytical standard errors. GLADIUS should be the first choice when the state space makes tabular methods infeasible, or when rewards are observed in the data (the paper's original setting, where the bi-conjugate Bellman error anchors Q-values and eliminates the identification ambiguity).

The R-squared diagnostic is a unique advantage. It provides a formal check of whether the researcher's linear utility specification captures the learned reward structure, bridging the gap between flexible neural estimation and interpretable structural parameters.

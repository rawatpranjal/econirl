# MaxEnt IRL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Ziebart et al. (2008) | Linear (state-only) | Yes | Bootstrap | No |

## What this estimator does

Maximum Entropy IRL recovers a reward function from demonstrations by applying the maximum entropy principle to the space of trajectories. Among all policies that reproduce the observed feature statistics, the algorithm selects the one with maximum entropy over trajectory distributions, yielding the least committed explanation consistent with the data. The solution is a Boltzmann distribution over trajectories weighted by cumulative reward, and the partition function factors via backward recursion through the MDP.

The key limitation is that MaxEnt IRL uses non-causal Shannon entropy over entire trajectories rather than conditioning each action on the agent's available information at decision time. The backward messages propagate future information into current decisions, giving the agent effective foresight. With action-dependent features, this foresight produces systematically biased reward estimates. On a 5 by 5 gridworld with direction-dependent step costs, MaxEnt IRL recovers a reward with cosine similarity of approximately negative 0.72 to the truth, while MCE-IRL achieves 0.9999. This is a structural bias, not a tuning problem. The only fix is the causal entropy objective of Ziebart (2010), implemented as MCE-IRL.

Rewards are restricted to be state-only, meaning $R(s;\theta) = \theta^\top \phi(s)$. Action-dependent features cannot be handled correctly because the non-causal backward messages do not factor into per-step decisions that respect the sequential timing of information arrival.

## How it works

The gradient of the log-likelihood is the difference between empirical and expected feature averages under the current policy:

$$
\nabla_\theta \log L = N(\tilde\phi - \mathbb{E}_\pi[\phi]).
$$

The expected features are computed by a forward pass that propagates the initial state distribution through the policy-weighted transitions. The policy itself comes from non-causal backward messages that compute a local partition function at each state. The optimizer updates $\theta$ via L-BFGS-B until the gradient vanishes, meaning the model's feature expectations match the data. Standard errors are available via bootstrap by resampling trajectories.

## When to use it

MaxEnt IRL should generally be replaced by MCE-IRL, which fixes the non-causal bias while keeping the same computational structure. MaxEnt IRL is useful as a historical reference and as a baseline to demonstrate the importance of causal entropy. When features are state-only and do not depend on actions, MaxEnt and MCE-IRL produce the same result because the non-causal bias only manifests with action-dependent features.

## References

- Ziebart, B. D., Maas, A., Bagnell, J. A., and Dey, A. K. (2008). Maximum Entropy Inverse Reinforcement Learning. *AAAI 2008*.

The full derivation, algorithm, and simulation results are in the [MaxEnt IRL primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/maxent_irl.pdf).

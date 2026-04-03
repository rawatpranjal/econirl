# AIRL

| Category | Citation | Reward | Transitions | SEs | Scales | Transfer |
|----------|----------|--------|-------------|-----|--------|----------|
| Inverse | Fu, Luo, and Levine (2018) | Linear or Tabular | No (adversarial) | No | No | Yes |

## Background

GAIL (Ho and Ermon 2016) framed IRL as a GAN: a discriminator tells apart expert and agent behavior, and the agent learns to fool it. But the "reward" that GAIL learns is entangled with the training environment. Move to a new environment and the learned reward stops working. Fu, Luo, and Levine (2018) fixed this by giving the discriminator a special structure. They split its output into a reward piece $g(s,a)$ and a shaping piece $\beta h(s') - h(s)$. The shaping piece absorbs everything that depends on the environment. What remains, $g$, is the true reward, and it transfers to new settings. Economists will recognize this as the potential-based shaping from Ng, Harada, and Russell (1999).

## Key Equations

$$
f_\phi(s,a,s') = g_\phi(s,a) + \beta \, h_\phi(s') - h_\phi(s), \qquad D_\phi = \frac{\exp(f_\phi)}{\exp(f_\phi) + \pi(a \mid s)}.
$$

At the optimum, $h_\phi(s) = V(s)$ (the soft value function defined above), so $f$ is the stochastic advantage $r(s,a) + \beta V(s') - V(s)$. The term $g_\phi(s,a)$ plays the role of the reward $r(s,a)$ and is the only piece that transfers across environments.

## Pseudocode

```
AIRL(D_expert, p, beta, sigma, max_rounds):
  1. Initialize reward network g, shaping network h, policy pi
  2. For each round:
     a. Collect transitions from expert and from current policy
     b. Train discriminator D to tell them apart
     c. Use discriminator signal as reward to improve policy
  3. Extract portable reward: r(s,a) = g(s,a)
  4. Return g, pi
```

## Strengths and Limitations

AIRL provides mathematical transfer guarantees. The recovered state rewards remain valid even when transition dynamics in downstream environments change. This makes it the only estimator in the package designed for sim-to-real transfer, where you learn on one environment and deploy in another.

The limitation is training instability. The adversarial training loop can be deeply unstable, and solving the full MDP at each iteration is slow. AIRL also does not produce standard errors, so it is not suitable when formal inference is required. Under clean conditions (deterministic transitions, state-only reward), the reward network $g$ recovers the true reward up to a constant (Theorem 5.1).

AIRL is the right choice when the learned reward needs to work in a different environment than the one it was trained in.

## In econirl

The package implementation is `NeuralAIRL`. It accepts either a pandas DataFrame or a `TrajectoryPanel`. `TrajectoryPanel` stores panel arrays in JAX, while `NeuralAIRL` trains its networks in Torch. The estimator bridges that boundary during minibatching, so user code can stay on the panel side.

See {doc}`../tutorials/neural_airl_trajectory_panel` for an end to end example.

## References

- Fu, J., Luo, K., & Levine, S. (2018). Learning Robust Rewards with Adversarial Inverse Reinforcement Learning. *ICLR 2018*.
- Ho, J. & Ermon, S. (2016). Generative Adversarial Imitation Learning. *NeurIPS 2016*.
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. *ICML 1999*.

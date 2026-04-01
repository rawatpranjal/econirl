# BC

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Baseline | Ross et al. (2011) | None | No | No | No |

## Background

Before trying anything fancy, check whether the data has a signal at all. Behavioral cloning just counts: how often was each action chosen in each state? No model, no optimization, no value function. If a sophisticated estimator cannot beat this, it is not learning anything useful. Ross et al. (2011) showed that BC errors grow as $O(T^2 \varepsilon)$ with horizon length $T$, while methods that recover the true reward achieve $O(\varepsilon)$ regardless of $T$. That gap is why structural estimation matters.

## Key Equations

$$
\hat\pi(a \mid s) = \frac{N(s,a)}{N(s)}.
$$

## Pseudocode

```
BC(D):
  1. For each state s, count how often each action a was chosen
  2. pi(a|s) = count(s,a) / count(s)
  3. Return pi
```

## Strengths and Limitations

BC is computationally instantaneous and validates whether fundamental data signals exist. It requires no model, no transition matrix, and no optimization. Every evaluation should start here.

The limitation is that BC has no structural capability and zero transferability. Errors compound as $O(T^2 \varepsilon)$ under distributional drift, compared to $O(\varepsilon)$ for structural methods. BC also cannot recover any parameters, rewards, or value functions. If structural estimators fail to beat BC on out-of-sample data, the environment fundamentally lacks the sequential forward-looking structure that model-based methods exploit.

BC should always be your first step as a mandatory baseline.

## References

- Ross, S., Gordon, G. J., & Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *AISTATS 2011*.

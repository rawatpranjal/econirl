# econirl

Benchmarking dynamic discrete choice and inverse RL algorithms on a variety of MDPs — comparing reward recovery, imitation, and generalization.

## Install

```bash
uv pip install -e .
```

## Try It

```python
from econirl.evaluation.benchmark import BenchmarkDGP, run_single, get_default_estimator_specs

# 5-state bus engine replacement MDP (Rust 1987)
dgp = BenchmarkDGP(n_states=5, discount_factor=0.95)
specs = get_default_estimator_specs()

# Run all 12 estimators with benchmark-tuned defaults
for spec in specs:
    result = run_single(dgp, spec, n_agents=100, n_periods=50, seed=42)
    print(f"{result.estimator:12s}  {result.pct_optimal:6.1f}%  {result.time_seconds:5.1f}s")
```

![5-State Bus Engine Replacement MDP](docs/mdp_data_generation.gif)

### Results

| Estimator      | Type        | % Optimal | % Transfer | Param RMSE | Policy RMSE | Time  |
|----------------|-------------|----------:|-----------:|------------|-------------|------:|
| BC             | Baseline    |     99.5% |      99.5% | —          | 0.0267      |  0.1s |
| NFXP           | Structural  |     99.7% |      99.8% | 0.3353     | 0.0233      |  7.1s |
| CCP            | Structural  |     99.7% |      99.8% | 0.3142     | 0.0228      | 14.5s |
| MCE IRL        | IRL         |     99.7% |      99.7% | 0.1023     | 0.0199      |  8.3s |
| MaxEnt IRL     | IRL         |     98.2% |      97.8% | —          | 0.0443      |  3.3s |
| Max Margin     | IRL         |     99.3% |      99.3% | 0.2427     | 0.0218      | 23.7s |
| Max Margin IRL | IRL         |     31.1% |      34.2% | —          | 0.2068      |  0.2s |
| TD-CCP         | Neural      |     99.7% |      99.7% | 0.1918     | 0.0184      | 11.6s |
| GLADIUS        | Neural      |     99.6% |      88.7% | 0.6919     | 0.0256      |  4.0s |
| GAIL           | Adversarial |     54.2% |      50.9% | —          | 0.1739      | 75.8s |
| AIRL           | Adversarial |     99.6% |      99.5% | —          | 0.0230      | 66.9s |
| GCL            | Neural      |     92.5% |      95.3% | —          | 0.0604      | 93.5s |

5-state MDP, 100 agents x 50 periods, seed=42. **% Optimal** = value achieved vs true optimal on training dynamics (baseline-normalized). **% Transfer** = same metric on held-out transition dynamics (same rewards, different wear rates). **BC** = behavioral cloning baseline (empirical P(a|s) from data, transfer evaluates the same fixed policy on new dynamics).

![Internal Validity — Policy Execution on Training Dynamics](docs/internal_validity.gif)

![External Validity — Policy Execution on Transfer Dynamics](docs/external_validity.gif)

![Estimated vs True Rewards](docs/reward_heatmaps.png)

## DDC and IRL Algorithms

| Algorithm      | Paper | Method |
|----------------|-------|--------|
| BC             | — | Supervised baseline: empirical P(a\|s) from demonstrations |
| NFXP           | [Rust (1987)](https://doi.org/10.2307/1911259) | Full-solution MLE via nested fixed point |
| CCP            | [Hotz & Miller (1993)](https://doi.org/10.2307/2298122) | Two-step conditional choice probability |
| MCE IRL        | [Ziebart (2010)](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf) | Maximum causal entropy IRL |
| MaxEnt IRL     | [Ziebart et al. (2008)](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf) | Maximum entropy IRL with state visitation |
| Max Margin     | [Ratliff et al. (2006)](https://doi.org/10.1145/1143844.1143936) | Structured max-margin planning |
| Max Margin IRL | [Abbeel & Ng (2004)](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf) | Apprenticeship learning via margin maximization |
| TD-CCP         | [Adusumilli & Eckardt (2022)](https://arxiv.org/abs/1912.09509) | TD-learning + CCP with neural approximation |
| GLADIUS        | [Kang, Yoganarasimhan & Jain (2025)](https://arxiv.org/abs/2502.14131) | Neural Bellman-consistent estimation |
| GAIL           | [Ho & Ermon (2016)](https://arxiv.org/abs/1611.03852) | Generative adversarial imitation learning |
| AIRL           | [Fu et al. (2018)](https://arxiv.org/abs/1710.11248) | Adversarial inverse reinforcement learning |
| GCL            | [Finn et al. (2016)](https://arxiv.org/abs/1603.00448) | Guided cost learning with importance sampling |

**Behavioral Cloning** (BC) is a supervised baseline that counts action frequencies per state — it sets the floor any IRL method should beat. **Structural estimators** (NFXP, CCP) recover flow utility parameters by maximum likelihood, assuming the econometrician knows the model. **IRL methods** (MCE IRL, MaxEnt IRL, Max Margin, Max Margin IRL) recover reward functions from demonstrations without requiring a parametric model of agent behavior. **Neural estimators** (TD-CCP, GLADIUS) approximate value functions with neural networks for scalability. **Adversarial methods** (GAIL, AIRL, GCL) learn reward or policy via a discriminator that distinguishes expert from generated behavior.

## License

MIT

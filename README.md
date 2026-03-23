# econirl

Structural Estimation meets Inverse Reinforcement Learning. 10 algorithms, one API.

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

# Run all 10 estimators with benchmark-tuned defaults
for spec in specs:
    result = run_single(dgp, spec, n_agents=100, n_periods=50, seed=42)
    print(f"{result.estimator:12s}  {result.pct_optimal:6.1f}%  {result.time_seconds:5.1f}s")
```

### Results

| Estimator  | Type        | % Optimal | Param RMSE | Policy RMSE | Time  |
|------------|-------------|----------:|------------|-------------|------:|
| NFXP       | Structural  |     99.7% | 0.3353     | 0.0233      |  10.1s |
| CCP        | Structural  |     99.7% | 0.3142     | 0.0228      |  20.3s |
| MCE IRL    | IRL         |     99.7% | 0.1023     | 0.0199      |  12.9s |
| MaxEnt IRL | IRL         |     98.2% | —          | 0.0443      |   4.7s |
| Max Margin | IRL         |     99.3% | 0.2427     | 0.0218      |  30.5s |
| TD-CCP     | Neural      |     99.8% | 0.1574     | 0.0172      |  15.3s |
| GLADIUS    | Neural      |     99.7% | 0.7133     | 0.0229      |   3.8s |
| GAIL       | Adversarial |     54.0% | —          | 0.1733      |  94.2s |
| AIRL       | Adversarial |     99.6% | —          | 0.0222      |  68.1s |
| GCL        | Neural      |     93.1% | —          | 0.0585      |  97.3s |

5-state MDP, 100 agents × 50 periods, seed=42. % Optimal = value achieved vs true optimal (baseline-normalized).

## The 10 Algorithms

| Algorithm  | Paper | Method |
|------------|-------|--------|
| NFXP       | Rust (1987) | Full-solution MLE via nested fixed point |
| CCP        | Hotz & Miller (1993) | Two-step conditional choice probability |
| MCE IRL    | Ziebart et al. (2010) | Maximum causal entropy IRL |
| MaxEnt IRL | Ziebart et al. (2008) | Maximum entropy IRL with state visitation |
| Max Margin | Ratliff et al. (2006) | Structured max-margin planning |
| TD-CCP     | Blevins (2014) | TD-learning + CCP with neural approximation |
| GLADIUS    | Semenova et al. (2024) | Neural Bellman-consistent estimation |
| GAIL       | Ho & Ermon (2016) | Generative adversarial imitation learning |
| AIRL       | Fu et al. (2018) | Adversarial inverse reinforcement learning |
| GCL        | Finn et al. (2016) | Guided cost learning with importance sampling |

## License

MIT

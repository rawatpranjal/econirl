# Keane & Wolpin (1997) Career Decisions — Replication Notes

## Paper

Keane, M. P. & Wolpin, K. I. (1997). "The Career Decisions of Young Men." *Journal of Political Economy*, 105(3), 473-522.

See also: Keane & Wolpin (1994). "The Solution and Estimation of Discrete Choice Dynamic Programming Models by Simulation and Interpolation." *Review of Economics and Statistics*, 76(4), 648-672.

## The Problem

Young men from the NLSY 1979 cohort make sequential career decisions. Each year from age 16-26, they choose one of:
- **School** (choice 0): Accumulate education
- **White-collar work** (choice 1): Accumulate WC experience, earn WC wage
- **Blue-collar work** (choice 2): Accumulate BC experience, earn BC wage
- **Home production** (choice 3): No state accumulation

State variables evolve endogenously based on choices: schooling, occupation-specific experience, and age determine future wages and opportunities.

## Model Specification

### State Variables
- `schooling`: Years of completed education (10-20)
- `exp_white_collar`: Years of white-collar experience (0-7+)
- `exp_blue_collar`: Years of blue-collar experience (0-7+)
- `age`: Current age (16-26 in estimation sample)

### Wage Equations (from paper eq. 2)
For occupation m in {WC, BC}:
```
ln(w_m) = ln(r_m) + e_m(16) + e_{m1}*schooling + e_{m2}*exp_m - e_{m3}*exp_m^2 + epsilon_m
```
where r_m is the occupation rental price, e_m(16) is the skill endowment, and epsilon_m is a shock.

### Reward Functions
- Work: Current wage w_m(a)
- School: Fixed effort cost + endowment + direct costs (tuition)
- Home: Fixed endowment + shock

### Transition Rules (Deterministic)
- School: schooling += 1
- White-collar: exp_wc += 1
- Blue-collar: exp_bc += 1
- Home: no state change

### Estimation
- Finite horizon: age 16 to ~65 (terminal value at end)
- Discount factor: beta ~ 0.95
- Shocks: Multivariate normal (not Type I EV like Rust)
- Original method: Simulation + interpolation (KW 1994)

## Our Approach

Since the state space in our discretized data is manageable (schooling: 11 levels x exp_wc: 8 levels x exp_bc: 8 levels = 704 states), we can run exact NFXP rather than simulation-based methods.

Key simplification: We use logit (Type I EV) shocks instead of the multivariate normal from the original paper. This changes the estimates somewhat but keeps the DDC framework intact.

## Data

- **Source**: Bundled synthetic sample matching NLSY 1979 structure
- **500 individuals**, 10 periods each (ages 17-26)
- **4 choices**: School (30%), WC work (29%), BC work (28%), Home (13%)
- State space: 704 discrete states

## Running

```bash
python examples/keane-wolpin-careers/replicate.py
```

## References

- Keane & Wolpin (1997). "The Career Decisions of Young Men." JPE 105(3).
- Keane & Wolpin (1994). "Solution and Estimation of DCDP Models." REStat 76(4).
- Aguirregabiria & Mira (2002). "Swapping the Nested Fixed Point Algorithm." Econometrica 70(4).
- respy package (OpenSourceEconomics) — Python reference for KW estimation

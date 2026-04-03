"""Run a lightweight NFXP finite-sample check and update results JSON.

This is a quick runner to add NFXP numbers to Analysis D for a few
sample sizes with one seed to keep runtime reasonable.
"""

from pathlib import Path
import json

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .run_full_simulation import (
    MDPConfig, build_mdp, solve_mdp, build_P_skip, generate_data,
    _build_panel_and_problem, cf_error,
)


def estimate_nfxp(data, mdp):
    from econirl.estimation.nfxp import NFXPEstimator
    panel, problem, utility, transitions_jnp = _build_panel_and_problem(data, mdp)
    est = NFXPEstimator(optimizer="BHHH", inner_solver="hybrid", inner_tol=1e-10,
                        inner_max_iter=20000, outer_max_iter=300, compute_hessian=False,
                        verbose=False)
    summary = est.estimate(panel, utility, problem, transitions_jnp)
    reward_table = utility.compute(summary.parameters)
    return {'r': jnp.array(reward_table), 'pi': summary.policy}


def main():
    cfg = MDPConfig()
    mdp = build_mdp(cfg)
    sol = solve_mdp(mdp['P'], mdp['r'], cfg.beta, mdp['ABS'])
    r_true, ABS, beta, nR = mdp['r'], mdp['ABS'], cfg.beta, mdp['n_regular']
    Pt3 = build_P_skip(mdp, skip=3)

    sizes = [200, 500, 2000, 5000, 10000]
    n_seeds = 3
    out_path = Path(__file__).parent / 'results' / 'full_simulation.json'
    js = json.loads(out_path.read_text()) if out_path.exists() else {"D": {}}
    js.setdefault('D', {})
    for N in sizes:
        errs = []
        for seed in range(n_seeds):
            data = generate_data(mdp, sol, N, jr.PRNGKey(1000 * seed + N))
            try:
                nfxp = estimate_nfxp(data, mdp)
                err = cf_error(Pt3, nfxp['r'], r_true, beta, ABS, nR)
                errs.append(float(err))
            except Exception:
                errs.append(float('nan'))
        row = js['D'].get(str(N), {})
        row['nfxp'] = round(np.nanmean(errs), 4)
        js['D'][str(N)] = row
        print(f"N={N}: NFXP Type II error (k=3) ~ {row['nfxp']} ({n_seeds} seeds)")

    out_path.write_text(json.dumps(js, indent=2))
    print(f"Updated {out_path}")


if __name__ == '__main__':
    main()

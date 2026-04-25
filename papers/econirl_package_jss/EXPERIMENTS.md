# Experiment matrix for the econirl JSS paper

This document is the authoritative list of code runs that will accompany the paper. Every numerical claim, listing, figure, and table in the paper traces to exactly one row here. Each row names the estimator, the dataset and example, the script that generates the output, and the single point the run is meant to make. If a row is in this document the run goes into the paper. If a run is in the paper it appears in this document. There is no third path.

The matrix follows the protocol in `BENCHMARK_PROTOCOL.md`. All runs use random seed 42, the hardware documented in the computational details section, and the package and JAX versions pinned at the top of each script. Every script writes its result to a JSON or CSV file named after the script so the paper can be recompiled without rerunning the experiments.

## Section 1. Introduction teaser

| ID | Estimator | Data | Script | Output | Emphasis |
| --- | --- | --- | --- | --- | --- |
| L1 | NFXP | Rust bus | `code_snippets/teaser_nfxp_rust.py` | console output (eight lines) shown verbatim as Listing 1 | The user-facing workflow fits an estimator and prints a regression-style summary in five lines of code. |

The teaser is the only code in the introduction. It shows that the unified API is real, not aspirational.

## Section 4.1. Worked example: structural likelihood on Rust bus

| ID | Estimator | Data | Script | Output | Emphasis |
| --- | --- | --- | --- | --- | --- |
| L2 | NFXP | Rust bus | `code_snippets/listing2_nfxp_example.py` | console (full `summary()` block) as Listing 2 | The standard workflow returns asymptotic standard errors, identification diagnostics, and a converged value function in one call. |
| F1 | NFXP | Rust bus | `code_snippets/fig1_rust_bus_ccp.py` | `figures/fig1_rust_bus_ccp.pdf` | Replacement probability rises monotonically in mileage. The 25 percent counterfactual reduction in $RC$ shifts the boundary leftward by roughly 30 mileage bins. |
| F2 | NFXP | Rust bus | `code_snippets/fig2_rust_bus_value.py` | `figures/fig2_rust_bus_value.pdf` | The converged $V(s)$ is concave and decreasing in mileage, consistent with rising maintenance cost being absorbed against the option value of replacement. |
| L2b | NFXP (bootstrap) | Rust bus | `code_snippets/listing2b_bootstrap.py` | console output as a follow-up listing | Switching from asymptotic to bootstrap standard errors requires a single argument change. Both methods agree to two significant figures on this panel. |

The Rust bus is the canonical structural example. Its purpose in the paper is not to show off a difficult fit but to walk the reader through the full workflow on a panel they already know.

## Section 4.2. Worked example: IRL on the same Rust bus panel

| ID | Estimator | Data | Script | Output | Emphasis |
| --- | --- | --- | --- | --- | --- |
| L3 | MCE-IRL | Rust bus | `code_snippets/listing3_mce_irl_example.py` | console (full `summary()` block) as Listing 3 | The same eight-line workflow recovers a reward function rather than a structural utility. The `EstimationSummary` contract does not change. |
| F3 | MCE-IRL vs NFXP | Rust bus | `code_snippets/fig3_mce_irl_reward.py` | `figures/fig3_mce_irl_reward.pdf` | The recovered MCE-IRL reward and the NFXP utility coincide up to an additive constant. The agreement is the point of the figure. |

The two-estimator illustration on the same panel is the structural argument for the unified result object. The user does not learn a separate workflow for IRL.

## Section 4.3. Worked example: neural value approximation at scale

| ID | Estimator | Data | Script | Output | Emphasis |
| --- | --- | --- | --- | --- | --- |
| L4 | NNES | Keane-Wolpin | `code_snippets/listing4_nnes_example.py` | console (full `summary()` block) as Listing 4 | The same workflow scales to four actions and over five thousand reachable states. Asymptotic standard errors are still available because automatic differentiation produces the Hessian regardless of the value function representation. |
| F4 | NNES | Keane-Wolpin | `code_snippets/fig4_keane_wolpin_policy.py` | `figures/fig4_keane_wolpin_policy.pdf` | At age twenty the modal occupational choice maps cleanly onto the schooling-by-experience plane, matching the reduced-form patterns in the original Keane-Wolpin sample. |
| L4b | NNES (GPU) | Keane-Wolpin | `code_snippets/listing4b_nnes_gpu.py` | console timing line | The same code runs without modification on a single A100 graphics processing unit and finishes roughly five times faster than the CPU reference. |

The Keane-Wolpin example is the scalability argument. The reader sees that the unified workflow holds even when tabular value iteration is no longer feasible.

## Section 4.4. Cross-estimator benchmark on Rust bus

The cross-estimator benchmark is the centerpiece of the paper because it is the only place where all twelve production estimators appear in the same experimental frame. Each row is a single run on the same panel with the same starting values.

| ID | Estimator | Data | Script | Output | Emphasis |
| --- | --- | --- | --- | --- | --- |
| T3a | NFXP    | Rust bus | `examples/rust-bus-engine/benchmark_all_estimators.py` | row of `benchmark_results.csv` | Reference structural likelihood. Recovers ground truth to four decimal places. |
| T3b | CCP     | Rust bus | same | same | Matches NFXP at lower wall-clock cost. The CCP-NFXP gap is the practical case for two-step estimation. |
| T3c | MPEC    | Rust bus | same | same | Joint optimization matches NFXP and CCP. Demonstrates that the constraint formulation is a viable third path. |
| T3d | SEES    | Rust bus | same | same | Sieve approximation reaches the same answer with lower effective dimensionality. The minor parameter gap reveals the approximation cost. |
| T3e | NNES    | Rust bus | same | same | Neural value approximation reaches the structural answer at the cost of additional wall-clock time. |
| T3f | TD-CCP  | Rust bus | same | same | Cross-fitted neural CCP recovers the structural answer but pays the highest CPU cost in the structural family. |
| T3g | MCE-IRL | Rust bus | same | same | Maximum causal entropy IRL recovers the same parameter vector as NFXP under linear features. The numerical confirmation of the asymptotic equivalence. |
| T3h | GLADIUS | Rust bus | same | same | Q-network IRL recovers the structural answer without taking the transition matrix as input. The unique selling point of the model-free family. |
| T3i | AIRL    | Rust bus | same | same | Adversarial IRL recovers the structural answer at the cost of two orders of magnitude more wall-clock time. |
| T3j | IQ-Learn| Rust bus | same | same | Inverse soft Q-learning recovers an approximate structural answer at near-zero wall-clock time. The chi-squared objective trades likelihood maximization for speed. |
| T3k | f-IRL   | Rust bus | same | same | Distribution matching does not recover the structural answer on this panel. The negative result is informative and is reported as such. |
| T3l | BC      | Rust bus | same | same | The behavioral cloning baseline. Any IRL or structural estimator that does not beat its log-likelihood is not learning from the dynamic structure. |

The benchmark table aggregates these twelve runs into a single artifact.

## Section 4.5 (optional appendix). Transfer experiment for IRL estimators

The transfer test is required for every IRL estimator under `BENCHMARK_PROTOCOL.md`. The results appear in an appendix table rather than in the main text because the headline benchmark already tells the story.

| ID | Estimator | Data | Script | Output | Emphasis |
| --- | --- | --- | --- | --- | --- |
| TT1 | MCE-IRL | Rust bus, perturbed transitions | `code_snippets/transfer_mce_irl.py` | row of `transfer_results.csv` | The recovered reward generalizes to a perturbed mileage transition matrix. Percent transfer value is within two points of percent optimal value. |
| TT2 | GLADIUS | Rust bus, perturbed transitions | `code_snippets/transfer_gladius.py` | same | Same transfer test for the model-free Q-network IRL estimator. The reward recovered without the transition matrix still transfers. |
| TT3 | AIRL    | Rust bus, perturbed transitions | `code_snippets/transfer_airl.py` | same | The disentangled reward parameterization in AIRL is designed to transfer. The test verifies the design works on this panel. |
| TT4 | IQ-Learn| Rust bus, perturbed transitions | `code_snippets/transfer_iq_learn.py` | same | The implicit reward recovered from the soft Q-function transfers within five percentage points of the structural reference. |
| TT5 | f-IRL   | Rust bus, perturbed transitions | `code_snippets/transfer_f_irl.py` | same | Negative result. Distribution matching does not transfer on this panel because the reward is degenerate. |

## Section 5. Pre-estimation diagnostics

The four pre-estimation diagnostics required by `BENCHMARK_PROTOCOL.md` run once per dataset, not once per estimator. They appear in a one-line summary at the top of every results CSV and in a single appendix table in the paper.

| ID | Dataset | Script | Output | Emphasis |
| --- | --- | --- | --- | --- |
| D1 | Rust bus | `code_snippets/diagnostics_rust_bus.py` | row of `diagnostics.csv` | Feature matrix has full rank 2, condition number 1.4e3, state coverage 1.0, single-action state count 0. The panel passes all four checks. |
| D2 | Keane-Wolpin | `code_snippets/diagnostics_keane_wolpin.py` | row of `diagnostics.csv` | Feature matrix has full rank 6, condition number 4.7e2, state coverage 0.83, single-action state count 19. The panel passes with a state-coverage warning that is reported in the caption. |

## Section 6. Summary plot

| ID | Content | Script | Output | Emphasis |
| --- | --- | --- | --- | --- |
| F5 | Time-vs-accuracy frontier | `code_snippets/fig5_time_accuracy.py` | `figures/fig5_time_accuracy.pdf` | Optional summary figure. Scatter of cosine similarity to ground truth versus log wall-clock time across all twelve estimators on Rust bus. The Pareto frontier is NFXP, CCP, GLADIUS, and IQ-Learn. |

The figure is optional because the benchmark table already carries both axes. We include it only if the page count needs filling.

## Tables

| ID | Content | Script | Output | Emphasis |
| --- | --- | --- | --- | --- |
| T1 | Estimator taxonomy across structural-assumption columns | `code_snippets/table1_estimator_taxonomy.py` | `figures/table1.tex` | Twelve rows derived from `econirl.estimation.categories.ESTIMATOR_REGISTRY`. The table is regenerated whenever a new estimator is added, never edited by hand. |
| T2 | Comparison to imitation, pyblp, respy, mlogit | `code_snippets/table2_library_comparison.py` | `figures/table2.tex` | Hand-curated. The script encodes our judgments about the four competing libraries and emits the LaTeX. The hand curation is documented in the script's docstring. |
| T3 | Cross-estimator benchmark on Rust bus | `examples/rust-bus-engine/benchmark_all_estimators.py` | `figures/table3.tex` (rendered from `benchmark_results.csv`) | The headline cross-estimator table. Generated end to end without hand editing. |

## Run order

The scripts can be run in any order, but the recommended order minimizes total wall-clock time on a single machine.

1. `code_snippets/diagnostics_rust_bus.py` and `code_snippets/diagnostics_keane_wolpin.py` first. They are cheap and they validate the datasets.
2. `code_snippets/teaser_nfxp_rust.py`, then `listing2_*`, `listing3_*`, `listing4_*`. Each runs a single estimator on a single dataset.
3. `code_snippets/fig1_rust_bus_ccp.py` through `fig4_keane_wolpin_policy.py`. The figures depend on the listings only through the conventions of the package, not through any persisted artifacts.
4. `examples/rust-bus-engine/benchmark_all_estimators.py`. The benchmark dominates the wall-clock budget at roughly forty minutes on a single CPU because of the AIRL row.
5. The five transfer scripts, in any order.
6. `code_snippets/fig5_time_accuracy.py` last. It consumes `benchmark_results.csv` and the transfer CSV.

A single shell script `run_all.sh` in this directory chains the eleven steps in order. The script writes a manifest of every output to `manifest.csv` so the LaTeX build can verify that nothing was skipped.

## Acceptance test

The paper is rebuilt by running `run_all.sh` followed by `pdflatex; bibtex; pdflatex; pdflatex`. The acceptance test is that every entry in this matrix has produced its named output file by the end of `run_all.sh`. A single missing output blocks the LaTeX build. There are no manual fix-ups between scripts.

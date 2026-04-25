# Replication map

Every numbered figure and table in the paper is produced by exactly one script. To regenerate the paper end to end, run each script below from the repository root.

| Element | Script | Output |
| --- | --- | --- |
| Listing 1 (teaser) | `code_snippets/teaser_nfxp_rust.py` | console output, copied verbatim into Section 1 |
| Table 1 (estimator taxonomy) | `code_snippets/table1_estimator_taxonomy.py` | `figures/table1.tex` |
| Table 2 (library comparison) | `code_snippets/table2_library_comparison.py` | `figures/table2.tex` |
| Figure 1 (Rust bus CCPs) | `code_snippets/fig1_rust_bus_ccp.py` | `figures/fig1_rust_bus_ccp.pdf` |
| Figure 2 (Rust bus value function) | `code_snippets/fig2_rust_bus_value.py` | `figures/fig2_rust_bus_value.pdf` |
| Listing 2 (NFXP example) | `code_snippets/listing2_nfxp_example.py` | console output, copied verbatim into Section 4.1 |
| Figure 3 (MCE-IRL recovered reward) | `code_snippets/fig3_mce_irl_reward.py` | `figures/fig3_mce_irl_reward.pdf` |
| Listing 3 (MCE-IRL example) | `code_snippets/listing3_mce_irl_example.py` | console output, copied verbatim into Section 4.2 |
| Figure 4 (Keane-Wolpin policy) | `code_snippets/fig4_keane_wolpin_policy.py` | `figures/fig4_keane_wolpin_policy.pdf` |
| Listing 4 (GLADIUS example) | `code_snippets/listing4_gladius_example.py` | console output, copied verbatim into Section 4.3 |
| Table 3 (cross-estimator benchmark) | `code_snippets/table3_benchmark_all.py` | `figures/table3.tex` |

All scripts are deterministic given a fixed random seed of 42 unless noted. Wall-clock numbers in Table 3 were measured on a single machine documented in Section 5.

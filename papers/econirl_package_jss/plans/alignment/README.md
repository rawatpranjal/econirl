# Per-estimator paper alignment audits

Each file in this directory audits one estimator's source code against
the formulas in its source paper. The format is fixed across files
so the JSS paper's capability table (Section 3, Table 2) can be
generated from them mechanically.

The audits cover fourteen estimators in the package: BC, NFXP, CCP,
MPEC, MCE-IRL, MCE-IRL Deep, AIRL, AIRL-Het, f-IRL, IQ-Learn,
GLADIUS, NNES, SEES, TD-CCP. GAIL is not in scope because the
package does not ship a standalone GAIL implementation; the
adversarial module ships AIRL and AIRL-Het.

## Template

```
## Estimator: <name>
## Paper(s): <citation>, located at <path or URL>
## Code: src/econirl/estimation/<file>.py

### Loss / objective
- Paper formula (with eq. number):
- Code implementation (file:line):
- Match: yes / no / approximate (explain)

### Gradient
- Paper formula:
- Code implementation:
- Match:

### Bellman / inner loop
- Paper algorithm:
- Code algorithm:
- Match:

### Identification assumptions
- Paper conditions:
- Code enforcement:
- Match:

### Hyperparameter defaults vs paper defaults
- Inner iterations / NPL steps / network width / learning rate:
- Match:

### Findings / fixes applied (commit refs):
```

## Status table (kept in sync with VALIDATION_LOG.md)

| Estimator     | Paper-side audit | Code fix needed | VALIDATION_LOG |
| ------------- | :---: | :---: | --- |
| BC            | done | no  | Pass |
| NFXP          | done | no  | Pass (already validated) |
| CCP           | done | no  | Pass with caveat (NPL>=10) |
| MPEC          | done | no  | Pass (theory only; verify on RunPod) |
| MCE-IRL       | done | yes | Fail (wrapper default unidentified) |
| MCE-IRL Deep  | done | no  | Pending (needs Tier 4 ss-neural-r) |
| AIRL          | done | no  | Pending (needs Tier 4 ss-neural-r) |
| AIRL-Het      | done | no  | Pending (needs Tier 3c) |
| f-IRL         | done | no  | Pending (needs Tier 4 ss-spine) |
| IQ-Learn      | done | no  | Pending (needs Tier 4 ss-spine) |
| GLADIUS       | done | no  | Pass with caveat (structural bias on tabular) |
| NNES          | done | yes | Fail (known-truth diagnostic) |
| SEES          | done | no  | Pending (needs Tier 4 ss-spine) |
| TD-CCP        | done | no  | Pass with caveat (cross-fitting variance fragile at high beta) |

The `Code fix needed` column flags audits where the paper formula
diverges from the code in a way that requires a code change. Those
fixes are applied in the same pass as the audit; the commit ref is
recorded in the relevant audit file's "Findings / fixes applied"
section. When a fix is applied, the corresponding ss-* cell is added
to `CLOUD_VERIFICATION_QUEUE.md` for dispatch verification.

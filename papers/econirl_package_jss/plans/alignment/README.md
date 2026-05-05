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

## Known-truth validation rule

TD-CCP is the template for the remaining estimator migrations.
Before a success claim enters RTD or the primer PDFs, the audit must
state the estimator's paper-identified target and then test that
target against a known-truth DGP. Flexible state representations are
allowed when the paper identifies a finite structural object through
known features. Raw neural rewards, unobserved reward bases, or other
objects outside the paper's identification result are diagnostic
stress tests, not pass/fail claims about the estimator.

Each migrated estimator should therefore separate:

- the paper-faithful hard case that can be gated;
- any out-of-scope diagnostic that is useful but not claimable;
- reward, policy, value, Q, and counterfactual gates against solver
  truth;
- documentation updates, which happen only after the gated artifact
  passes.

## Status table (kept in sync with VALIDATION_LOG.md)

| Estimator     | Paper-side audit | Code fix needed | VALIDATION_LOG |
| ------------- | :---: | :---: | --- |
| BC            | done | no  | Pass |
| NFXP          | done | no  | Pass (already validated) |
| CCP           | done | no  | Pass with caveat (NPL>=10) |
| MPEC          | done | no  | Pass (theory only; verify on RunPod) |
| MCE-IRL       | done | no  | Pass (low-level root feature matching; wrapper default fixed) |
| MCE-IRL Deep  | done | no  | Pass (projected state-reward path) |
| AIRL          | done | no  | Pending (needs Tier 4 ss-neural-r) |
| AIRL-Het      | done | no  | Pending (needs Tier 3c) |
| f-IRL         | done | no  | Pending (needs Tier 4 ss-spine) |
| IQ-Learn      | done | no  | Pending (needs Tier 4 ss-spine) |
| GLADIUS       | done | no  | Pass with caveat (structural bias on tabular) |
| NNES          | done | no  | Pass (low + high known-truth gates) |
| SEES          | done | no  | Pass with caveat (optimizer flag false, structural gates pass) |
| TD-CCP        | done | no  | Pass (finite-theta neural-feature hard case; raw neural reward diagnostic fails) |

The `Code fix needed` column flags audits where the paper formula
diverges from the code in a way that requires a code change. Those
fixes are applied in the same pass as the audit; the commit ref is
recorded in the relevant audit file's "Findings / fixes applied"
section. When a fix is applied, the corresponding ss-* cell is added
to `CLOUD_VERIFICATION_QUEUE.md` for dispatch verification.

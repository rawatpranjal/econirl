# Code Review: Task 3 - Numerical Hessian for Inference

## Review Scope
- **Implementation:** `_numerical_hessian` method in `/src/econirl/estimation/mce_irl.py` (lines 521-633)
- **Test Suite:** `TestMCEIRLInference` class in `/tests/test_mce_irl_core.py` (lines 583-632)

---

## STRENGTHS

### 1. Numerical Stability Measures
- **Adaptive step sizing** (lines 580-585): Uses parameter-dependent step sizes (`h_i = max(eps, abs(params[i]) * eps)`) to accommodate both small and large parameter values. This is a sophisticated approach that avoids scale-dependent issues.
- **Central difference formula** (line 589): Uses the symmetric 3-point formula for diagonal elements, which has O(h²) error and is more accurate than forward differences.
- **4-point mixed partial formula** (line 619): The off-diagonal computation uses a proper 4-point stencil:
  ```
  f_ij = (f_pp - f_pm - f_mp + f_mm) / (4*h_i*h_j)
  ```
  This correctly computes mixed partials from 4 evaluations.
- **Symmetry enforcement** (lines 620-621): The Hessian is forced to be symmetric by copying upper triangle to lower triangle, which is mathematically correct.

### 2. Negative Semi-Definite Projection
- **Eigendecomposition check** (lines 625-631): The code correctly checks if the Hessian is negative semi-definite (required at a maximum) and projects it if needed.
- **Proper clamping** (line 630): Uses `torch.clamp(eigenvalues, max=-1e-8)` to ensure negative definiteness with a numerical tolerance of 1e-8.
- **Reconstruction** (line 631): Properly reconstructs the Hessian using `U @ Λ @ U^T` where U are eigenvectors and Λ are clamped eigenvalues.

### 3. Code Clarity
- **Comprehensive docstring** (lines 531-560): Clearly explains the algorithm, parameters, and guarantees about the output.
- **Inline comments** (lines 589, 602, 623-624): Key computational steps are explained.
- **Variable naming**: Clear variable names (`h_i`, `ll_plus`, `p_pp`) make the logic traceable.
- **Closure function** (lines 565-576): Encapsulates log-likelihood computation cleanly, reducing parameter passing.

### 4. Test Quality
- **End-to-end integration** (lines 586-613): The `fitted_result` fixture trains an actual MCE IRL estimator with `se_method="hessian"`, testing the full pipeline.
- **Multiple validation angles**:
  - `test_standard_errors_computed` verifies that SEs are computed and finite
  - `test_confidence_interval_valid` verifies statistical properties of the estimates
- **Real data usage**: Tests use actual panel data (`synthetic_panel`) rather than mocks.

---

## ISSUES (Critical)

### 1. Missing Test for Negative Semi-Definiteness
**Severity:** HIGH
**Location:** `test_mce_irl_core.py`

The current tests do NOT verify that the returned Hessian is actually negative semi-definite. This is critical because:
- The method explicitly guarantees negative semi-definiteness in its docstring
- This guarantee is essential for the asymptotic SEs computation (uses `-H` in inversion)
- A positive semi-definite Hessian would produce inverted variance estimates

**Evidence:**
```python
# No test for this guarantee exists:
# eigenvalues, _ = torch.linalg.eigh(hessian)
# assert (eigenvalues <= 1e-8).all()
```

**Recommendation:**
Add explicit test:
```python
def test_hessian_negative_semidefinite(self, fitted_result):
    """Hessian must be negative semi-definite for valid inference."""
    hessian = fitted_result.hessian
    if hessian is not None:
        eigenvalues = torch.linalg.eigvalsh(hessian)
        assert (eigenvalues <= 1e-8).all(), \
            f"Hessian has positive eigenvalues: {eigenvalues[eigenvalues > 1e-8]}"
```

### 2. No Test for Hessian Accuracy
**Severity:** HIGH
**Location:** `test_mce_irl_core.py`

The tests verify that standard errors are computed and finite, but do NOT validate that the Hessian itself is accurate. This is problematic because:
- Numerical differentiation introduces truncation error
- Poor Hessian accuracy directly impacts inference quality
- No comparison against analytical Hessian (if available) or perturbed estimates

**Recommendation:**
Add finite difference validation:
```python
def test_hessian_finite_difference_convergence(self, fitted_result):
    """Validate Hessian accuracy with finer step sizes."""
    # Recompute with smaller eps and verify consistency
    hessian1 = self._numerical_hessian(..., eps=1e-3)
    hessian2 = self._numerical_hessian(..., eps=1e-4)

    # Expect some consistency as eps decreases
    diff = torch.norm(hessian1 - hessian2).item()
    assert diff < 0.1 * torch.norm(hessian1).item()
```

### 3. Step Size Calculation May Be Suboptimal
**Severity:** MEDIUM-HIGH
**Location:** `mce_irl.py`, lines 582, 585

The adaptive step sizing uses:
```python
h_i = max(eps, abs(params[i]) * eps)
```

This has an issue: **when `abs(params[i])` is much larger than 1, the step becomes very large**, potentially leading to:
- Truncation error from the nonlinearity of the log-likelihood
- Numerical overflow in intermediate computations
- Loss of accuracy for small derivatives

**Example problem:**
- If a parameter is 100 and `eps=1e-3`, then `h_i = 0.1` (10% of parameter value)
- This is often TOO LARGE for accurate 2nd derivative estimation

**Better approach:** Use scale-relative but bounded step sizing:
```python
h_i = max(eps, min(0.01 * abs(params[i]), eps * 100))
```

This keeps the step size in a more reasonable range while adapting to parameter scale.

### 4. No Handling of Flat or Nearly-Flat Directions
**Severity:** MEDIUM
**Location:** `mce_irl.py`, lines 625-631

When the Hessian has near-zero eigenvalues (flat directions), the projection code sets them to `-1e-8`, which could be:
- Too aggressive (loses information about weak curvature)
- Arbitrary (the 1e-8 threshold has no principled justification)

**Evidence:**
```python
eigenvalues_clamped = torch.clamp(eigenvalues, max=-1e-8)
```

**Risk:** If the true Hessian has near-singular directions, the artificial negative definiteness can create spurious inference.

**Recommendation:** Add diagnostics:
```python
def _numerical_hessian(...):
    # ... computation ...

    # Project to negative semi-definite
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)

    # Log diagnostics
    n_positive = (eigenvalues > 1e-8).sum()
    if n_positive > 0:
        self._log(f"Warning: {n_positive} positive eigenvalues found in Hessian")
        self._log(f"Largest positive eigenvalue: {eigenvalues[eigenvalues > 1e-8].max().item():.6e}")

        # More conservative clamping
        eigenvalues_clamped = torch.clamp(eigenvalues, max=-torch.abs(eigenvalues).min() * 0.1)
```

### 5. Missing Documentation on Step Size Choice
**Severity:** MEDIUM
**Location:** `mce_irl.py`, lines 529, 553

The docstring mentions:
```
eps : float
    Step size for finite differences. Default 1e-3 for stability.
```

But:
- Does NOT justify why 1e-3 is "stable"
- Does NOT explain the scale-dependent behavior (line 582)
- Does NOT provide guidance on when to adjust eps

**Recommendation:** Expand docstring to:
```
eps : float, default 1e-3
    Nominal step size for finite differences. The actual step size h is computed
    adaptively as h = max(eps, |param| * eps), ensuring adequate scaling for
    both small and large parameter values. Increase if Hessian elements vary
    wildly; decrease if step size estimates appear unstable. Note: Very small
    eps (< 1e-4) may lead to cancellation error in finite differences.
```

---

## ISSUES (Minor)

### 1. Inefficient Matrix Cloning in Loop
**Severity:** LOW
**Location:** `mce_irl.py`, lines 590-617

For each Hessian element computation, the code clones the entire parameter vector multiple times:
```python
p_plus = params.clone()
p_plus[i] += h_i
p_minus = params.clone()
p_minus[i] -= h_i
# ... 4 more clones for off-diagonals
```

For large parameter vectors, this creates unnecessary memory overhead. However, given typical parameter counts (1-100), this is acceptable.

**Impact:** Negligible for most applications, but could matter for very high-dimensional problems.

### 2. Off-Diagonal Computation Loop Could Be Optimized
**Severity:** LOW
**Location:** `mce_irl.py`, lines 584-621

The nested loop `for j in range(i, n_params)` recomputes each off-diagonal pair. For high-dimensional problems, could cache function evaluations.

**Current:** For n_params = 10, computes ~55 distinct function evaluations + 10 diagonals = ~65 evals
**Potential:** Could reduce to ~45 with smarter caching

**Recommendation:** Use explicit caching grid:
```python
# Precompute all perturbations
evals = {}
for i in range(n_params):
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            p_pert = params.clone()
            if di != 0:
                p_pert[i] += di * h_i
            if dj != 0:
                p_pert[j] += dj * h_j
            evals[(di, dj)] = ll_at(p_pert)
```

However, this is premature optimization for most use cases.

### 3. No Logging of Hessian Diagnostics
**Severity:** LOW
**Location:** `mce_irl.py`, lines 521-633

The method logs a warning if projection is needed (line 629), but does NOT log:
- Condition number of original Hessian
- Number and magnitude of eigenvalues corrected
- Total number of function evaluations
- Elapsed time for Hessian computation

**Recommendation:** Add verbose logging:
```python
if self.config.verbose:
    self._log(f"Hessian computation:")
    self._log(f"  Function evaluations: {n_evals}")
    self._log(f"  Eigenvalue range: [{eigenvalues.min():.3e}, {eigenvalues.max():.3e}]")
    if (eigenvalues > 0).any():
        self._log(f"  Eigenvalues fixed: {(eigenvalues > 0).sum()}")
```

### 4. Type Consistency
**Severity:** VERY LOW
**Location:** `mce_irl.py`, line 563

The Hessian is initialized with `dtype=params.dtype`, which is correct. However, if `params` is float64 and intermediate computations cast to float32, this could cause silent precision loss.

**Mitigation:** Already handled by PyTorch's automatic type promotion. Not a real issue.

### 5. No Parameter Validation
**Severity:** LOW
**Location:** `mce_irl.py`, lines 521-530

The method accepts `eps` as a parameter but does NOT validate:
```python
if eps <= 0:
    raise ValueError(f"eps must be positive, got {eps}")
```

**Impact:** Invalid eps silently fails in numerical computation.

---

## TEST QUALITY ASSESSMENT

### Positive Aspects:
1. **Tests use real fitted models** - not artificial Hessians
2. **Integration test** - tests the full pipeline (fit + SE computation)
3. **Multiple assertions** - checks for NaN, finiteness, CI validity
4. **Proper fixtures** - uses shared `synthetic_panel` for reproducibility

### Gaps:
1. **No direct Hessian validation** - Tests never examine the actual Hessian matrix
2. **No sensitivity analysis** - Doesn't test behavior with different eps values
3. **No edge cases** - Doesn't test with:
   - Very small parameters (near zero)
   - Very large parameters (e.g., 1e6)
   - Nearly-flat log-likelihood (nearly singular Hessian)
   - Well-conditioned vs ill-conditioned problems
4. **No accuracy benchmarking** - Doesn't compare numerical Hessian against:
   - Analytic Hessian (if available)
   - Finite difference with different step sizes
   - Convergence as step size → 0

### Minimum Test Coverage Needed:
```python
def test_hessian_negative_semidefinite(self):
    """Hessian must satisfy H ≤ 0 (all eigenvalues ≤ 0)."""

def test_hessian_shape_and_symmetry(self):
    """Hessian should be square and symmetric."""

def test_finite_difference_convergence(self):
    """Finer step sizes should give similar Hessians."""
```

---

## VERDICT

### **NEEDS FIXES (Approval withheld)**

### Summary:
The implementation demonstrates **good understanding of numerical Hessian computation** with appropriate techniques (central differences, adaptive step sizing, negative semi-definite projection). The code is **well-documented and reasonably efficient**.

However, **critical gaps in testing** prevent approval:
1. No verification that returned Hessian is actually negative semi-definite
2. No validation of Hessian accuracy
3. Potential numerical issues with step sizing for large parameters
4. Insufficient documentation of step size selection

The method will likely work for most cases, but **lacks the validation infrastructure** needed for a robust inference tool where the Hessian directly impacts statistical inference quality.

### Required Fixes (Before Merge):
1. **Add test for negative semi-definiteness** (critical)
2. **Add finite difference convergence test** (critical)
3. **Document step size selection** (high priority)
4. **Improve step size bounds** (medium priority)

### Nice-to-Have Improvements (Post-Merge):
1. Add Hessian diagnostic logging
2. Add parameter validation (eps > 0)
3. Optimize caching for high-dimensional problems
4. Add edge case tests (very small/large parameters)

---

## DETAILED RECOMMENDATIONS

### Priority 1: Add Critical Tests
Add to `test_mce_irl_core.py::TestMCEIRLInference`:

```python
def test_hessian_negative_semidefinite(self, fitted_result):
    """Hessian must be negative semi-definite (for valid MLE inference)."""
    hessian = fitted_result.hessian
    if hessian is not None:
        eigenvalues = torch.linalg.eigvalsh(hessian)
        # Allow small positive perturbations due to numerical error
        assert torch.all(eigenvalues <= 1e-6), \
            f"Hessian has {(eigenvalues > 1e-6).sum()} positive eigenvalues. " \
            f"Max eigenvalue: {eigenvalues.max().item():.2e}"

def test_hessian_symmetric(self, fitted_result):
    """Hessian must be symmetric."""
    hessian = fitted_result.hessian
    if hessian is not None:
        diff = torch.norm(hessian - hessian.T).item()
        assert diff < 1e-6 * torch.norm(hessian).item(), \
            f"Hessian not symmetric: {diff:.2e}"

def test_hessian_shape(self, fitted_result):
    """Hessian must be square with dimension = n_params."""
    hessian = fitted_result.hessian
    n_params = len(fitted_result.parameters)
    if hessian is not None:
        assert hessian.shape == (n_params, n_params), \
            f"Hessian shape {hessian.shape} != ({n_params}, {n_params})"
```

### Priority 2: Document Step Size Selection
Update `mce_irl.py` docstring:

```python
eps : float, default 1e-3
    Nominal step size for central differences. The actual step size h_i for
    parameter i is computed adaptively as:
        h_i = max(eps, |theta_i| * eps)

    This ensures adequate scaling: small parameters use eps, large parameters
    scale proportionally. For a parameter θ=0.5, h=1e-3; for θ=100, h=0.1.

    Guidance on adjustment:
    - Decrease eps (→ 1e-4) if estimated Hessian appears unstable
    - Increase eps (→ 1e-2) if step size estimates are too small relative
      to parameter variation scales
    - Note: eps < 1e-4 risks cancellation error in finite differences

    Default 1e-3 is conservative for typical economic parameters (ranging
    from 0.01 to 100).
```

### Priority 3: Improve Step Size Bounds
Modify `mce_irl.py` lines 582-585:

```python
# Adaptive step: larger for larger params, but bounded
# Bounds: [eps, eps * 100] gives range [1e-3, 0.1] for default eps
h_i = max(eps, min(abs(params[i].item()) * eps, eps * 100))

for j in range(i, n_params):
    h_j = max(eps, min(abs(params[j].item()) * eps, eps * 100))
```

This prevents step sizes from becoming too large for large parameters.

---

## Files Affected
- `/src/econirl/estimation/mce_irl.py` - Implementation
- `/tests/test_mce_irl_core.py` - Test suite

## Estimated Effort to Fix
- Add tests: 30 min
- Update documentation: 15 min
- Improve step sizing: 10 min
- **Total: ~1 hour**


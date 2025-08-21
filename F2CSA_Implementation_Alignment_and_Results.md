## F2CSA implementation: alignment with F2CSA.tex and deterministic.py, changes, and results

### Overview
This document summarizes the current F2CSA implementation, what changed during this work session, how it aligns with the paper (F2CSA.tex) and the deterministic reference (deterministic.py), and the experimental results (stationarity gap curves and tuning sweeps). It also records practical notes such as encoding fixes and reproducibility.

---

### What we changed (code structure and behavior)
- Moved all problem formulation and CVXPy build into the Problem class
  - New helper: Problem.solve_f2csa_penalty_lagrangian(...) builds and solves the penalized Lagrangian exactly as in Eq. (396–401) with active constraints h(x,y) ≤ 0.
- Paper-accurate gating
  - Implemented piecewise gating ρ(x) = σ_h(h(x,y*))·σ_λ(λ̃) per definitions in F2CSA.tex with tunables tau_coeff and eps_lambda.
- Parameterized inner problems via cp.Parameter
  - Injected mini-batch style stochasticity (objectives only) for both y*(x) and ỹ(x) solves; constraints h(x,y) stay deterministic.
  - Controlled by inner_batch_size in F2CSA; noise magnitude scales as noise_std / sqrt(batch_size).
- Simplified noise model to default Gaussian
  - upper_objective/lower_objective now only support add_noise flag with Gaussian noise (removed adversarial/custom variants as per request).
- Added non-smooth Algorithm 2 (Goldstein-style) outer update
  - New method: F2CSA.optimize_nonsmooth(...)
  - Uses z = x_base + s·Δ, updates Δ ← clip_D(Δ − η_t g(z)), and advances x_base ← x_base + Δ_t.
  - Added exponential and polynomial step-size schedules.
  - Fixed base-point update bug to use Δ_t (not Δ_{t−1}).
- Redefined “gap” as stationarity gap
  - gap := ||∇F(x)|| (stochastic proxy via the hypergradient oracle); always nonnegative.
  - Objective F(x) continues to be logged separately and can be negative due to linear terms.
- Plotting and experiments
  - Scripts to run 10k or 2k iterations and plot linear/log curves.
  - A parallel harness that tags outputs to avoid overwrites and writes markdown summaries.
- Console encoding fixes (Windows)
  - Replaced Unicode symbols in prints (e.g., Greek letters, checkmark, ≤) with ASCII to prevent UnicodeEncodeError on cp125x consoles.

Key code pointers:
- Penalty Lagrangian inner solve:
<augment_code_snippet path="problem.py" mode="EXCERPT">
````python
objective = cp.Minimize(
    f_cp + alpha1 * (g_cp - g_opt_cp + lam_np.T @ h_cp)
    + 0.5 * alpha2 * cp.sum(cp.multiply(rho_np, cp.square(h_cp)))
)
````
</augment_code_snippet>

- Calling the inner solve from F2CSA:
<augment_code_snippet path="algorithm.py" mode="EXCERPT">
````python
y_tilde = self.problem.solve_f2csa_penalty_lagrangian(
    x=x, y_star=y_star, lam_tilde=lam_tilde, rho=rho,
    alpha1=self.alpha1, alpha2=self.alpha2, batch_size=self.inner_batch_size,
)
````
</augment_code_snippet>

- Non-smooth Algorithm 2 update and base-point fix:
<augment_code_snippet path="algorithm.py" mode="EXCERPT">
````python
z = (x_base + s * delta_vec).detach()
g = self.compute_hypergradient(z)
...
# advance with updated Δ_t
y_base = (x_base + delta_vec).detach()
````
</augment_code_snippet>

- Stationarity gap usage in logs:
<augment_code_snippet path="algorithm.py" mode="EXCERPT">
````python
bilevel_obj = float(self.problem.upper_objective(x_cur, y_star, add_noise=False))
gap = g_norm  # ||∇F|| proxy
````
</augment_code_snippet>

---

### Alignment with F2CSA.tex
- Inner objective (Eq. 396–401) is implemented exactly:
  - L(x,y) = f(x,y) + α1 [ g(x,y) − g(x,y*) + λ̃ᵀ h(x,y) ] + (α2/2) Σ ρ_i h_i(x,y)^2
  - Constraints enforced: h(x,y) ≤ 0 during the inner penalized solve.
- Parameter scaling and δ relationship:
  - α1 = α^(−2), α2 = α^(−4), δ = α^3 are used consistently.
- Gating ρ:
  - Implemented the paper’s piecewise σ_h and σ_λ gating. τ is proportional to δ via tau_coeff; λ threshold eps_lambda exposed as a parameter.
- Stochasticity in practice:
  - Objectives only via cp.Parameter; constraints deterministic.
- Non-smooth outer method:
  - Added z = x + s·Δ evaluation and clipped Δ updates in the code per Algorithm 2.

---

### Relationship to deterministic.py
- deterministic.py’s “loss” is the upper-level objective f(x,y) (can be negative because of linear terms). We kept “objective” for that and separated “gap” as ||∇F||.
- FFO_COMPLEX in deterministic.py uses a Goldstein-style update with z = x + sΔ and Δ clipping. We added the analogous non-smooth path in F2CSA with schedules and corrected base-point advancement.
- deterministic.py active-set vs ρ-gating:
  - deterministic.py triggers a quadratic penalty on active h rows detected via thresholds on |h| and γ.
  - F2CSA uses the smoother paper-accurate gating ρ = σ_h·σ_λ.

Reference code in deterministic.py:
<augment_code_snippet path="deterministic.py" mode="EXCERPT">
````python
# Loss definition in deterministic version
loss = f(xx, y_opt).item()
````
</augment_code_snippet>

---

### Experiments and results (dim=100, 2,000 iterations, non-smooth Alg. 2)
- Common: constraints=5, inner_batch_size default 32 unless noted, schedule exponential with eta0=0.002.
- Gap is always stationarity gap ||∇F|| (stochastic proxy). Objective F(x) is logged separately.

Results (from per-run summaries):
- BASELINE (N_g=30, alpha=0.35, D_clip≈alpha^3/6)
  - final_gap 3.3148e-02, best_gap 7.9474e-03, time 534.7 s
- BS64 (N_g=30, BS=64)
  - final_gap 3.3146e-02, best_gap 7.9460e-03, time 536.1 s
- BS128 (N_g=30, BS=128)
  - final_gap 3.3144e-02, best_gap 7.9450e-03, time 547.7 s
- NG50 (N_g=50)
  - final_gap 3.4005e-02, best_gap 3.6355e-03, time 624.8 s
- NG100 (N_g=100)
  - final_gap 3.3353e-02, best_gap 2.5659e-03, time 845.6 s
- ALPHA30 (alpha=0.30, D_clip=alpha^3/6)
  - final_gap 3.2845e-02, best_gap 1.9325e-03, time 547.2 s
- ALPHA25 (alpha=0.25, D_clip=alpha^3/6)
  - final_gap 2.7346e-02, best_gap 2.3255e-03, time 565.6 s
- GAMMA3E4 (exp_gamma=3e-4)
  - final_gap 3.3671e-02, best_gap 6.5089e-03, time 520.4 s
- GAMMA5E4 (exp_gamma=5e-4)
  - final_gap 3.2413e-02, best_gap 4.2780e-03, time 420.4 s

Observations:
- Lower alpha (reducing bias) has the strongest effect on reducing the stationarity gap floor, especially best_gap.
- Increasing N_g reduces best_gap but increases runtime; final_gap improvements at 2k iters are modest.
- Larger inner_batch_size had little effect compared to N_g and alpha in this setup.
- Slightly larger exp decay (exp_gamma=5e-4 vs 2e-4) improved final_gap modestly.

Artifacts (examples):
- ns_gap_history_dim100_2000iters_ALPHA25.csv
- ns_gap_linear_2000_dim100_ALPHA25.png
- ns_gap_history_dim100_2000iters_ALPHA25_summary.json

---

### Notes on 10k smooth/Adam run (dim=100)
- The original 10k experiment with the smooth update and early stopping converged around 130 iterations.
- With stationarity gap adopted, the “gap” reported in plots and CSVs is ||∇F||, and “objective” F(x) is separate (can be negative). The 10k linear/log plotting scripts can show both curves.

---

### Practical fixes: encoding and reproducibility
- Removed non-ASCII characters from prints in problem.py and algorithm.py to avoid UnicodeEncodeError with Windows console code pages.
- Added tagging (ST_TAG) to output filenames to avoid overwrites in parallel or batch runs.
- Wrote a harness (run_parallel_tests.py) that runs a matrix of configurations and writes timestamped markdown reports (test_results_YYYYMMDD_HHMMSS.md).

---

### Immediate next steps (optional)
- Combine best settings for non-smooth runs: alpha=0.30, N_g=50, inner_batch_size=64, exp_gamma=3e-4, D_clip=alpha^3/6 and run 5,000 iterations; plot objective and stationarity gap (linear).
- Add EMA and CI to reported stationarity gaps to visualize stochastic floors.
- If we want convex certificates, add a mode to compute a primal–dual gap when the upper problem has usable dual information or linear oracle (e.g., Frank–Wolfe for convex sets).

---

### Appendix: small code references
- ASCII-only init print to avoid encoding issues:
<augment_code_snippet path="algorithm.py" mode="EXCERPT">
````python
print(f"F2CSA initialized: N_g={N_g}, alpha={alpha}, alpha1={self.alpha1:.3f}, alpha2={self.alpha2:.3f}, delta={self.delta:.3e}")
````
</augment_code_snippet>

- Objective vs gap separation in history entries:
<augment_code_snippet path="algorithm.py" mode="EXCERPT">
````python
history.append({
    'iteration': t,
    'bilevel_objective': bilevel_obj,
    'gap': g_norm,
    'time': time.time() - start_time,
})
````
</augment_code_snippet>


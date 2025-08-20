## F2CSA sweep: effect of alpha and N_g on convergence plateau

- Problem: StronglyConvexBilevelProblem(dim=20, constraints=5, noise_std=1e-4)
- Sweep: alpha in [0.25, 0.35, 0.5], N_g in [30, 100], Adam lr=0.005, max_iters=1200, conv_thr=0.001

### Config: alpha=0.25, N_g=30, lr=0.005

- Outcome: converged=False, iters=1200, final_gap=3.648791e-03, best_gap=2.383489e-03
- Plateau (Δ100<1e-3): 140
- Plateau (Δ100<1e-5): 142
- Grad norm: init=8.687e-01 → final=1.138e-16 (ratio=0.000)

- [Inner] cond(H)=1.030e+00 | raw_grad=1.138e-16
- [Gating] ρ: min=2.948e-03 mean=1.356e-02 max=2.512e-02 std=1.005e-02; active(>0.9)=0
- [Components] |grad_f|=1.194e-01 |grad_pen_lin|=9.109e-03 |grad_pen_quad|=1.176e-01
- [LL] status=optimal converged=True cv=nan opt_gap=7.70e-07

- Time: 95.01s

### Config: alpha=0.25, N_g=100, lr=0.005

- Outcome: converged=False, iters=1200, final_gap=3.648791e-03, best_gap=2.383489e-03
- Plateau (Δ100<1e-3): 140
- Plateau (Δ100<1e-5): 142
- Grad norm: init=8.687e-01 → final=1.091e-16 (ratio=0.000)

- [Inner] cond(H)=1.030e+00 | raw_grad=1.091e-16
- [Gating] ρ: min=2.948e-03 mean=1.356e-02 max=2.512e-02 std=1.005e-02; active(>0.9)=0
- [Components] |grad_f|=1.194e-01 |grad_pen_lin|=9.109e-03 |grad_pen_quad|=1.176e-01
- [LL] status=optimal converged=True cv=nan opt_gap=7.70e-07

- Time: 236.20s

### Config: alpha=0.35, N_g=30, lr=0.005

- Outcome: converged=False, iters=1200, final_gap=1.105040e-02, best_gap=8.432891e-03
- Plateau (Δ100<1e-3): 139
- Plateau (Δ100<1e-5): 140
- Grad norm: init=8.616e-01 → final=1.121e-15 (ratio=0.000)

- [Inner] cond(H)=1.030e+00 | raw_grad=1.121e-15
- [Gating] ρ: min=6.711e-02 mean=9.954e-02 max=1.277e-01 std=2.898e-02; active(>0.9)=0
- [Components] |grad_f|=2.089e-01 |grad_pen_lin|=1.631e-02 |grad_pen_quad|=2.064e-01
- [LL] status=optimal converged=True cv=nan opt_gap=7.67e-07

- Time: 95.81s

### Config: alpha=0.35, N_g=100, lr=0.005

- Outcome: converged=False, iters=1200, final_gap=1.105040e-02, best_gap=8.432891e-03
- Plateau (Δ100<1e-3): 139
- Plateau (Δ100<1e-5): 140
- Grad norm: init=8.616e-01 → final=1.945e-15 (ratio=0.000)

- [Inner] cond(H)=1.030e+00 | raw_grad=1.945e-15
- [Gating] ρ: min=6.711e-02 mean=9.954e-02 max=1.277e-01 std=2.898e-02; active(>0.9)=0
- [Components] |grad_f|=2.089e-01 |grad_pen_lin|=1.631e-02 |grad_pen_quad|=2.064e-01
- [LL] status=optimal converged=True cv=nan opt_gap=7.67e-07

- Time: 217.63s

### Config: alpha=0.5, N_g=30, lr=0.005

- Outcome: converged=False, iters=1200, final_gap=2.566397e-03, best_gap=1.979134e-03
- Plateau (Δ100<1e-3): 141
- Plateau (Δ100<1e-5): 145
- Grad norm: init=8.559e-01 → final=1.109e-16 (ratio=0.000)

- [Inner] cond(H)=1.030e+00 | raw_grad=1.109e-16
- [Gating] ρ: min=1.727e-01 mean=1.900e-01 max=2.043e-01 std=1.534e-02; active(>0.9)=0
- [Components] |grad_f|=9.917e-02 |grad_pen_lin|=9.224e-03 |grad_pen_quad|=9.837e-02
- [LL] status=optimal converged=True cv=nan opt_gap=7.69e-07

- Time: 94.74s

### Config: alpha=0.5, N_g=100, lr=0.005

- Outcome: converged=False, iters=1200, final_gap=2.566397e-03, best_gap=1.979134e-03
- Plateau (Δ100<1e-3): 141
- Plateau (Δ100<1e-5): 145
- Grad norm: init=8.559e-01 → final=6.268e-17 (ratio=0.000)

- [Inner] cond(H)=1.030e+00 | raw_grad=6.268e-17
- [Gating] ρ: min=1.727e-01 mean=1.900e-01 max=2.043e-01 std=1.534e-02; active(>0.9)=0
- [Components] |grad_f|=9.917e-02 |grad_pen_lin|=9.224e-03 |grad_pen_quad|=9.837e-02
- [LL] status=optimal converged=True cv=nan opt_gap=7.69e-07

- Time: 209.40s

## Observations

- Best final gap: 2.566397e-03 at alpha=0.50, N_g=30
- Earliest plateau (Δ100<1e-3): it=139 at alpha=0.35, N_g=30
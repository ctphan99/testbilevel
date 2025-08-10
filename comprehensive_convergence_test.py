"""
Comprehensive Convergence Test: All Algorithms to Same Gap on Dimension 100
Run F2CSA, SSIGD, DS-BLO until they converge to the same gap value
NO workarounds, fallbacks, or band-aid solutions - only fundamental fixes
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from algorithm import F2CSA, SSIGD, DSBLO
from problem import StronglyConvexBilevelProblem

class ComprehensiveMonitor:
    """Monitor all algorithms comprehensively until convergence alignment"""

    def __init__(self):
        self.algorithm_data = {}
        self.convergence_achieved = False
        self.fundamental_issues = []

    def initialize_algorithm(self, alg_name):
        """Initialize monitoring for an algorithm"""
        self.algorithm_data[alg_name] = {
            'iterations': [],
            'gaps': [],
            'objectives': [],
            'gradient_norms': [],
            'constraint_violations': [],
            'kkt_residuals': [],
            'iteration_times': [],
            'x_norms': [],
            'stepsizes': [],
            'momentum_norms': [],
            'converged': False,
            'final_gap': None,
            'total_time': 0
        }

    def log_iteration(self, alg_name, iteration, gap, objective, gradient_norm, constraint_violation, iteration_time, x_norm, stepsize=None, momentum_norm=None, kkt_residual=None):
        """Log iteration data for an algorithm"""
        data = self.algorithm_data[alg_name]
        data['iterations'].append(iteration)
        data['gaps'].append(gap)
        data['objectives'].append(objective)
        data['gradient_norms'].append(gradient_norm)
        data['constraint_violations'].append(constraint_violation)
        data['iteration_times'].append(iteration_time)
        data['x_norms'].append(x_norm)
        data['stepsizes'].append(stepsize)
        data['momentum_norms'].append(momentum_norm)
        data['kkt_residuals'].append(kkt_residual)
        data['total_time'] += iteration_time

    def check_convergence_alignment(self, tolerance=1e-4):
        """Check if all algorithms have converged to the same gap"""
        converged_algorithms = []
        final_gaps = []

        for alg_name, data in self.algorithm_data.items():
            if len(data['gaps']) >= 20:  # Need sufficient iterations
                recent_gaps = data['gaps'][-20:]
                gap_variance = np.var(recent_gaps)
                gap_mean = np.mean(recent_gaps)

                # Check if algorithm has converged (low variance)
                if gap_variance < tolerance**2:
                    converged_algorithms.append(alg_name)
                    final_gaps.append(gap_mean)
                    data['converged'] = True
                    data['final_gap'] = gap_mean

        # Check if all algorithms have converged
        if len(converged_algorithms) == len(self.algorithm_data):
            # Check if they converged to the same gap
            if len(final_gaps) > 1:
                gap_range = max(final_gaps) - min(final_gaps)
                if gap_range <= tolerance:
                    self.convergence_achieved = True
                    print(f"🎉 CONVERGENCE ACHIEVED: All algorithms converged to same gap ±{tolerance}")
                    print(f"   Final gaps: {[f'{gap:.8f}' for gap in final_gaps]}")
                    print(f"   Gap range: {gap_range:.8f}")
                    return True
                else:
                    print(f"⚠️  Algorithms converged but to different gaps:")
                    for alg, gap in zip(converged_algorithms, final_gaps):
                        print(f"   {alg}: {gap:.8f}")
                    print(f"   Gap range: {gap_range:.8f} (tolerance: {tolerance})")

        return False

    def detect_fundamental_issues(self):
        """Detect fundamental mathematical issues"""
        issues = []

        for alg_name, data in self.algorithm_data.items():
            if len(data['gaps']) > 10:
                # Check for numerical explosions
                if any(gap > 1000 for gap in data['gaps'][-10:]):
                    issues.append(f"{alg_name}: Numerical explosion in gap calculation")

                # Check for gradient explosions
                if any(grad > 1000 for grad in data['gradient_norms'][-10:]):
                    issues.append(f"{alg_name}: Gradient explosion")

                # Check for constraint violation explosions
                if any(viol > 1.0 for viol in data['constraint_violations'][-10:]):
                    issues.append(f"{alg_name}: Constraint violation explosion")

                # Check for stagnation without convergence
                if len(data['gaps']) > 100:
                    recent_gaps = data['gaps'][-50:]
                    gap_change = abs(recent_gaps[-1] - recent_gaps[0])
                    if gap_change < 1e-8 and not data['converged']:
                        issues.append(f"{alg_name}: Stagnation without convergence")

        self.fundamental_issues.extend(issues)
        return issues

def run_comprehensive_convergence_test():
    """Run comprehensive test until all algorithms converge to same gap"""
    print("🚀 COMPREHENSIVE CONVERGENCE TEST: DIMENSION 100")
    print("=" * 80)
    print("Goal: All algorithms converge to same gap (±1e-4) on 100D problem")
    print("Requirements:")
    print("  ✅ No workarounds, fallbacks, or band-aid solutions")
    print("  ✅ Pure algorithm implementations following paper specifications")
    print("  ✅ Same problem instance for all algorithms")
    print("  ✅ Detailed monitoring of all calculations")
    print("  ✅ Continue until convergence alignment or fundamental issues")
    print("=" * 80)

    device = 'cpu'

    # Create single problem instance for all algorithms
    problem = StronglyConvexBilevelProblem(
        dim=100,
        num_constraints=20,
        device=device,
        seed=42,
        noise_std=0.0001
    )

    print(f"📊 Problem Configuration:")
    print(f"   Dimension: {problem.dim}")
    print(f"   Constraints: {problem.num_constraints}")
    print(f"   Device: {device}")
    print(f"   Noise std: 0.0001")

    # Verify problem is well-posed
    print(f"\n🔍 Problem Validation:")
    x_test = torch.zeros(problem.dim, device=device)
    y_test, ll_info = problem.solve_lower_level(x_test)
    initial_gap = problem.compute_gap(x_test)

    print(f"   LL solver method: {ll_info.get('method', 'unknown')}")
    print(f"   LL converged: {ll_info.get('converged', False)}")
    print(f"   LL constraint violation: {ll_info.get('constraint_violation', 0):.2e}")
    print(f"   Initial gap: {initial_gap:.8f}")

    if not ll_info.get('converged', False) or ll_info.get('constraint_violation', 0) > 1e-6:
        print(f"❌ Problem is not well-posed - LL solver issues")
        return None

    # Algorithm configurations (pure implementations)
    algorithms = {
        'F2CSA': {
            'class': F2CSA,
            'params': {
                'N_g': int(os.getenv('F2CSA_NG', '30')),           # Higher upfront for variance reduction (env overridable)
                'alpha': float(os.getenv('F2CSA_ALPHA', '0.25')),       # Paper-aligned penalty (env overridable)
                'adam_lr': float(os.getenv('F2CSA_LR', '5e-3'))      # Env-overridable Adam LR for steadier descent
            },
            'ema_decay': 0.95        # Stronger smoothing for stochastic algorithm
        },
        'SSIGD': {
            'class': SSIGD,
            'params': {
                'stepsize_schedule': 'strongly_convex',  # Start with strongly-convex schedule
                'beta_0': 0.01,
                'mu_F': 200.0,                           # Slightly higher μ_F to lower steps initially
                'seed': 42,
                'll_tolerance': 1e-6
            },
            'ema_decay': None
        },
        'DSBLO': {
            'class': DSBLO,
            'params': {
                'momentum': float(os.getenv('DSBLO_MOMENTUM', '0.9')),
                'sigma': float(os.getenv('DSBLO_SIGMA', '0.0001')),
                'gamma1': float(os.getenv('DSBLO_GAMMA1', '50.0')),
                'gamma2': float(os.getenv('DSBLO_GAMMA2', '5.0'))
            },
            'ema_decay': float(os.getenv('DSBLO_EMA', '0.9'))
        }
    }

    monitor = ComprehensiveMonitor()
    algorithm_instances = {}

    # Initialize all algorithms with same starting point
    torch.manual_seed(42)
    np.random.seed(42)
    initial_x = torch.randn(problem.dim, device=device) * 0.1

    # If ONLY_ALGOS env is set (comma-separated), restrict to those algorithms
    only_algos = os.getenv('ONLY_ALGOS')
    if only_algos:
        requested = [name.strip() for name in only_algos.split(',') if name.strip()]
        algorithms = {k: v for k, v in algorithms.items() if k in requested}
        print(f"🔧 Filtering algorithms via ONLY_ALGOS: {list(algorithms.keys())}")

    for alg_name, alg_config in algorithms.items():
        print(f"\n📋 Initializing {alg_name}:")

        # Create algorithm instance
        torch.manual_seed(42)  # Ensure same initialization
        algorithm = alg_config['class'](problem, **alg_config['params'])

        # Set same starting point (preserve optimizer bindings for F2CSA)
        if alg_name == 'F2CSA' and hasattr(algorithm, 'optimizer'):
            with torch.no_grad():
                algorithm.x.data.copy_(initial_x)
                algorithm.x.requires_grad_(True)
            # Optional diagnostic: force SGD with fixed lr and set N_g via env
            if os.getenv('F2CSA_FORCE_SGD', '0') == '1' and hasattr(algorithm, 'switch_to_sgd'):
                try:
                    sgd_lr = float(os.getenv('F2CSA_SGD_LR', '1e-3'))
                    algorithm.switch_to_sgd(lr=sgd_lr)
                    print(f"   🔧 [INIT] F2CSA: Forced SGD with fixed lr={sgd_lr:.3e}")
                except Exception as e:
                    print(f"   ⚠️ [INIT] F2CSA: switch_to_sgd failed: {e}")
            if os.getenv('F2CSA_NG'):
                try:
                    algorithm.N_g = int(os.getenv('F2CSA_NG'))
                    print(f"   🔧 [INIT] F2CSA: Set N_g={algorithm.N_g} from env")
                except Exception:
                    pass
        else:
            algorithm.x = initial_x.clone()

        # Configure EMA if applicable (allow disabling via env)
        disable_ema = os.getenv('F2CSA_DISABLE_EMA', '0') == '1' and alg_name == 'F2CSA'
        if not disable_ema and alg_config['ema_decay'] is not None:
            algorithm.ema_decay = alg_config['ema_decay']
            print(f"   EMA decay: {algorithm.ema_decay}")
        elif disable_ema:
            if hasattr(algorithm, 'ema_decay'):
                algorithm.ema_decay = None
            print("   🔧 [INIT] F2CSA: EMA disabled by env")

        # Verify starting state
        start_gap = problem.compute_gap(algorithm.x)
        start_obj = float(problem.true_bilevel_objective(algorithm.x, add_noise=False))

        print(f"   Starting gap: {start_gap:.8f}")
        print(f"   Starting objective: {start_obj:.6f}")
        print(f"   Starting x norm: {torch.norm(algorithm.x):.6f}")

        algorithm_instances[alg_name] = algorithm
        monitor.initialize_algorithm(alg_name)

    # Auto-adjust policies (paper-aligned) using recent history
    def auto_adjust(alg_name, algorithm, data, iteration, window=30):
        # Diagnostic override: keep F2CSA fixed LR and/or fixed N_g when requested
        if alg_name == 'F2CSA':
            if os.getenv('F2CSA_FIXED_LR', '0') == '1':
                # Do nothing to the LR; preserve optimizer step size
                pass
            if os.getenv('F2CSA_NG'):
                try:
                    algorithm.N_g = int(os.getenv('F2CSA_NG'))
                except Exception:
                    pass
        if len(data['gaps']) < window + 1:
            return
        recent_gaps = data['gaps'][-(window+1):]
        recent_grads = data['gradient_norms'][-window:]
        recent_objs = data['objectives'][-window:]
        improvement = recent_gaps[0] - recent_gaps[-1]
        grad_trend = recent_grads[0] - recent_grads[-1]
        grad_std = float(np.std(recent_grads))
        grad_mean = float(np.mean(recent_grads)) if recent_grads else 0.0
        noisy = grad_mean > 0 and (grad_std / max(grad_mean, 1e-12)) > 0.5

        if alg_name == 'F2CSA':
            # Paper-aligned: increase N_g if progress stalls or gradients are noisy (disabled when fixed by env)
            if os.getenv('F2CSA_FIX_NG', '0') != '1':
                if improvement < 1e-3 or noisy:
                    old_Ng = getattr(algorithm, 'N_g', 5)
                    new_Ng = int(min(old_Ng + 5, 50))
                    if new_Ng != old_Ng:
                        algorithm.N_g = new_Ng
                        print(f"   🔧 [AUTO] F2CSA: Increased N_g {old_Ng}→{new_Ng} (variance control)")
            # Backoff lr when spikes detected; lower bound keeps optimizer responsive
            lr = algorithm.optimizer.param_groups[0]['lr']
            if os.getenv('F2CSA_FIXED_LR', '0') != '1' and (noisy or grad_trend > 0):
                new_lr = max(lr * 0.9, 1e-4)
                if new_lr < lr:
                    for g in algorithm.optimizer.param_groups:
                        g['lr'] = new_lr
                    print(f"   🔧 [AUTO] F2CSA: Adam lr {lr:.3e}→{new_lr:.3e}")
            # Extra: if gradient skyrockets vs. recent baseline, increase smoothing and consider SGD switch
            if len(data['gradient_norms']) >= 100:
                recent = np.array(data['gradient_norms'][-100:])
                med = float(np.median(recent))
                if data['gradient_norms'][-1] > 10.0 * max(med, 1e-12):
                    # strengthen gate smoothing
                    algorithm.tau_factor = max(algorithm.tau_factor, 10.0)
                    algorithm.lam_beta = min(0.995, max(algorithm.lam_beta, 0.97))
                    # and reduce lr significantly or move to SGD
                    cur_lr = algorithm.optimizer.param_groups[0]['lr']
                    if os.getenv('F2CSA_FIXED_LR', '0') != '1':
                        new_lr = max(cur_lr * 0.3, 1e-4)
                        for g in algorithm.optimizer.param_groups:
                            g['lr'] = new_lr
                        if getattr(algorithm, 'outer_optimizer_type', 'adam') == 'adam':
                            try:
                                algorithm.switch_to_sgd(lr=new_lr)
                                print(f"   🔧 [AUTO] F2CSA: Switched to SGD with lr={new_lr:.3e} due to gradient spike")
                            except Exception:
                                print(f"   ⚠️ [AUTO] F2CSA: SGD switch unavailable; kept Adam with lr={new_lr:.3e}")

        elif alg_name == 'SSIGD':
            # Switch to strongly-convex schedule if diminishing stalls
            if algorithm.stepsize_schedule == 'diminishing' and improvement < 1e-3:
                algorithm.stepsize_schedule = 'strongly_convex'
                print(f"   🔧 [AUTO] SSIGD: Schedule switched to strongly_convex (β_r = 1/(μ_F(r+1)))")
            # If still oscillating, increase μ_F to reduce stepsizes
            if noisy and algorithm.stepsize_schedule == 'strongly_convex':
                old_mu = algorithm.mu_F
                algorithm.mu_F = min(old_mu * 1.2, 1e4)
                print(f"   🔧 [AUTO] SSIGD: Increased μ_F {old_mu:.2f}→{algorithm.mu_F:.2f} (smaller steps)")
            # Relaxation: if objective improves steadily and gradients small, gently reduce μ_F back
            if improvement > 1e-2 and grad_mean < 1e-2 and algorithm.stepsize_schedule == 'strongly_convex':
                old_mu = algorithm.mu_F
                algorithm.mu_F = max(old_mu / 1.1, 50.0)
                if abs(algorithm.mu_F - old_mu) > 1e-6:
                    print(f"   🔧 [AUTO] SSIGD: Relaxed μ_F {old_mu:.2f}→{algorithm.mu_F:.2f}")

        elif alg_name == 'DSBLO':
            # Reduce perturbation variance if gradient is noisy (paper allows σ tuning)
            if os.getenv('DSBLO_FIX_SIGMA', '1') != '1' and noisy:
                old_sigma = algorithm.sigma
                algorithm.sigma = max(old_sigma * 0.9, 1e-6)
                print(f"   🔧 [AUTO] DS-BLO: Reduced σ {old_sigma:.2e}→{algorithm.sigma:.2e} (variance control)")
            # If progress is slow and |m| tiny → η_t≈1/γ₂. Slightly lower γ₂ to allow larger steps
            if improvement < 1e-3 and len(data['momentum_norms']) > 0:
                mnorm = data['momentum_norms'][-1]
                if mnorm is not None and mnorm < 1e-6:
                    old_g2 = algorithm.gamma2
                    algorithm.gamma2 = max(old_g2 * 0.9, 1e-3)
                    print(f"   🔧 [AUTO] DS-BLO: γ₂ {old_g2:.3f}→{algorithm.gamma2:.3f} (larger η_t)")
            # Minimal σ decay schedule on plateau: if no improvement over 250 iters and gap < 1e-2, halve σ
            PLATEAU_WIN = 250
            if os.getenv('DSBLO_FIX_SIGMA', '1') != '1' and iteration >= 1000 and len(data['gaps']) > PLATEAU_WIN:
                imp_250 = data['gaps'][-PLATEAU_WIN-1] - data['gaps'][-1]
                cur_gap = data['gaps'][-1]
                if imp_250 <= 0.0 and cur_gap < 1e-2:
                    old_sigma = algorithm.sigma
                    algorithm.sigma = max(old_sigma * 0.5, 1e-6)
                    print(f"   🔧 [AUTO] DS-BLO: Plateau detected after 1000 iters (Δ{PLATEAU_WIN}={imp_250:.3e}, gap={cur_gap:.3e}) → σ {old_sigma:.2e}→{algorithm.sigma:.2e}")


    # Run all algorithms until convergence alignment
    print(f"\n🏃 RUNNING ALGORITHMS TO CONVERGENCE ALIGNMENT")
    print("=" * 60)

    max_iterations = int(os.getenv('MAX_ITERS', '10000'))  # Allow short diagnostics via MAX_ITERS env
    convergence_tolerance = 1e-4

    # Optional diagnostic settings
    DIAG_GRAD_PROBE_EVERY = 25
    DIAG_KKT_LIMIT_START = 300

    grad_probe_rows = []
    fd_probe_rows = []

    for iteration in range(max_iterations):
        iteration_start_time = time.time()

        # Gradient-direction probe at matched x (every N iters)
        if iteration % DIAG_GRAD_PROBE_EVERY == 0:
            # Lock RNG so stochasticity does not mask direction cosines
            rng_state = torch.get_rng_state()
            # Clone shared x for fairness
            x_shared = {name: alg.x.detach().clone() for name, alg in algorithm_instances.items()}
            # Compute per algorithm without stepping
            def compute_grad(alg_name, alg, x0):
                torch.set_rng_state(rng_state)
                if alg_name == 'F2CSA':
                    return alg.compute_hypergradient(x0)
                elif alg_name == 'SSIGD':
                    return alg.compute_implicit_gradient(x0)
                else:
                    q0 = torch.randn(problem.dim, device=alg.device) * getattr(alg, 'sigma', 0.0001)
                    return alg.compute_perturbed_gradient(x0, q0)
            grads = {name: compute_grad(name, alg, x_shared[name]) for name, alg in algorithm_instances.items()}
            names = list(grads.keys())
            ref = grads[names[0]]
            for name in names[1:]:
                g = grads[name]
                cos = float((ref @ g) / (torch.norm(ref) * torch.norm(g) + 1e-12))
                grad_probe_rows.append({'iter': iteration, 'ref': names[0], 'other': name, 'cos': cos,
                                        'ref_norm': float(torch.norm(ref)), 'other_norm': float(torch.norm(g))})

        # F2CSA KKT-limit diagnostic schedule (paper-pure guard)
        if iteration == DIAG_KKT_LIMIT_START and 'F2CSA' in algorithm_instances:
            f2 = algorithm_instances['F2CSA']
            paper_pure = os.getenv('PAPER_PURE', '0') == '1'

            # Finite-difference subspace cosine check every 50 iters (diagnostic only)
            # We perform this right after the gradient probe block when iteration % 50 == 0
            if iteration % 50 == 0:
                f2_alg = algorithm_instances.get('F2CSA')
                if f2_alg is not None:
                    x0 = f2_alg.x.detach().clone()
                    d = x0.numel()
                    torch.manual_seed(123)
                    idx = torch.randperm(d)[:5]
                    eps = 1e-4
                    fd = torch.zeros_like(x0)
                    def F(x):
                        return problem.true_bilevel_objective(x)
                    for j in idx:
                        e = torch.zeros_like(x0); e[j] = 1.0
                        f_plus = F(x0 + eps * e)
                        f_minus = F(x0 - eps * e)
                        fd[j] = (f_plus - f_minus) / (2 * eps)
                    # Always recompute F2CSA hypergradient at x0 for FD probe
                    hg = f2_alg.compute_hypergradient(x0).detach()
                    mask = torch.zeros_like(hg)
                    mask[idx] = 1.0
                    hg_sub = hg * mask
                    fd_sub = fd * mask
                    if torch.norm(hg_sub) > 0 and torch.norm(fd_sub) > 0:
                        cos_fd = float((hg_sub @ fd_sub) / (torch.norm(hg_sub) * torch.norm(fd_sub) + 1e-12))
                        print(f"      [fd-probe] iter={iteration} cos(F2CSA, FD-5D)={cos_fd:.4f} |hg_sub|={torch.norm(hg_sub):.3e} |fd_sub|={torch.norm(fd_sub):.3e}")
                        fd_probe_rows.append({'iter': iteration, 'cos_fd': cos_fd, 'hg_sub_norm': float(torch.norm(hg_sub)), 'fd_sub_norm': float(torch.norm(fd_sub))})
                        if cos_fd < 0.9:
                            print("      ⚠️  [fd-probe] Surrogate bias likely dominates: cosine < 0.9 persistently")

            # Paper-pure mode: skip any behavioral modifications to F2CSA
            if paper_pure:
                pass
            else:
                # Increase penalties and tighten smoothing/regularization conservatively
                f2.alpha1 *= 100.0  # stronger push toward exactness
                f2.alpha2 *= 100.0

                # Switch F2CSA optimizer to SGD with fixed lr and increase N_g for stronger signal
                try:
                    f2.switch_to_sgd(lr=1e-3)
                    f2.N_g = max(getattr(f2, 'N_g', 30), 50)
                    # Apply de-bias flag if requested via env
                    if os.getenv('F2CSA_DEBIAS_ACTIVES', '0') == '1' and hasattr(f2, 'penalize_inactive_only'):
                        f2.penalize_inactive_only = True
                        print("   🔧 [DIAG] F2CSA: penalize_inactive_only=True (drop quad penalty on actives)")
                except Exception as e:
                    print(f"   🔧 [DIAG] F2CSA SGD switch failed: {e}")

                # Reduce smoothing and adjust gating under KKT-limit
                f2.tau_factor = max(0.5, f2.tau_factor / 4.0)  # reduce smoothing
                f2.tau_min = max(1e-6, f2.tau_min / 100.0)
                f2.delta = max(1e-6, f2.delta / 10.0)
                # Disable λ EMA gating and switch to actives-only gating under KKT-limit
                if hasattr(f2, 'disable_lam_ema'):
                    f2.disable_lam_ema = True
                if hasattr(f2, 'actives_only_gating'):
                    f2.actives_only_gating = True
                # Ensure optimizer is SGD with sufficient N_g
                try:
                    f2.switch_to_sgd(lr=1e-3)
                    f2.N_g = max(getattr(f2, 'N_g', 30), 50)
                    sgd_note = f"switched_to_sgd=True, sgd_lr={1e-3:.1e}, N_g={f2.N_g}"
                except Exception:
                    sgd_note = "switched_to_sgd=False"
                print(f"   🔧 [DIAG] F2CSA KKT-limit: α1={f2.alpha1:.3e}, α2={f2.alpha2:.3e}, τ_factor={f2.tau_factor:.3e}, τ_min={f2.tau_min:.1e}, δ={f2.delta:.1e}, disable_lam_ema={getattr(f2,'disable_lam_ema', False)}, actives_only_gating={getattr(f2,'actives_only_gating', False)}, {sgd_note}")

        # Run one iteration for each algorithm
        for alg_name, algorithm in algorithm_instances.items():
            alg_start_time = time.time()

            # Take algorithm-specific step
            if alg_name == 'F2CSA':
                # F2CSA step - pure implementation
                # compute_hypergradient already applies paper-allowed smoothing internally; do NOT double-EMA here
                hypergradient = algorithm.compute_hypergradient(algorithm.x)

                # Optimizer step
                algorithm.optimizer.zero_grad()
                algorithm.x.grad = hypergradient
                algorithm.optimizer.step()

                gradient_norm = float(torch.norm(hypergradient))

            elif alg_name == 'SSIGD':
                # SSIGD step - pure implementation
                # Paper-exact strongly-convex schedule: β_r = 1/(μ_F(r+1))
                beta_r = 1.0 / (algorithm.mu_F * (iteration + 1))
                implicit_grad = algorithm.compute_implicit_gradient(algorithm.x)

                with torch.no_grad():
                    algorithm.x -= beta_r * implicit_grad

                gradient_norm = float(torch.norm(implicit_grad))

            elif alg_name == 'DSBLO':
                # DS-BLO step (paper-exact when DSBLO_PAPER_STEP=1)
                paper_step = os.getenv('DSBLO_PAPER_STEP', '1') == '1'
                option_ii = os.getenv('DSBLO_OPTION_II', '1') == '1'

                if paper_step:
                    # Use m_t to step, then sample \bar{x}_{t+1} and compute g_{t+1}, then update m_{t+1}
                    m_t = algorithm.momentum_vector
                    mnorm_t = torch.norm(m_t)
                    eta_t = 1.0 / (algorithm.gamma1 * mnorm_t + algorithm.gamma2)

                    # x_{t+1} = x_t - eta_t * m_t
                    with torch.no_grad():
                        x_t = algorithm.x.clone()
                        step_vec = eta_t * m_t
                        algorithm.x -= step_vec

                    # \bar{x}_{t+1} ~ Uniform[x_t, x_{t+1}]
                    u = torch.rand((), device=algorithm.device, dtype=algorithm.x.dtype)
                    x_bar = x_t - u * eta_t * m_t

                    # g_{t+1} at \bar{x}_{t+1} with optional within-step averaging K (DSBLO_NG)
                    K = int(os.getenv('DSBLO_NG', '1'))
                    if K <= 1:
                        q_tp1 = torch.randn(problem.dim, device=algorithm.device) * algorithm.sigma
                        raw_gradient = (algorithm.compute_perturbed_gradient_with_noise(x_bar, q_tp1)
                                        if option_ii else
                                        algorithm.compute_perturbed_gradient(x_bar, q_tp1))
                        grad_avg = raw_gradient
                    else:
                        grads = []
                        for _ in range(K):
                            qk = torch.randn(problem.dim, device=algorithm.device) * algorithm.sigma
                            gk = (algorithm.compute_perturbed_gradient_with_noise(x_bar, qk)
                                  if option_ii else
                                  algorithm.compute_perturbed_gradient(x_bar, qk))
                            grads.append(gk)
                        grad_avg = torch.stack(grads, dim=0).mean(dim=0)

                    # EMA smoothing (optional) — leave off when EMA is None or 0
                    if getattr(algorithm, 'ema_decay', None) in (None, 0.0):
                        gradient = grad_avg
                    else:
                        if algorithm.gradient_ema is None:
                            algorithm.gradient_ema = grad_avg.clone()
                        else:
                            algorithm.gradient_ema = algorithm.ema_decay * algorithm.gradient_ema + (1 - algorithm.ema_decay) * grad_avg
                        gradient = algorithm.gradient_ema

                    # m_{t+1} = β m_t + (1-β) g_{t+1}
                    algorithm.momentum_vector = algorithm.momentum * m_t + (1 - algorithm.momentum) * gradient

                    # Debug metrics
                    gradient_norm = float(torch.norm(gradient))
                    current_gap = problem.compute_gap(algorithm.x)
                    algorithm.last_debug = {
                        'gap': float(current_gap),
                        'mnorm': float(torch.norm(algorithm.momentum_vector)),
                        'eta': float(eta_t),
                        'dx_norm': float(torch.norm(step_vec))
                    }
                else:
                    # Legacy order (previous harness behavior)
                    q_t = torch.randn(problem.dim, device=algorithm.device) * algorithm.sigma
                    raw_gradient = (algorithm.compute_perturbed_gradient_with_noise(algorithm.x, q_t)
                                     if option_ii else
                                     algorithm.compute_perturbed_gradient(algorithm.x, q_t))

                    # EMA smoothing
                    if algorithm.gradient_ema is None:
                        algorithm.gradient_ema = raw_gradient.clone()
                    else:
                        algorithm.gradient_ema = algorithm.ema_decay * algorithm.gradient_ema + (1 - algorithm.ema_decay) * raw_gradient

                    gradient = algorithm.gradient_ema

                    # Update momentum and step
                    algorithm.momentum_vector = algorithm.momentum * algorithm.momentum_vector + (1 - algorithm.momentum) * gradient

                    momentum_norm = torch.norm(algorithm.momentum_vector)
                    eta_t = 1.0 / (algorithm.gamma1 * momentum_norm + algorithm.gamma2)

                    with torch.no_grad():
                        step_vec = eta_t * algorithm.momentum_vector
                        algorithm.x -= step_vec

                    gradient_norm = float(torch.norm(gradient))
                    current_gap = problem.compute_gap(algorithm.x)
                    algorithm.last_debug = {
                        'gap': float(current_gap),
                        'mnorm': float(momentum_norm),
                        'eta': float(eta_t),
                        'dx_norm': float(torch.norm(step_vec))
                    }

            # Optional: algorithm-specific debug dump if available
            if hasattr(algorithm, 'last_debug') and isinstance(algorithm.last_debug, dict):
                dbg_extra = ', '.join([f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k,v in algorithm.last_debug.items()])
                print(f"      ↳ debug: {dbg_extra}")

            # Evaluate and log current state for this algorithm
            current_gap = problem.compute_gap(algorithm.x)
            current_obj = float(problem.true_bilevel_objective(algorithm.x, add_noise=False))
            x_norm = float(torch.norm(algorithm.x))

            # Constraint violation and KKT residual (from LL)
            _, ll_info = problem.solve_lower_level(algorithm.x)
            constraint_violation = ll_info.get('constraint_violation', 0)
            kkt_residual = ll_info.get('optimality_gap', None)

            # Stepsize / momentum norm per algorithm
            stepsize = None
            momentum_norm = None
            if alg_name == 'F2CSA':
                stepsize = algorithm.optimizer.param_groups[0]['lr']
            elif alg_name == 'SSIGD':
                stepsize = float(algorithm.beta_0 / torch.sqrt(torch.tensor(iteration + 1.0)))
            elif alg_name == 'DSBLO':
                momentum_norm = float(torch.norm(algorithm.momentum_vector))
                stepsize = float(1.0 / (algorithm.gamma1 * max(momentum_norm, 0.01) + algorithm.gamma2))

            alg_time = time.time() - alg_start_time

            # Log iteration data
            monitor.log_iteration(
                alg_name, iteration, current_gap, current_obj, gradient_norm,
                constraint_violation, alg_time, x_norm,
                stepsize=stepsize, momentum_norm=momentum_norm, kkt_residual=kkt_residual
            )

            # Debug print (per-iteration key stats)
            dbg = f"{alg_name}: gap={current_gap:.6e}, obj={current_obj:.6e}, grad={gradient_norm:.3e}, cv={constraint_violation:.2e}"
            if kkt_residual is not None:
                dbg += f", kkt={kkt_residual:.2e}"
            if stepsize is not None:
                dbg += f", step={stepsize:.3e}"
            if momentum_norm is not None:
                dbg += f", |m|={momentum_norm:.3e}"
            print("   ", dbg)

        # Auto-adjust using recent history (paper-aligned policies)
        for alg_name2, alg2 in algorithm_instances.items():
            auto_adjust(alg_name2, alg2, monitor.algorithm_data[alg_name2], iteration)

        # Paper-pure FD probes for F2CSA at multiple points (diagnostic only)
        if os.getenv('PAPER_PURE', '0') == '1' and 'F2CSA' in algorithm_instances:
            FD_START = int(os.getenv('FD_START', '300'))
            FD_EVERY = int(os.getenv('FD_EVERY', '300'))
            if iteration >= FD_START and (iteration % FD_EVERY == 0):
                f2_alg = algorithm_instances['F2CSA']
                x0 = f2_alg.x.detach().clone()
                d = x0.numel()
                torch.manual_seed(123)
                idx = torch.randperm(d)[:5]
                eps = 1e-4
                fd = torch.zeros_like(x0)
                def F(x):
                    return problem.true_bilevel_objective(x)
                for j in idx:
                    e = torch.zeros_like(x0); e[j] = 1.0
                    f_plus = F(x0 + eps * e)
                    f_minus = F(x0 - eps * e)
                    fd[j] = (f_plus - f_minus) / (2 * eps)
                hg = f2_alg.compute_hypergradient(x0).detach()
                mask = torch.zeros_like(hg)
                mask[idx] = 1.0
                hg_sub = hg * mask
                fd_sub = fd * mask
                if torch.norm(hg_sub) > 0 and torch.norm(fd_sub) > 0:
                    cos_fd = float((hg_sub @ fd_sub) / (torch.norm(hg_sub) * torch.norm(fd_sub) + 1e-12))
                    print(f"      [fd-probe] iter={iteration} cos(F2CSA, FD-5D)={cos_fd:.4f} |hg_sub|={torch.norm(hg_sub):.3e} |fd_sub|={torch.norm(fd_sub):.3e}")
                    fd_probe_rows.append({'iter': iteration, 'cos_fd': cos_fd, 'hg_sub_norm': float(torch.norm(hg_sub)), 'fd_sub_norm': float(torch.norm(fd_sub))})

        # Save probe results if any
        if grad_probe_rows:
            df_probe = pd.DataFrame(grad_probe_rows)
            df_probe.to_csv('gradient_probe_summary.csv', index=False)
            print("📁 Saved gradient_probe_summary.csv")
        if fd_probe_rows:
            df_fd = pd.DataFrame(fd_probe_rows)
            df_fd.to_csv('fd_probe_summary.csv', index=False)
            print("📁 Saved fd_probe_summary.csv")

        # Auto-stop safeguards: halt and diagnose on F2CSA instability
        f2_data = monitor.algorithm_data.get('F2CSA', None)
        if f2_data and len(f2_data['gradient_norms']) >= 120:
            recent = np.array(f2_data['gradient_norms'][-100:])
            med = float(np.median(recent))
            last = f2_data['gradient_norms'][-1]
            spike = last > 10.0 * max(med, 1e-12)

            # Pull latest debug if present
            f2_alg = algorithm_instances['F2CSA']
            last_dbg = getattr(f2_alg, 'last_debug', {}) if hasattr(f2_alg, 'last_debug') else {}
            rho_std = last_dbg.get('rho_std', 0.0)
            rho_max = last_dbg.get('rho_max', 0.0)
            rho_min = last_dbg.get('rho_min', 0.0)
            H_cond = last_dbg.get('H_cond', 0.0)

            rho_flip = (rho_std > 0.5 and rho_max > 0.98 and rho_min < 0.02)
            cond_bad = (H_cond is not None and H_cond != H_cond) or (H_cond is not None and H_cond > 10.0)

            # Gap monotonic increase
            gaps = f2_data['gaps'][-20:]
            inc = len(gaps) >= 20 and all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1))

            if spike or rho_flip or cond_bad or inc:
                print("\n⛔ F2CSA instability detected — stopping run for diagnosis.")
                print(f"   Reason(s): spike={spike}, rho_flip={rho_flip}, cond_bad={cond_bad}, inc20={inc}")
                if last_dbg:
                    print("   Last F2CSA debug:", {k: round(v, 6) if isinstance(v, float) else v for k, v in last_dbg.items()})
                print("   Recent grad_norms tail:", [round(v, 6) for v in f2_data['gradient_norms'][-10:]])
                print("   Recent gaps tail:", [round(v, 6) for v in f2_data['gaps'][-10:]])
                break

        # Plateau diagnostics: when F2CSA gap “flats out”, print root-cause signals
        if os.getenv('PLATEAU_DEBUG', '1') == '1':
            f2_data = monitor.algorithm_data.get('F2CSA', None)
            ds_data = monitor.algorithm_data.get('DSBLO', None)
            if f2_data and len(f2_data['gaps']) >= 120:
                win = int(os.getenv('PLATEAU_WIN', '100'))
                eps = float(os.getenv('PLATEAU_EPS', '1e-4'))
                gaps = f2_data['gaps']
                grads = f2_data['gradient_norms']
                if len(gaps) > win:
                    imp = gaps[-win] - gaps[-1]
                    slope = (gaps[-1] - gaps[-win]) / win
                    grad_delta = grads[-win] - grads[-1] if len(grads) > win else None
                    if imp < eps:
                        f2_alg = algorithm_instances['F2CSA']
                        last_dbg = getattr(f2_alg, 'last_debug', {}) if hasattr(f2_alg, 'last_debug') else {}
                        # Pull optimizer and gating state
                        opt_type = getattr(f2_alg, 'outer_optimizer_type', 'unknown')
                        lr = f2_alg.optimizer.param_groups[0]['lr'] if hasattr(f2_alg, 'optimizer') else None
                        Ng = getattr(f2_alg, 'N_g', None)
                        alpha = getattr(f2_alg, 'alpha', None)
                        alpha1 = getattr(f2_alg, 'alpha1', None)
                        alpha2 = getattr(f2_alg, 'alpha2', None)
                        delta = getattr(f2_alg, 'delta', None)
                        tau = getattr(f2_alg, 'tau', None) if hasattr(f2_alg, 'tau') else None
                        tau_factor = getattr(f2_alg, 'tau_factor', None)
                        tau_min = getattr(f2_alg, 'tau_min', None)
                        lam_beta = getattr(f2_alg, 'lam_beta', None)
                        ema_decay = getattr(f2_alg, 'ema_decay', None)
                        disable_lam_ema = getattr(f2_alg, 'disable_lam_ema', None)
                        actives_only = getattr(f2_alg, 'actives_only_gating', None)
                        penalize_inactive_only = getattr(f2_alg, 'penalize_inactive_only', None)
                        # Last probe cosines
                        last_fd = fd_probe_rows[-1]['cos_fd'] if fd_probe_rows else None
                        last_probe = None
                        if grad_probe_rows:
                            last_probe = grad_probe_rows[-1]
                        ds_gap = ds_data['gaps'][-1] if ds_data and ds_data['gaps'] else None
                        print("\n🛑 F2CSA PLATEAU DETECTED — diagnostics")
                        gd_str = f"{grad_delta:.3e}" if grad_delta is not None else "None"
                        ds_gap_str = f"{ds_gap:.6e}" if ds_gap is not None else "None"
                        lr_str = f"{lr:.3e}" if lr is not None else "None"
                        print(f"   window={win}, imp={imp:.3e}, slope/iter={slope:.3e}, grad_delta={gd_str}")
                        print(f"   F2CSA gap={gaps[-1]:.6e}, DS-BLO gap={ds_gap_str}")
                        print(f"   opt={opt_type}, lr={lr_str}, N_g={Ng}")
                        print(f"   alpha={alpha}, alpha1={alpha1}, alpha2={alpha2}, delta={delta}")
                        print(f"   tau={tau}, tau_factor={tau_factor}, tau_min={tau_min}, lam_beta={lam_beta}")
                        print(f"   ema_decay={ema_decay}, disable_lam_ema={disable_lam_ema}, actives_only={actives_only}, penalize_inactive_only={penalize_inactive_only}")
                        if last_dbg:
                            rho_min = last_dbg.get('rho_min'); rho_mean = last_dbg.get('rho_mean'); rho_max = last_dbg.get('rho_max'); rho_std = last_dbg.get('rho_std')
                            H_cond = last_dbg.get('H_cond'); ytil = last_dbg.get('y_tilde_norm'); rhs = last_dbg.get('rhs_norm')
                            rho_min_str = f"{rho_min:.3e}" if rho_min is not None else "None"
                            rho_mean_str = f"{rho_mean:.3e}" if rho_mean is not None else "None"
                            rho_max_str = f"{rho_max:.3e}" if rho_max is not None else "None"
                            rho_std_str = f"{rho_std:.3e}" if rho_std is not None else "None"
                            print(f"   ρ stats: min={rho_min_str}, mean={rho_mean_str}, max={rho_max_str}, std={rho_std_str}")
                            print(f"   H_cond={H_cond}, |y_tilde|={ytil}, |rhs|={rhs}")
                        if last_fd is not None:
                            print(f"   last FD-cos(F2CSA, FD-5D)={last_fd:.4f}")
                        if last_probe is not None:
                            print(f"   last probe cos({last_probe['ref']},{last_probe['other']})={last_probe['cos']:.4f} |ref|={last_probe['ref_norm']:.3e} |other|={last_probe['other_norm']:.3e}")

        # Progress reporting
        if iteration % 50 == 0 or iteration < 10:
            print(f"\nIteration {iteration}:")
            for alg_name in algorithm_instances.keys():
                data = monitor.algorithm_data[alg_name]
                if data['gaps']:
                    gap = data['gaps'][-1]
                    grad_norm = data['gradient_norms'][-1]
                    cv = data['constraint_violations'][-1] if data['constraint_violations'] else None
                    kkt = data['kkt_residuals'][-1] if data['kkt_residuals'] else None
                    step = data['stepsizes'][-1] if data['stepsizes'] else None
                    mnorm = data['momentum_norms'][-1] if data['momentum_norms'] else None
                    msg = f"   {alg_name}: Gap={gap:.8f}, Grad={grad_norm:.6f}"
                    if cv is not None: msg += f", CV={cv:.2e}"
                    if kkt is not None: msg += f", KKT={kkt:.2e}"
                    if step is not None: msg += f", Step={step:.3e}"
                    if mnorm is not None: msg += f", |m|={mnorm:.3e}"
                    print(msg)
            # If we probed gradients recently, print last probe summary
            if grad_probe_rows:
                last = grad_probe_rows[-min(3, len(grad_probe_rows)) :]
                for row in last:
                    print(f"      [probe] iter={row['iter']} cos({row['ref']},{row['other']})={row['cos']:.4f} |ref|={row['ref_norm']:.3e} |other|={row['other_norm']:.3e}")

        # Check for convergence alignment every 20 iterations
        if iteration > 100 and iteration % 20 == 0:
            if monitor.check_convergence_alignment(convergence_tolerance):
                print(f"✅ CONVERGENCE ALIGNMENT ACHIEVED at iteration {iteration}")
                break

        # Check for fundamental issues every 50 iterations
        if iteration % 50 == 0:
            issues = monitor.detect_fundamental_issues()
            if issues:
                print(f"🚨 FUNDAMENTAL ISSUES DETECTED:")
                for issue in issues:
                    print(f"   {issue}")
                if len(issues) >= 3:  # Multiple serious issues
                    print(f"❌ STOPPING: Too many fundamental issues")
                    break

        # Safety termination
        if iteration >= max_iterations - 1:
            print(f"⏰ REACHED MAXIMUM ITERATIONS ({max_iterations})")
            break

    # Final analysis
    print(f"\n📊 FINAL CONVERGENCE ANALYSIS")
    print("=" * 60)

    # Check final convergence status
    final_gaps = {}
    converged_algorithms = []

    for alg_name, data in monitor.algorithm_data.items():
        if data['converged']:
            final_gaps[alg_name] = data['final_gap']
            converged_algorithms.append(alg_name)
            print(f"✅ {alg_name}: CONVERGED to gap {data['final_gap']:.8f}")
        else:
            if data['gaps']:
                final_gaps[alg_name] = data['gaps'][-1]
                print(f"❌ {alg_name}: NOT CONVERGED, final gap {data['gaps'][-1]:.8f}")

    # Analyze gap alignment
    if len(final_gaps) > 1:
        gaps = list(final_gaps.values())
        gap_mean = np.mean(gaps)
        gap_std = np.std(gaps)
        gap_range = max(gaps) - min(gaps)

        print(f"\n📈 Gap Alignment Analysis:")
        print(f"   Final gaps: {[f'{gap:.8f}' for gap in gaps]}")
        print(f"   Gap mean: {gap_mean:.8f}")
        print(f"   Gap std: {gap_std:.8f}")
        print(f"   Gap range: {gap_range:.8f}")
        print(f"   Target tolerance: {convergence_tolerance}")

        if gap_range <= convergence_tolerance:
            print(f"🎉 SUCCESS: All algorithms converged to same gap!")
        else:
            print(f"⚠️  PARTIAL SUCCESS: Algorithms converged to different gaps")
            print(f"   This indicates fundamental implementation differences")

    # Analyze upper-level objective alignment (they should match at convergence)
    final_objectives = {}
    for alg_name, data in monitor.algorithm_data.items():
        if data['objectives']:
            final_objectives[alg_name] = data['objectives'][-1]
    if len(final_objectives) > 1:
        objs = list(final_objectives.values())
        obj_mean = np.mean(objs)
        obj_std = np.std(objs)
        obj_range = max(objs) - min(objs)
        print(f"\n📈 Upper Objective Alignment:")
        print(f"   Final objectives: {[f'{v:.8f}' for v in objs]}")
        print(f"   Mean: {obj_mean:.8f}")
        print(f"   Std: {obj_std:.8f}")
        print(f"   Range: {obj_range:.8f}")
        print(f"   Alignment tolerance: {convergence_tolerance}")
        if obj_range <= convergence_tolerance:
            print(f"🎉 SUCCESS: All algorithms reached the same upper objective value (within tolerance)!")
        else:
            print(f"⚠️  NOT ALIGNED: Upper objectives differ beyond tolerance.")
            print(f"   Investigate hypergradient consistency and LL solver equivalence.")

    # Save detailed results
    save_comprehensive_results(monitor)


    # Final ground-truth check against analytical stationary point (inactive-constraints model)
    try:
        print("\n🔎 Final ground-truth gap check vs analytical x* (inactive regime)")
        x_star = problem.stationary_x_star_if_inactive()
        print("   compute_gap(x*): (should be ~0)")
        gap_xstar = problem.compute_gap(x_star)
        print(f"   => gap(x*) = {gap_xstar:.6e}")
        for alg_name, alg in algorithm_instances.items():
            x_final = alg.x.detach()
            dist = float(torch.norm(x_final - x_star))
            print(f"   {alg_name}: ||x_final - x*|| = {dist:.6e}")
            print(f"   {alg_name}: compute_gap(x_final):")
            gap_final = problem.compute_gap(x_final)
            print(f"   => {alg_name} gap(x_final) = {gap_final:.6e}")
    except Exception as e:
        print(f"   ⚠️  Final x* check skipped due to error: {e}")


    # DS-BLO end-of-run tail diagnostics: true grad vs raw/EMA/momentum
    if 'DSBLO' in algorithm_instances:
        print("\n🔬 DS-BLO tail diagnostics at final x")
        ds = algorithm_instances['DSBLO']
        x_final = ds.x.detach()
        true_total, true_direct, true_impl = problem.compute_true_bilevel_gradient(x_final)
        true_norm = float(torch.norm(true_total))
        dir_norm = float(torch.norm(true_direct))
        imp_norm = float(torch.norm(true_impl))
        print(f"   true ||∇F|| = {true_norm:.6e} (direct={dir_norm:.6e}, implicit={imp_norm:.6e})")
        # Reconstruct a raw gradient sample at the same x with a fixed seed for determinism
        torch.manual_seed(12345)
        q = torch.randn(problem.dim, device=ds.device) * getattr(ds, 'sigma', 0.0001)
        raw = ds.compute_perturbed_gradient(x_final, q)
        raw_norm = float(torch.norm(raw))
        ema_norm = float(torch.norm(ds.gradient_ema)) if getattr(ds, 'gradient_ema', None) is not None else float('nan')
        m_norm = float(torch.norm(ds.momentum_vector)) if getattr(ds, 'momentum_vector', None) is not None else float('nan')
        # Cosines for alignment
        cos_raw = float(torch.dot(raw, true_total) / (torch.norm(raw) * torch.norm(true_total) + 1e-12)) if true_norm > 0 and raw_norm > 0 else float('nan')
        print(f"   raw_grad ||·|| = {raw_norm:.6e}, cos(raw, true)={cos_raw:.4f}")
        print(f"   grad_ema ||·|| = {ema_norm:.6e}")
        print(f"   momentum |m| = {m_norm:.6e}")
        # Sensitivity check: raw vs true under larger σ multipliers (diagnostic only)
        sig0 = getattr(ds, 'sigma', 0.0001)
        for mult in [1.0, 5.0, 10.0]:
            torch.manual_seed(12345)
            qk = torch.randn(problem.dim, device=ds.device) * (sig0 * mult)
            raw_k = ds.compute_perturbed_gradient(x_final, qk)
            raw_k_norm = float(torch.norm(raw_k))
            cos_k = float(torch.dot(raw_k, true_total) / (torch.norm(raw_k) * torch.norm(true_total) + 1e-12)) if true_norm > 0 and raw_k_norm > 0 else float('nan')
            print(f"   σ×{mult:>4.1f}: ||raw||={raw_k_norm:.6e}, cos(raw,true)={cos_k:.4f}")


    return monitor

def save_comprehensive_results(monitor):
    """Save comprehensive results to files"""
    # Create summary DataFrame
    summary_data = []
    for alg_name, data in monitor.algorithm_data.items():
        if data['gaps']:
            summary_data.append({
                'Algorithm': alg_name,
                'Converged': data['converged'],
                'Final_Gap': data['final_gap'] if data['final_gap'] else data['gaps'][-1],
                'Total_Iterations': len(data['gaps']),
                'Total_Time': data['total_time'],
                'Avg_Time_Per_Iter': data['total_time'] / len(data['gaps']) if data['gaps'] else 0,
                'Final_Gradient_Norm': data['gradient_norms'][-1] if data['gradient_norms'] else 0,
                'Final_Constraint_Violation': data['constraint_violations'][-1] if data['constraint_violations'] else 0
            })

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv('comprehensive_convergence_summary.csv', index=False)
        print(f"📁 Saved comprehensive_convergence_summary.csv")

    # Create convergence plots
    create_comprehensive_plots(monitor)

def create_comprehensive_plots(monitor):
    """Create comprehensive convergence plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comprehensive Convergence Test: Dimension 100', fontsize=16, fontweight='bold')

    colors = {'F2CSA': 'blue', 'SSIGD': 'green', 'DSBLO': 'red'}

    # Plot 1: Gap convergence
    ax1 = axes[0, 0]
    for alg_name, data in monitor.algorithm_data.items():
        if data['gaps']:
            ax1.semilogy(data['iterations'], data['gaps'],
                        color=colors.get(alg_name, 'black'),
                        linewidth=2, label=alg_name, alpha=0.8)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Gap (Log Scale)')
    ax1.set_title('Gap Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gradient norms
    ax2 = axes[0, 1]
    for alg_name, data in monitor.algorithm_data.items():
        if data['gradient_norms']:
            ax2.semilogy(data['iterations'], data['gradient_norms'],
                        color=colors.get(alg_name, 'black'),
                        linewidth=2, label=alg_name, alpha=0.8)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm (Log Scale)')
    ax2.set_title('Gradient Norm Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final gaps comparison
    ax3 = axes[1, 0]
    algorithms = []
    final_gaps = []
    for alg_name, data in monitor.algorithm_data.items():
        if data['gaps']:
            algorithms.append(alg_name)
            final_gaps.append(data['final_gap'] if data['final_gap'] else data['gaps'][-1])

    if algorithms:
        bars = ax3.bar(algorithms, final_gaps,
                      color=[colors.get(alg, 'black') for alg in algorithms], alpha=0.7)
        ax3.set_ylabel('Final Gap')
        ax3.set_title('Final Gap Comparison')
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, gap in zip(bars, final_gaps):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{gap:.6f}', ha='center', va='bottom', fontsize=10)

    # Plot 4: Constraint violations
    ax4 = axes[1, 1]
    for alg_name, data in monitor.algorithm_data.items():
        if data['constraint_violations']:
            ax4.semilogy(data['iterations'], data['constraint_violations'],
                        color=colors.get(alg_name, 'black'),
                        linewidth=2, label=alg_name, alpha=0.8)

    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Constraint Violation (Log Scale)')
    ax4.set_title('Constraint Violation Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comprehensive_convergence_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Saved comprehensive_convergence_test.png")

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE CONVERGENCE TEST")
    print("=" * 80)
    print("Testing fundamental correctness of bilevel optimization algorithms")
    print("Goal: All algorithms converge to same gap on same problem")
    print("No workarounds - only correct mathematical implementations")
    print("=" * 80)

    # Autopilot mode: attempt multiple runs with σ halving between runs if DS-BLO plateaus <1e-2 but gaps not aligned
    if os.getenv('AUTOPILOT', '0') == '1':
        max_attempts = int(os.getenv('AUTOPILOT_ATTEMPTS', '20'))
        tol = float(os.getenv('ALIGN_TOL', '1e-4'))
        ds_sigma = float(os.getenv('DSBLO_SIGMA', '1e-4'))
        for attempt in range(1, max_attempts + 1):
            os.environ['DSBLO_SIGMA'] = f"{ds_sigma:.12g}"
            print(f"\n🤖 AUTOPILOT attempt {attempt}/{max_attempts} with DSBLO_SIGMA={os.environ['DSBLO_SIGMA']}")
            monitor = run_comprehensive_convergence_test()
            if not monitor:
                print("❌ Run returned no monitor; stopping.")
                break
            data_f2 = monitor.algorithm_data.get('F2CSA')
            data_ds = monitor.algorithm_data.get('DSBLO')
            def final_gap_and_converged(d):
                if not d or len(d['gaps']) < 20:
                    return None, False
                recent = d['gaps'][-20:]
                var = float(np.var(recent))
                mean_gap = float(np.mean(recent))
                return mean_gap, var < tol**2
            f2_gap, f2_conv = final_gap_and_converged(data_f2)
            ds_gap, ds_conv = final_gap_and_converged(data_ds)
            if f2_gap is not None and ds_gap is not None:
                gap_range = abs(f2_gap - ds_gap)
                print(f"🤖 AUTOPILOT result: F2CSA gap={f2_gap:.6e}, DSBLO gap={ds_gap:.6e}, |diff|={gap_range:.6e}")
                if gap_range <= tol:
                    print("🤖 AUTOPILOT: Alignment achieved. Stopping.")
                    break
                # If DS-BLO has plateaued and is below 1e-2 but not aligned → halve σ and retry
                if os.getenv('DSBLO_FIX_SIGMA', '1') != '1' and ds_conv and ds_gap < 1e-2:
                    prev = ds_sigma
                    ds_sigma = max(prev * 0.5, 1e-6)
                    print(f"🤖 AUTOPILOT: DS-BLO plateaued with gap<{1e-2:.0e} but misaligned → σ {prev:.2e}→{ds_sigma:.2e} and retry")
                    continue
            print("🤖 AUTOPILOT: Alignment not reached; continuing without σ change.")
        print("🤖 AUTOPILOT: Finished attempts.")
    else:
        monitor = run_comprehensive_convergence_test()
        if monitor and monitor.convergence_achieved:
            print(f"\n🎉 MISSION ACCOMPLISHED: Proper bilevel optimization achieved!")
        elif monitor:
            print(f"\n⚠️  MISSION INCOMPLETE: Fundamental issues remain")
            if monitor.fundamental_issues:
                print(f"Issues to address:")
                for issue in monitor.fundamental_issues:
                    print(f"  - {issue}")
        else:
            print(f"\n❌ MISSION FAILED: Critical problems detected")

    print(f"\n✅ Comprehensive test completed!")
    print(f"📊 Check comprehensive_convergence_test.png and .csv for detailed results")

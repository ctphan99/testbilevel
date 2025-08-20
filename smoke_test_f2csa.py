import os
import torch
from problem import StronglyConvexBilevelProblem
from algorithm import F2CSA


def main():
    # Small, fast configuration
    dim = int(os.getenv('ST_DIM', '10'))
    cons = int(os.getenv('ST_CONS', '3'))
    iters = int(os.getenv('ST_ITERS', '50'))
    inner_bs = int(os.getenv('ST_INNER_BS', '32'))

    prob = StronglyConvexBilevelProblem(dim=dim, num_constraints=cons, noise_std=1e-4, device='cpu', seed=123)
    algo = F2CSA(prob, N_g=5, alpha=0.35, adam_lr=5e-3, inner_batch_size=inner_bs)

    res = algo.optimize(max_iterations=iters, convergence_threshold=1e-4)

    # Compute a noisy gap estimate at the final iterate by sampling f(x, y*(x)) with noise
    nsamp = int(os.getenv('ST_GAP_NOISE_SAMPLES', '50'))
    with torch.no_grad():
        y_star, _ = prob.solve_lower_level(algo.x)
        samples = []
        for _ in range(nsamp):
            samples.append(float(prob.upper_objective(algo.x, y_star, add_noise=True)))
        import numpy as np
        noisy_gap_mean = float(np.mean(samples))
        noisy_gap_std = float(np.std(samples))

    print("=== Smoke Test Summary (stochastic inner) ===")
    print(f"inner_batch_size={inner_bs}")
    print(f"algo={res['algorithm']} iters={res['total_iterations']} converged={res['converged']}")
    print(f"final_obj={res['final_objective']:.6f} final_gap={res['final_gap']:.6e} best_gap={res['best_gap']:.6e}")
    print(f"noised_gap_mean={noisy_gap_mean:.6e} noised_gap_std={noisy_gap_std:.6e} (samples={nsamp})")


if __name__ == '__main__':
    main()


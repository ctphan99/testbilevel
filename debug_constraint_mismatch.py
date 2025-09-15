#!/usr/bin/env python3
"""
Debug constraint computation mismatch
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm import F2CSAAlgorithm

def debug_constraint_mismatch():
    """Debug why constraint computations don't match"""
    print("=" * 80)
    print("DEBUGGING CONSTRAINT COMPUTATION MISMATCH")
    print("=" * 80)
    
    # Create problem with constraint tightening
    problem = StronglyConvexBilevelProblem(dim=5, num_constraints=3, seed=42)
    
    # Apply constraint tightening
    problem.b = problem.b - 0.2
    problem.B = problem.B * 2.5
    problem.Q_lower = problem.Q_lower * 1.8
    
    print(f"Problem setup:")
    print(f"  A: {problem.A.detach().numpy()}")
    print(f"  B: {problem.B.detach().numpy()}")
    print(f"  b: {problem.b.detach().numpy()}")
    
    # Test x
    x_test = torch.randn(problem.dim, dtype=torch.float64)
    print(f"\nTesting x: {x_test.detach().numpy()}")
    
    # Solve lower level
    y_opt, info = problem.solve_lower_level(x_test)
    dual_vars = info.get('lambda', None)
    
    print(f"Lower-level solution y*: {y_opt.detach().numpy()}")
    print(f"Dual variables λ*: {dual_vars.detach().numpy() if dual_vars is not None else 'None'}")
    
    # Compute constraint values
    h_val = problem.A @ x_test - problem.B @ y_opt - problem.b
    print(f"Constraint values h = A@x - B@y - b: {h_val.detach().numpy()}")
    print(f"Max violation: {torch.max(h_val).item():.6f}")
    print(f"Min violation: {torch.min(h_val).item():.6f}")
    
    # Check if constraints are satisfied
    feasible = torch.all(h_val <= 1e-6)
    print(f"Constraints satisfied: {feasible}")
    
    # Now test with F2CSA algorithm
    print(f"\n" + "=" * 50)
    print("TESTING WITH F2CSA ALGORITHM")
    print("=" * 50)
    
    f2csa = F2CSAAlgorithm(
        problem=problem,
        alpha_override=0.08,
        eta_override=0.001,
        D_override=0.01,
        Ng_override=64,
        grad_ema_beta_override=0.9,
        prox_weight_override=0.1,
        grad_clip_override=1.0
    )
    
    # Test oracle_sample
    try:
        sample = f2csa.oracle_sample(x_test, 0.08, 1)
        print(f"F2CSA hypergradient: {sample.detach().numpy()}")
    except Exception as e:
        print(f"Error in F2CSA oracle_sample: {e}")
        
        # Debug the oracle_sample manually
        print(f"\nDebugging oracle_sample manually...")
        
        xx = x_test.detach().clone().requires_grad_(True)
        
        # Use CVXPY solver
        y_opt_f2csa, info_f2csa = problem.solve_lower_level(xx)
        lambda_opt_f2csa = info_f2csa.get('lambda', torch.zeros(problem.num_constraints, device=problem.device, dtype=problem.dtype))
        
        print(f"F2CSA y*: {y_opt_f2csa.detach().numpy()}")
        print(f"F2CSA λ*: {lambda_opt_f2csa.detach().numpy()}")
        
        # Compute constraint violations
        h_val_f2csa = problem.A @ xx - problem.B @ y_opt_f2csa - problem.b
        print(f"F2CSA constraint values h: {h_val_f2csa.detach().numpy()}")
        print(f"F2CSA max violation: {torch.max(h_val_f2csa).item():.6f}")
        print(f"F2CSA min violation: {torch.min(h_val_f2csa).item():.6f}")
        
        # Check if they match
        print(f"\nComparison:")
        print(f"Direct y*: {y_opt.detach().numpy()}")
        print(f"F2CSA y*:  {y_opt_f2csa.detach().numpy()}")
        print(f"Difference: {torch.norm(y_opt - y_opt_f2csa).item():.6f}")
        
        print(f"Direct h: {h_val.detach().numpy()}")
        print(f"F2CSA h:  {h_val_f2csa.detach().numpy()}")
        print(f"Difference: {torch.norm(h_val - h_val_f2csa).item():.6f}")

if __name__ == "__main__":
    debug_constraint_mismatch()

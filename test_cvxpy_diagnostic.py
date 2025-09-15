#!/usr/bin/env python3
"""
Diagnostic to check why CVXPY solver is not finding feasible solutions
"""

import torch
import numpy as np
from problem import StronglyConvexBilevelProblem
import warnings

warnings.filterwarnings('ignore')

def diagnose_cvxpy_issue():
    """Check why CVXPY is not finding feasible solutions"""
    print("🔍 CVXPY DIAGNOSTIC: Why are constraints not satisfied?")
    print("=" * 60)
    
    # Create problem
    problem = StronglyConvexBilevelProblem(
        dim=5, num_constraints=3, noise_std=0.1, 
        strong_convex=True, device='cpu'
    )
    
    # Test with a specific point
    x = torch.randn(5, dtype=torch.float64)
    alpha = 0.05
    
    print(f"Testing with x = {x}")
    print(f"α = {alpha}")
    print()
    
    # 1. Check the constraint matrix
    print("1️⃣ Constraint Matrix Analysis")
    print(f"   A shape: {problem.A.shape}")
    print(f"   B shape: {problem.B.shape}")
    print(f"   b shape: {problem.b.shape}")
    print(f"   A = {problem.A}")
    print(f"   B = {problem.B}")
    print(f"   b = {problem.b}")
    print()
    
    # 2. Check constraint definition
    print("2️⃣ Constraint Definition Check")
    print("   Constraints: A*x + B*y - b ≤ 0")
    print(f"   A*x = {torch.mv(problem.A, x)}")
    print(f"   b = {problem.b}")
    print(f"   A*x - b = {torch.mv(problem.A, x) - problem.b}")
    print()
    
    # 3. Try to solve with different tolerances
    print("3️⃣ CVXPY Solver with Different Tolerances")
    
    for tol in [1e-6, 1e-8, 1e-10, 1e-12]:
        print(f"   Testing with tolerance = {tol}")
        try:
            y_star, info = problem.solve_lower_level(x, 'accurate', 1000, tol, alpha)
            h_val = problem.constraints(x, y_star)
            print(f"     y* = {y_star}")
            print(f"     h(x,y*) = {h_val}")
            print(f"     ||h(x,y*)|| = {torch.norm(h_val).item():.8f}")
            print(f"     Status: {info.get('status', 'unknown')}")
            print(f"     Max violation: {info.get('max_violation', 'N/A')}")
            print()
        except Exception as e:
            print(f"     Error: {e}")
            print()
    
    # 4. Check if the problem is actually feasible
    print("4️⃣ Feasibility Check")
    print("   Checking if there exists y such that A*x + B*y - b ≤ 0")
    
    # Try to find a feasible point by solving: min ||A*x + B*y - b||²
    y_feasible = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([y_feasible], lr=0.01)
    
    for iteration in range(1000):
        optimizer.zero_grad()
        h_val = problem.constraints(x, y_feasible)
        loss = torch.sum(h_val**2)
        loss.backward()
        optimizer.step()
        
        if iteration % 200 == 0:
            print(f"     Iter {iteration}: ||h|| = {torch.norm(h_val).item():.8f}")
    
    h_final = problem.constraints(x, y_feasible)
    print(f"   Final feasible y: {y_feasible.detach()}")
    print(f"   Final h(x,y): {h_final}")
    print(f"   Final ||h(x,y)||: {torch.norm(h_final).item():.8f}")
    print()
    
    # 5. Check the lower-level objective
    print("5️⃣ Lower-Level Objective Analysis")
    f_lower = problem.lower_objective(x, y_feasible.detach())
    print(f"   f(x, y_feasible) = {f_lower:.6f}")
    
    # Try CVXPY with the feasible point as initial guess
    print("6️⃣ CVXPY with Feasible Initial Guess")
    try:
        y_cvxpy, info = problem.solve_lower_level(x, 'accurate', 1000, 1e-8, alpha)
        h_cvxpy = problem.constraints(x, y_cvxpy)
        f_cvxpy = problem.lower_objective(x, y_cvxpy)
        
        print(f"   CVXPY y: {y_cvxpy}")
        print(f"   CVXPY h: {h_cvxpy}")
        print(f"   CVXPY ||h||: {torch.norm(h_cvxpy).item():.8f}")
        print(f"   CVXPY f: {f_cvxpy:.6f}")
        print(f"   Feasible f: {f_lower:.6f}")
        print(f"   Difference: {abs(f_cvxpy - f_lower).item():.6f}")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    diagnose_cvxpy_issue()

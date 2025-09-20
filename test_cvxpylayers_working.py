#!/usr/bin/env python3
"""
Working CVXPYLayers example for Hessian computation
"""

import torch
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

def test_cvxpylayers_working():
    """Test CVXPYLayers with a working DPP-compliant problem"""
    
    print("🧪 Testing CVXPYLayers with Working DPP Problem")
    print("=" * 50)
    
    # Set random seed
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Test parameters
    dim = 5
    print(f"Testing with dimension: {dim}")
    
    try:
        # Create a working DPP-compliant problem
        print("1. Creating DPP-compliant problem...")
        
        # Parameters
        A = cp.Parameter((dim, dim), symmetric=True)  # Symmetric parameter
        b = cp.Parameter(dim)
        
        # Variables
        x = cp.Variable(dim)
        
        # Objective: 0.5 * x^T * A * x + b^T * x
        objective = cp.Minimize(0.5 * cp.quad_form(x, A) + b.T @ x)
        
        # Constraints: box constraints -1 <= x <= 1
        constraints = [x >= -1, x <= 1]
        
        # Create the problem
        problem = cp.Problem(objective, constraints)
        
        # Create CVXPYLayer
        layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        
        print("   ✓ CVXPYLayer created successfully")
        
        # Test with sample data
        A_torch = torch.eye(dim, dtype=torch.float64, requires_grad=True)
        b_torch = torch.randn(dim, dtype=torch.float64, requires_grad=True)
        
        print(f"   A_torch norm: {torch.norm(A_torch).item():.6f}")
        print(f"   b_torch norm: {torch.norm(b_torch).item():.6f}")
        
        # Forward pass
        print(f"\n2. Testing forward pass...")
        
        solution, = layer(A_torch, b_torch)
        print(f"   ✓ Forward pass successful")
        print(f"   Solution norm: {torch.norm(solution).item():.6f}")
        
        # Test gradient computation
        print(f"\n3. Testing gradient computation...")
        
        loss = torch.sum(solution)
        loss.backward()
        
        print(f"   ✓ Gradient computation successful")
        print(f"   A gradient norm: {torch.norm(A_torch.grad).item():.6f}")
        print(f"   b gradient norm: {torch.norm(b_torch.grad).item():.6f}")
        
        # Test Hessian computation
        print(f"\n4. Testing Hessian computation...")
        
        # Reset gradients
        A_torch.grad = None
        b_torch.grad = None
        
        # Compute Hessian using torch.autograd.functional.hessian
        def loss_fn(A, b):
            sol, = layer(A, b)
            return torch.sum(sol)
        
        try:
            # Compute Hessian with respect to A
            hessian_A = torch.autograd.functional.hessian(
                lambda A: loss_fn(A, b_torch.detach()), 
                A_torch.detach()
            )
            print(f"   ✓ Hessian w.r.t. A computed")
            print(f"   Hessian A shape: {hessian_A.shape}")
            print(f"   Hessian A norm: {torch.norm(hessian_A).item():.6f}")
            
            # Compute Hessian with respect to b
            hessian_b = torch.autograd.functional.hessian(
                lambda b: loss_fn(A_torch.detach(), b), 
                b_torch.detach()
            )
            print(f"   ✓ Hessian w.r.t. b computed")
            print(f"   Hessian b shape: {hessian_b.shape}")
            print(f"   Hessian b norm: {torch.norm(hessian_b).item():.6f}")
            
        except Exception as e:
            print(f"   ❌ Hessian computation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test with different solvers
        print(f"\n5. Testing different solvers...")
        
        solvers_to_test = ["Clarabel", "SCS"]
        
        for solver_name in solvers_to_test:
            try:
                # Reset gradients
                A_torch.grad = None
                b_torch.grad = None
                
                # Solve with specific solver
                solution_solver, = layer(A_torch, b_torch, 
                                        solver_args={"solve_method": solver_name})
                
                print(f"   ✓ {solver_name}: solution norm = {torch.norm(solution_solver).item():.6f}")
                
                # Test gradient
                loss_solver = torch.sum(solution_solver)
                loss_solver.backward()
                
                print(f"   ✓ {solver_name}: gradient norm = {torch.norm(A_torch.grad).item():.6f}")
                
            except Exception as e:
                print(f"   ❌ {solver_name} failed: {e}")
        
        # Compare with direct CVXPY
        print(f"\n6. Comparing with direct CVXPY...")
        
        # Direct CVXPY solution
        x_direct = cp.Variable(dim)
        A_direct = A_torch.detach().cpu().numpy()
        b_direct = b_torch.detach().cpu().numpy()
        
        objective_direct = cp.Minimize(0.5 * cp.quad_form(x_direct, A_direct) + 
                                      cp.sum(cp.multiply(b_direct, x_direct)))
        constraints_direct = [x_direct >= -1, x_direct <= 1]
        problem_direct = cp.Problem(objective_direct, constraints_direct)
        problem_direct.solve(solver=cp.CLARABEL, verbose=False)
        
        if problem_direct.status == cp.OPTIMAL:
            x_direct_sol = torch.tensor(x_direct.value, dtype=torch.float64, device='cpu')
            print(f"   Direct CVXPY solution norm: {torch.norm(x_direct_sol).item():.6f}")
            
            # Compare solutions
            diff = torch.norm(solution - x_direct_sol).item()
            print(f"   Difference from direct CVXPY: {diff:.6f}")
            
            if diff < 1e-6:
                print(f"   ✅ CVXPYLayers matches direct CVXPY very well")
            elif diff < 1e-4:
                print(f"   ✅ CVXPYLayers matches direct CVXPY well")
            else:
                print(f"   ⚠️  CVXPYLayers differs from direct CVXPY")
        
        print("\n" + "=" * 50)
        print("✅ CVXPYLayers working test completed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cvxpylayers_working()

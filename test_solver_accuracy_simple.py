#!/usr/bin/env python3
"""
Simple test of solver accuracy for SSIGD
"""

import torch
import numpy as np
import cvxpy as cp
from problem import StronglyConvexBilevelProblem
from ssigd_correct_final import CorrectSSIGD

def test_solver_accuracy():
    """Test solver accuracy for SSIGD"""
    
    print("🧪 Testing Solver Accuracy for SSIGD")
    print("=" * 50)
    
    # Set random seed
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Test parameters
    dim = 20
    print(f"Testing with dimension: {dim}")
    
    try:
        # Create problem
        problem = StronglyConvexBilevelProblem(dim=dim, device='cpu')
        x_test = torch.randn(dim, dtype=torch.float64) * 0.1
        q_test = torch.randn(dim, dtype=torch.float64) * 0.01
        
        print(f"Test x norm: {torch.norm(x_test).item():.6f}")
        print(f"Test q norm: {torch.norm(q_test).item():.6f}")
        print()
        
        # Test different solvers
        print("1. Testing different solvers...")
        
        y = cp.Variable(dim)
        c_modified = problem.c_lower.cpu().numpy() + q_test.cpu().numpy()
        objective = cp.Minimize(0.5 * cp.quad_form(y, problem.Q_lower.cpu().numpy()) + 
                               cp.sum(cp.multiply(c_modified, y)))
        constraints = [y >= -1, y <= 1]
        problem_cp = cp.Problem(objective, constraints)
        
        solvers_to_test = [
            ("Clarabel", cp.CLARABEL),
            ("SCS", cp.SCS),
            ("GUROBI", cp.GUROBI)
        ]
        
        results = {}
        
        for solver_name, solver in solvers_to_test:
            try:
                problem_cp.solve(solver=solver, verbose=False)
                if problem_cp.status == cp.OPTIMAL:
                    y_sol = torch.tensor(y.value, dtype=torch.float64, device='cpu')
                    results[solver_name] = y_sol
                    print(f"   ✓ {solver_name}: norm={torch.norm(y_sol).item():.6f}, status={problem_cp.status}")
                else:
                    print(f"   ❌ {solver_name}: status={problem_cp.status}")
            except Exception as e:
                print(f"   ❌ {solver_name}: {e}")
        
        # Compare results
        print(f"\n2. Accuracy Comparison:")
        print("-" * 50)
        print(f"{'Solver':<10} {'Solution Norm':<15} {'Difference from Clarabel':<25}")
        print("-" * 50)
        
        if "Clarabel" in results:
            clarabel_norm = torch.norm(results["Clarabel"]).item()
            print(f"{'Clarabel':<10} {clarabel_norm:<15.6f} {'0.000000':<25}")
            
            for solver_name, y_sol in results.items():
                if solver_name != "Clarabel":
                    diff = torch.norm(results["Clarabel"] - y_sol).item()
                    print(f"{solver_name:<10} {torch.norm(y_sol).item():<15.6f} {diff:<25.6f}")
        
        # Test SSIGD accuracy
        print(f"\n3. SSIGD Accuracy Test:")
        print("-" * 30)
        
        try:
            ssigd = CorrectSSIGD(problem)
            
            # Test solve_ll_with_q
            y_ssigd = ssigd.solve_ll_with_q(x_test, q_test)
            print(f"   SSIGD solve_ll_with_q norm: {torch.norm(y_ssigd).item():.6f}")
            
            # Compare with different solvers
            if "Clarabel" in results:
                clarabel_diff = torch.norm(y_ssigd - results["Clarabel"]).item()
                print(f"   Difference from Clarabel: {clarabel_diff:.6f}")
                
                if clarabel_diff < 1e-6:
                    print(f"   ✅ SSIGD matches Clarabel very well")
                elif clarabel_diff < 1e-4:
                    print(f"   ✅ SSIGD matches Clarabel well")
                else:
                    print(f"   ⚠️  SSIGD differs from Clarabel")
            
            if "GUROBI" in results:
                gurobi_diff = torch.norm(y_ssigd - results["GUROBI"]).item()
                print(f"   Difference from GUROBI: {gurobi_diff:.6f}")
                
                if gurobi_diff < 1e-6:
                    print(f"   ✅ SSIGD matches GUROBI very well")
                elif gurobi_diff < 1e-4:
                    print(f"   ✅ SSIGD matches GUROBI well")
                else:
                    print(f"   ⚠️  SSIGD differs from GUROBI")
            
            # Test gradient accuracy
            print(f"\n4. Gradient Accuracy Test:")
            print("-" * 30)
            
            # Test with different solvers
            x_test_grad = x_test.clone().requires_grad_(True)
            q_test_grad = q_test.clone().requires_grad_(True)
            
            # Manual gradient computation with Clarabel
            y_manual = cp.Variable(dim)
            c_modified_grad = problem.c_lower.cpu().numpy() + q_test_grad.detach().cpu().numpy()
            objective_grad = cp.Minimize(0.5 * cp.quad_form(y_manual, problem.Q_lower.cpu().numpy()) + 
                                        cp.sum(cp.multiply(c_modified_grad, y_manual)))
            constraints_grad = [y_manual >= -1, y_manual <= 1]
            problem_grad = cp.Problem(objective_grad, constraints_grad)
            problem_grad.solve(solver=cp.CLARABEL, verbose=False)
            
            if problem_grad.status == cp.OPTIMAL:
                y_manual_sol = torch.tensor(y_manual.value, dtype=torch.float64, device='cpu')
                loss_manual = torch.sum(y_manual_sol)
                loss_manual.backward()
                grad_manual = q_test_grad.grad
                
                # SSIGD gradient
                x_test_ssigd = x_test.clone().requires_grad_(True)
                q_test_ssigd = q_test.clone().requires_grad_(True)
                
                y_ssigd_grad = ssigd.solve_ll_with_q(x_test_ssigd, q_test_ssigd)
                loss_ssigd = torch.sum(y_ssigd_grad)
                loss_ssigd.backward()
                grad_ssigd = q_test_ssigd.grad
                
                grad_diff = torch.norm(grad_manual - grad_ssigd).item()
                print(f"   Manual gradient norm: {torch.norm(grad_manual).item():.6f}")
                print(f"   SSIGD gradient norm: {torch.norm(grad_ssigd).item():.6f}")
                print(f"   Gradient difference: {grad_diff:.6f}")
                
                if grad_diff < 1e-6:
                    print(f"   ✅ Gradients are very close (diff < 1e-6)")
                elif grad_diff < 1e-4:
                    print(f"   ✅ Gradients are close (diff < 1e-4)")
                else:
                    print(f"   ⚠️  Gradients differ significantly (diff >= 1e-4)")
            
        except Exception as e:
            print(f"   ❌ SSIGD test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 50)
        print("✅ Solver accuracy test completed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_solver_accuracy()

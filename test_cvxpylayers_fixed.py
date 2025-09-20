#!/usr/bin/env python3
"""
Test CVXPYLayers accuracy vs direct CVXPY (fixed parameter issue)
"""

import torch
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from problem import StronglyConvexBilevelProblem
from ssigd_correct_final import CorrectSSIGD

def test_cvxpylayers_fixed():
    """Test CVXPYLayers accuracy vs direct CVXPY with fixed parameters"""
    
    print("🧪 Testing CVXPYLayers Accuracy vs Direct CVXPY (Fixed)")
    print("=" * 60)
    
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
        
        # Method 1: Direct CVXPY (current implementation)
        print("1. Testing Direct CVXPY method...")
        
        y = cp.Variable(dim)
        c_modified = problem.c_lower.cpu().numpy() + q_test.cpu().numpy()
        objective = cp.Minimize(0.5 * cp.quad_form(y, problem.Q_lower.cpu().numpy()) + 
                               cp.sum(cp.multiply(c_modified, y)))
        constraints = [y >= -1, y <= 1]
        problem_cp = cp.Problem(objective, constraints)
        
        # Test Clarabel (best for high-dim)
        problem_cp.solve(solver=cp.CLARABEL, verbose=False)
        if problem_cp.status == cp.OPTIMAL:
            y_direct = torch.tensor(y.value, dtype=torch.float64, device='cpu')
            print(f"   ✓ Direct Clarabel: norm={torch.norm(y_direct).item():.6f}")
        else:
            print(f"   ❌ Direct Clarabel failed: {problem_cp.status}")
            return
        
        # Method 2: CVXPYLayers (fixed parameter issue)
        print(f"\n2. Testing CVXPYLayers method...")
        
        # Create CVXPYLayers problem with correct parameter structure
        y_cp = cp.Variable(dim)
        q_param = cp.Parameter(dim)  # Only q as parameter, not x
        
        # Objective with q parameter
        objective_cp = cp.Minimize(0.5 * cp.quad_form(y_cp, problem.Q_lower.cpu().numpy()) + 
                                  cp.sum(cp.multiply(problem.c_lower.cpu().numpy() + q_param, y_cp)))
        constraints_cp = [y_cp >= -1, y_cp <= 1]
        problem_cp_layer = cp.Problem(objective_cp, constraints_cp)
        
        try:
            # Create layer with correct parameters
            layer = CvxpyLayer(problem_cp_layer, variables=[y_cp], parameters=[q_param])
            
            # Solve with different solvers
            solvers_to_test = ["Clarabel", "SCS"]
            cvxpylayers_results = {}
            
            for solver_name in solvers_to_test:
                try:
                    y_sol = layer(q_test, solver_args={"solve_method": solver_name})
                    cvxpylayers_results[solver_name] = y_sol
                    print(f"   ✓ CVXPYLayers {solver_name}: norm={torch.norm(y_sol).item():.6f}")
                except Exception as e:
                    print(f"   ❌ CVXPYLayers {solver_name}: {e}")
            
            # Compare results
            print(f"\n3. Accuracy Comparison:")
            print("-" * 50)
            print(f"{'Method':<20} {'Solution Norm':<15} {'Difference':<15}")
            print("-" * 50)
            print(f"{'Direct Clarabel':<20} {torch.norm(y_direct).item():<15.6f} {'0.000000':<15}")
            
            for solver_name, y_sol in cvxpylayers_results.items():
                diff = torch.norm(y_direct - y_sol).item()
                print(f"{'CVXPYLayers ' + solver_name:<20} {torch.norm(y_sol).item():<15.6f} {diff:<15.6f}")
            
            # Test gradient accuracy
            print(f"\n4. Gradient Accuracy Test:")
            print("-" * 40)
            
            # Direct CVXPY gradient (manual)
            q_test_grad = q_test.clone().requires_grad_(True)
            
            # Manual gradient computation
            y_manual = cp.Variable(dim)
            c_modified_grad = problem.c_lower.cpu().numpy() + q_test_grad.cpu().numpy()
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
                
                # CVXPYLayers gradient
                q_test_layer = q_test.clone().requires_grad_(True)
                
                y_layer_sol = layer(q_test_layer, solver_args={"solve_method": "Clarabel"})
                loss_layer = torch.sum(y_layer_sol)
                loss_layer.backward()
                grad_layer = q_test_layer.grad
                
                grad_diff = torch.norm(grad_manual - grad_layer).item()
                print(f"   Manual gradient norm: {torch.norm(grad_manual).item():.6f}")
                print(f"   Layer gradient norm: {torch.norm(grad_layer).item():.6f}")
                print(f"   Gradient difference: {grad_diff:.6f}")
                
                if grad_diff < 1e-6:
                    print(f"   ✅ Gradients are very close (diff < 1e-6)")
                elif grad_diff < 1e-4:
                    print(f"   ✅ Gradients are close (diff < 1e-4)")
                else:
                    print(f"   ⚠️  Gradients differ significantly (diff >= 1e-4)")
            
            # Test with SSIGD
            print(f"\n5. SSIGD Integration Test:")
            print("-" * 30)
            
            try:
                ssigd = CorrectSSIGD(problem)
                
                # Test solve_ll_with_q
                y_ssigd = ssigd.solve_ll_with_q(x_test, q_test)
                print(f"   SSIGD solve_ll_with_q norm: {torch.norm(y_ssigd).item():.6f}")
                
                # Compare with direct method
                ssigd_diff = torch.norm(y_ssigd - y_direct).item()
                print(f"   Difference from direct Clarabel: {ssigd_diff:.6f}")
                
                if ssigd_diff < 1e-6:
                    print(f"   ✅ SSIGD matches direct method very well")
                elif ssigd_diff < 1e-4:
                    print(f"   ✅ SSIGD matches direct method well")
                else:
                    print(f"   ⚠️  SSIGD differs from direct method")
                
                # Compare with CVXPYLayers
                if "Clarabel" in cvxpylayers_results:
                    layer_diff = torch.norm(y_ssigd - cvxpylayers_results["Clarabel"]).item()
                    print(f"   Difference from CVXPYLayers Clarabel: {layer_diff:.6f}")
                    
                    if layer_diff < 1e-6:
                        print(f"   ✅ SSIGD matches CVXPYLayers very well")
                    elif layer_diff < 1e-4:
                        print(f"   ✅ SSIGD matches CVXPYLayers well")
                    else:
                        print(f"   ⚠️  SSIGD differs from CVXPYLayers")
                
            except Exception as e:
                print(f"   ❌ SSIGD test failed: {e}")
                import traceback
                traceback.print_exc()
        
        except Exception as e:
            print(f"   ❌ CVXPYLayers setup failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("✅ CVXPYLayers accuracy test completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cvxpylayers_fixed()

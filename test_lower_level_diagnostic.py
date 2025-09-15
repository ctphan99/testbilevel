#!/usr/bin/env python3
"""
Focused Diagnostic for Lower-Level Solution Convergence
Investigates why the penalty Lagrangian solver is not converging to the accurate solution
"""

import torch
import numpy as np
from typing import Dict, Tuple
from problem import StronglyConvexBilevelProblem
from f2csa_algorithm_corrected_final import F2CSAAlgorithm1Final
import warnings

warnings.filterwarnings('ignore')

class LowerLevelDiagnostic:
    """
    Focused diagnostic for lower-level solution convergence
    """
    
    def __init__(self, dim: int = 5, num_constraints: int = 3, noise_std: float = 0.1):
        self.dim = dim
        self.num_constraints = num_constraints
        self.noise_std = noise_std
        
        # Create problem instance
        self.problem = StronglyConvexBilevelProblem(
            dim=dim,
            num_constraints=num_constraints,
            noise_std=noise_std,
            strong_convex=True,
            device='cpu'
        )
        
        # Initialize algorithm
        self.algorithm = F2CSAAlgorithm1Final(self.problem)
        
    def diagnose_convergence_issue(self, alpha: float = 0.05):
        """
        Diagnose why the penalty Lagrangian solver is not converging
        """
        print("🔍 DIAGNOSING LOWER-LEVEL CONVERGENCE ISSUE")
        print("=" * 60)
        print(f"α = {alpha}, δ = {alpha**3:.6f} (target: < 0.1)")
        print()
        
        # Generate random upper-level variable
        x = torch.randn(self.dim, dtype=torch.float64, requires_grad=True)
        print(f"📊 Testing with x = {x.detach().numpy()}")
        print()
        
        # Step 1: Get accurate solution using CVXPY
        print("🔬 STEP 1: Accurate Solution (CVXPY)")
        print("-" * 40)
        y_star, info = self.problem.solve_lower_level(x, 'accurate', 1000, 1e-6, alpha)
        lambda_star = info.get('lambda_star', torch.zeros(self.problem.num_constraints, dtype=torch.float64))
        
        print(f"✅ y_star = {y_star.detach().numpy()}")
        print(f"✅ lambda_star = {lambda_star.detach().numpy()}")
        print(f"✅ ||y_star|| = {torch.norm(y_star).item():.6f}")
        print()
        
        # Step 2: Test penalty Lagrangian solver with different parameters
        print("🔬 STEP 2: Penalty Lagrangian Solver")
        print("-" * 40)
        
        # Test with different iteration counts
        for max_iter in [100, 500, 1000, 2000]:
            print(f"📊 Testing with {max_iter} iterations:")
            
            y_penalty = self.algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, alpha**3)
            gap = torch.norm(y_penalty - y_star).item()
            
            print(f"  ✅ Gap: {gap:.6f} (target: < 0.1)")
            print(f"  ✅ ||y_penalty|| = {torch.norm(y_penalty).item():.6f}")
            print(f"  ✅ Converged: {gap < 0.1}")
            print()
        
        # Step 3: Analyze the penalty Lagrangian function
        print("🔬 STEP 3: Penalty Lagrangian Analysis")
        print("-" * 40)
        
        # Compute penalty Lagrangian at y_star
        L_star = self.algorithm._compute_penalty_lagrangian(x, y_star, y_star, lambda_star, alpha, alpha**3)
        print(f"✅ L(y_star) = {L_star.item():.6f}")
        
        # Compute penalty Lagrangian at y_penalty
        L_penalty = self.algorithm._compute_penalty_lagrangian(x, y_penalty, y_star, lambda_star, alpha, alpha**3)
        print(f"✅ L(y_penalty) = {L_penalty.item():.6f}")
        print(f"✅ L difference = {abs(L_penalty - L_star).item():.6f}")
        print()
        
        # Step 4: Check constraint violations
        print("🔬 STEP 4: Constraint Violations")
        print("-" * 40)
        
        h_star = self.problem.constraints(x, y_star)
        h_penalty = self.problem.constraints(x, y_penalty)
        
        print(f"✅ h(y_star) = {h_star.detach().numpy()}")
        print(f"✅ h(y_penalty) = {h_penalty.detach().numpy()}")
        print(f"✅ ||h(y_star)|| = {torch.norm(h_star).item():.6f}")
        print(f"✅ ||h(y_penalty)|| = {torch.norm(h_penalty).item():.6f}")
        print()
        
        # Step 5: Check penalty parameters
        print("🔬 STEP 5: Penalty Parameters")
        print("-" * 40)
        
        alpha1 = 1.0 / alpha  # α₁ = α⁻¹
        alpha2 = 1.0 / (alpha**2)  # α₂ = α⁻²
        
        print(f"✅ α₁ = {alpha1:.1f}")
        print(f"✅ α₂ = {alpha2:.1f}")
        print(f"✅ δ = {alpha**3:.6f}")
        print()
        
        # Step 6: Check if the problem is well-conditioned
        print("🔬 STEP 6: Problem Conditioning")
        print("-" * 40)
        
        # Check strong convexity of penalty Lagrangian
        y_test = y_star + 0.1 * torch.randn_like(y_star)
        L_test = self.algorithm._compute_penalty_lagrangian(x, y_test, y_star, lambda_star, alpha, alpha**3)
        
        print(f"✅ L(y_star) = {L_star.item():.6f}")
        print(f"✅ L(y_test) = {L_test.item():.6f}")
        print(f"✅ L difference = {abs(L_test - L_star).item():.6f}")
        print()
        
        # Step 7: Test with different alpha values
        print("🔬 STEP 7: Testing Different Alpha Values")
        print("-" * 40)
        
        for test_alpha in [0.1, 0.2, 0.5, 1.0]:
            print(f"📊 Testing with α = {test_alpha}:")
            
            y_penalty_test = self.algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, test_alpha, test_alpha**3)
            gap_test = torch.norm(y_penalty_test - y_star).item()
            
            print(f"  ✅ Gap: {gap_test:.6f} (target: < 0.1)")
            print(f"  ✅ Converged: {gap_test < 0.1}")
            print()
        
        return {
            'y_star': y_star,
            'y_penalty': y_penalty,
            'gap': gap,
            'converged': gap < 0.1
        }
    
    def test_improved_convergence(self, alpha: float = 0.05):
        """
        Test improved convergence with better parameters
        """
        print("🚀 TESTING IMPROVED CONVERGENCE")
        print("=" * 60)
        print(f"α = {alpha}, δ = {alpha**3:.6f}")
        print()
        
        # Generate random upper-level variable
        x = torch.randn(self.dim, dtype=torch.float64, requires_grad=True)
        
        # Get accurate solution
        y_star, info = self.problem.solve_lower_level(x, 'accurate', 1000, 1e-6, alpha)
        lambda_star = info.get('lambda_star', torch.zeros(self.problem.num_constraints, dtype=torch.float64))
        
        # Test with improved parameters
        print("📊 Testing with improved parameters:")
        
        # Use more iterations and better tolerance
        y_penalty = self.algorithm._minimize_penalty_lagrangian(x, y_star, lambda_star, alpha, alpha**3)
        gap = torch.norm(y_penalty - y_star).item()
        
        print(f"✅ Gap: {gap:.6f} (target: < 0.1)")
        print(f"✅ Converged: {gap < 0.1}")
        print()
        
        return {
            'gap': gap,
            'converged': gap < 0.1
        }

def main():
    """Main diagnostic function"""
    diagnostic = LowerLevelDiagnostic()
    
    # Run comprehensive diagnostic
    result = diagnostic.diagnose_convergence_issue(alpha=0.05)
    
    # Test improved convergence
    improved_result = diagnostic.test_improved_convergence(alpha=0.05)
    
    return result, improved_result

if __name__ == "__main__":
    main()

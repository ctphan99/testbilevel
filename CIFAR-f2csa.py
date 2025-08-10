#!/usr/bin/env python3
"""
F2CSA Implementation
======================================
- CIFAR-10/100 datasets with ResNet-18
- Epsilon values: 8/255, 16/255
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import time
import json
import argparse
import os


class ResNet18(nn.Module):
    """ResNet-18 for CIFAR-10/100 (-compliant)"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        # CIFAR-10 adapted ResNet-18 (no initial max pooling)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # weight initialization
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class F2CSATrainer:
    """F2CSA (First-order Constrained Stochastic Approximation) Trainer """

    def __init__(self, dataset='cifar10', epsilon=8/255, alpha=0.3, device='cuda'):
        self.dataset = dataset
        self.epsilon = epsilon
        self.alpha = alpha
        self.device = device

        # F2CSA parameters from 
        self.alpha_1 = alpha ** (-2)  # Penalty parameter 1
        self.alpha_2 = alpha ** (-4)  # Penalty parameter 2
        self.delta = alpha ** 3       # Smoothing parameter

        # Rho activation function parameters
        self.tau = self.delta         # τ = Θ(δ) from 
        self.epsilon_lambda = 0.01    # ε_λ > 0 small positive parameter

        # setup
        self.num_classes = 10 if dataset == 'cifar10' else 100
        self.batch_size = 128  # Standard for CIFAR experiments

        # Setup data loaders
        self._setup_data()

        print(f"🔧 F2CSA Trainer (-Compliant)")
        print(f"   Dataset: {dataset.upper()}")
        print(f"   Epsilon: {epsilon:.4f} ({int(epsilon*255)}/255)")
        print(f"   Alpha: {alpha:.3f}")
        print(f"   Alpha_1: {self.alpha_1:.3f}")
        print(f"   Alpha_2: {self.alpha_2:.3f}")
        print(f"   Delta: {self.delta:.6f}")
        print(f"   Tau: {self.tau:.6f}")
        print(f"   Epsilon_λ: {self.epsilon_lambda:.3f}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Device: {device}")
    
    def _setup_data(self):
        """Setup CIFAR-10/100 data loaders (-compliant)"""
        # Standard CIFAR transforms (no normalization for adversarial training)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        if self.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        else:  # cifar100
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        
        self.train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, 
                                     num_workers=2, pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, 
                                    num_workers=2, pin_memory=True)
        
        print(f"   Training samples: {len(trainset)}")
        print(f"   Test samples: {len(testset)}")

    def sigma_h(self, z):
        """
        Smooth constraint activation function σ_h(z) from F2CSA 

        σ_h(z) = {
            0                    if z < -τδ
            (τδ + z)/(τδ)       if -τδ ≤ z < 0
            1                    if z ≥ 0
        }
        """
        tau_delta = self.tau * self.delta

        # Vectorized implementation for batch processing
        result = torch.zeros_like(z)

        # Case 1: z < -τδ → σ_h(z) = 0 (already initialized)

        # Case 2: -τδ ≤ z < 0 → σ_h(z) = (τδ + z)/(τδ)
        mask_middle = (z >= -tau_delta) & (z < 0)
        result[mask_middle] = (tau_delta + z[mask_middle]) / tau_delta

        # Case 3: z ≥ 0 → σ_h(z) = 1
        mask_active = z >= 0
        result[mask_active] = 1.0

        return result

    def sigma_lambda(self, z):
        """
        Smooth dual variable gating function σ_λ(z) from F2CSA 

        σ_λ(z) = {
            0                    if z ≤ 0
            z/ε_λ               if 0 < z < ε_λ
            1                    if z ≥ ε_λ
        }
        """
        # Vectorized implementation for batch processing
        result = torch.zeros_like(z)

        # Case 1: z ≤ 0 → σ_λ(z) = 0 (already initialized)

        # Case 2: 0 < z < ε_λ → σ_λ(z) = z/ε_λ
        mask_middle = (z > 0) & (z < self.epsilon_lambda)
        result[mask_middle] = z[mask_middle] / self.epsilon_lambda

        # Case 3: z ≥ ε_λ → σ_λ(z) = 1
        mask_active = z >= self.epsilon_lambda
        result[mask_active] = 1.0

        return result

    def rho_i(self, h_values, lambda_values):
        """
        Rho activation function ρ_i(x) from F2CSA 

        ρ_i(x) = σ_h(h_i(x, ỹ*(x))) · σ_λ(λ̃*_i(x))

        Args:
            h_values: Constraint values h_i(x, ỹ*(x))
            lambda_values: Dual variable values λ̃*_i(x)

        Returns:
            ρ_i(x) activation values
        """
        sigma_h_vals = self.sigma_h(h_values)
        sigma_lambda_vals = self.sigma_lambda(lambda_values)

        return sigma_h_vals * sigma_lambda_vals

    def pgd_attack(self, model, images, labels, eps, steps=10, alpha=None):
        """PGD attack for adversarial example generation - FIXED gradient handling"""
        if alpha is None:
            alpha = eps / 4

        # Store original states
        original_mode = model.training
        original_grad_states = {}
        for name, param in model.named_parameters():
            original_grad_states[name] = param.requires_grad

        # Set model to eval mode but keep gradients enabled
        model.eval()
        for param in model.parameters():
            param.requires_grad_(True)

        # Initialize adversarial images
        adv_images = images.clone().detach()

        # Add random initialization
        noise = torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = adv_images + noise
        adv_images = torch.clamp(adv_images, 0, 1)

        for step in range(steps):
            # Create a fresh copy that requires gradients
            adv_images_var = adv_images.clone().detach().requires_grad_(True)

            # Forward pass
            outputs = model(adv_images_var)
            loss = F.cross_entropy(outputs, labels)

            # Compute gradients w.r.t. adversarial images
            grad = torch.autograd.grad(loss, adv_images_var,
                                     retain_graph=False, create_graph=False)[0]

            # Update adversarial images
            with torch.no_grad():
                adv_images = adv_images + alpha * grad.sign()

                # Project back to epsilon ball
                delta = torch.clamp(adv_images - images, -eps, eps)
                adv_images = torch.clamp(images + delta, 0, 1)

        # Restore original states
        model.train(original_mode)
        for name, param in model.named_parameters():
            param.requires_grad_(original_grad_states[name])

        return adv_images.detach()
    
    def compute_f2csa_penalty_lagrangian(self, model, images, labels):
        """
        Compute F2CSA penalty lagrangian with rho activation -  compliant

        L_λ,α(x,y) = f(x,y) + α₁[g(x,y) + λᵀh(x,y) - g(x,ỹ*(x))] + (α₂/2)Σ ρᵢ(x)·hᵢ(x,y)²
        """
        # Ensure input tensors require gradients
        images = images.clone().detach().requires_grad_(True)

        # Solve lower-level problems (generate adversarial examples)
        y_star = self.pgd_attack(model, images, labels, self.epsilon, steps=10)
        y_tilde = self.pgd_attack(model, images, labels, self.epsilon, steps=10)

        # Ensure adversarial perturbations require gradients
        y_star = y_star.clone().detach().requires_grad_(True)
        y_tilde = y_tilde.clone().detach().requires_grad_(True)

        # Upper-level objective f(x, y_tilde)
        outputs_tilde = model(y_tilde)
        f_xy = F.cross_entropy(outputs_tilde, labels)

        # Lower-level objectives
        g_xy = -f_xy  # g(x, y_tilde) = -f(x, y_tilde)
        outputs_star = model(y_star)
        g_x_ystar = -F.cross_entropy(outputs_star, labels)  # g(x, y_star)

        # Constraint functions h_i(x, y) = ||y - x||_∞ - ε (per sample)
        delta_tilde = y_tilde - images
        delta_star = y_star - images

        # Compute constraint values for each sample
        delta_tilde_flat = delta_tilde.view(delta_tilde.size(0), -1)
        delta_star_flat = delta_star.view(delta_star.size(0), -1)

        # h_i(x, y_tilde) and h_i(x, y_star) for each sample i
        h_tilde = torch.max(torch.abs(delta_tilde_flat), dim=1)[0] - self.epsilon
        h_star = torch.max(torch.abs(delta_star_flat), dim=1)[0] - self.epsilon

        # Approximate dual variables λ̃*_i(x) (simplified for adversarial training)
        # In practice, these would be computed from KKT conditions
        lambda_tilde = torch.clamp(h_star, min=0)  # Simplified dual approximation

        # Compute rho activation ρ_i(x) = σ_h(h_i(x, ỹ*(x))) · σ_λ(λ̃*_i(x))
        rho_values = self.rho_i(h_star, lambda_tilde)

        # F2CSA penalty terms with rho activation
        # Linear penalty: α₁ * (g(x,y) + λᵀh(x,y) - g(x,y*))
        linear_penalty = self.alpha_1 * (g_xy + torch.sum(lambda_tilde * h_tilde) - g_x_ystar)

        # Quadratic penalty with rho weighting: (α₂/2) * Σ ρᵢ(x) · hᵢ(x,y)²
        weighted_violations = rho_values * (h_tilde ** 2)
        quadratic_penalty = (self.alpha_2 / 2) * torch.sum(weighted_violations)

        # F2CSA penalty lagrangian (formulation)
        penalty_lagrangian = f_xy + linear_penalty + quadratic_penalty

        return penalty_lagrangian
    
    def evaluate_performance(self, model, test_batches=None):
        """Evaluate standard accuracy (SA) and robust accuracy (RA) -  metrics"""
        model.eval()
        clean_correct = 0
        robust_correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.test_loader):
            if test_batches and batch_idx >= test_batches:
                break

            images, labels = images.to(self.device), labels.to(self.device)

            # Standard Accuracy (SA) - can use no_grad for inference
            with torch.no_grad():
                clean_outputs = model(images)
                clean_pred = clean_outputs.argmax(dim=1)
                clean_correct += (clean_pred == labels).sum().item()

            # Robust Accuracy (RA) - PGD-20 attack (needs gradients)
            adv_images = self.pgd_attack(model, images, labels, self.epsilon, steps=20)

            # Inference on adversarial images (can use no_grad)
            with torch.no_grad():
                robust_outputs = model(adv_images)
                robust_pred = robust_outputs.argmax(dim=1)
                robust_correct += (robust_pred == labels).sum().item()

            total += labels.size(0)

        sa = 100.0 * clean_correct / total
        ra = 100.0 * robust_correct / total

        return sa, ra
    
    def train_f2csa(self, epochs=100, learning_rate=1e-3):
        """Train using F2CSA algorithm ( implementation)"""
        print(f"\n F2CSA Training (-Compliant)")
        print(f"   Epochs: {epochs}")
        print(f"   Learning Rate: {learning_rate:.1e}")
        print(f"   Alpha: {self.alpha:.3f}")
        print(f"   Target RA: ~51.79% (CIFAR-10, ε=8/255)")
        print("-" * 60)
        
        # Initialize model and optimizer
        model = ResNet18(num_classes=self.num_classes).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        
        # Learning rate scheduler ( setup)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90], gamma=0.1)
        
        # Training history
        history = []
        best_ra = 0.0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            epoch_start = time.time()
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Ensure model parameters require gradients
                for param in model.parameters():
                    param.requires_grad_(True)

                # DS-BLO training step
                optimizer.zero_grad()

                try:
                    loss = self.compute_f2csa_penalty_lagrangian(model, images, labels)

                    # Check if loss requires gradients
                    if not loss.requires_grad:
                        print(f"    Loss doesn't require gradients at epoch {epoch}, batch {batch_idx}")
                        continue

                    loss.backward()

                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

                except Exception as e:
                    print(f"    Error at epoch {epoch}, batch {batch_idx}: {e}")
                    print(f"      Error type: {type(e).__name__}")
                    continue
            
            scheduler.step()
            
            avg_loss = epoch_loss / max(batch_count, 1)
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            
            # Evaluation every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                sa, ra = self.evaluate_performance(model, test_batches=50)
                
                if ra > best_ra:
                    best_ra = ra
                
                history.append({
                    'epoch': epoch,
                    'loss': avg_loss,
                    'standard_accuracy': sa,
                    'robust_accuracy': ra,
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time
                })
                
                print(f"   Epoch {epoch:3d}: Loss={avg_loss:.4f}, SA={sa:.2f}%, "
                      f"RA={ra:.2f}%, LR={current_lr:.1e}, Time={epoch_time:.1f}s")
                
                #  target check
                if ra >= 51.0:
                    print(f"     target achieved: {ra:.2f}% RA!")
            else:
                print(f"   Epoch {epoch:3d}: Loss={avg_loss:.4f}, LR={current_lr:.1e}, Time={epoch_time:.1f}s")
        
        total_time = time.time() - start_time
        
        # Final evaluation
        print(f"\n Final Evaluation...")
        final_sa, final_ra = self.evaluate_performance(model)
        
        result = {
            'dataset': self.dataset,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'alpha_1': self.alpha_1,
            'alpha_2': self.alpha_2,
            'delta': self.delta,
            'epochs_completed': epoch + 1,
            'final_standard_accuracy': final_sa,
            'final_robust_accuracy': final_ra,
            'best_robust_accuracy': best_ra,
            'total_training_time': total_time,
            '_target_achieved': final_ra >= 51.0,  #  benchmark
            'training_history': history,
            '_comparison': {
                'target_ra_cifar10_8_255': 51.79,
                'achieved_ra': final_ra,
                'performance_ratio': final_ra / 51.79 if final_ra > 0 else 0
            }
        }
        
        return result


def main():
    """Main function with experimental setup"""
    parser = argparse.ArgumentParser(description='F2CSA Implementation')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10',
                       help='Dataset (default: cifar10)')
    parser.add_argument('--epsilon', type=float, default=8/255,
                       help='Attack budget (default: 8/255)')
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='F2CSA alpha parameter (default: 0.3)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--device', default='cuda',
                       help='Device (default: cuda)')

    args = parser.parse_args()

    # Display F2CSA parameter calculations
    alpha_1 = args.alpha ** (-2)
    alpha_2 = args.alpha ** (-4)
    delta = args.alpha ** 3

    print(" F2CSA Parameters:")
    print(f"   Alpha (α): {args.alpha}")
    print(f"   Alpha_1 (α₁): α⁻² = {args.alpha}⁻² = {alpha_1:.6f}")
    print(f"   Alpha_2 (α₂): α⁻⁴ = {args.alpha}⁻⁴ = {alpha_2:.6f}")
    print(f"   Delta (δ): α³ = {args.alpha}³ = {delta:.6f}")
    print("")

    # Initialize trainer
    trainer = F2CSATrainer(
        dataset=args.dataset,
        epsilon=args.epsilon,
        alpha=args.alpha,
        device=args.device
    )

    # Train F2CSA
    result = trainer.train_f2csa(
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Results

    print(f"\nPerformance Metrics:")
    print(f"   Standard Accuracy (SA): {result['final_standard_accuracy']:.2f}%")
    print(f"   Robust Accuracy (RA): {result['final_robust_accuracy']:.2f}%")
    print(f"   Best RA: {result['best_robust_accuracy']:.2f}%")
    print(f"   Training Time: {result['total_training_time']:.1f}s")
    
    print(f"\n Comparison:")
    print(f"   Achieved RA: {result['_comparison']['achieved_ra']:.2f}%")
    print(f"   Performance Ratio: {result['_comparison']['performance_ratio']:.3f}")
    
    # Save results with SLURM-compatible filename format
    # Convert epsilon to fraction format (8/255 -> 8255, 16/255 -> 16255)
    epsilon_255 = int(round(args.epsilon * 255))
    if epsilon_255 == 8:
        epsilon_label = "8255"
    elif epsilon_255 == 16:
        epsilon_label = "16255"
    else:
        # Fallback for other epsilon values
        epsilon_label = f"{epsilon_255}255"

    # Add command-line arguments to results for traceability
    result['command_line_args'] = {
        'dataset': args.dataset,
        'epsilon': args.epsilon,
        'epsilon_fraction': f"{int(round(args.epsilon * 255))}/255",
        'alpha': args.alpha,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'device': args.device
    }

    result_filename = f"f2csa_results_{args.dataset}_eps{epsilon_label}_alpha{args.alpha}.json"
    with open(result_filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n Results saved to: {result_filename}")
    print(f"   Format: f2csa_results_<dataset>_eps<epsilon_label>_alpha<alpha>.json")

    if result['_target_achieved']:
        print(f"   F2CSA RA: {result['final_robust_accuracy']:.2f}% ≥ 51.0%")
        print(f"   Alpha parameters: α₁={result['alpha_1']:.6f}, α₂={result['alpha_2']:.6f}, δ={result['delta']:.6f}")
    else:
        print(f"   Current RA: {result['final_robust_accuracy']:.2f}%")
        print(f"   Alpha parameters: α₁={result['alpha_1']:.6f}, α₂={result['alpha_2']:.6f}, δ={result['delta']:.6f}")


if __name__ == "__main__":
    main()

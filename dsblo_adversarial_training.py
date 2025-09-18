#!/usr/bin/env python3
"""
DS-BLO Adversarial Training Implementation
Based on the DS-BLO paper for bilevel optimization with linear constraints

Implements adversarial training as:
- Upper level: min_x F(x) = E[f(φ(x; c_i + y_i*(x)), d_i)]  (classification loss)
- Lower level: min_y g(φ(x; c_i + y_i), d_i) + q^T y subject to ||y||_∞ ≤ ε

Uses CIFAR-10/100 datasets with ResNet-18 model as in the paper experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
from typing import Dict, Tuple, Optional, List
import time
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResNet18(nn.Module):
    """ResNet-18 model for CIFAR experiments"""
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        # Modify for CIFAR (32x32 images, 3 channels)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool for CIFAR
        self.model.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.model(x)

class DSBLOAdversarialTraining:
    """
    DS-BLO implementation for adversarial training with linear constraints
    """
    
    def __init__(self, model, epsilon=8/255, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.epsilon = epsilon  # Attack budget (L∞ constraint)
        
        # DS-BLO hyperparameters
        self.gamma1 = 1.0
        self.gamma2 = 1.0
        self.beta = 0.8  # Momentum parameter
        self.grad_clip = 5.0
        self.eta_cap = 1e-3  # Step size cap
        
        # Perturbation parameters
        self.perturbation_std = 1e-4  # Standard deviation for q perturbation
        self.inner_steps = 10  # Steps for solving lower-level problem
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.robust_accs = []
        self.clean_accs = []
        
    def lower_level_objective(self, x, y, q, inputs, targets):
        """
        Lower-level objective: g(x,y) + q^T y
        where g is the adversarial loss and q is the perturbation
        """
        # Adversarial loss (cross-entropy on perturbed inputs)
        perturbed_inputs = inputs + y
        outputs = self.model(perturbed_inputs)
        adv_loss = F.cross_entropy(outputs, targets)
        
        # Add perturbation term q^T y
        perturbation_term = torch.dot(q.flatten(), y.flatten())
        
        return adv_loss + perturbation_term
    
    def upper_level_objective(self, x, y_star, inputs, targets):
        """
        Upper-level objective: f(x, y*(x)) - classification loss on adversarial examples
        """
        perturbed_inputs = inputs + y_star
        outputs = self.model(perturbed_inputs)
        return F.cross_entropy(outputs, targets)
    
    def solve_lower_level(self, inputs, targets, q):
        """
        Solve lower-level problem: min_y g(x,y) + q^T y subject to ||y||_∞ ≤ ε
        Using projected gradient descent
        """
        batch_size, channels, height, width = inputs.shape
        y = torch.zeros_like(inputs, requires_grad=True, device=self.device)
        
        # Use Adam optimizer for inner problem
        inner_optimizer = optim.Adam([y], lr=0.01)
        
        for step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            
            # Compute objective
            loss = self.lower_level_objective(None, y, q, inputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([y], self.grad_clip)
            
            inner_optimizer.step()
            
            # Project to constraint set: ||y||_∞ ≤ ε
            with torch.no_grad():
                y.data = torch.clamp(y.data, -self.epsilon, self.epsilon)
        
        return y.detach()
    
    def compute_implicit_gradient(self, inputs, targets, y_star, q):
        """
        Compute implicit gradient: ∇F(x) = ∇_x f + [∇y*(x)]^T ∇_y f
        
        For DS-BLO, we use the stochastic gradient approximation
        """
        # Enable gradients for model parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Compute upper-level loss
        upper_loss = self.upper_level_objective(None, y_star, inputs, targets)
        
        # Compute gradients w.r.t. model parameters
        grad_x = torch.autograd.grad(upper_loss, self.model.parameters(), 
                                    create_graph=True, retain_graph=True)
        
        # Flatten gradients
        grad_x_flat = torch.cat([g.flatten() for g in grad_x])
        
        # Add stochastic noise for DS-BLO
        noise = torch.randn_like(grad_x_flat) * self.perturbation_std
        
        return grad_x_flat + noise
    
    def dsblo_step(self, inputs, targets):
        """
        Single DS-BLO step
        """
        # Step 1: Sample perturbation q for smoothing
        q = torch.randn_like(inputs) * self.perturbation_std
        
        # Step 2: Solve perturbed lower-level problem
        y_star = self.solve_lower_level(inputs, targets, q)
        
        # Step 3: Compute implicit gradient
        grad_F = self.compute_implicit_gradient(inputs, targets, y_star, q)
        
        # Step 4: Update with momentum and adaptive step size
        if not hasattr(self, 'momentum'):
            self.momentum = torch.zeros_like(grad_F)
        
        # Momentum update
        self.momentum = self.beta * self.momentum + (1 - self.beta) * grad_F
        
        # Adaptive step size: η = 1 / (γ₁||m|| + γ₂)
        step_size = 1.0 / (self.gamma1 * torch.norm(self.momentum) + self.gamma2)
        step_size = min(step_size, self.eta_cap)  # Cap step size
        
        # Update model parameters
        param_idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param_grad = self.momentum[param_idx:param_idx + param_size].view(param.shape)
            param.data -= step_size * param_grad
            param_idx += param_size
        
        return step_size
    
    def evaluate_robustness(self, dataloader, attack_steps=20, attack_lr=0.01):
        """
        Evaluate model robustness using PGD attack
        """
        self.model.eval()
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Generate adversarial examples using PGD
            adv_inputs = inputs.clone()
            adv_inputs.requires_grad = True
            
            for _ in range(attack_steps):
                outputs = self.model(adv_inputs)
                loss = F.cross_entropy(outputs, targets)
                grad = torch.autograd.grad(loss, adv_inputs)[0]
                
                # Update adversarial examples
                adv_inputs = adv_inputs + attack_lr * grad.sign()
                adv_inputs = torch.clamp(adv_inputs, inputs - self.epsilon, inputs + self.epsilon)
                adv_inputs = torch.clamp(adv_inputs, 0, 1)  # Valid image range
                adv_inputs = adv_inputs.detach().requires_grad_(True)
            
            # Evaluate on adversarial examples
            with torch.no_grad():
                outputs = self.model(adv_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return 100.0 * correct / total
    
    def evaluate_clean_accuracy(self, dataloader):
        """
        Evaluate model accuracy on clean data
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return 100.0 * correct / total

def load_cifar_data(dataset='cifar10', batch_size=128, val_split=0.1):
    """
    Load CIFAR-10 or CIFAR-100 dataset with train/test/val splits
    """
    if dataset == 'cifar10':
        num_classes = 10
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset == 'cifar100':
        num_classes = 100
        dataset_class = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    
    # Create validation split from training data
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, num_classes

def train_dsblo_adversarial(model, train_loader, val_loader, test_loader, 
                          epochs=100, epsilon=8/255, patience=10, 
                          save_dir='./dsblo_results'):
    """
    Train model using DS-BLO adversarial training
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize DS-BLO trainer
    trainer = DSBLOAdversarialTraining(model, epsilon=epsilon)
    
    # Training history
    best_val_loss = float('inf')
    best_robust_acc = 0
    patience_counter = 0
    
    logger.info(f"Starting DS-BLO adversarial training for {epochs} epochs")
    logger.info(f"Epsilon (attack budget): {epsilon}")
    logger.info(f"Device: {trainer.device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(trainer.device), targets.to(trainer.device)
            
            # DS-BLO step
            step_size = trainer.dsblo_step(inputs, targets)
            
            # Compute training loss for monitoring
            with torch.no_grad():
                # Generate adversarial examples for loss computation
                q = torch.randn_like(inputs) * trainer.perturbation_std
                y_star = trainer.solve_lower_level(inputs, targets, q)
                loss = trainer.upper_level_objective(None, y_star, inputs, targets)
                train_loss += loss.item()
                num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'StepSize': f'{step_size:.6f}'
            })
        
        avg_train_loss = train_loss / num_batches
        trainer.train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(trainer.device), targets.to(trainer.device)
                
                # Generate adversarial examples for validation
                q = torch.randn_like(inputs) * trainer.perturbation_std
                y_star = trainer.solve_lower_level(inputs, targets, q)
                loss = trainer.upper_level_objective(None, y_star, inputs, targets)
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        trainer.val_losses.append(avg_val_loss)
        
        # Evaluate robustness and clean accuracy
        robust_acc = trainer.evaluate_robustness(val_loader)
        clean_acc = trainer.evaluate_clean_accuracy(val_loader)
        
        trainer.robust_accs.append(robust_acc)
        trainer.clean_accs.append(clean_acc)
        
        logger.info(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, '
                   f'Val Loss: {avg_val_loss:.4f}, '
                   f'Robust Acc: {robust_acc:.2f}%, '
                   f'Clean Acc: {clean_acc:.2f}%')
        
        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_robust_acc = robust_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'robust_acc': robust_acc,
                'clean_acc': clean_acc,
                'trainer_state': {
                    'momentum': trainer.momentum,
                    'train_losses': trainer.train_losses,
                    'val_losses': trainer.val_losses,
                    'robust_accs': trainer.robust_accs,
                    'clean_accs': trainer.clean_accs
                }
            }, os.path.join(save_dir, 'best_model.pth'))
            
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f'Early stopping at epoch {epoch+1}')
            break
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_robust_acc = trainer.evaluate_robustness(test_loader)
    test_clean_acc = trainer.evaluate_clean_accuracy(test_loader)
    
    logger.info(f'Final Test Results:')
    logger.info(f'  Clean Accuracy: {test_clean_acc:.2f}%')
    logger.info(f'  Robust Accuracy: {test_robust_acc:.2f}%')
    
    # Save final results
    results = {
        'final_test_clean_acc': test_clean_acc,
        'final_test_robust_acc': test_robust_acc,
        'best_val_loss': best_val_loss,
        'best_robust_acc': best_robust_acc,
        'epochs_trained': epoch + 1,
        'epsilon': epsilon,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'robust_accs': trainer.robust_accs,
        'clean_accs': trainer.clean_accs
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    plot_training_curves(trainer, save_dir)
    
    return results

def plot_training_curves(trainer, save_dir):
    """
    Plot training curves
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training and validation losses
    axes[0, 0].plot(trainer.train_losses, label='Train Loss')
    axes[0, 0].plot(trainer.val_losses, label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Robust accuracy
    axes[0, 1].plot(trainer.robust_accs, label='Robust Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Robust Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Clean accuracy
    axes[1, 0].plot(trainer.clean_accs, label='Clean Accuracy', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Clean Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined accuracy plot
    axes[1, 1].plot(trainer.robust_accs, label='Robust Accuracy', color='red')
    axes[1, 1].plot(trainer.clean_accs, label='Clean Accuracy', color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Accuracy Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='DS-BLO Adversarial Training')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--epsilon', type=float, default=8/255, 
                       help='Attack budget (default: 8/255)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=10, 
                       help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='./dsblo_results', 
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default=None, 
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, test_loader, num_classes = load_cifar_data(
        dataset=args.dataset, 
        batch_size=args.batch_size
    )
    
    # Create model
    model = ResNet18(num_classes=num_classes)
    logger.info(f"Created ResNet-18 model for {num_classes} classes")
    
    # Train model
    results = train_dsblo_adversarial(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        epsilon=args.epsilon,
        patience=args.patience,
        save_dir=args.save_dir
    )
    
    logger.info("Training completed!")
    logger.info(f"Results saved to: {args.save_dir}")

if __name__ == '__main__':
    main()

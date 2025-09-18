#!/usr/bin/env python3
"""
Test script for DS-BLO adversarial training
Runs a quick test to verify the implementation works correctly
"""

import torch
import os
import sys
import logging
from dsblo_adversarial_training import (
    ResNet18, DSBLOAdversarialTraining, load_cifar_data, 
    train_dsblo_adversarial
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dsblo_implementation():
    """Test DS-BLO implementation with a small subset of data"""
    
    logger.info("Starting DS-BLO adversarial training test...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load a small subset of CIFAR-10 for testing
    logger.info("Loading CIFAR-10 dataset (small subset for testing)...")
    train_loader, val_loader, test_loader, num_classes = load_cifar_data(
        dataset='cifar10', 
        batch_size=32,  # Smaller batch size for testing
        val_split=0.2   # Larger validation split for testing
    )
    
    # Create a smaller subset for quick testing
    train_subset_size = min(1000, len(train_loader.dataset))
    val_subset_size = min(200, len(val_loader.dataset))
    test_subset_size = min(200, len(test_loader.dataset))
    
    # Create subset loaders
    from torch.utils.data import Subset
    train_subset = Subset(train_loader.dataset, range(train_subset_size))
    val_subset = Subset(val_loader.dataset, range(val_subset_size))
    test_subset = Subset(test_loader.dataset, range(test_subset_size))
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False)
    
    logger.info(f"Training subset: {len(train_subset)} samples")
    logger.info(f"Validation subset: {len(val_subset)} samples")
    logger.info(f"Test subset: {len(test_subset)} samples")
    
    # Create model
    model = ResNet18(num_classes=num_classes)
    logger.info(f"Created ResNet-18 model for {num_classes} classes")
    
    # Test DS-BLO components
    logger.info("Testing DS-BLO components...")
    trainer = DSBLOAdversarialTraining(model, epsilon=8/255, device=device)
    
    # Test with a single batch
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        logger.info(f"Testing with batch shape: {inputs.shape}")
        
        # Test lower-level objective
        q = torch.randn_like(inputs) * trainer.perturbation_std
        y_star = trainer.solve_lower_level(inputs, targets, q)
        logger.info(f"Lower-level solution shape: {y_star.shape}")
        
        # Test upper-level objective
        upper_loss = trainer.upper_level_objective(None, y_star, inputs, targets)
        logger.info(f"Upper-level loss: {upper_loss.item():.4f}")
        
        # Test implicit gradient computation
        grad_F = trainer.compute_implicit_gradient(inputs, targets, y_star, q)
        logger.info(f"Implicit gradient norm: {torch.norm(grad_F).item():.4f}")
        
        # Test DS-BLO step
        step_size = trainer.dsblo_step(inputs, targets)
        logger.info(f"DS-BLO step size: {step_size:.6f}")
        
        break  # Only test with first batch
    
    logger.info("DS-BLO components test passed!")
    
    # Run short training
    logger.info("Running short training test (5 epochs)...")
    results = train_dsblo_adversarial(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=5,  # Short training for testing
        epsilon=8/255,
        patience=3,
        save_dir='./dsblo_test_results'
    )
    
    logger.info("Training test completed!")
    logger.info(f"Final results: {results}")
    
    return results

def run_full_experiment():
    """Run full DS-BLO experiment as in the paper"""
    
    logger.info("Starting full DS-BLO adversarial training experiment...")
    
    # Parameters matching the paper experiments
    datasets = ['cifar10', 'cifar100']
    epsilons = [8/255, 16/255]
    
    for dataset in datasets:
        for epsilon in epsilons:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running experiment: {dataset} with ε={epsilon}")
            logger.info(f"{'='*60}")
            
            # Load data
            train_loader, val_loader, test_loader, num_classes = load_cifar_data(
                dataset=dataset, 
                batch_size=128,
                val_split=0.1
            )
            
            # Create model
            model = ResNet18(num_classes=num_classes)
            
            # Create save directory
            save_dir = f'./dsblo_results_{dataset}_eps{int(epsilon*255)}'
            
            # Train model
            results = train_dsblo_adversarial(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=100,
                epsilon=epsilon,
                patience=10,
                save_dir=save_dir
            )
            
            logger.info(f"Completed {dataset} with ε={epsilon}")
            logger.info(f"Results: Clean Acc={results['final_test_clean_acc']:.2f}%, "
                       f"Robust Acc={results['final_test_robust_acc']:.2f}%")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DS-BLO Adversarial Training Test')
    parser.add_argument('--test-only', action='store_true', 
                       help='Run only component tests (quick)')
    parser.add_argument('--full-experiment', action='store_true',
                       help='Run full experiment as in paper')
    
    args = parser.parse_args()
    
    if args.test_only:
        logger.info("Running component tests only...")
        test_dsblo_implementation()
    elif args.full_experiment:
        logger.info("Running full experiment...")
        run_full_experiment()
    else:
        logger.info("Running quick test...")
        test_dsblo_implementation()
        
        logger.info("\nTo run full experiment, use: python run_dsblo_test.py --full-experiment")
        logger.info("To run only component tests, use: python run_dsblo_test.py --test-only")

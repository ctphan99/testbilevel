#!/usr/bin/env python3
"""
Quick GPU test for DS-BLO adversarial training
Verifies GPU usage and basic functionality
"""

import torch
import os
import sys
import logging
from dsblo_adversarial_training import (
    ResNet18, DSBLOAdversarialTraining, load_cifar_data
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpu_availability():
    """Test GPU availability and configuration"""
    logger.info("Testing GPU availability...")
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return False
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Test basic GPU operations
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.mm(x, y)
        logger.info(f"GPU tensor operations working: {z.shape}")
        return True
    except Exception as e:
        logger.error(f"GPU operations failed: {e}")
        return False

def test_dsblo_gpu():
    """Test DS-BLO with GPU"""
    logger.info("Testing DS-BLO with GPU...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create model
    model = ResNet18(num_classes=10)
    logger.info(f"Created ResNet-18 model")
    
    # Initialize DS-BLO trainer
    trainer = DSBLOAdversarialTraining(model, epsilon=8/255, device=device)
    logger.info(f"Initialized DS-BLO trainer on {trainer.device}")
    
    # Test with dummy data
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32, device=device)
    targets = torch.randint(0, 10, (batch_size,), device=device)
    
    logger.info(f"Testing with batch shape: {inputs.shape}")
    
    # Test lower-level objective
    q = torch.randn_like(inputs) * trainer.perturbation_std
    y_star = trainer.solve_lower_level(inputs, targets, q)
    logger.info(f"Lower-level solution shape: {y_star.shape}")
    logger.info(f"Lower-level solution device: {y_star.device}")
    
    # Test upper-level objective
    upper_loss = trainer.upper_level_objective(None, y_star, inputs, targets)
    logger.info(f"Upper-level loss: {upper_loss.item():.4f}")
    logger.info(f"Upper-level loss device: {upper_loss.device}")
    
    # Test implicit gradient computation
    grad_F = trainer.compute_implicit_gradient(inputs, targets, y_star, q)
    logger.info(f"Implicit gradient norm: {torch.norm(grad_F).item():.4f}")
    logger.info(f"Implicit gradient device: {grad_F.device}")
    
    # Test DS-BLO step
    step_size = trainer.dsblo_step(inputs, targets)
    logger.info(f"DS-BLO step size: {step_size:.6f}")
    
    # Test GPU memory usage
    if device == 'cuda':
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
        logger.info(f"GPU memory allocated: {memory_allocated:.2f} MB")
        logger.info(f"GPU memory reserved: {memory_reserved:.2f} MB")
    
    logger.info("DS-BLO GPU test passed!")
    return True

def test_data_loading_gpu():
    """Test data loading with GPU"""
    logger.info("Testing data loading with GPU...")
    
    try:
        # Load small subset for testing
        train_loader, val_loader, test_loader, num_classes = load_cifar_data(
            dataset='cifar10', 
            batch_size=8,  # Small batch for testing
            val_split=0.2
        )
        
        logger.info(f"Loaded CIFAR-10 with {num_classes} classes")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Test moving data to GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logger.info(f"Data moved to {device}: {inputs.shape}, {targets.shape}")
            break  # Only test first batch
        
        logger.info("Data loading GPU test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        return False

def main():
    """Run all GPU tests"""
    logger.info("Starting DS-BLO GPU tests...")
    
    # Test 1: GPU availability
    if not test_gpu_availability():
        logger.error("GPU not available, skipping GPU tests")
        return False
    
    # Test 2: DS-BLO with GPU
    if not test_dsblo_gpu():
        logger.error("DS-BLO GPU test failed")
        return False
    
    # Test 3: Data loading with GPU
    if not test_data_loading_gpu():
        logger.error("Data loading GPU test failed")
        return False
    
    logger.info("All GPU tests passed! 🚀")
    logger.info("DS-BLO is ready for GPU training")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

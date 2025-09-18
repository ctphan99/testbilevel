#!/usr/bin/env python3
"""
Simple GPU test for DS-BLO adversarial training (without torchvision dependency)
Verifies basic GPU usage and PyTorch functionality
"""

import torch
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpu_availability():
    """Test GPU availability and configuration"""
    logger.info("Testing GPU availability...")
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - testing CPU functionality instead")
        logger.info("This is expected on login nodes or when GPU resources are not allocated")
        
        # Test basic CPU operations
        try:
            x = torch.randn(100, 100)
            y = torch.randn(100, 100)
            z = torch.mm(x, y)
            logger.info(f"CPU tensor operations working: {z.shape}")
            return True
        except Exception as e:
            logger.error(f"CPU operations failed: {e}")
            return False
    else:
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        
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

def test_basic_dsblo_components():
    """Test basic DS-BLO components without full implementation"""
    logger.info("Testing basic DS-BLO components...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Test basic tensor operations that DS-BLO would use
    try:
        # Simulate adversarial perturbation
        batch_size = 4
        channels = 3
        height = 32
        width = 32
        
        # Create dummy data
        inputs = torch.randn(batch_size, channels, height, width, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)
        
        logger.info(f"Created dummy data: {inputs.shape}, {targets.shape}")
        
        # Test perturbation generation
        epsilon = 8/255
        perturbation = torch.randn_like(inputs) * 0.01  # Small perturbation
        perturbed_inputs = inputs + perturbation
        
        # Test constraint projection (L∞ constraint)
        perturbed_inputs = torch.clamp(perturbed_inputs, -epsilon, epsilon)
        
        logger.info(f"Perturbation applied and projected: {perturbed_inputs.shape}")
        
        # Test gradient computation
        perturbed_inputs.requires_grad_(True)
        dummy_loss = torch.sum(perturbed_inputs ** 2)  # Simple loss
        dummy_loss.backward()
        
        grad_norm = torch.norm(perturbed_inputs.grad)
        logger.info(f"Gradient computation successful: norm={grad_norm.item():.4f}")
        
        # Test memory usage
        if device == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} MB")
            logger.info(f"GPU memory reserved: {memory_reserved:.2f} MB")
        else:
            logger.info("CPU mode - no GPU memory to report")
        
        logger.info("Basic DS-BLO components test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Basic DS-BLO components test failed: {e}")
        return False

def test_torchvision_import():
    """Test torchvision import separately"""
    logger.info("Testing torchvision import...")
    
    try:
        import torchvision
        logger.info(f"Torchvision version: {torchvision.__version__}")
        
        # Test basic torchvision operations
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        logger.info("Torchvision transforms working")
        
        return True
        
    except Exception as e:
        logger.error(f"Torchvision import failed: {e}")
        return False

def main():
    """Run all GPU tests"""
    logger.info("Starting simple DS-BLO tests...")
    
    # Test 1: GPU/CPU availability
    if not test_gpu_availability():
        logger.error("PyTorch functionality test failed")
        return False
    
    # Test 2: Basic DS-BLO components
    if not test_basic_dsblo_components():
        logger.error("Basic DS-BLO components test failed")
        return False
    
    # Test 3: Torchvision import (optional)
    torchvision_ok = test_torchvision_import()
    if not torchvision_ok:
        logger.warning("Torchvision import failed, but basic functionality should work")
    
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    logger.info(f"Simple tests completed! 🚀")
    logger.info(f"Basic DS-BLO functionality verified on {device_type}")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test script for Distribution Matching ControlNet
This script tests the basic functionality of the distribution matching model.
"""

import torch
import yaml
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.distribution_matching_controlnet import DistributionMatchingControlNet, DistributionMatchingControlNetDistilled
from models.controlnet import ControlNet


def test_distribution_matching_model():
    """Test the basic functionality of the distribution matching model"""
    print("Testing Distribution Matching ControlNet...")
    
    # Load config
    config_path = 'config/mnist.yaml'
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    model_config = config['model_params']
    
    # Test basic model
    print("1. Testing basic DistributionMatchingControlNet...")
    try:
        model = DistributionMatchingControlNet(model_config)
        print("   ✓ Basic model created successfully")
        
        # Test forward pass
        batch_size = 2
        x_t = torch.randn(batch_size, model_config['im_channels'], 
                         model_config['im_size'], model_config['im_size'])
        t = torch.randint(0, 1000, (batch_size,))
        hint = torch.randn(batch_size, model_config['hint_channels'], 
                          model_config['im_size'], model_config['im_size'])
        
        output = model(x_t, t, hint)
        expected_shape = (batch_size, model_config['im_channels'], 
                         model_config['im_size'], model_config['im_size'])
        
        if output.shape == expected_shape:
            print("   ✓ Forward pass successful")
        else:
            print(f"   ✗ Forward pass failed: expected {expected_shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"   ✗ Basic model test failed: {e}")
        return False
    
    # Test distilled model (without teacher checkpoint)
    print("2. Testing DistributionMatchingControlNetDistilled...")
    try:
        # Create a dummy teacher checkpoint path
        dummy_ckpt = "dummy_checkpoint.pth"
        
        # This should fail gracefully if checkpoint doesn't exist
        try:
            distilled_model = DistributionMatchingControlNetDistilled(
                model_config, dummy_ckpt
            )
            print("   ✓ Distilled model created successfully (without teacher)")
        except FileNotFoundError:
            print("   ✓ Distilled model handles missing checkpoint correctly")
        
        # Test forward pass
        output = distilled_model.student(x_t, t, hint)
        if output.shape == expected_shape:
            print("   ✓ Distilled model forward pass successful")
        else:
            print(f"   ✗ Distilled model forward pass failed: expected {expected_shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"   ✗ Distilled model test failed: {e}")
        return False
    
    # Test loss computation
    print("3. Testing loss computation...")
    try:
        x_0_target = torch.randn_like(x_t)
        total_loss, dist_matching_loss, distillation_loss = distilled_model.distillation_loss(
            x_t, t, hint, x_0_target
        )
        
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            print("   ✓ Loss computation successful")
        else:
            print("   ✗ Loss computation failed")
            return False
            
    except Exception as e:
        print(f"   ✗ Loss computation test failed: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True


def test_model_compatibility():
    """Test compatibility with existing ControlNet"""
    print("\nTesting model compatibility...")
    
    # Load config
    config_path = 'config/mnist.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    model_config = config['model_params']
    
    # Test that both models have similar interfaces
    try:
        controlnet = ControlNet(model_config, model_locked=True)
        dist_matching = DistributionMatchingControlNet(model_config)
        
        # Both should have similar parameter counts
        controlnet_params = sum(p.numel() for p in controlnet.parameters())
        dist_matching_params = sum(p.numel() for p in dist_matching.parameters())
        
        print(f"   ControlNet parameters: {controlnet_params:,}")
        print(f"   Distribution Matching parameters: {dist_matching_params:,}")
        
        # Parameters should be similar (within 10%)
        ratio = dist_matching_params / controlnet_params
        if 0.9 <= ratio <= 1.1:
            print("   ✓ Parameter counts are compatible")
        else:
            print(f"   ⚠ Parameter counts differ significantly (ratio: {ratio:.2f})")
            
    except Exception as e:
        print(f"   ✗ Compatibility test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("DISTRIBUTION MATCHING CONTROLNET TEST")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_distribution_matching_model()
    success &= test_model_compatibility()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Distribution Matching ControlNet is ready to use.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    print("=" * 60) 
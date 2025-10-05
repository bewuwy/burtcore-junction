#!/usr/bin/env python3
"""
Script to reset CUDA context and test GPU availability.
Run this before starting your server if you get CUDA errors.
"""
import os
import sys

# Set environment variables BEFORE importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch

print("=" * 80)
print("CUDA RESET & TEST SCRIPT")
print("=" * 80)

try:
    # Check CUDA availability
    print(f"\n1. Checking CUDA availability...")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("   ❌ CUDA is not available!")
        print("   This might be due to:")
        print("   - Another process holding GPU lock")
        print("   - Driver/library mismatch")
        print("   - Environment variable issues")
        sys.exit(1)
    
    # Get device count
    print(f"\n2. Getting device count...")
    device_count = torch.cuda.device_count()
    print(f"   Device count: {device_count}")
    
    # Get device properties
    print(f"\n3. Getting device properties...")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"   Device {i}: {props.name}")
        print(f"   - Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   - Compute capability: {props.major}.{props.minor}")
    
    # Test tensor creation
    print(f"\n4. Testing tensor creation on GPU...")
    device = torch.device('cuda:0')
    test_tensor = torch.randn(100, 100).to(device)
    print(f"   ✓ Successfully created tensor on GPU")
    print(f"   Tensor shape: {test_tensor.shape}")
    print(f"   Tensor device: {test_tensor.device}")
    
    # Test computation
    print(f"\n5. Testing GPU computation...")
    result = torch.matmul(test_tensor, test_tensor)
    print(f"   ✓ Successfully performed computation on GPU")
    
    # Clear cache
    print(f"\n6. Clearing GPU cache...")
    torch.cuda.empty_cache()
    print(f"   ✓ Cache cleared")
    
    # Memory info
    print(f"\n7. GPU Memory Status:")
    print(f"   - Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"   - Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - CUDA IS WORKING!")
    print("=" * 80)
    print("\nYou can now start your server with GPU support.")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTroubleshooting steps:")
    print("1. Make sure no other processes are using the GPU:")
    print("   nvidia-smi")
    print("2. Kill any blocking processes:")
    print("   kill <PID>")
    print("3. Try setting DEVICE='auto' in config.py instead of 'cuda'")
    print("4. Restart your terminal/IDE")
    sys.exit(1)


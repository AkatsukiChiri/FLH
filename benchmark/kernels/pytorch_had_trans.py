#!/usr/bin/env python3
"""
PyTorch implementation of Hadamard Transform as baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class HadamardTransform(nn.Module):
    """
    PyTorch implementation of Hadamard Transform
    """
    
    def __init__(self, n: int):
        """
        Initialize Hadamard Transform
        
        Args:
            n: Size of the transform (must be power of 2)
        """
        super().__init__()
        self.n = n
        self.log2_n = int(torch.log2(torch.tensor(n, dtype=torch.float32)).item())
        
        if 2 ** self.log2_n != n:
            raise ValueError(f"n must be a power of 2, got {n}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Hadamard Transform
        
        Args:
            x: Input tensor of shape (..., n)
            
        Returns:
            Transformed tensor of same shape
        """
        return self._hadamard_transform_recursive(x, self.n)
    
    def _hadamard_transform_recursive(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Recursive implementation of Hadamard Transform
        """
        if n == 1:
            return x
        
        # Split into two halves
        half = n // 2
        left = self._hadamard_transform_recursive(x[..., :half], half)
        right = self._hadamard_transform_recursive(x[..., half:], half)
        
        # Apply butterfly operation
        sum_part = left + right
        diff_part = left - right
        
        # Concatenate results
        return torch.cat([sum_part, diff_part], dim=-1)

def hadamard_transform_vectorized(x: torch.Tensor) -> torch.Tensor:
    """
    Vectorized implementation of Hadamard Transform using matrix multiplication
    
    Args:
        x: Input tensor of shape (..., n) where n is power of 2
        
    Returns:
        Transformed tensor of same shape
    """
    n = x.shape[-1]
    log2_n = int(torch.log2(torch.tensor(n, dtype=torch.float32)).item())
    
    if 2 ** log2_n != n:
        raise ValueError(f"n must be a power of 2, got {n}")
    
    # Generate Hadamard matrix
    H = _generate_hadamard_matrix(n, device=x.device, dtype=x.dtype)
    
    # Apply transform: y = x @ H^T
    return torch.matmul(x, H.T)

def _generate_hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Generate Hadamard matrix of size n x n
    
    Args:
        n: Size of the matrix (must be power of 2)
        device: Device to create tensor on
        dtype: Data type of the tensor
        
    Returns:
        Hadamard matrix of shape (n, n)
    """
    if n == 1:
        return torch.ones((1, 1), device=device, dtype=dtype)
    
    if n == 2:
        return torch.tensor([[1, 1], [1, -1]], device=device, dtype=dtype)
    
    # Recursive construction: H_n = [H_{n/2}, H_{n/2}; H_{n/2}, -H_{n/2}]
    half = n // 2
    H_half = _generate_hadamard_matrix(half, device, dtype)
    
    top = torch.cat([H_half, H_half], dim=1)
    bottom = torch.cat([H_half, -H_half], dim=1)
    
    return torch.cat([top, bottom], dim=0)

def hadamard_transform_inplace(x: torch.Tensor) -> torch.Tensor:
    """
    In-place implementation of Hadamard Transform
    
    Args:
        x: Input tensor of shape (..., n) where n is power of 2
        
    Returns:
        Transformed tensor (same as input, modified in-place)
    """
    n = x.shape[-1]
    log2_n = int(torch.log2(torch.tensor(n, dtype=torch.float32)).item())
    
    if 2 ** log2_n != n:
        raise ValueError(f"n must be a power of 2, got {n}")
    
    # Apply log2(n) stages of butterfly operations
    for stage in range(log2_n):
        step = 1 << stage
        half_step = step
        
        # Apply butterfly operations
        for i in range(0, n, 2 * step):
            for j in range(half_step):
                idx1 = i + j
                idx2 = i + j + half_step
                
                if idx2 < n:
                    # Butterfly operation: [a, b] -> [a+b, a-b]
                    a = x[..., idx1]
                    b = x[..., idx2]
                    
                    x[..., idx1] = a + b
                    x[..., idx2] = a - b
    
    return x

def hadamard_transform_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Batch implementation optimized for multiple rows
    
    Args:
        x: Input tensor of shape (batch_size, n) where n is power of 2
        
    Returns:
        Transformed tensor of same shape
    """
    batch_size, n = x.shape
    log2_n = int(torch.log2(torch.tensor(n, dtype=torch.float32)).item())
    
    if 2 ** log2_n != n:
        raise ValueError(f"n must be a power of 2, got {n}")
    
    # Work on a copy to avoid modifying input
    result = x.clone()
    
    # Apply log2(n) stages of butterfly operations
    for stage in range(log2_n):
        step = 1 << stage
        half_step = step
        
        # Vectorized butterfly operations for all rows
        for i in range(0, n, 2 * step):
            for j in range(half_step):
                idx1 = i + j
                idx2 = i + j + half_step
                
                if idx2 < n:
                    # Butterfly operation: [a, b] -> [a+b, a-b]
                    a = result[:, idx1]
                    b = result[:, idx2]
                    
                    result[:, idx1] = a + b
                    result[:, idx2] = a - b
    
    return result

# Convenience functions for different implementations
def pytorch_had_trans_recursive(x: torch.Tensor) -> torch.Tensor:
    """Recursive implementation"""
    transform = HadamardTransform(x.shape[-1])
    return transform(x)

def pytorch_had_trans_vectorized(x: torch.Tensor) -> torch.Tensor:
    """Vectorized implementation using matrix multiplication"""
    return hadamard_transform_vectorized(x)

def pytorch_had_trans_inplace(x: torch.Tensor) -> torch.Tensor:
    """In-place implementation"""
    return hadamard_transform_inplace(x.clone())

def pytorch_had_trans_batch(x: torch.Tensor) -> torch.Tensor:
    """Batch-optimized implementation"""
    return hadamard_transform_batch(x)

# Default implementation (batch-optimized)
def pytorch_had_trans(x: torch.Tensor) -> torch.Tensor:
    """
    Default PyTorch Hadamard Transform implementation
    
    Args:
        x: Input tensor of shape (..., n) where n is power of 2
        
    Returns:
        Transformed tensor of same shape
    """
    return pytorch_had_trans_batch(x)

if __name__ == "__main__":
    # Test the implementations
    print("Testing PyTorch Hadamard Transform implementations...")
    
    # Test data
    torch.manual_seed(42)
    x = torch.randn(2, 128, device='cuda')
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
    
    # Test different implementations
    implementations = [
        ("Recursive", pytorch_had_trans_recursive),
        ("Vectorized", pytorch_had_trans_vectorized),
        ("In-place", pytorch_had_trans_inplace),
        ("Batch", pytorch_had_trans_batch),
        ("Default", pytorch_had_trans)
    ]
    
    results = {}
    for name, func in implementations:
        try:
            result = func(x.clone())
            results[name] = result
            print(f"{name:12}: Shape {result.shape}, Range [{result.min().item():.4f}, {result.max().item():.4f}]")
        except Exception as e:
            print(f"{name:12}: Error - {e}")
    
    # Check if all results are the same
    if len(results) > 1:
        ref_result = list(results.values())[0]
        all_same = all(torch.allclose(ref_result, result, atol=1e-6) for result in results.values())
        print(f"\nAll implementations produce same result: {all_same}")
        
        if not all_same:
            print("Differences found between implementations!")
            for name, result in results.items():
                diff = torch.abs(ref_result - result).max().item()
                print(f"  {name}: Max diff = {diff:.2e}")
    
    print("\nPyTorch Hadamard Transform test completed!")

#!/usr/bin/env python3
"""
Linear4bit 性能测试
比较多种量化方法的推理性能：
1. FP16 baseline
2. W4A4: 权重4bit，激活4bit
3. W4A16: 权重4bit，激活16bit（反量化后计算）
4. Hadamard+W4A4 (PyTorch): PyTorch实现的Hadamard变换 + W4A4
5. Hadamard+W4A4 (Fast): fast-hadamard-transform库实现的Hadamard变换 + W4A4
6. HQL: 我们的 Hadamard + 4bit 量化方法
"""

import torch
import torch.nn as nn
import time
import hql
from hql.nn import Linear4bit
import numpy as np
import pandas as pd
from datetime import datetime
import os

import quarot

# 尝试导入 fast-hadamard-transform
try:
    from fast_hadamard_transform import hadamard_transform
    FAST_HADAMARD_AVAILABLE = True
except ImportError:
    FAST_HADAMARD_AVAILABLE = False
    print("Warning: fast-hadamard-transform not available. Will skip Fast Hadamard baseline.")


class W4A4Linear(nn.Module):
    """
    W4A4 量化线性层
    - 权重：量化到4bit范围 [-8, 7]，直接用 int8 存储（不打包）
    - 激活：运行时量化到4bit范围 [-8, 7]，用 int8 存储
    - 计算：使用 int8 存储的值进行矩阵乘法
    """
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 4-bit 权重（直接用 int8 存储，不打包）
        self.register_buffer('weight_q', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features, dtype=dtype))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    
    def quantize_weight(self, weight_fp16):
        """将 FP16 权重量化为 4-bit 范围，直接存储为 int8"""
        # 对每一行进行对称量化
        weight_max = weight_fp16.abs().max(dim=1, keepdim=True)[0]
        scale = weight_max / 7.0  # 4-bit 有符号范围 [-8, 7]
        scale = torch.clamp(scale, min=1e-5)
        
        # 量化到 4-bit 范围并直接用 int8 存储
        weight_q = torch.round(weight_fp16 / scale).clamp(-8, 7).to(torch.int8)
        
        self.weight_q.copy_(weight_q)
        self.weight_scale.copy_(scale.squeeze())
    
    def quantize_activation(self, x):
        """将激活量化到 4-bit 范围，用 int8 存储"""
        # 逐 token 量化
        x_max = x.abs().max(dim=-1, keepdim=True)[0]
        scale = x_max / 7.0
        scale = torch.clamp(scale, min=1e-5)
        
        x_q = torch.round(x / scale).clamp(-8, 7).to(torch.int8)
        return x_q, scale
    
    def forward(self, x):
        """W4A4 前向传播 - 使用 int8 存储的 4-bit 范围值进行计算"""
        # 1. 量化输入激活为 int8（4-bit 范围）
        x_q, x_scale = self.quantize_activation(x)
        
        # 2. 使用 float 进行矩阵乘法（int8 matmul 支持有限）
        # 将 int8 转为 float 进行计算，模拟量化计算的数值行为
        output = quarot.matmul(x_q.to(torch.uint8), self.weight_q.to(torch.uint8).contiguous())
        # output = gemm_int8.matmul(x_q, self.weight_q.t())
        
        # 3. 反量化
        output_scale = x_scale * self.weight_scale.unsqueeze(0)
        output = output * output_scale
        
        if self.bias is not None:
            output = output + self.bias
        
        return output.to(x.dtype)


class W4A16Linear(nn.Module):
    """
    W4A16 量化线性层
    - 权重：4bit 存储
    - 激活：保持 FP16
    - 计算：先将权重反量化为 FP16，然后进行 FP16 矩阵乘法
    """
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 4-bit 权重（打包为 uint8）
        self.register_buffer('weight_packed', torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer('weight_scale', torch.ones(out_features, dtype=dtype))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    
    def quantize_weight(self, weight_fp16):
        """将 FP16 权重量化为 4-bit"""
        # 对每一行进行对称量化
        weight_max = weight_fp16.abs().max(dim=1, keepdim=True)[0]
        scale = weight_max / 7.0
        scale = torch.clamp(scale, min=1e-5)
        
        weight_q = torch.round(weight_fp16 / scale).clamp(-8, 7).to(torch.int8)
        
        # 打包
        weight_q = weight_q.view(self.out_features, -1, 2)
        weight_packed = ((weight_q[:, :, 0] & 0xF) | ((weight_q[:, :, 1] & 0xF) << 4)).to(torch.uint8)
        
        self.weight_packed.copy_(weight_packed)
        self.weight_scale.copy_(scale.squeeze())
    
    def forward(self, x):
        """W4A16 前向传播：反量化权重后进行 FP16 计算"""
        # 1. 解包权重
        weight_low = (self.weight_packed & 0xF).to(torch.int8)
        weight_high = ((self.weight_packed >> 4) & 0xF).to(torch.int8)
        
        # 处理符号位
        weight_low = torch.where(weight_low > 7, weight_low - 16, weight_low)
        weight_high = torch.where(weight_high > 7, weight_high - 16, weight_high)
        
        weight_q = torch.stack([weight_low, weight_high], dim=-1).view(self.out_features, -1)
        
        # 2. 反量化权重
        weight_fp16 = weight_q.to(x.dtype) * self.weight_scale.unsqueeze(1)
        
        # 3. FP16 矩阵乘法
        output = torch.matmul(x, weight_fp16.t())
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class HadamardW4A4LinearPyTorch(nn.Module):
    """
    Hadamard+W4A4 量化线性层 (PyTorch实现)
    - 权重：先进行Hadamard变换，然后4bit量化存储  
    - 激活：直接量化为4bit（不进行Hadamard变换以避免输出维度限制）
    - 计算：W4A4矩阵乘法
    - 输出：反量化
    """
    def __init__(self, in_features, out_features, bias=False, group_size=128, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        assert in_features % group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        
        # 4-bit 权重（打包为 uint8）
        self.register_buffer('weight_packed', torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer('weight_scale', torch.ones(out_features, dtype=dtype))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    
    def hadamard_transform_pytorch(self, x, group_size):
        """PyTorch实现的Hadamard变换"""
        # x: (..., d) -> (..., d)
        # 按 group_size 分组进行Hadamard变换
        orig_shape = x.shape
        x = x.reshape(-1, group_size)
        
        # 递归实现Hadamard变换
        n = group_size
        assert n & (n - 1) == 0, "group_size must be power of 2"
        
        h = x.clone()
        step = 1
        while step < n:
            for i in range(0, n, step * 2):
                for j in range(step):
                    u = h[:, i + j]
                    v = h[:, i + j + step]
                    h[:, i + j] = u + v
                    h[:, i + j + step] = u - v
            step *= 2
        
        # 归一化
        h = h / np.sqrt(n)
        
        return h.reshape(orig_shape)
    
    def quantize_weight(self, weight_fp16):
        """将FP16权重进行Hadamard变换并量化为4-bit"""
        # weight_fp16: (out_features, in_features)
        
        # 1. Hadamard变换（按group_size分组）
        weight_had = self.hadamard_transform_pytorch(weight_fp16, self.group_size)
        
        # 2. Per-channel量化
        weight_max = weight_had.abs().max(dim=1, keepdim=True)[0]
        scale = weight_max / 7.0
        scale = torch.clamp(scale, min=1e-5)
        
        weight_q = torch.round(weight_had / scale).clamp(-8, 7).to(torch.int8)
        
        # 3. 打包
        weight_q = weight_q.view(self.out_features, -1, 2)
        weight_packed = ((weight_q[:, :, 0] & 0xF) | ((weight_q[:, :, 1] & 0xF) << 4)).to(torch.uint8)
        
        self.weight_packed.copy_(weight_packed)
        self.weight_scale.copy_(scale.squeeze())
    
    def forward(self, x):
        """前向传播"""
        # 1. 量化激活（per-token量化）
        x_max = x.abs().max(dim=-1, keepdim=True)[0]
        x_scale = x_max / 7.0
        x_scale = torch.clamp(x_scale, min=1e-5)
        
        x_q = torch.round(x / x_scale).clamp(-8, 7).to(torch.int8)
        
        # 2. 解包权重
        weight_low = (self.weight_packed & 0xF).to(torch.int8)
        weight_high = ((self.weight_packed >> 4) & 0xF).to(torch.int8)
        weight_low = torch.where(weight_low > 7, weight_low - 16, weight_low)
        weight_high = torch.where(weight_high > 7, weight_high - 16, weight_high)
        weight_q = torch.stack([weight_low, weight_high], dim=-1).view(self.out_features, -1)
        
        # 3. 矩阵乘法（int计算）
        output = torch.matmul(x_q.float(), weight_q.t().float())
        
        # 4. 反量化
        output_scale = x_scale * self.weight_scale.unsqueeze(0)
        output = output * output_scale
        
        if self.bias is not None:
            output = output + self.bias
        
        return output.to(x.dtype)


class HadamardW4A4LinearFast(nn.Module):
    """
    Hadamard+W4A4 量化线性层 (fast-hadamard-transform实现)
    - 使用fast-hadamard-transform库加速Hadamard变换
    - 权重：先进行Hadamard变换，然后4bit量化存储
    - 激活：直接量化为4bit（不进行Hadamard变换以避免输出维度限制）
    - 计算：W4A4矩阵乘法
    - 输出：反量化
    """
    def __init__(self, in_features, out_features, bias=False, group_size=128, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        assert in_features % group_size == 0, f"in_features must be divisible by group_size"
        
        # 4-bit 权重（打包为 uint8）
        self.register_buffer('weight_packed', torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer('weight_scale', torch.ones(out_features, dtype=dtype))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    
    def hadamard_transform_fast(self, x, group_size):
        """使用fast-hadamard-transform库的Hadamard变换"""
        orig_shape = x.shape
        x = x.reshape(-1, group_size)
        
        # 使用fast库
        h = hadamard_transform(x, scale=1.0/np.sqrt(group_size))
        
        return h.reshape(orig_shape)
    
    def quantize_weight(self, weight_fp16):
        """将FP16权重进行Hadamard变换并量化为4-bit"""
        # weight_fp16: (out_features, in_features)
        
        # 1. Hadamard变换（按group_size分组）
        weight_had = self.hadamard_transform_fast(weight_fp16, self.group_size)
        
        # 2. Per-channel量化
        weight_max = weight_had.abs().max(dim=1, keepdim=True)[0]
        scale = weight_max / 7.0
        scale = torch.clamp(scale, min=1e-5)
        
        weight_q = torch.round(weight_had / scale).clamp(-8, 7).to(torch.int8)
        
        # 3. 打包
        weight_q = weight_q.view(self.out_features, -1, 2)
        weight_packed = ((weight_q[:, :, 0] & 0xF) | ((weight_q[:, :, 1] & 0xF) << 4)).to(torch.uint8)
        
        self.weight_packed.copy_(weight_packed)
        self.weight_scale.copy_(scale.squeeze())
    
    def forward(self, x):
        """前向传播"""
        # 1. 量化激活（per-token量化）
        x_max = x.abs().max(dim=-1, keepdim=True)[0]
        x_scale = x_max / 7.0
        x_scale = torch.clamp(x_scale, min=1e-5)
        
        x_q = torch.round(x / x_scale).clamp(-8, 7).to(torch.int8)
        
        # 2. 解包权重
        weight_low = (self.weight_packed & 0xF).to(torch.int8)
        weight_high = ((self.weight_packed >> 4) & 0xF).to(torch.int8)
        weight_low = torch.where(weight_low > 7, weight_low - 16, weight_low)
        weight_high = torch.where(weight_high > 7, weight_high - 16, weight_high)
        weight_q = torch.stack([weight_low, weight_high], dim=-1).view(self.out_features, -1)
        
        # 3. 矩阵乘法（int计算）
        output = torch.matmul(x_q.float(), weight_q.t().float())
        
        # 4. 反量化
        output_scale = x_scale * self.weight_scale.unsqueeze(0)
        output = output * output_scale
        
        if self.bias is not None:
            output = output + self.bias
        
        return output.to(x.dtype)


def benchmark_layer(layer, x, warmup=10, iters=100, name="Layer"):
    """
    测试层的推理性能
    
    Args:
        layer: 要测试的层
        x: 输入张量
        warmup: 预热迭代次数
        iters: 测试迭代次数
        name: 层的名称
    
    Returns:
        平均推理时间（毫秒）
    """
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(x)
    
    # 同步
    torch.cuda.synchronize()
    
    # 创建 CUDA 事件用于精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 测试
    with torch.no_grad():
        start_event.record()
        for _ in range(iters):
            _ = layer(x)
        end_event.record()
    
    # 同步并计算时间
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / iters
    
    return avg_time_ms


def test_accuracy(linear_fp16, linear_4bit, x, name="Test"):
    """
    测试 4-bit 层的精度
    
    Args:
        linear_fp16: FP16 线性层
        linear_4bit: 4-bit 线性层
        x: 输入张量
        name: 测试名称
    """
    with torch.no_grad():
        output_fp16 = linear_fp16(x)
        output_4bit = linear_4bit(x)
    
    # 计算误差
    abs_error = (output_fp16 - output_4bit).abs()
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()
    
    rel_error = abs_error / (output_fp16.abs() + 1e-8)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()
    
    print(f"\n{name} - 精度分析:")
    print(f"  最大绝对误差: {max_abs_error:.6f}")
    print(f"  平均绝对误差: {mean_abs_error:.6f}")
    print(f"  最大相对误差: {max_rel_error:.2%}")
    print(f"  平均相对误差: {mean_rel_error:.2%}")
    
    # 计算余弦相似度
    cos_sim = nn.functional.cosine_similarity(
        output_fp16.flatten(), 
        output_4bit.flatten(), 
        dim=0
    ).item()
    print(f"  余弦相似度: {cos_sim:.6f}")


def run_benchmark(batch_size, in_features, out_features, group_size=128, 
                  warmup=20, iters=100):
    """
    运行单个配置的基准测试 - 对比 6 种方法
    
    Args:
        batch_size: 批大小
        in_features: 输入特征数
        out_features: 输出特征数
        group_size: 量化组大小（用于 HQL 和 Hadamard baselines）
        warmup: 预热次数
        iters: 测试迭代次数
    """
    print("\n" + "="*80)
    print(f"Config: Batch={batch_size}, In={in_features}, Out={out_features}, Group={group_size}")
    print("="*80)
    
    # 创建输入
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')
    
    # ========================================
    # 1. FP16 Baseline
    # ========================================
    linear_fp16 = nn.Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    weight_fp16 = linear_fp16.weight.data  # (out_features, in_features)
    
    # ========================================
    # 2. W4A4 Linear
    # ========================================
    print("\n[1/6] Creating W4A4 Linear...")
    linear_w4a4 = W4A4Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    with torch.no_grad():
        linear_w4a4.quantize_weight(weight_fp16)
    print("  ✓ W4A4 Linear created")
    
    # ========================================
    # 3. W4A16 Linear
    # ========================================
    print("\n[2/6] Creating W4A16 Linear...")
    linear_w4a16 = W4A16Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    with torch.no_grad():
        linear_w4a16.quantize_weight(weight_fp16)
    print("  ✓ W4A16 Linear created")
    
    # ========================================
    # 4. Hadamard+W4A4 (PyTorch) Linear
    # ========================================
    print("\n[3/6] Creating Hadamard+W4A4 (PyTorch) Linear...")
    linear_hadpt = HadamardW4A4LinearPyTorch(in_features, out_features, bias=False, 
                                             group_size=group_size, dtype=torch.float16).cuda()
    with torch.no_grad():
        linear_hadpt.quantize_weight(weight_fp16)
    print("  ✓ Hadamard+W4A4 (PyTorch) Linear created")
    
    # ========================================
    # 5. Hadamard+W4A4 (Fast) Linear
    # ========================================
    linear_hadfast = None
    if FAST_HADAMARD_AVAILABLE:
        print("\n[4/6] Creating Hadamard+W4A4 (Fast) Linear...")
        linear_hadfast = HadamardW4A4LinearFast(in_features, out_features, bias=False,
                                                group_size=group_size, dtype=torch.float16).cuda()
        with torch.no_grad():
            linear_hadfast.quantize_weight(weight_fp16)
        print("  ✓ Hadamard+W4A4 (Fast) Linear created")
    else:
        print("\n[4/6] Skipping Hadamard+W4A4 (Fast) - library not available")
    
    # ========================================
    # 6. HQL Linear4bit
    # ========================================
    print(f"\n[5/6] Creating HQL Linear4bit...")
    linear_hql = Linear4bit(in_features, out_features, bias=False, 
                           dtype=torch.float16, group_size=group_size).cuda()
    
    with torch.no_grad():
        had_weight = weight_fp16.view(-1, group_size)
        quantized_x, scales_x = hql.had_and_quant(had_weight)
        quantized_x = quantized_x.reshape(-1, in_features // 2)
        scales_x = scales_x.reshape(-1, in_features // group_size)
        
        linear_hql.weight.copy_(quantized_x)
        linear_hql.weight_scales.copy_(scales_x)
    print("  ✓ HQL Linear4bit created")
    
    # ========================================
    # Performance Testing
    # ========================================
    print("\n[6/6] Performance Testing...")
    print("\nBenchmarking...")
    
    time_fp16 = benchmark_layer(linear_fp16, x, warmup=warmup, iters=iters, name="FP16")
    print(f"  FP16:          {time_fp16:.4f} ms")
    
    time_w4a4 = benchmark_layer(linear_w4a4, x, warmup=warmup, iters=iters, name="W4A4")
    print(f"  W4A4:          {time_w4a4:.4f} ms")
    
    time_w4a16 = benchmark_layer(linear_w4a16, x, warmup=warmup, iters=iters, name="W4A16")
    print(f"  W4A16:         {time_w4a16:.4f} ms")
    
    time_hadpt = benchmark_layer(linear_hadpt, x, warmup=warmup, iters=iters, name="Hadamard+W4A4 (PyTorch)")
    print(f"  Had+W4A4 (PT): {time_hadpt:.4f} ms")
    
    time_hadfast = None
    if linear_hadfast is not None:
        time_hadfast = benchmark_layer(linear_hadfast, x, warmup=warmup, iters=iters, name="Hadamard+W4A4 (Fast)")
        print(f"  Had+W4A4 (F):  {time_hadfast:.4f} ms")
    
    time_hql = benchmark_layer(linear_hql, x, warmup=warmup, iters=iters, name="HQL")
    print(f"  HQL:           {time_hql:.4f} ms")
    
    # ========================================
    # Calculate Metrics
    # ========================================
    flops = 2 * batch_size * in_features * out_features
    
    throughput_fp16 = (flops / (time_fp16 / 1000)) / 1e12  # TFLOPS
    throughput_w4a4 = (flops / (time_w4a4 / 1000)) / 1e12
    throughput_w4a16 = (flops / (time_w4a16 / 1000)) / 1e12
    throughput_hadpt = (flops / (time_hadpt / 1000)) / 1e12
    throughput_hadfast = (flops / (time_hadfast / 1000)) / 1e12 if time_hadfast else None
    throughput_hql = (flops / (time_hql / 1000)) / 1e12
    
    speedup_w4a4 = time_fp16 / time_w4a4
    speedup_w4a16 = time_fp16 / time_w4a16
    speedup_hadpt = time_fp16 / time_hadpt
    speedup_hadfast = time_fp16 / time_hadfast if time_hadfast else None
    speedup_hql = time_fp16 / time_hql
    
    # ========================================
    # Print Results
    # ========================================
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Method':<20} {'Time (ms)':<12} {'Throughput':<15} {'Speedup':<12}")
    print("-"*80)
    print(f"{'FP16':<20} {time_fp16:<12.4f} {throughput_fp16:<15.2f} {'1.00x':<12}")
    print(f"{'W4A4':<20} {time_w4a4:<12.4f} {throughput_w4a4:<15.2f} {speedup_w4a4:<12.2f}x")
    print(f"{'W4A16':<20} {time_w4a16:<12.4f} {throughput_w4a16:<15.2f} {speedup_w4a16:<12.2f}x")
    print(f"{'Had+W4A4 (PT)':<20} {time_hadpt:<12.4f} {throughput_hadpt:<15.2f} {speedup_hadpt:<12.2f}x")
    if time_hadfast:
        print(f"{'Had+W4A4 (Fast)':<20} {time_hadfast:<12.4f} {throughput_hadfast:<15.2f} {speedup_hadfast:<12.2f}x")
    print(f"{'HQL':<20} {time_hql:<12.4f} {throughput_hql:<15.2f} {speedup_hql:<12.2f}x")
    print("="*80)
    
    # ========================================
    # Accuracy Testing
    # ========================================
    print("\nACCURACY ANALYSIS:")
    
    with torch.no_grad():
        output_fp16 = linear_fp16(x)
        output_w4a4 = linear_w4a4(x)
        output_w4a16 = linear_w4a16(x)
        output_hadpt = linear_hadpt(x)
        output_hadfast = linear_hadfast(x) if linear_hadfast else None
        output_hql = linear_hql(x)
    
    def calc_error(ref, test, name):
        abs_error = (ref - test).abs()
        max_abs = abs_error.max().item()
        mean_abs = abs_error.mean().item()
        cos_sim = nn.functional.cosine_similarity(ref.flatten(), test.flatten(), dim=0).item()
        print(f"  {name:<18}: Max Abs Err = {max_abs:.6f}, Mean Abs Err = {mean_abs:.6f}, Cos Sim = {cos_sim:.6f}")
    
    calc_error(output_fp16, output_w4a4, "W4A4")
    calc_error(output_fp16, output_w4a16, "W4A16")
    calc_error(output_fp16, output_hadpt, "Had+W4A4 (PT)")
    if output_hadfast is not None:
        calc_error(output_fp16, output_hadfast, "Had+W4A4 (Fast)")
    calc_error(output_fp16, output_hql, "HQL")
    
    # ========================================
    # Memory Footprint
    # ========================================
    memory_fp16 = in_features * out_features * 2  # FP16 = 2 bytes
    memory_w4a4 = (in_features * out_features * 1) + (out_features * 2)  # int8 + per-channel scales
    memory_w4_packed = (in_features * out_features // 2)  # Packed 4-bit
    memory_w4a16 = memory_w4_packed + (out_features * 2)  # + per-channel scales
    memory_hql = memory_w4_packed + (out_features * (in_features // group_size) * 2)  # + group scales
    
    print(f"\nMEMORY FOOTPRINT (Weight only):")
    print(f"  FP16:          {memory_fp16 / 1024 / 1024:.2f} MB")
    print(f"  W4A4:          {memory_w4a4 / 1024 / 1024:.2f} MB ({memory_w4a4/memory_fp16:.2%}) [int8 storage]")
    print(f"  W4A16:         {memory_w4a16 / 1024 / 1024:.2f} MB ({memory_w4a16/memory_fp16:.2%}) [packed 4-bit]")
    print(f"  Had+W4A4 (PT): {memory_hql / 1024 / 1024:.2f} MB ({memory_hql/memory_fp16:.2%}) [packed 4-bit]")
    if linear_hadfast:
        print(f"  Had+W4A4 (F):  {memory_hql / 1024 / 1024:.2f} MB ({memory_hql/memory_fp16:.2%}) [packed 4-bit]")
    print(f"  HQL:           {memory_hql / 1024 / 1024:.2f} MB ({memory_hql/memory_fp16:.2%}) [packed 4-bit]")
    
    result = {
        'config': f"B{batch_size}_I{in_features}_O{out_features}",
        'time_fp16': time_fp16,
        'time_w4a4': time_w4a4,
        'time_w4a16': time_w4a16,
        'time_hadpt': time_hadpt,
        'time_hql': time_hql,
        'speedup_w4a4': speedup_w4a4,
        'speedup_w4a16': speedup_w4a16,
        'speedup_hadpt': speedup_hadpt,
        'speedup_hql': speedup_hql,
        'throughput_fp16': throughput_fp16,
        'throughput_w4a4': throughput_w4a4,
        'throughput_w4a16': throughput_w4a16,
        'throughput_hadpt': throughput_hadpt,
        'throughput_hql': throughput_hql,
    }
    
    if time_hadfast:
        result['time_hadfast'] = time_hadfast
        result['speedup_hadfast'] = speedup_hadfast
        result['throughput_hadfast'] = throughput_hadfast
    
    return result


def save_results_to_excel(results, output_file="benchmark_results.xlsx"):
    """
    将基准测试结果保存到 Excel 文件
    
    Args:
        results: 结果列表
        output_file: 输出文件名
    """
    if not results:
        print("没有结果可保存")
        return
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 解析配置信息
    df['batch_size'] = df['config'].apply(lambda x: int(x.split('_')[0][1:]))
    df['in_features'] = df['config'].apply(lambda x: int(x.split('_')[1][1:]))
    df['out_features'] = df['config'].apply(lambda x: int(x.split('_')[2][1:]))
    
    # 重新排列列
    columns_order = [
        'config', 'batch_size', 'in_features', 'out_features',
        'time_fp16', 'time_w4a4', 'time_w4a16', 'time_hadpt', 'time_hql',
        'speedup_w4a4', 'speedup_w4a16', 'speedup_hadpt', 'speedup_hql',
        'throughput_fp16', 'throughput_w4a4', 'throughput_w4a16', 'throughput_hadpt', 'throughput_hql'
    ]
    
    # 添加 hadfast 列（如果存在）
    if 'time_hadfast' in df.columns:
        columns_order.insert(columns_order.index('time_hql'), 'time_hadfast')
        columns_order.insert(columns_order.index('speedup_hql'), 'speedup_hadfast')
        columns_order.insert(columns_order.index('throughput_hql'), 'throughput_hadfast')
    
    # 只保留存在的列
    columns_order = [col for col in columns_order if col in df.columns]
    df = df[columns_order]
    
    # 计算统计信息
    stats_data = {
        '指标': [
            '平均时间 (ms)',
            '平均加速比',
            '最大加速比',
            '最小加速比',
            '平均吞吐量 (TFLOPS)'
        ],
        'FP16': [
            df['time_fp16'].mean(),
            1.0,
            1.0,
            1.0,
            df['throughput_fp16'].mean()
        ],
        'W4A4': [
            df['time_w4a4'].mean(),
            df['speedup_w4a4'].mean(),
            df['speedup_w4a4'].max(),
            df['speedup_w4a4'].min(),
            df['throughput_w4a4'].mean()
        ],
        'W4A16': [
            df['time_w4a16'].mean(),
            df['speedup_w4a16'].mean(),
            df['speedup_w4a16'].max(),
            df['speedup_w4a16'].min(),
            df['throughput_w4a16'].mean()
        ],
        'Had+W4A4(PT)': [
            df['time_hadpt'].mean(),
            df['speedup_hadpt'].mean(),
            df['speedup_hadpt'].max(),
            df['speedup_hadpt'].min(),
            df['throughput_hadpt'].mean()
        ],
        'HQL': [
            df['time_hql'].mean(),
            df['speedup_hql'].mean(),
            df['speedup_hql'].max(),
            df['speedup_hql'].min(),
            df['throughput_hql'].mean()
        ]
    }
    
    # 添加 fast hadamard 统计（如果存在）
    if 'time_hadfast' in df.columns:
        stats_data['Had+W4A4(F)'] = [
            df['time_hadfast'].mean(),
            df['speedup_hadfast'].mean(),
            df['speedup_hadfast'].max(),
            df['speedup_hadfast'].min(),
            df['throughput_hadfast'].mean()
        ]
    
    stats_df = pd.DataFrame(stats_data)
    
    # 添加系统信息
    system_info = {
        '项目': ['GPU', 'PyTorch', 'CUDA', '测试时间', '测试配置数量'],
        '值': [
            torch.cuda.get_device_name(0),
            torch.__version__,
            torch.version.cuda,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            len(results)
        ]
    }
    system_df = pd.DataFrame(system_info)
    
    # 保存到 Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 详细结果
        df.to_excel(writer, sheet_name='详细结果', index=False)
        
        # 统计汇总
        stats_df.to_excel(writer, sheet_name='统计汇总', index=False)
        
        # 系统信息
        system_df.to_excel(writer, sheet_name='系统信息', index=False)
        
        # 按 batch_size 分组的结果
        for batch_size in sorted(df['batch_size'].unique()):
            sheet_name = f'Batch_{batch_size}'
            df_batch = df[df['batch_size'] == batch_size].copy()
            df_batch.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\n✓ 结果已保存到: {output_file}")
    print(f"  - 详细结果: 包含所有测试配置的完整数据")
    print(f"  - 统计汇总: 平均性能、加速比等统计信息")
    print(f"  - 系统信息: GPU、PyTorch、CUDA 版本等")
    print(f"  - 按 Batch 分组: 每个 batch size 单独一个工作表")


def main():
    """主测试函数"""
    print("\n" + "="*100)
    print("4-bit Linear Layer Performance Benchmark")
    print("Comparing: FP16 | W4A4 | W4A16 | Had+W4A4(PT) | Had+W4A4(Fast) | HQL")
    print("="*100)
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # 测试配置列表
    test_configs = [
        # (1, 4096, 4096, 128),      # Large model
        # (1, 4096, 11008, 128),     # LLaMA FFN (up)
        # (1, 11008, 4096, 128),     # LLaMA FFN (down)
        # (2, 4096, 4096, 128),      # Large model
        # (2, 4096, 11008, 128),     # LLaMA FFN (up)
        # (2, 11008, 4096, 128),     # LLaMA FFN (down)
        # (4, 4096, 4096, 128),      # Large model
        # (4, 4096, 11008, 128),     # LLaMA FFN (up)
        # (4, 11008, 4096, 128),     # LLaMA FFN (down)
        # (8, 4096, 4096, 128),      # Large model
        # (8, 4096, 11008, 128),     # LLaMA FFN (up)
        # (8, 11008, 4096, 128),     # LLaMA FFN (down)
        
        (128, 4096, 4096, 128),      # Large model
        (128, 4096, 14336, 128),     # LLaMA FFN (up)
        (128, 14336, 4096, 128),     # LLaMA FFN (down)
        (256, 4096, 4096, 128),      # Large model
        (256, 4096, 14336, 128),     # LLaMA FFN (up)
        (256, 14336, 4096, 128),     # LLaMA FFN (down)
        (512, 4096, 4096, 128),      # Large model
        (512, 4096, 14336, 128),     # LLaMA FFN (up)
        (512, 14336, 4096, 128),     # LLaMA FFN (down)
        (1024, 4096, 4096, 128),      # Large model
        (1024, 4096, 14336, 128),     # LLaMA FFN (up)
        (1024, 14336, 4096, 128),     # LLaMA FFN (down)
    ]
    
    # 运行所有测试
    results = []
    for batch_size, in_features, out_features, group_size in test_configs:
        try:
            result = run_benchmark(
                batch_size, in_features, out_features, 
                group_size=group_size, warmup=20, iters=100
            )
            results.append(result)
        except Exception as e:
            print(f"\nError: Config (B{batch_size}, I{in_features}, O{out_features}) failed")
            print(f"Error message: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总结果
    print("\n" + "="*130)
    print("SUMMARY RESULTS")
    print("="*130)
    
    # 检查是否有 hadfast 结果
    has_hadfast = any('time_hadfast' in r for r in results)
    
    if has_hadfast:
        print(f"\n{'Config':<20} {'FP16(ms)':<11} {'W4A4(ms)':<11} {'W4A16(ms)':<11} {'HadPT(ms)':<11} {'HadF(ms)':<11} {'HQL(ms)':<11} {'W4A4x':<8} {'W4A16x':<8} {'HadPTx':<8} {'HadFx':<8} {'HQLx':<8}")
        print("-"*130)
        
        for r in results:
            if 'time_hadfast' in r:
                print(f"{r['config']:<20} "
                      f"{r['time_fp16']:<11.4f} "
                      f"{r['time_w4a4']:<11.4f} "
                      f"{r['time_w4a16']:<11.4f} "
                      f"{r['time_hadpt']:<11.4f} "
                      f"{r['time_hadfast']:<11.4f} "
                      f"{r['time_hql']:<11.4f} "
                      f"{r['speedup_w4a4']:<8.2f} "
                      f"{r['speedup_w4a16']:<8.2f} "
                      f"{r['speedup_hadpt']:<8.2f} "
                      f"{r['speedup_hadfast']:<8.2f} "
                      f"{r['speedup_hql']:<8.2f}")
            else:
                print(f"{r['config']:<20} "
                      f"{r['time_fp16']:<11.4f} "
                      f"{r['time_w4a4']:<11.4f} "
                      f"{r['time_w4a16']:<11.4f} "
                      f"{r['time_hadpt']:<11.4f} "
                      f"{'N/A':<11} "
                      f"{r['time_hql']:<11.4f} "
                      f"{r['speedup_w4a4']:<8.2f} "
                      f"{r['speedup_w4a16']:<8.2f} "
                      f"{r['speedup_hadpt']:<8.2f} "
                      f"{'N/A':<8} "
                      f"{r['speedup_hql']:<8.2f}")
    else:
        print(f"\n{'Config':<20} {'FP16(ms)':<11} {'W4A4(ms)':<11} {'W4A16(ms)':<11} {'HadPT(ms)':<11} {'HQL(ms)':<11} {'W4A4x':<8} {'W4A16x':<8} {'HadPTx':<8} {'HQLx':<8}")
        print("-"*120)
        
        for r in results:
            print(f"{r['config']:<20} "
                  f"{r['time_fp16']:<11.4f} "
                  f"{r['time_w4a4']:<11.4f} "
                  f"{r['time_w4a16']:<11.4f} "
                  f"{r['time_hadpt']:<11.4f} "
                  f"{r['time_hql']:<11.4f} "
                  f"{r['speedup_w4a4']:<8.2f} "
                  f"{r['speedup_w4a16']:<8.2f} "
                  f"{r['speedup_hadpt']:<8.2f} "
                  f"{r['speedup_hql']:<8.2f}")
    
    # 计算平均值
    if results:
        avg_speedup_w4a4 = np.mean([r['speedup_w4a4'] for r in results])
        avg_speedup_w4a16 = np.mean([r['speedup_w4a16'] for r in results])
        avg_speedup_hadpt = np.mean([r['speedup_hadpt'] for r in results])
        avg_speedup_hql = np.mean([r['speedup_hql'] for r in results])
        
        if has_hadfast:
            avg_speedup_hadfast = np.mean([r.get('speedup_hadfast', 0) for r in results if 'speedup_hadfast' in r])
            print("-"*130)
            print(f"{'Average':<20} {'':<11} {'':<11} {'':<11} {'':<11} {'':<11} {'':<11} "
                  f"{avg_speedup_w4a4:<8.2f} {avg_speedup_w4a16:<8.2f} {avg_speedup_hadpt:<8.2f} {avg_speedup_hadfast:<8.2f} {avg_speedup_hql:<8.2f}")
        else:
            print("-"*120)
            print(f"{'Average':<20} {'':<11} {'':<11} {'':<11} {'':<11} {'':<11} "
                  f"{avg_speedup_w4a4:<8.2f} {avg_speedup_w4a16:<8.2f} {avg_speedup_hadpt:<8.2f} {avg_speedup_hql:<8.2f}")
    
    print("\n" + "="*100)
    print("KEY FINDINGS:")
    print("="*100)
    if results:
        print(f"  - W4A4 average speedup:        {avg_speedup_w4a4:.2f}x over FP16")
        print(f"  - W4A16 average speedup:       {avg_speedup_w4a16:.2f}x over FP16")
        print(f"  - Had+W4A4(PT) average speedup:{avg_speedup_hadpt:.2f}x over FP16")
        if has_hadfast:
            print(f"  - Had+W4A4(F) average speedup: {avg_speedup_hadfast:.2f}x over FP16")
        print(f"  - HQL average speedup:         {avg_speedup_hql:.2f}x over FP16")
        print(f"\n  HQL vs W4A4:        {avg_speedup_hql/avg_speedup_w4a4:.2f}x")
        print(f"  HQL vs W4A16:       {avg_speedup_hql/avg_speedup_w4a16:.2f}x")
        print(f"  HQL vs Had+W4A4(PT):{avg_speedup_hql/avg_speedup_hadpt:.2f}x")
        if has_hadfast:
            print(f"  HQL vs Had+W4A4(F): {avg_speedup_hql/avg_speedup_hadfast:.2f}x")
    
    print("\n" + "="*100)
    print("✓ Benchmark completed")
    print("="*100 + "\n")
    
    # 保存结果到 Excel
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"benchmark_linear4bit_{timestamp}.xlsx"
    save_results_to_excel(results, output_file)


if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    main()


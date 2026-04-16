#!/usr/bin/env python3
"""
LinearFLH 性能测试
比较多种量化方法的推理性能：
1. FP16 baseline
2. W4A4: 权重4bit，激活4bit
3. W4A16: 权重4bit，激活16bit（反量化后计算）
4. Hadamard+W4A4 (PyTorch): PyTorch实现的Hadamard变换 + W4A4
5. Hadamard+W4A4 (Fast): fast-hadamard-transform库实现的Hadamard变换 + W4A4
6. LinearFLH: 我们的 Hadamard + 4bit 量化方法
"""

import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import flh
from flh.nn.linear import LinearFLH
from flh.nn.quantization import ActQuantizer

import quarot

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
        
        self.register_buffer('weight_q', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features, dtype=dtype))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    
    def quantize_weight(self, weight_fp16):
        weight_max = weight_fp16.abs().max(dim=1, keepdim=True)[0]
        scale = weight_max / 7.0
        scale = torch.clamp(scale, min=1e-5)
        
        weight_q = torch.round(weight_fp16 / scale).clamp(-8, 7).to(torch.int8)
        
        self.weight_q.copy_(weight_q)
        self.weight_scale.copy_(scale.squeeze())
    
    def quantize_activation(self, x):
        x_max = x.abs().max(dim=-1, keepdim=True)[0]
        scale = x_max / 7.0
        scale = torch.clamp(scale, min=1e-5)
        
        x_q = torch.round(x / scale).clamp(-8, 7).to(torch.int8)
        return x_q, scale
    
    def forward(self, x):
        x_q, x_scale = self.quantize_activation(x)
        output = quarot.matmul(x_q.to(torch.uint8), self.weight_q.to(torch.uint8).contiguous())
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
        
        self.register_buffer('weight_packed', torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer('weight_scale', torch.ones(out_features, dtype=dtype))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    
    def quantize_weight(self, weight_fp16):
        weight_max = weight_fp16.abs().max(dim=1, keepdim=True)[0]
        scale = weight_max / 7.0
        scale = torch.clamp(scale, min=1e-5)
        
        weight_q = torch.round(weight_fp16 / scale).clamp(-8, 7).to(torch.int8)
        
        weight_q = weight_q.view(self.out_features, -1, 2)
        weight_packed = ((weight_q[:, :, 0] & 0xF) | ((weight_q[:, :, 1] & 0xF) << 4)).to(torch.uint8)
        
        self.weight_packed.copy_(weight_packed)
        self.weight_scale.copy_(scale.squeeze())
    
    def forward(self, x):
        weight_low = (self.weight_packed & 0xF).to(torch.int8)
        weight_high = ((self.weight_packed >> 4) & 0xF).to(torch.int8)
        
        weight_low = torch.where(weight_low > 7, weight_low - 16, weight_low)
        weight_high = torch.where(weight_high > 7, weight_high - 16, weight_high)
        
        weight_q = torch.stack([weight_low, weight_high], dim=-1).view(self.out_features, -1)
        
        weight_fp16 = weight_q.to(x.dtype) * self.weight_scale.unsqueeze(1)
        output = torch.matmul(x, weight_fp16.t())
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class HadamardW4A4LinearPyTorch(nn.Module):
    """
    Hadamard+W4A4 量化线性层 (PyTorch实现)
    - 权重：先进行Hadamard变换，然后4bit量化���储  
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
        
        self.register_buffer('weight_packed', torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer('weight_scale', torch.ones(out_features, dtype=dtype))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    
    def hadamard_transform_pytorch(self, x, group_size):
        orig_shape = x.shape
        x = x.reshape(-1, group_size)
        
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
        
        h = h / np.sqrt(n)
        
        return h.reshape(orig_shape)
    
    def quantize_weight(self, weight_fp16):
        weight_had = self.hadamard_transform_pytorch(weight_fp16, self.group_size)
        
        weight_max = weight_had.abs().max(dim=1, keepdim=True)[0]
        scale = weight_max / 7.0
        scale = torch.clamp(scale, min=1e-5)
        
        weight_q = torch.round(weight_had / scale).clamp(-8, 7).to(torch.int8)
        
        weight_q = weight_q.view(self.out_features, -1, 2)
        weight_packed = ((weight_q[:, :, 0] & 0xF) | ((weight_q[:, :, 1] & 0xF) << 4)).to(torch.uint8)
        
        self.weight_packed.copy_(weight_packed)
        self.weight_scale.copy_(scale.squeeze())
    
    def forward(self, x):
        x_max = x.abs().max(dim=-1, keepdim=True)[0]
        x_scale = x_max / 7.0
        x_scale = torch.clamp(x_scale, min=1e-5)
        
        x_q = torch.round(x / x_scale).clamp(-8, 7).to(torch.int8)
        
        weight_low = (self.weight_packed & 0xF).to(torch.int8)
        weight_high = ((self.weight_packed >> 4) & 0xF).to(torch.int8)
        weight_low = torch.where(weight_low > 7, weight_low - 16, weight_low)
        weight_high = torch.where(weight_high > 7, weight_high - 16, weight_high)
        weight_q = torch.stack([weight_low, weight_high], dim=-1).view(self.out_features, -1)
        
        output = torch.matmul(x_q.float(), weight_q.t().float())
        
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
        
        self.register_buffer('weight_packed', torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer('weight_scale', torch.ones(out_features, dtype=dtype))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    
    def hadamard_transform_fast(self, x, group_size):
        orig_shape = x.shape
        x = x.reshape(-1, group_size)
        
        h = hadamard_transform(x, scale=1.0/np.sqrt(group_size))
        
        return h.reshape(orig_shape)
    
    def quantize_weight(self, weight_fp16):
        weight_had = self.hadamard_transform_fast(weight_fp16, self.group_size)
        
        weight_max = weight_had.abs().max(dim=1, keepdim=True)[0]
        scale = weight_max / 7.0
        scale = torch.clamp(scale, min=1e-5)
        
        weight_q = torch.round(weight_had / scale).clamp(-8, 7).to(torch.int8)
        
        weight_q = weight_q.view(self.out_features, -1, 2)
        weight_packed = ((weight_q[:, :, 0] & 0xF) | ((weight_q[:, :, 1] & 0xF) << 4)).to(torch.uint8)
        
        self.weight_packed.copy_(weight_packed)
        self.weight_scale.copy_(scale.squeeze())
    
    def forward(self, x):
        x_max = x.abs().max(dim=-1, keepdim=True)[0]
        x_scale = x_max / 7.0
        x_scale = torch.clamp(x_scale, min=1e-5)
        
        x_q = torch.round(x / x_scale).clamp(-8, 7).to(torch.int8)
        
        weight_low = (self.weight_packed & 0xF).to(torch.int8)
        weight_high = ((self.weight_packed >> 4) & 0xF).to(torch.int8)
        weight_low = torch.where(weight_low > 7, weight_low - 16, weight_low)
        weight_high = torch.where(weight_high > 7, weight_high - 16, weight_high)
        weight_q = torch.stack([weight_low, weight_high], dim=-1).view(self.out_features, -1)
        
        output = torch.matmul(x_q.float(), weight_q.t().float())
        
        output_scale = x_scale * self.weight_scale.unsqueeze(0)
        output = output * output_scale
        
        if self.bias is not None:
            output = output + self.bias
        
        return output.to(x.dtype)


class LinearFLHWrapper(nn.Module):
    """
    LinearFLH 包装器
    - 输入：先使用 ActQuantizer 进行量化
    - 使用 LinearFLH 进行推理
    """
    def __init__(self, in_features, out_features, bias=False, group_size=128, dtype=torch.float16,
                 act_bits=4, act_group_size=128, act_sym=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dtype = dtype
        
        assert in_features % group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        
        self.linear_flh = LinearFLH(
            in_features, out_features, bias=bias, 
            dtype=dtype, device='cuda',
            group_size=group_size,
            in_group_size=group_size
        )
        
        self.act_quantizer = ActQuantizer(
            bits=act_bits,
            group_size=act_group_size,
            sym=act_sym,
            use_hadamard=False,
            packed_output=True
        )
    
    def quantize_weight(self, weight_fp16):
        """对权重进行 Hadamard 变换和量化"""
        from flh.nn import quantization as _quant
        
        weight_quantizer = _quant.WeightQuantizer(
            bits=4,
            group_size=self.group_size,
            sym=True,
            channel_wise=False,
            use_hadamard=False,
            clip_ratio=1.0
        )
        
        weight_had = _quant.fast_hadamard_transform(
            weight_fp16,
            group_size=self.group_size,
            normalize=True,
            use_cuda=False,
        )
        
        weight_quantizer.calibrate(weight_had)
        scale, zero, w_int = weight_quantizer.quantize(weight_had)
        
        if scale.dim() == 3:
            scale = scale.squeeze(-1)
            zero = zero.squeeze(-1) if zero is not None else torch.zeros_like(scale, dtype=torch.int32)
        
        self.linear_flh._pack_weights(w_int, scale, zero)
    
    def forward(self, x):
        """
        前向传播：输入 x 需要是 (batch, in_features)
        每次前向都对输入进行量化
        """
        scales, zeros, x_packed = self.act_quantizer(x)
        
        output = self.linear_flh(
            x_packed,
            a_scale=scales,
            a_zero=zeros,
            x_is_packed=True,
            is_symmetric=True
        )
        
        return output


def benchmark_layer(layer, x, warmup=10, iters=100, name="Layer"):
    """测试层的推理性能"""
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(x)
    
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        start_event.record()
        for _ in range(iters):
            _ = layer(x)
        end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / iters
    
    return avg_time_ms


def test_accuracy(linear_fp16, linear_4bit, x, name="Test"):
    """测试 4-bit 层的精度"""
    with torch.no_grad():
        output_fp16 = linear_fp16(x)
        output_4bit = linear_4bit(x)
    
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
    
    cos_sim = nn.functional.cosine_similarity(
        output_fp16.flatten(), 
        output_4bit.flatten(), 
        dim=0
    ).item()
    print(f"  余弦相似度: {cos_sim:.6f}")


def run_benchmark(batch_size, in_features, out_features, group_size=128, 
                  warmup=20, iters=100):
    """运行单个配置的基准测试"""
    print("\n" + "="*80)
    print(f"Config: Batch={batch_size}, In={in_features}, Out={out_features}, Group={group_size}")
    print("="*80)
    
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')
    
    # 1. FP16 Baseline
    linear_fp16 = nn.Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    weight_fp16 = linear_fp16.weight.data
    
    # 2. W4A4 Linear
    print("\n[1/6] Creating W4A4 Linear...")
    linear_w4a4 = W4A4Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    with torch.no_grad():
        linear_w4a4.quantize_weight(weight_fp16)
    print("  ✓ W4A4 Linear created")
    
    # 3. W4A16 Linear
    print("\n[2/6] Creating W4A16 Linear...")
    linear_w4a16 = W4A16Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    with torch.no_grad():
        linear_w4a16.quantize_weight(weight_fp16)
    print("  ✓ W4A16 Linear created")
    
    # 4. Hadamard+W4A4 (PyTorch) Linear
    print("\n[3/6] Creating Hadamard+W4A4 (PyTorch) Linear...")
    linear_hadpt = HadamardW4A4LinearPyTorch(in_features, out_features, bias=False, 
                                             group_size=group_size, dtype=torch.float16).cuda()
    with torch.no_grad():
        linear_hadpt.quantize_weight(weight_fp16)
    print("  ✓ Hadamard+W4A4 (PyTorch) Linear created")
    
    # 5. Hadamard+W4A4 (Fast) Linear
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
    
    # 6. LinearFLH
    print(f"\n[5/6] Creating LinearFLH...")
    linear_flh = LinearFLHWrapper(in_features, out_features, bias=False, 
                                   group_size=group_size, dtype=torch.float16).cuda()
    with torch.no_grad():
        linear_flh.quantize_weight(weight_fp16)
    print("  ✓ LinearFLH created")
    
    # Performance Testing
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
    
    time_flh = benchmark_layer(linear_flh, x, warmup=warmup, iters=iters, name="LinearFLH")
    print(f"  LinearFLH:     {time_flh:.4f} ms")
    
    # Calculate Metrics
    flops = 2 * batch_size * in_features * out_features
    
    throughput_fp16 = (flops / (time_fp16 / 1000)) / 1e12
    throughput_w4a4 = (flops / (time_w4a4 / 1000)) / 1e12
    throughput_w4a16 = (flops / (time_w4a16 / 1000)) / 1e12
    throughput_hadpt = (flops / (time_hadpt / 1000)) / 1e12
    throughput_hadfast = (flops / (time_hadfast / 1000)) / 1e12 if time_hadfast else None
    throughput_flh = (flops / (time_flh / 1000)) / 1e12
    
    speedup_w4a4 = time_fp16 / time_w4a4
    speedup_w4a16 = time_fp16 / time_w4a16
    speedup_hadpt = time_fp16 / time_hadpt
    speedup_hadfast = time_fp16 / time_hadfast if time_hadfast else None
    speedup_flh = time_fp16 / time_flh
    
    # Print Results
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
    print(f"{'LinearFLH':<20} {time_flh:<12.4f} {throughput_flh:<15.2f} {speedup_flh:<12.2f}x")
    print("="*80)
    
    # Accuracy Testing
    print("\nACCURACY ANALYSIS:")
    
    with torch.no_grad():
        output_fp16 = linear_fp16(x)
        output_w4a4 = linear_w4a4(x)
        output_w4a16 = linear_w4a16(x)
        output_hadpt = linear_hadpt(x)
        output_hadfast = linear_hadfast(x) if linear_hadfast else None
        output_flh = linear_flh(x)
    
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
    calc_error(output_fp16, output_flh, "LinearFLH")
    
    # Memory Footprint
    memory_fp16 = in_features * out_features * 2
    memory_w4a4 = (in_features * out_features * 1) + (out_features * 2)
    memory_w4_packed = (in_features * out_features // 2)
    memory_w4a16 = memory_w4_packed + (out_features * 2)
    memory_flh = memory_w4_packed + (out_features * (in_features // group_size) * 2)
    
    print(f"\nMEMORY FOOTPRINT (Weight only):")
    print(f"  FP16:          {memory_fp16 / 1024 / 1024:.2f} MB")
    print(f"  W4A4:          {memory_w4a4 / 1024 / 1024:.2f} MB ({memory_w4a4/memory_fp16:.2%}) [int8 storage]")
    print(f"  W4A16:         {memory_w4a16 / 1024 / 1024:.2f} MB ({memory_w4a16/memory_fp16:.2%}) [packed 4-bit]")
    print(f"  Had+W4A4 (PT): {memory_flh / 1024 / 1024:.2f} MB ({memory_flh/memory_fp16:.2%}) [packed 4-bit]")
    if linear_hadfast:
        print(f"  Had+W4A4 (F):  {memory_flh / 1024 / 1024:.2f} MB ({memory_flh/memory_fp16:.2%}) [packed 4-bit]")
    print(f"  LinearFLH:     {memory_flh / 1024 / 1024:.2f} MB ({memory_flh/memory_fp16:.2%}) [packed 4-bit]")
    
    result = {
        'config': f"B{batch_size}_I{in_features}_O{out_features}",
        'time_fp16': time_fp16,
        'time_w4a4': time_w4a4,
        'time_w4a16': time_w4a16,
        'time_hadpt': time_hadpt,
        'time_flh': time_flh,
        'speedup_w4a4': speedup_w4a4,
        'speedup_w4a16': speedup_w4a16,
        'speedup_hadpt': speedup_hadpt,
        'speedup_flh': speedup_flh,
        'throughput_fp16': throughput_fp16,
        'throughput_w4a4': throughput_w4a4,
        'throughput_w4a16': throughput_w4a16,
        'throughput_hadpt': throughput_hadpt,
        'throughput_flh': throughput_flh,
    }
    
    if time_hadfast:
        result['time_hadfast'] = time_hadfast
        result['speedup_hadfast'] = speedup_hadfast
        result['throughput_hadfast'] = throughput_hadfast
    
    return result


def main():
    """主测试函数"""
    print("\n" + "="*100)
    print("LinearFLH Performance Benchmark")
    print("Comparing: FP16 | W4A4 | W4A16 | Had+W4A4(PT) | Had+W4A4(Fast) | LinearFLH")
    print("="*100)
    
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    test_configs = [
        # (512, 4096, 4096, 128),
        # (512, 4096, 11008, 128),
        # (512, 11008, 4096, 128),
        
        (512, 4096, 4096, 128),
        (512, 4096, 14336, 128),
        (512, 14336, 4096, 128),
    ]
    
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
    
    print("\n" + "="*130)
    print("SUMMARY RESULTS")
    print("="*130)
    
    has_hadfast = any('time_hadfast' in r for r in results)
    
    if has_hadfast:
        print(f"\n{'Config':<20} {'FP16(ms)':<11} {'W4A4(ms)':<11} {'W4A16(ms)':<11} {'HadPT(ms)':<11} {'HadF(ms)':<11} {'FLH(ms)':<11} {'W4A4x':<8} {'W4A16x':<8} {'HadPTx':<8} {'HadFx':<8} {'FLHx':<8}")
        print("-"*130)
        
        for r in results:
            if 'time_hadfast' in r:
                print(f"{r['config']:<20} "
                      f"{r['time_fp16']:<11.4f} "
                      f"{r['time_w4a4']:<11.4f} "
                      f"{r['time_w4a16']:<11.4f} "
                      f"{r['time_hadpt']:<11.4f} "
                      f"{r['time_hadfast']:<11.4f} "
                      f"{r['time_flh']:<11.4f} "
                      f"{r['speedup_w4a4']:<8.2f} "
                      f"{r['speedup_w4a16']:<8.2f} "
                      f"{r['speedup_hadpt']:<8.2f} "
                      f"{r['speedup_hadfast']:<8.2f} "
                      f"{r['speedup_flh']:<8.2f}")
            else:
                print(f"{r['config']:<20} "
                      f"{r['time_fp16']:<11.4f} "
                      f"{r['time_w4a4']:<11.4f} "
                      f"{r['time_w4a16']:<11.4f} "
                      f"{r['time_hadpt']:<11.4f} "
                      f"{'N/A':<11} "
                      f"{r['time_flh']:<11.4f} "
                      f"{r['speedup_w4a4']:<8.2f} "
                      f"{r['speedup_w4a16']:<8.2f} "
                      f"{r['speedup_hadpt']:<8.2f} "
                      f"{'N/A':<8} "
                      f"{r['speedup_flh']:<8.2f}")
    else:
        print(f"\n{'Config':<20} {'FP16(ms)':<11} {'W4A4(ms)':<11} {'W4A16(ms)':<11} {'HadPT(ms)':<11} {'FLH(ms)':<11} {'W4A4x':<8} {'W4A16x':<8} {'HadPTx':<8} {'FLHx':<8}")
        print("-"*110)
        
        for r in results:
            print(f"{r['config']:<20} "
                  f"{r['time_fp16']:<11.4f} "
                  f"{r['time_w4a4']:<11.4f} "
                  f"{r['time_w4a16']:<11.4f} "
                  f"{r['time_hadpt']:<11.4f} "
                  f"{r['time_flh']:<11.4f} "
                  f"{r['speedup_w4a4']:<8.2f} "
                  f"{r['speedup_w4a16']:<8.2f} "
                  f"{r['speedup_hadpt']:<8.2f} "
                  f"{r['speedup_flh']:<8.2f}")
    
    if results:
        avg_speedup_w4a4 = np.mean([r['speedup_w4a4'] for r in results])
        avg_speedup_w4a16 = np.mean([r['speedup_w4a16'] for r in results])
        avg_speedup_hadpt = np.mean([r['speedup_hadpt'] for r in results])
        avg_speedup_flh = np.mean([r['speedup_flh'] for r in results])
        
        if has_hadfast:
            avg_speedup_hadfast = np.mean([r.get('speedup_hadfast', 0) for r in results if 'speedup_hadfast' in r])
            print("-"*130)
            print(f"{'Average':<20} {'':<11} {'':<11} {'':<11} {'':<11} {'':<11} {'':<11} "
                  f"{avg_speedup_w4a4:<8.2f} {avg_speedup_w4a16:<8.2f} {avg_speedup_hadpt:<8.2f} {avg_speedup_hadfast:<8.2f} {avg_speedup_flh:<8.2f}")
        else:
            print("-"*110)
            print(f"{'Average':<20} {'':<11} {'':<11} {'':<11} {'':<11} {'':<11} "
                  f"{avg_speedup_w4a4:<8.2f} {avg_speedup_w4a16:<8.2f} {avg_speedup_hadpt:<8.2f} {avg_speedup_flh:<8.2f}")
    
    print("\n" + "="*100)
    print("KEY FINDINGS:")
    print("="*100)
    if results:
        print(f"  - W4A4 average speedup:        {avg_speedup_w4a4:.2f}x over FP16")
        print(f"  - W4A16 average speedup:       {avg_speedup_w4a16:.2f}x over FP16")
        print(f"  - Had+W4A4(PT) average speedup:{avg_speedup_hadpt:.2f}x over FP16")
        if has_hadfast:
            print(f"  - Had+W4A4(F) average speedup: {avg_speedup_hadfast:.2f}x over FP16")
        print(f"  - LinearFLH average speedup:    {avg_speedup_flh:.2f}x over FP16")
        print(f"\n  LinearFLH vs W4A4:        {avg_speedup_flh/avg_speedup_w4a4:.2f}x")
        print(f"  LinearFLH vs W4A16:       {avg_speedup_flh/avg_speedup_w4a16:.2f}x")
        print(f"  LinearFLH vs Had+W4A4(PT): {avg_speedup_flh/avg_speedup_hadpt:.2f}x")
        if has_hadfast:
            print(f"  LinearFLH vs Had+W4A4(F): {avg_speedup_flh/avg_speedup_hadfast:.2f}x")
    
    print("\n" + "="*100)
    print("✓ Benchmark completed")
    print("="*100 + "\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    main()

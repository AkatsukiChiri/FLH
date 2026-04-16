#!/usr/bin/env python3
"""
LinearFLH-GS 全组别对比测试
对比 group_size = 32, 64, 128 的性能差异
"""

from turtle import window_height
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
from flh.nn.linear_gs import LinearFLHGS
from flh.nn.quantization_gs import ActQuantizerGS
from flh.nn.quantization import WeightQuantizer, fast_hadamard_transform

torch.manual_seed(42)
np.random.seed(42)


class W4A4LinearQuarot(nn.Module):
    """W4A4 量化线性层（使用 quarot matmul）"""
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
        import quarot
        x_q, x_scale = self.quantize_activation(x)
        output = quarot.matmul(x_q.to(torch.uint8), self.weight_q.to(torch.uint8).contiguous())
        output_scale = x_scale * self.weight_scale.unsqueeze(0)
        output = output * output_scale
        
        if self.bias is not None:
            output = output + self.bias
        
        return output.to(x.dtype)


class W4A16Linear(nn.Module):
    """W4A16 量化线性层"""
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
        weight_quantizer = WeightQuantizer(
            bits=4,
            group_size=self.in_features,
            sym=True,
            channel_wise=True,
            use_hadamard=False,
            clip_ratio=1.0
        )
        weight_quantizer.calibrate(weight_fp16)
        scale, zero, w_int = weight_quantizer.quantize(weight_fp16)
        
        # 打包权重
        w_clamped = torch.clamp(w_int.flatten(), -8, 7).int()
        w_unsigned = torch.where(w_clamped < 0, w_clamped + 16, w_clamped)
        if w_unsigned.size(0) % 2 != 0:
            w_unsigned = torch.cat([w_unsigned, torch.zeros(1, dtype=w_unsigned.dtype, device=w_unsigned.device)])
        w_reshaped = w_unsigned.view(-1, 2)
        w_packed_vals = w_reshaped[:, 0] + (w_reshaped[:, 1] << 4)
        
        self.weight_packed.view(-1)[:w_packed_vals.size(0)].copy_(w_packed_vals.to(torch.uint8))
        self.weight_scale.copy_(scale.squeeze())
    
    def forward(self, x):
        out_features, in_features = self.out_features, self.in_features
        device = self.weight_packed.device
        
        total_elements = out_features * in_features
        element_indices = torch.arange(total_elements, dtype=torch.int32, device=device)
        packed_indices = element_indices // 2
        bit_offsets = (element_indices % 2) * 4
        
        w_4bit = torch.zeros(total_elements, dtype=torch.int32, device=device)
        valid_mask = packed_indices < self.weight_packed.numel()
        
        if valid_mask.any():
            packed_vals = self.weight_packed.view(-1)[packed_indices[valid_mask]].int()
            bit_offs = bit_offsets[valid_mask]
            extracted_4bit = (packed_vals >> bit_offs) & 0xF
            w_4bit[valid_mask] = extracted_4bit
        
        w_int = torch.where(w_4bit >= 8, w_4bit - 16, w_4bit).view(out_features, in_features).to(x.dtype)
        w_float = w_int * self.weight_scale.view(-1, 1).to(x.dtype)
        
        return torch.nn.functional.linear(x, w_float, self.bias)


class HadamardW4A4LinearPyTorch(nn.Module):
    """Hadamard + W4A4 (PyTorch 实现)"""
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
        # 应用 Hadamard 变换
        weight_had = weight_fp16
        # weight_had = fast_hadamard_transform(weight_fp16, group_size=weight_fp16.size(-1), normalize=True, use_cuda=False)
        
        # 量化
        weight_max = weight_had.abs().max(dim=1, keepdim=True)[0]
        scale = weight_max / 7.0
        scale = torch.clamp(scale, min=1e-5)
        
        weight_q = torch.round(weight_had / scale).clamp(-8, 7).to(torch.int8)
        
        self.weight_q.copy_(weight_q)
        self.weight_scale.copy_(scale.squeeze())
    
    def forward(self, x):
        # 对输入应用 Hadamard 变换
        x_had = x
        # x_had = fast_hadamard_transform(x, group_size=x.size(-1), normalize=True, use_cuda=False)
        
        # 量化激活值
        x_max = x_had.abs().max(dim=-1, keepdim=True)[0]
        x_scale = x_max / 7.0
        x_scale = torch.clamp(x_scale, min=1e-5)
        x_q = torch.round(x_had / x_scale).clamp(-8, 7).to(torch.int8)
        
        # GEMM
        output = torch.matmul(x_q.float(), self.weight_q.float().T) * (x_scale * self.weight_scale).unsqueeze(0)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output.to(x.dtype)


class LinearFLHGSWrapper(nn.Module):
    """LinearFLH-GS 包装器（使用 ActQuantizerGS）"""
    def __init__(self, in_features, out_features, group_size=64, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dtype = dtype
        
        self.linear_gs = LinearFLHGS(
            in_features, out_features, group_size=group_size, bias=False, dtype=dtype, device='cuda'
        )
        
        self.act_quantizer = ActQuantizerGS(
            bits=4,
            group_size=group_size,
            sym=True,
            use_hadamard=False
        ).cuda()
    
    def quantize_weight(self, weight_fp16):
        weight_quantizer = WeightQuantizer(
            bits=4,
            group_size=self.group_size,
            sym=True,
            channel_wise=False,
            use_hadamard=False,
            clip_ratio=1.0
        )
        
        weight_quantizer.calibrate(weight_fp16)
        scale, zero, w_int = weight_quantizer.quantize(weight_fp16)
        
        if scale.dim() == 3:
            scale = scale.squeeze(-1)
            zero = zero.squeeze(-1) if zero is not None else torch.zeros_like(scale, dtype=torch.int32)
        
        self.linear_gs._pack_weights(w_int, scale, zero)
    
    def forward(self, x):
        scales, zeros, x_packed = self.act_quantizer(x)
        output = self.linear_gs(x_packed, a_scale=scales.squeeze(-1))
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


def benchmark_linear_gs_all():
    """
    对比所有 group_size (32, 64, 128) 的性能
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, using CPU (tests will be slow)")

    print("=" * 100)
    print("LinearFLH-GS 全组别对比测试")
    print("Methods: FP16 | W4A4 (quarot) | W4A16 | Had+W4A4(PT) | LinearFLH-GS")
    print("=" * 100)

    # 配置
    batch_sizes = [512]
    in_features = 2048
    out_features = 8192
    group_sizes = [32, 64]
    iterations = 100

    print(f"\n配置:")
    print(f"  in_features: {in_features}")
    print(f"  out_features: {out_features}")
    print(f"  group_sizes: {group_sizes}")
    print(f"  batch_sizes: {batch_sizes}")
    print(f"  device: {device}")
    print(f"  dtype: float16")

    # FP16 baseline
    print("\n创建 FP16 baseline...")
    linear_fp16 = nn.Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    weight_fp16 = linear_fp16.weight.data.clone()
    
    # W4A4 quarot baseline
    print("创建 W4A4 (quarot) baseline...")
    linear_w4a4 = W4A4LinearQuarot(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    linear_w4a4.quantize_weight(weight_fp16)
    
    # W4A16 baseline
    print("创建 W4A16 baseline...")
    linear_w4a16 = W4A16Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    linear_w4a16.quantize_weight(weight_fp16)
    
    # Hadamard+W4A4 baseline
    print("创建 Hadamard+W4A4 (PyTorch) baseline...")
    linear_hadpt = HadamardW4A4LinearPyTorch(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    linear_hadpt.quantize_weight(weight_fp16)

    results = []

    for gs in group_sizes:
        print(f"\n{'='*60}")
        print(f"group_size = {gs}")
        print(f"{'='*60}")

        # 创建 LinearFLHGS wrapper
        linear_gs = LinearFLHGSWrapper(in_features, out_features, group_size=gs, dtype=torch.float16).cuda()
        linear_gs.quantize_weight(weight_fp16)
        linear_gs.eval()

        # 预热
        x_warmup = torch.randn(8, in_features, dtype=torch.float16, device=device)
        with torch.no_grad():
            _ = linear_fp16(x_warmup)
            _ = linear_w4a4(x_warmup)
            _ = linear_w4a16(x_warmup)
            _ = linear_hadpt(x_warmup)
            _ = linear_gs(x_warmup)
        torch.cuda.synchronize()

        for batch_size in batch_sizes:
            print(f"\n  Testing batch_size={batch_size}...")
            x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

            with torch.no_grad():
                # 预热
                for _ in range(3):
                    _ = linear_fp16(x)
                    _ = linear_w4a4(x)
                    _ = linear_w4a16(x)
                    _ = linear_hadpt(x)
                    _ = linear_gs(x)

                torch.cuda.synchronize()
                
                # 测试所有方法
                time_fp16 = benchmark_layer(linear_fp16, x, warmup=0, iters=iterations, name="FP16")
                time_w4a4 = benchmark_layer(linear_w4a4, x, warmup=0, iters=iterations, name="W4A4")
                time_w4a16 = benchmark_layer(linear_w4a16, x, warmup=0, iters=iterations, name="W4A16")
                time_hadpt = benchmark_layer(linear_hadpt, x, warmup=0, iters=iterations, name="Had+W4A4(PT)")
                time_flhgs = benchmark_layer(linear_gs, x, warmup=0, iters=iterations, name="LinearFLH-GS")

            flops = batch_size * in_features * out_features * 2
            
            def calc_throughput(time_ms):
                return flops / (time_ms / 1000) / 1e9
            
            results.append({
                'group_size': gs,
                'batch_size': batch_size,
                'method': 'FP16',
                'avg_time_ms': time_fp16,
                'throughput_gflops': calc_throughput(time_fp16)
            })
            results.append({
                'group_size': gs,
                'batch_size': batch_size,
                'method': 'W4A4',
                'avg_time_ms': time_w4a4,
                'throughput_gflops': calc_throughput(time_w4a4)
            })
            results.append({
                'group_size': gs,
                'batch_size': batch_size,
                'method': 'W4A16',
                'avg_time_ms': time_w4a16,
                'throughput_gflops': calc_throughput(time_w4a16)
            })
            results.append({
                'group_size': gs,
                'batch_size': batch_size,
                'method': 'Had+W4A4(PT)',
                'avg_time_ms': time_hadpt,
                'throughput_gflops': calc_throughput(time_hadpt)
            })
            results.append({
                'group_size': gs,
                'batch_size': batch_size,
                'method': 'LinearFLH-GS',
                'avg_time_ms': time_flhgs,
                'throughput_gflops': calc_throughput(time_flhgs)
            })

    # 保存结果
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'benchmark_linear_gs_all_{timestamp}.csv')
    df.to_csv(output_file, index=False)
    print(f"\n结果保存到: {output_file}")

    # 生成对比表格
    print("\n" + "=" * 120)
    print("性能对比总结 (Time in ms, Speedup vs FP16)")
    print("=" * 120)
    
    print(f"\n{'Batch':>8} | {'GS=32':^35} | {'GS=64':^35} | {'GS=128':^35}")
    print(f"{'':>8} | {'Method':<18} {'Time(ms)':>8} {'Speedup':>6} | {'Method':<18} {'Time(ms)':>8} {'Speedup':>6} | {'Method':<18} {'Time(ms)':>8} {'Speedup':>6}")
    print("-" * 120)
    
    pivot = df.pivot_table(
        index='batch_size',
        columns=['group_size', 'method'],
        values='avg_time_ms',
        aggfunc='first'
    )
    
    for batch in sorted(df['batch_size'].unique()):
        print(f"{batch:>8}", end="")
        for gs in group_sizes:
            if gs in pivot.columns.get_level_values(0):
                for method in ['FP16', 'W4A4', 'W4A16', 'Had+W4A4(PT)', 'LinearFLH-GS']:
                    if (gs, method) in pivot.columns:
                        time_val = pivot.loc[batch, (gs, method)]
                        fp16_val = pivot.loc[batch, (gs, 'FP16')]
                        speedup = fp16_val / time_val if time_val > 0 else 0
                        print(f" | {method:<18} {time_val:>8.3f} {speedup:>6.2f}x", end="")
        print()
    
    print("\n" + "=" * 120)
    print("吞吐量对比 (GFLOPS)")
    print("=" * 120)
    
    pivot_tp = df.pivot_table(
        index='batch_size',
        columns=['group_size', 'method'],
        values='throughput_gflops',
        aggfunc='first'
    )
    print(pivot_tp.to_string())
    
    print("=" * 120)
    return df


if __name__ == "__main__":
    df = benchmark_linear_gs_all()

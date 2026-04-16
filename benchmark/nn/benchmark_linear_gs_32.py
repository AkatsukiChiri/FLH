#!/usr/bin/env python3
"""
LinearFLH-GS Benchmark (group_size=32)
比较多种量化方法的推理性能：
1. FP16 baseline
2. W4A4 (quarot)
3. W4A16
4. Hadamard+W4A4 (PyTorch)
5. LinearFLH-GS (group_size=32)
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
from flh.nn.linear_gs import LinearFLHGS
from flh.nn.quantization_gs import ActQuantizerGS
from flh.nn.quantization import WeightQuantizer

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
        
        # 打包
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
        from flh.nn.quantization import fast_hadamard_transform
        
        # 应用 Hadamard 变换
        weight_had = fast_hadamard_transform(weight_fp16, group_size=weight_fp16.size(-1), normalize=True, use_cuda=False)
        
        # 量化
        weight_max = weight_had.abs().max(dim=1, keepdim=True)[0]
        scale = weight_max / 7.0
        scale = torch.clamp(scale, min=1e-5)
        
        weight_q = torch.round(weight_had / scale).clamp(-8, 7).to(torch.int8)
        
        self.weight_q.copy_(weight_q)
        self.weight_scale.copy_(scale.squeeze())
    
    def forward(self, x):
        from flh.nn.quantization import fast_hadamard_transform
        
        # 对输入应用 Hadamard 变换
        x_had = fast_hadamard_transform(x, group_size=x.size(-1), normalize=True, use_cuda=False)
        
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
    def __init__(self, in_features, out_features, group_size=32, dtype=torch.float16):
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


def benchmark_linear_gs_32():
    """
    测试 group_size=32 的 LinearFLHGS 性能
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, using CPU (tests will be slow)")

    print("=" * 80)
    print("LinearFLH-GS Benchmark (group_size=32)")
    print("=" * 80)

    # 配置
    batch_sizes = [128,256,512,1024,2048]
    in_features = 4096
    out_features = 14336
    group_size = 32

    print(f"\n配置:")
    print(f"  in_features: {in_features}")
    print(f"  out_features: {out_features}")
    print(f"  group_size: {group_size}")
    print(f"  device: {device}")
    print(f"  dtype: float16")

    # 创建 FP16 baseline
    print("\n创建 FP16 baseline...")
    linear_fp16 = nn.Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    weight_fp16 = linear_fp16.weight.data.clone()
    
    # 创建 W4A4 baseline
    print("创建 W4A4 baseline...")
    linear_w4a4 = W4A4LinearQuarot(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    linear_w4a4.quantize_weight(weight_fp16)
    
    # 创建 W4A16 baseline
    print("创建 W4A16 baseline...")
    linear_w4a16 = W4A16Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    linear_w4a16.quantize_weight(weight_fp16)
    
    # 创建 Hadamard+W4A4 baseline
    print("创建 Hadamard+W4A4 (PyTorch) baseline...")
    linear_hadpt = HadamardW4A4LinearPyTorch(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    linear_hadpt.quantize_weight(weight_fp16)
    
    # 创建 LinearFLH-GS
    print("创建 LinearFLH-GS...")
    linear_flhgs = LinearFLHGSWrapper(in_features, out_features, group_size=group_size, dtype=torch.float16).cuda()
    linear_flhgs.quantize_weight(weight_fp16)

    # 预热
    print("\n预热...")
    x_warmup = torch.randn(8, in_features, dtype=torch.float16, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = linear_fp16(x_warmup)
            _ = linear_w4a4(x_warmup)
            _ = linear_w4a16(x_warmup)
            _ = linear_hadpt(x_warmup)
            _ = linear_flhgs(x_warmup)
    torch.cuda.synchronize()

    # 基准测试
    results = []

    for batch_size in batch_sizes:
        print(f"\n测试 batch_size={batch_size}:")

        x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

        with torch.no_grad():
            # 预热
            for _ in range(3):
                _ = linear_fp16(x)
                _ = linear_w4a4(x)
                _ = linear_w4a16(x)
                _ = linear_hadpt(x)
                _ = linear_flhgs(x)

            torch.cuda.synchronize()
            
            # 测试 FP16
            start = time.time()
            for _ in range(100):
                output_fp16 = linear_fp16(x)
            torch.cuda.synchronize()
            time_fp16 = (time.time() - start) / 100 * 1000
            
            # 测试 W4A4
            start = time.time()
            for _ in range(100):
                output_w4a4 = linear_w4a4(x)
            torch.cuda.synchronize()
            time_w4a4 = (time.time() - start) / 100 * 1000
            
            # 测试 W4A16
            start = time.time()
            for _ in range(100):
                output_w4a16 = linear_w4a16(x)
            torch.cuda.synchronize()
            time_w4a16 = (time.time() - start) / 100 * 1000
            
            # 测试 Hadamard+W4A4
            start = time.time()
            for _ in range(100):
                output_hadpt = linear_hadpt(x)
            torch.cuda.synchronize()
            time_hadpt = (time.time() - start) / 100 * 1000
            
            # 测试 LinearFLH-GS
            start = time.time()
            for _ in range(100):
                output_flhgs = linear_flhgs(x)
            torch.cuda.synchronize()
            time_flhgs = (time.time() - start) / 100 * 1000

        flops = batch_size * in_features * out_features * 2
        tp_fp16 = flops / (time_fp16 / 1000) / 1e9
        tp_w4a4 = flops / (time_w4a4 / 1000) / 1e9
        tp_w4a16 = flops / (time_w4a16 / 1000) / 1e9
        tp_hadpt = flops / (time_hadpt / 1000) / 1e9
        tp_flhgs = flops / (time_flhgs / 1000) / 1e9

        print(f"  FP16:         {time_fp16:.3f} ms, {tp_fp16:.2f} GFLOPS")
        print(f"  W4A4:         {time_w4a4:.3f} ms, {tp_w4a4:.2f} GFLOPS")
        print(f"  W4A16:        {time_w4a16:.3f} ms, {tp_w4a16:.2f} GFLOPS")
        print(f"  Had+W4A4(PT): {time_hadpt:.3f} ms, {tp_hadpt:.2f} GFLOPS")
        print(f"  LinearFLH-GS: {time_flhgs:.3f} ms, {tp_flhgs:.2f} GFLOPS")

        results.append({
            'batch_size': batch_size,
            'group_size': group_size,
            'method': 'FP16',
            'avg_time_ms': time_fp16,
            'throughput_gflops': tp_fp16
        })
        results.append({
            'batch_size': batch_size,
            'group_size': group_size,
            'method': 'W4A4',
            'avg_time_ms': time_w4a4,
            'throughput_gflops': tp_w4a4
        })
        results.append({
            'batch_size': batch_size,
            'group_size': group_size,
            'method': 'W4A16',
            'avg_time_ms': time_w4a16,
            'throughput_gflops': tp_w4a16
        })
        results.append({
            'batch_size': batch_size,
            'group_size': group_size,
            'method': 'Had+W4A4(PT)',
            'avg_time_ms': time_hadpt,
            'throughput_gflops': tp_hadpt
        })
        results.append({
            'batch_size': batch_size,
            'group_size': group_size,
            'method': 'LinearFLH-GS',
            'avg_time_ms': time_flhgs,
            'throughput_gflops': tp_flhgs
        })

    # 保存结果
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'benchmark_linear_gs_32_{timestamp}.csv')
    df.to_csv(output_file, index=False)
    print(f"\n结果保存到: {output_file}")

    # 打印汇总表格
    print("\n" + "="*100)
    print(f"SUMMARY (group_size={group_size})")
    print("="*100)
    print(f"{'Config':<25} {'Method':<18} {'Time(ms)':>10} {'TP(GFLOPS)':>12} {'Speedup':>8}")
    print("-"*100)
    
    all_results = []
    for batch_size in sorted(df['batch_size'].unique()):
        config_str = f"B{batch_size}_I{in_features}_O{out_features}_GS{group_size}"
        fp16_row = df[(df['batch_size'] == batch_size) & (df['method'] == 'FP16')].iloc[0]
        fp16_time = fp16_row['avg_time_ms']
        fp16_tp = fp16_row['throughput_gflops']
        
        methods_data = []
        for method in ['W4A4', 'W4A16', 'Had+W4A4(PT)', 'LinearFLH-GS']:
            row = df[(df['batch_size'] == batch_size) & (df['method'] == method)]
            if len(row) > 0:
                row = row.iloc[0]
                speedup = fp16_time / row['avg_time_ms'] if row['avg_time_ms'] > 0 else 0
                methods_data.append((method, row['avg_time_ms'], row['throughput_gflops'], speedup))
        
        all_results.append({'config': config_str, 'fp16_time': fp16_time, 'fp16_tp': fp16_tp, 'methods': methods_data})
    
    for res in all_results:
        config_str = res['config']
        print(f"{config_str:<25} FP16                      {res['fp16_time']:>10.3f} {res['fp16_tp']:>12.2f} {'1.00x':>8}")
        for method, time_ms, tp, speedup in res['methods']:
            print(f"{' '*25} {method:<18} {time_ms:>10.3f} {tp:>12.2f} {speedup:>7.2f}x")
        print("-"*100)
    
    print("="*100)
    print("✓ Benchmark completed")
    print("="*100)

    return df


if __name__ == "__main__":
    df = benchmark_linear_gs_32()

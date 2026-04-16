#!/usr/bin/env python3
"""
LinearFLH-GS Benchmark (group_size=64)
比较多种量化方法的推理性能：
1. FP16 baseline
2. W4A4: 权重4bit，激活4bit
3. W4A16: 权重4bit，激活16bit
4. Hadamard+W4A4 (PyTorch): PyTorch实现的Hadamard变换 + W4A4
5. LinearFLH-GS: 我们的 group-size-flexible Hadamard + 4bit 量化方法
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
        from flh.nn.quantization import WeightQuantizer
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
        # 解包权重
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
        
        w_int = torch.where(w_4bit >= 8, w_4bit - 16, w_4bit).view(out_features, in_features)
        w_float = w_int.to(x.dtype) * self.weight_scale.view(-1, 1).to(x.dtype)
        
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
        weight_had = fast_hadamard_transform(weight_fp16, group_size=weight_fp16.size(-1), normalize=True, use_cuda=False)
        
        # 量化
        weight_max = weight_had.abs().max(dim=1, keepdim=True)[0]
        scale = weight_max / 7.0
        scale = torch.clamp(scale, min=1e-5)
        
        weight_q = torch.round(weight_had / scale).clamp(-8, 7).to(torch.int8)
        
        self.weight_q.copy_(weight_q)
        self.weight_scale.copy_(scale.squeeze())
    
    def forward(self, x):
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
    """
    LinearFLH-GS 包装器
    - 输入：先使用 ActQuantizerGS 进行量化
    - 使用 LinearFLHGS 进行推理
    """
    def __init__(self, in_features, out_features, group_size=64, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dtype = dtype
        
        assert in_features % group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        
        self.linear_gs = LinearFLHGS(
            in_features, out_features, group_size=group_size, bias=False, dtype=dtype, device='cuda'
        )
        
        self.act_quantizer = ActQuantizerGS(
            bits=4,
            group_size=group_size,
            sym=True,
            use_hadamard=False  # 不使用 Hadamard
        ).cuda()
    
    def quantize_weight(self, weight_fp16):
        """量化权重"""
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
        """
        前向传播：输入 x 需要是 (batch, in_features) FP16
        每次前向都对输入进行量化
        """
        scales, zeros, x_packed = self.act_quantizer(x)
        
        output = self.linear_gs(
            x_packed,
            a_scale=scales.squeeze(-1)
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


def run_benchmark(batch_size, in_features, out_features, group_size=64,
                  warmup=20, iters=100):
    """运行单个配置的基准测试"""
    print("\n" + "="*80)
    print(f"Config: Batch={batch_size}, In={in_features}, Out={out_features}, Group={group_size}")
    print("="*80)
    
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')
    
    # 1. FP16 Baseline
    print("\n[1/5] Creating FP16 Baseline...")
    linear_fp16 = nn.Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    weight_fp16 = linear_fp16.weight.data
    print("  ✓ FP16 Baseline created")
    
    # 2. W4A4 Linear
    print("\n[2/5] Creating W4A4 Linear...")
    linear_w4a4 = W4A4LinearQuarot(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    with torch.no_grad():
        linear_w4a4.quantize_weight(weight_fp16)
    print("  ✓ W4A4 Linear created")
    
    # 3. W4A16 Linear
    print("\n[3/5] Creating W4A16 Linear...")
    linear_w4a16 = W4A16Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    with torch.no_grad():
        linear_w4a16.quantize_weight(weight_fp16)
    print("  ✓ W4A16 Linear created")
    
    # 4. Hadamard+W4A4 (PyTorch) Linear
    print("\n[4/5] Creating Hadamard+W4A4 (PyTorch) Linear...")
    linear_hadpt = HadamardW4A4LinearPyTorch(in_features, out_features, bias=False, dtype=torch.float16).cuda()
    with torch.no_grad():
        linear_hadpt.quantize_weight(weight_fp16)
    print("  ✓ Hadamard+W4A4 (PyTorch) created")
    
    # 5. LinearFLH-GS
    print("\n[5/5] Creating LinearFLH-GS...")
    linear_flhgs = LinearFLHGSWrapper(in_features, out_features, group_size=group_size, dtype=torch.float16).cuda()
    with torch.no_grad():
        linear_flhgs.quantize_weight(weight_fp16)
    print("  ✓ LinearFLH-GS created")
    
    # Performance Testing
    print("\n" + "="*80)
    print("Performance Testing...")
    print("="*80)
    
    time_fp16 = benchmark_layer(linear_fp16, x, warmup=warmup, iters=iters, name="FP16")
    print(f"  FP16 Baseline: {time_fp16:.4f} ms")
    
    time_w4a4 = benchmark_layer(linear_w4a4, x, warmup=warmup, iters=iters, name="W4A4")
    print(f"  W4A4:          {time_w4a4:.4f} ms")
    
    time_w4a16 = benchmark_layer(linear_w4a16, x, warmup=warmup, iters=iters, name="W4A16")
    print(f"  W4A16:         {time_w4a16:.4f} ms")
    
    time_hadpt = benchmark_layer(linear_hadpt, x, warmup=warmup, iters=iters, name="Had+W4A4(PT)")
    print(f"  Had+W4A4(PT):  {time_hadpt:.4f} ms")
    
    time_flhgs = benchmark_layer(linear_flhgs, x, warmup=warmup, iters=iters, name="LinearFLH-GS")
    print(f"  LinearFLH-GS:  {time_flhgs:.4f} ms")
    
    # Calculate Metrics
    flops = 2 * batch_size * in_features * out_features
    
    def calc_throughput(time_ms):
        return flops / (time_ms / 1000) / 1e9  # GFLOPS
    
    tp_fp16 = calc_throughput(time_fp16)
    tp_w4a4 = calc_throughput(time_w4a4)
    tp_w4a16 = calc_throughput(time_w4a16)
    tp_hadpt = calc_throughput(time_hadpt)
    tp_flhgs = calc_throughput(time_flhgs)
    
    # Speedup relative to FP16
    speedup_w4a4 = time_fp16 / time_w4a4 if time_w4a4 > 0 else 0
    speedup_w4a16 = time_fp16 / time_w4a16 if time_w4a16 > 0 else 0
    speedup_hadpt = time_fp16 / time_hadpt if time_hadpt > 0 else 0
    speedup_flhgs = time_fp16 / time_flhgs if time_flhgs > 0 else 0
    
    # Accuracy Testing
    print("\n" + "="*80)
    print("Accuracy Testing...")
    print("="*80)
    test_accuracy(linear_fp16, linear_w4a4, x, "W4A4")
    test_accuracy(linear_fp16, linear_w4a16, x, "W4A16")
    test_accuracy(linear_fp16, linear_hadpt, x, "Had+W4A4(PT)")
    test_accuracy(linear_fp16, linear_flhgs, x, "LinearFLH-GS")
    
    # Memory Footprint
    print("\n" + "="*80)
    print("Memory Footprint (weights only)")
    print("="*80)
    memory_fp16 = in_features * out_features * 2
    memory_4bit = in_features * out_features * 0.5
    
    print(f"  FP16:          {memory_fp16/1024/1024:.2f} MB (100%)")
    print(f"  W4A4/W4A16:    {memory_4bit/1024/1024:.2f} MB ({memory_4bit/memory_fp16:.2%}) [packed 4-bit]")
    print(f"  LinearFLH-GS:  {memory_4bit/1024/1024:.2f} MB ({memory_4bit/memory_fp16:.2%}) [packed 4-bit]")
    
    result = {
        'config': f"B{batch_size}_I{in_features}_O{out_features}_GS{group_size}",
        'time_fp16_ms': time_fp16,
        'time_w4a4_ms': time_w4a4,
        'time_w4a16_ms': time_w4a16,
        'time_hadpt_ms': time_hadpt,
        'time_flhgs_ms': time_flhgs,
        'tp_fp16_gflops': tp_fp16,
        'tp_w4a4_gflops': tp_w4a4,
        'tp_w4a16_gflops': tp_w4a16,
        'tp_hadpt_gflops': tp_hadpt,
        'tp_flhgs_gflops': tp_flhgs,
        'speedup_w4a4': speedup_w4a4,
        'speedup_w4a16': speedup_w4a16,
        'speedup_hadpt': speedup_hadpt,
        'speedup_flhgs': speedup_flhgs,
    }
    
    return result


def main():
    """主测试函数"""
    print("\n" + "="*100)
    print("LinearFLH-GS Performance Benchmark")
    print("Comparing: FP16 | W4A4 | W4A16 | Had+W4A4(PT) | LinearFLH-GS")
    print("="*100)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    # 测试不同配置
    configs = [
        # (batch_size, in_features, out_features, group_size)
        (128, 4096, 14336, 64),
        (256, 4096, 14336, 64),
        (512, 4096, 14336, 64),
        (1024, 4096, 14336, 64),
        (2048, 4096, 14336, 64),
    ]
    
    all_results = []
    
    for batch_size, in_features, out_features, group_size in configs:
        result = run_benchmark(batch_size, in_features, out_features, group_size, warmup=20, iters=100)
        all_results.append(result)
    
    # 打印汇总表格
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"{'Config':<25} {'Method':<18} {'Time(ms)':>10} {'TP(GFLOPS)':>12} {'Speedup':>8}")
    print("-"*100)
    
    for res in all_results:
        config_str = res['config']
        methods = [
            ('FP16', res['time_fp16_ms'], res['tp_fp16_gflops'], 1.0),
            ('W4A4', res['time_w4a4_ms'], res['tp_w4a4_gflops'], res['speedup_w4a4']),
            ('W4A16', res['time_w4a16_ms'], res['tp_w4a16_gflops'], res['speedup_w4a16']),
            ('Had+W4A4(PT)', res['time_hadpt_ms'], res['tp_hadpt_gflops'], res['speedup_hadpt']),
            ('LinearFLH-GS', res['time_flhgs_ms'], res['tp_flhgs_gflops'], res['speedup_flhgs']),
        ]
        
        for i, (method, time_ms, tp, speedup) in enumerate(methods):
            if i == 0:
                print(f"{config_str:<25} {method:<18} {time_ms:>10.3f} {tp:>12.2f} {speedup:>8.2f}x")
            else:
                print(f"{' '*25} {method:<18} {time_ms:>10.3f} {tp:>12.2f} {speedup:>8.2f}x")
        print("-"*100)
    
    # 保存结果
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'benchmark_linear_gs_64_comprehensive_{timestamp}.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print("="*100)
    print("✓ Benchmark completed")
    print("="*100)


if __name__ == "__main__":
    main()

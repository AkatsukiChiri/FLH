"""
Benchmark script for comparing FLH quantized model vs original FP16 model.

This script tests:
- Inference latency (prefill and decode phases)
- Memory usage
- Throughput
- Different batch sizes, input lengths, and generation lengths
"""

import argparse
import gc
import re
import numpy as np
import torch
import time
import transformers
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from flh.quantized_model.modeling_llama import FLH_FP16LlamaForCausalLM, FLH_LlamaForCausalLM, FLH_LlamaConfig


def parse_list_or_range(s: str) -> List[int]:
    """Parse a comma-separated list or range (e.g., '1,2,4' or '1-8' or '1,2-4:128,8')"""
    values = []
    parts = s.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Check for step range syntax: "1-8:2" or "512-2048:128"
        if '-' in part or ':' in part:
            # Handle step range: "start-end:step" or "start-end"
            if ':' in part:
                range_part, step_part = part.split(':')
                step = int(step_part)
            else:
                range_part = part
                step = 1
            
            if '-' in range_part:
                start, end = range_part.split('-')
                values.extend(range(int(start), int(end) + 1, step))
            else:
                values.append(int(range_part))
        else:
            values.append(int(part))
    return sorted(list(set(values)))  # Remove duplicates and sort


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run"""
    batch_size: int
    prefill_seq_len: int
    decode_steps: int


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    prefill_time_ms: float
    prefill_time_std: float
    prefill_memory_gb: float
    decode_time_ms: float
    decode_time_std: float
    decode_memory_gb: float
    e2e_time_ms: float
    e2e_time_std: float
    throughput_prefill: float
    throughput_decode: float
    throughput_e2e: float


# Benchmark defaults
num_warmup_steps = 3
num_bench_steps = 10


def _cleanup():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()


def cuda_time_ms(fn, iters: int = 100, warmup: int = 20) -> float:
    """Measure GPU time in milliseconds using CUDA events"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def get_model_fp16(config_name: str) -> FLH_FP16LlamaForCausalLM:
    """Load FLH FP16 model"""
    config = transformers.AutoConfig.from_pretrained(
        config_name,
        attn_implementation="flash_attention_2"
    )
    config._attn_implementation = "flash_attention_2"
    
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    
    with transformers.modeling_utils.no_init_weights():
        model = FLH_FP16LlamaForCausalLM(config=config)
    
    torch.set_default_dtype(dtype_old)
    model = model.to(dtype=torch.float16)
    
    return model


def get_model_quantized(config_name: str, load_quantized_path: Optional[str] = None,
                        weight_bits: int = 4, weight_group_size: int = 128,
                        act_bits: int = 16, act_group_size: int = -1,
                        weight_sym: bool = False) -> FLH_LlamaForCausalLM:
    """Load or create FLH quantized model"""
    if load_quantized_path:
        print(f"  Loading quantized model from {load_quantized_path}...")
        return FLH_LlamaForCausalLM.load_quantized(load_quantized_path, target_device="cuda")
    
    print(f"  Creating quantized model with W{weight_bits}G{weight_group_size}/A{act_bits}G{act_group_size}...")
    from transformers import LlamaForCausalLM
    
    config = transformers.AutoConfig.from_pretrained(
        config_name,
        attn_implementation="flash_attention_2"
    )
    config._attn_implementation = "flash_attention_2"
    
    print("  Loading FP16 model for weight conversion...")
    float_model = LlamaForCausalLM.from_pretrained(config_name, attnn_implementation="flash_attention_2")
    float_model = float_model.to(dtype=torch.float16)
    
    print("  Converting to FLH quantized model...")
    quantized_model = FLH_LlamaForCausalLM.from_float(
        float_model,
        target_device="cuda",
        weight_bits=weight_bits,
        weight_group_size=weight_group_size,
        act_bits=act_bits,
        act_group_size=act_group_size,
        weight_sym=weight_sym,
        act_sym=True,
        clip_ratio=1.0
    )
    
    del float_model
    _cleanup()
    
    return quantized_model


@torch.no_grad()
def benchmark_prefill(model, bsz: int, prefill_length: int) -> Tuple[List[float], float]:
    """Benchmark prefill phase"""
    device = next(model.parameters()).device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    
    # Warmup
    print(f"    Prefill warmup ({num_warmup_steps} iterations)...")
    for _ in range(num_warmup_steps):
        _ = model(test_input)
    torch.cuda.synchronize()
    _cleanup()
    
    # Benchmark
    print(f"    Prefill benchmark ({num_bench_steps} iterations)...")
    times = []
    memories = []
    
    for _ in tqdm(range(num_bench_steps), desc="      Prefill", leave=False):
        torch.cuda.reset_max_memory_allocated()
        start = time.perf_counter()
        
        _ = model(test_input)
        torch.cuda.synchronize()
        
        end = time.perf_counter()
        peak_memory = torch.cuda.max_memory_allocated()
        
        times.append((end - start) * 1000)
        memories.append(peak_memory)
    
    return times, np.mean(memories)


@torch.no_grad()
def benchmark_decode(model, bsz: int, prefill_length: int, decode_steps: int) -> Tuple[List[float], float]:
    """Benchmark decode phase"""
    device = next(model.parameters()).device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    next_input = torch.tensor([[100] for _ in range(bsz)], dtype=torch.int32, device=device)
    
    def _run_decode():
        # Prefill phase
        out = model(test_input, use_cache=True)
        past_key_values = out.past_key_values
        
        # Decode phase
        for _ in range(decode_steps):
            out = model(next_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
    
    # Warmup
    print(f"    Decode warmup ({num_warmup_steps} iterations)...")
    for _ in range(num_warmup_steps):
        _run_decode()
    torch.cuda.synchronize()
    _cleanup()
    
    # Benchmark decode only (measure just the decode steps, not prefill)
    print(f"    Decode benchmark ({num_bench_steps} iterations)...")
    times = []
    memories = []
    
    for _ in tqdm(range(num_bench_steps), desc="      Decode", leave=False):
        # First do prefill to setup KV cache
        out = model(test_input, use_cache=True)
        past_key_values = out.past_key_values
        
        # Now measure just the decode steps
        torch.cuda.reset_max_memory_allocated()
        start = time.perf_counter()
        for _ in range(decode_steps):
            out = model(next_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
        torch.cuda.synchronize()
        end = time.perf_counter()
        peak_memory = torch.cuda.max_memory_allocated()
        
        times.append((end - start) * 1000)
        memories.append(peak_memory)
    
    return times, np.mean(memories)


@torch.no_grad()
def benchmark_e2e(model, bsz: int, prefill_length: int, decode_steps: int) -> Tuple[List[float], float]:
    """Benchmark end-to-end (prefill + decode)"""
    device = next(model.parameters()).device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    next_input = torch.tensor([[100] for _ in range(bsz)], dtype=torch.int32, device=device)
    
    def _run_e2e():
        # Prefill
        out = model(test_input, use_cache=True)
        past_key_values = out.past_key_values
        
        # Decode
        for _ in range(decode_steps):
            out = model(next_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
    
    # Warmup
    print(f"    E2E warmup ({num_warmup_steps} iterations)...")
    for _ in range(num_warmup_steps):
        _run_e2e()
    torch.cuda.synchronize()
    _cleanup()
    
    # Benchmark
    print(f"    E2E benchmark ({num_bench_steps} iterations)...")
    times = []
    memories = []
    
    for _ in tqdm(range(num_bench_steps), desc="      E2E", leave=False):
        torch.cuda.reset_max_memory_allocated()
        start = time.perf_counter()
        
        _run_e2e()
        torch.cuda.synchronize()
        
        end = time.perf_counter()
        peak_memory = torch.cuda.max_memory_allocated()
        
        times.append((end - start) * 1000)
        memories.append(peak_memory)
    
    return times, np.mean(memories)


@torch.no_grad()
def run_benchmark_for_model(model, config: BenchmarkConfig, model_name: str) -> BenchmarkResult:
    """Run all benchmarks for a model with given configuration"""
    model.eval()
    model = model.cuda()
    
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Prefill seq len: {config.prefill_seq_len}")
    print(f"  Decode steps: {config.decode_steps}")
    print(f"{'='*70}")
    
    # Prefill benchmark
    print(f"\n[1/3] Prefill Phase")
    time_prefill, mem_prefill = benchmark_prefill(model, config.batch_size, config.prefill_seq_len)
    _cleanup()
    
    # Decode benchmark
    print(f"\n[2/3] Decode Phase ({config.decode_steps} steps)")
    time_decode, mem_decode = benchmark_decode(model, config.batch_size, config.prefill_seq_len, config.decode_steps)
    _cleanup()
    
    # E2E benchmark
    print(f"\n[3/3] End-to-End (Prefill + Decode)")
    time_e2e, mem_e2e = benchmark_e2e(model, config.batch_size, config.prefill_seq_len, config.decode_steps)
    _cleanup()
    
    # Calculate statistics
    result = BenchmarkResult(
        prefill_time_ms=np.mean(time_prefill),
        prefill_time_std=np.std(time_prefill),
        prefill_memory_gb=mem_prefill / (1024**3),
        decode_time_ms=np.mean(time_decode),
        decode_time_std=np.std(time_decode),
        decode_memory_gb=mem_decode / (1024**3),
        e2e_time_ms=np.mean(time_e2e),
        e2e_time_std=np.std(time_e2e),
        throughput_prefill=config.batch_size * config.prefill_seq_len / (np.mean(time_prefill) / 1000),
        throughput_decode=config.batch_size * config.decode_steps / (np.mean(time_decode) / 1000),
        throughput_e2e=config.batch_size * (config.prefill_seq_len + config.decode_steps) / (np.mean(time_e2e) / 1000)
    )
    
    return result


def print_comparison_table(results_fp16: Dict[str, BenchmarkResult],
                            results_quant: Dict[str, BenchmarkResult],
                            configs: List[BenchmarkConfig]):
    """Print a comparison table between FP16 and quantized models"""
    print(f"\n{'='*100}")
    print(f"{'MODEL COMPARISON RESULTS':^100}")
    print(f"{'='*100}\n")
    
    # Table header
    header = f"{'Config':<25} {'Model':<10} {'Prefill (ms)':<15} {'Decode (ms)':<15} {'E2E (ms)':<15} {'Mem (GB)':<10} {'Speedup':<10}"
    print(header)
    print("-" * 100)
    
    for config in configs:
        config_key = f"bs{config.batch_size}_pre{config.prefill_seq_len}_dec{config.decode_steps}"
        
        fp16_result = results_fp16.get(config_key)
        quant_result = results_quant.get(config_key)
        
        if fp16_result:
            speedup = fp16_result.e2e_time_ms / quant_result.e2e_time_ms if quant_result else 1.0
            mem_avg = (fp16_result.prefill_memory_gb + fp16_result.decode_memory_gb) / 2
            print(f"{config_key:<25} {'FP16':<10} {fp16_result.prefill_time_ms:<15.2f} {fp16_result.decode_time_ms:<15.2f} "
                  f"{fp16_result.e2e_time_ms:<15.2f} {mem_avg:<10.2f} {'-':<10}")
        
        if quant_result:
            speedup = fp16_result.e2e_time_ms / quant_result.e2e_time_ms if fp16_result else 1.0
            mem_avg = (quant_result.prefill_memory_gb + quant_result.decode_memory_gb) / 2
            speedup_str = f"{speedup:.2f}x" if fp16_result else "-"
            print(f"{config_key:<25} {'FLH-Q':<10} {quant_result.prefill_time_ms:<15.2f} {quant_result.decode_time_ms:<15.2f} "
                  f"{quant_result.e2e_time_ms:<15.2f} {mem_avg:<10.2f} {speedup_str:<10}")
        
        print("-" * 100)
    
    print(f"\n{'='*100}")


def print_latex_table(results_fp16: Dict[str, BenchmarkResult],
                      results_quant: Dict[str, BenchmarkResult],
                      configs: List[BenchmarkConfig]):
    """Print results in LaTeX table format"""
    print(f"\n{'='*70}")
    print(f"{'LATEX TABLE FORMAT':^70}")
    print(f"{'='*70}\n")
    
    print("% Comparison table: Batch Size, Prefill, Decode, FP16-E2E, FP16-Mem, Quant-E2E, Quant-Mem, Speedup")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|c|c|c|c|}")
    print("\\hline")
    print("Config & BS & Prefill & Decode & FP16 E2E (ms) & Quant E2E (ms) & Memory Saved & Speedup \\\\")
    print("\\hline")
    
    for config in configs:
        config_key = f"bs{config.batch_size}_pre{config.prefill_seq_len}_dec{config.decode_steps}"
        fp16_result = results_fp16.get(config_key)
        quant_result = results_quant.get(config_key)
        
        if fp16_result and quant_result:
            mem_saved = ((fp16_result.prefill_memory_gb + fp16_result.decode_memory_gb) / 2 -
                        (quant_result.prefill_memory_gb + quant_result.decode_memory_gb) / 2)
            speedup = fp16_result.e2e_time_ms / quant_result.e2e_time_ms
            print(f"{config_key} & {config.batch_size} & {config.prefill_seq_len} & {config.decode_steps} & "
                  f"{fp16_result.e2e_time_ms:.2f} & {quant_result.e2e_time_ms:.2f} & "
                  f"{mem_saved:.2f}GB & {speedup:.2f}x \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{FLH Quantized Model vs FP16 Model Performance Comparison}")
    print("\\end{table}")


def print_detailed_results(result: BenchmarkResult, model_name: str):
    """Print detailed benchmark results"""
    print(f"\n{'='*70}")
    print(f"{f'DETAILED RESULTS: {model_name}':^70}")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<30} {'Value':<20}")
    print(f"{'-'*50}")
    
    print(f"{'Prefill Time (ms)':<30} {result.prefill_time_ms:.3f} ± {result.prefill_time_std:.3f}")
    print(f"{'Prefill Memory (GB)':<30} {result.prefill_memory_gb:.3f}")
    print(f"{'Prefill Throughput (tok/s)':<30} {result.throughput_prefill:.2f}")
    
    print(f"{'-'*50}")
    print(f"{'Decode Time (ms)':<30} {result.decode_time_ms:.3f} ± {result.decode_time_std:.3f}")
    print(f"{'Decode Time/token (ms)':<30} {result.decode_time_ms / num_bench_steps:.3f}")
    print(f"{'Decode Memory (GB)':<30} {result.decode_memory_gb:.3f}")
    print(f"{'Decode Throughput (tok/s)':<30} {result.throughput_decode:.2f}")
    
    print(f"{'-'*50}")
    print(f"{'E2E Time (ms)':<30} {result.e2e_time_ms:.3f} ± {result.e2e_time_std:.3f}")
    print(f"{'E2E Throughput (tok/s)':<30} {result.throughput_e2e:.2f}")
    
    print(f"{'='*70}\n")


def run_comparison(args):
    """Run comparison benchmark between FP16 and quantized models"""
    print(f"\n{'█'*70}")
    print(f"█{' '*68}█")
    print(f"█{' '*10}FLH QUANTIZED MODEL VS FP16 BENCHMARK{' '*22}█")
    print(f"█{' '*68}█")
    print(f"{'█'*70}\n")
    
    # Parse configurations - support both comma-separated and range syntax
    # Example: "1,2,4" or "1-8" or "1,2-4,8"
    batch_sizes = parse_list_or_range(args.batch_sizes)
    prefill_lens = parse_list_or_range(args.prefill_lens)
    decode_steps_list = parse_list_or_range(args.decode_steps)
    
    # Support explicit combinations via --configs "bs,prefill,decode;bs,prefill,decode"
    explicit_configs = []
    if args.configs:
        for cfg in args.configs.split(';'):
            cfg = cfg.strip()
            if cfg:
                parts = cfg.split(',')
                if len(parts) == 3:
                    explicit_configs.append(BenchmarkConfig(
                        batch_size=int(parts[0]),
                        prefill_seq_len=int(parts[1]),
                        decode_steps=int(parts[2])
                    ))
    
    # If explicit configs provided, use only those; otherwise use Cartesian product
    if explicit_configs:
        configs = explicit_configs
    else:
        configs = []
        for bsz in batch_sizes:
            for pre_len in prefill_lens:
                for dec_len in decode_steps_list:
                    configs.append(BenchmarkConfig(
                        batch_size=bsz,
                        prefill_seq_len=pre_len,
                        decode_steps=dec_len
                    ))
    
    print(f"Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Batch sizes: {batch_sizes if not explicit_configs else 'custom'}")
    print(f"  Prefill lengths: {prefill_lens if not explicit_configs else 'custom'}")
    print(f"  Decode steps: {decode_steps_list if not explicit_configs else 'custom'}")
    print(f"  Total configurations: {len(configs)}")
    if explicit_configs:
        print(f"  Explicit configs: {[f'bs{c.batch_size}_pre{c.prefill_seq_len}_dec{c.decode_steps}' for c in configs]}")
    print(f"  Warmup iterations: {num_warmup_steps}")
    print(f"  Benchmark iterations: {num_bench_steps}")
    
    results_fp16 = {}
    results_quant = {}
    failed_configs_fp16 = []
    failed_configs_quant = []
    
    # Benchmark FP16 model if requested
    if not args.quant_only:
        print(f"\n{'='*70}")
        print(f"{'LOADING FP16 MODEL':=^70}")
        print(f"{'='*70}")
        
        try:
            model_fp16 = get_model_fp16(args.model_path)
            print(f"✓ FP16 model loaded successfully")
            
            # Get model size
            param_size = sum(p.nelement() * p.element_size() for p in model_fp16.parameters())
            print(f"  Model size: {param_size / (1024**3):.2f} GB")
            
            # Run benchmarks for each configuration (continue on error)
            for i, config in enumerate(configs):
                config_key = f"bs{config.batch_size}_pre{config.prefill_seq_len}_dec{config.decode_steps}"
                print(f"\n[{i+1}/{len(configs)}] Testing config: {config_key}")
                try:
                    result = run_benchmark_for_model(model_fp16, config, "FP16")
                    results_fp16[config_key] = result
                    print_detailed_results(result, "FP16")
                except Exception as e:
                    print(f"✗ Config {config_key} failed: {e}")
                    failed_configs_fp16.append(config_key)
                    import traceback
                    traceback.print_exc()
                _cleanup()
            
            del model_fp16
            _cleanup()
            
            if failed_configs_fp16:
                print(f"\n⚠ FP16: {len(failed_configs_fp16)}/{len(configs)} configs failed: {failed_configs_fp16}")
                
        except Exception as e:
            print(f"✗ Failed to load FP16 model: {e}")
            import traceback
            traceback.print_exc()
    
    # Benchmark quantized model
    if not args.fp16_only:
        print(f"\n{'='*70}")
        print(f"{'LOADING QUANTIZED MODEL':=^70}")
        print(f"{'='*70}")
        
        try:
            model_quant = get_model_quantized(
                args.model_path,
                load_quantized_path=args.load_quantized,
                weight_bits=args.weight_bits,
                weight_group_size=args.weight_group_size,
                act_bits=args.act_bits,
                act_group_size=args.act_group_size,
                weight_sym=args.weight_sym
            )
            print(f"✓ Quantized model loaded successfully")
            
            # Get model size
            param_size = sum(p.nelement() * p.element_size() for p in model_quant.parameters())
            print(f"  Model size: {param_size / (1024**3):.2f} GB")
            
            # Run benchmarks for each configuration (continue on error)
            for i, config in enumerate(configs):
                config_key = f"bs{config.batch_size}_pre{config.prefill_seq_len}_dec{config.decode_steps}"
                print(f"\n[{i+1}/{len(configs)}] Testing config: {config_key}")
                try:
                    result = run_benchmark_for_model(model_quant, config, "FLH-Quantized")
                    results_quant[config_key] = result
                    print_detailed_results(result, "FLH-Quantized")
                except Exception as e:
                    print(f"✗ Config {config_key} failed: {e}")
                    failed_configs_quant.append(config_key)
                    import traceback
                    traceback.print_exc()
                _cleanup()
            
            del model_quant
            _cleanup()
            
            if failed_configs_quant:
                print(f"\n⚠ Quantized: {len(failed_configs_quant)}/{len(configs)} configs failed: {failed_configs_quant}")
                
        except Exception as e:
            print(f"✗ Failed to load quantized model: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"{'BENCHMARK SUMMARY':=^70}")
    print(f"{'='*70}")
    print(f"  FP16 configs passed: {len(results_fp16)}/{len(configs)}")
    print(f"  Quantized configs passed: {len(results_quant)}/{len(configs)}")
    if failed_configs_fp16:
        print(f"  FP16 failed: {failed_configs_fp16}")
    if failed_configs_quant:
        print(f"  Quantized failed: {failed_configs_quant}")
    
    # Print comparison (only for configs that have both results)
    if results_fp16 and results_quant:
        # Filter configs that have both results
        valid_configs = [c for c in configs 
                        if f"bs{c.batch_size}_pre{c.prefill_seq_len}_dec{c.decode_steps}" in results_fp16 
                        and f"bs{c.batch_size}_pre{c.prefill_seq_len}_dec{c.decode_steps}" in results_quant]
        if valid_configs:
            print_comparison_table(results_fp16, results_quant, valid_configs)
            print_latex_table(results_fp16, results_quant, valid_configs)
        else:
            print("\n⚠ No configs have both FP16 and Quantized results for comparison")
    
    print(f"\n{'='*70}")
    print("Benchmark completed!")
    print(f"{'='*70}\n")
    
    return results_fp16, results_quant


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark FLH Quantized Model vs FP16 Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare FP16 and quantized model with Cartesian product of all values
  python benchmark/bench_quant_vs_fp16.py --model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      --batch_sizes 1,2 --prefill_lens 512,1024 --decode_steps 32,64
  
  # Use range syntax: test batch sizes 1 through 4
  python benchmark/bench_quant_vs_fp16.py --model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      --batch_sizes 1-4 --prefill_lens 512-2048:128 --decode_steps 32,64
  
  # Explicit specific combinations only (no Cartesian product)
  python benchmark/bench_quant_vs_fp16.py --model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      --configs "1,512,32;1,512,64;2,1024,32"
  
  # Benchmark only FP16 model with specific settings
  python benchmark/bench_quant_vs_fp16.py --model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      --fp16_only --batch_sizes 1 --prefill_lens 2048 --decode_steps 128
  
  # Load pre-quantized model
  python benchmark/bench_quant_vs_fp16.py --model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      --load_quantized ./quantized_model --batch_sizes 1 --prefill_lens 1024 --decode_steps 64

Input Format:
  - Comma-separated: "1,2,4,8"
  - Range syntax: "1-8" (1 through 8)
  - Mixed: "1,2-4,8" (1,2,3,4,8)
  - Step range: "512-2048:128" (512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,2048)
  
  Use --configs to specify exact combinations instead of Cartesian product:
  --configs "bs,prefill,decode;bs,prefill,decode;..."
        """
    )
    
    parser.add_argument(
        '--model_path', type=str,
        help='Model config name or path (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)',
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    )
    parser.add_argument(
        '--batch_sizes', type=str,
        help='Comma-separated batch sizes, range, or step range (e.g., "1,2,4,8" or "1-8" or "1-8:2")',
        default='1',
    )
    parser.add_argument(
        '--prefill_lens', type=str,
        help='Comma-separated prefill lengths, range, or step range (e.g., "512,1024,2048" or "512-2048")',
        default='512',
    )
    parser.add_argument(
        '--decode_steps', type=str,
        help='Comma-separated decode steps, range, or step range (e.g., "32,64,128" or "32-128")',
        default='32',
    )
    parser.add_argument(
        '--configs', type=str,
        help='Explicit configurations as "bs,prefill,decode;bs,prefill,decode;..." to avoid Cartesian product',
        default=None,
    )
    parser.add_argument(
        '--load_quantized', type=str,
        help='Path to pre-quantized model (skip quantization)',
        default=None,
    )
    parser.add_argument(
        '--weight_bits', type=int,
        help='Weight quantization bits (default: 4)',
        default=4,
    )
    parser.add_argument(
        '--weight_group_size', type=int,
        help='Weight quantization group size (default: 128)',
        default=128,
    )
    parser.add_argument(
        '--act_bits', type=int,
        help='Activation quantization bits (default: 16 = no quant)',
        default=16,
    )
    parser.add_argument(
        '--act_group_size', type=int,
        help='Activation quantization group size (default: -1)',
        default=-1,
    )
    parser.add_argument(
        '--weight_sym', action='store_true',
        help='Use symmetric weight quantization',
    )
    parser.add_argument(
        '--fp16_only', action='store_true',
        help='Benchmark only FP16 model',
    )
    parser.add_argument(
        '--quant_only', action='store_true',
        help='Benchmark only quantized model',
    )
    
    args = parser.parse_args()
    run_comparison(args)

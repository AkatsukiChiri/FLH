"""
Benchmark script for FLH_FP16LlamaForCausalLM

This script tests the FP16 version of the quantized Llama model including:
- Inference latency (prefill and decode phases)
- Memory usage
- Throughput
"""

import argparse
import gc
import numpy as np
import torch
import time
import transformers
from tqdm import tqdm

from flh.quantized_model.modeling_llama import FLH_FP16LlamaForCausalLM, FLH_LlamaConfig

# Benchmark configuration
num_warmup_steps = 3
num_bench_steps = 10


def repeated_run(num_repeats=10):
    """Decorator to run a function multiple times and collect results"""
    def func(module):
        def _f(*args, **kwargs):
            times = []
            for i in range(num_repeats):
                times.append(module(*args, **kwargs))
            return tuple(zip(*times))
        return _f
    return func


def _cleanup():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()


def module_benchmark(module, desc="Benchmarking"):
    """Run benchmark on a module with warmup"""
    # Warmup - only once before all benchmark runs
    print(f"  Warming up ({num_warmup_steps} iterations)...")
    for i in range(num_warmup_steps):
        out = module()
    torch.cuda.synchronize()
    _cleanup()
    
    # Actual benchmark runs
    print(f"  Running benchmark ({num_bench_steps} iterations)...")
    times = []
    memories = []
    
    with tqdm(total=num_bench_steps, desc=f"  {desc}", leave=False, unit="iter") as pbar:
        for i in range(num_bench_steps):
            torch.cuda.reset_max_memory_allocated()
            start_time = time.perf_counter()
            
            out = module()
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # ms
            memories.append(peak_memory)  # bytes
            pbar.update(1)
    
    return tuple(times), tuple(memories)


def get_model_fp16(config_name):
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
    
    # Explicitly convert all parameters to float16 to ensure Flash Attention 2 compatibility
    model = model.to(dtype=torch.float16)
    
    return model


def run_prefill(model, bsz, prefill_length):
    """Benchmark prefill phase"""
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    return module_benchmark(lambda: model(test_input), desc="Prefill")


def run_decode(model, bsz, prefill_length, decode_steps):
    """Benchmark decode phase - measures decode time by doing prefill+decode and subtracting prefill time"""
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    next_input = torch.tensor([[100] for _ in range(bsz)], dtype=torch.int32, device=device)
    
    def _benchmark_decode():
        # Do prefill to initialize cache
        with torch.no_grad():
            out = model(test_input, use_cache=True)
            past_key_values = out.past_key_values
        
        # Measure decode steps
        torch.cuda.reset_max_memory_allocated()
        start = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(decode_steps):
                out = model(next_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        decode_time_ms = (end - start) * 1000
        peak_memory = torch.cuda.max_memory_allocated()
        
        return decode_time_ms, peak_memory
    
    # Warmup
    print(f"  Warming up ({num_warmup_steps} iterations)...")
    for _ in range(num_warmup_steps):
        _benchmark_decode()
    torch.cuda.synchronize()
    _cleanup()
    
    # Benchmark
    print(f"  Running benchmark ({num_bench_steps} iterations)...")
    times = []
    memories = []
    
    with tqdm(total=num_bench_steps, desc=f"  Decode", leave=False, unit="iter") as pbar:
        for _ in range(num_bench_steps):
            t, m = _benchmark_decode()
            times.append(t)
            memories.append(m)
            pbar.update(1)
    
    return tuple(times), tuple(memories)


def run_e2e(model, bsz, prefill_length, decode_steps):
    """Benchmark end-to-end (prefill + decode)"""
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    next_input = torch.tensor([[100] for _ in range(bsz)], dtype=torch.int32, device=device)
    
    def _prefill_and_decode_for_multiple_steps():
        # Prefill phase
        with torch.no_grad():
            out = model(test_input, use_cache=True)
            past_key_values = out.past_key_values
        
        # Decode phase
        for _ in range(decode_steps):
            with torch.no_grad():
                out = model(next_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
    
    return module_benchmark(_prefill_and_decode_for_multiple_steps, desc="E2E")


@torch.no_grad()
def run_all_for_model(model, bsz, prefill, decode):
    """Run all benchmarks for a model"""
    model.eval()
    model = model.cuda()
    
    print(f"\n{'='*70}")
    print("Running Benchmarks")
    print(f"{'='*70}")
    
    # Prefill benchmark
    print(f"\n[1/3] Prefill Phase (batch={bsz}, seq_len={prefill})")
    time_prefill, memory_prefill = run_prefill(model, bsz, prefill)
    _cleanup()
    
    if decode is not None:
        # Decode benchmark
        print(f"\n[2/3] Decode Phase ({decode} steps)")
        time_decode, memory_decode = run_decode(model, bsz, prefill, decode)
        _cleanup()
        
        # E2E benchmark
        print(f"\n[3/3] End-to-End (Prefill + Decode)")
        time_e2e, memory_e2e = run_e2e(model, bsz, prefill, decode)
        _cleanup()
    else:
        time_decode = time_e2e = memory_decode = memory_e2e = None
    
    return time_prefill, time_decode, time_e2e, memory_prefill, memory_decode


def benchmark(args):
    """Run benchmark with given arguments"""
    print(f"\n{'█'*70}")
    print(f"█{' '*68}█")
    print(f"█{' '*15}FLH FP16 LLAMA BENCHMARK{' '*28}█")
    print(f"█{' '*68}█")
    print(f"{'█'*70}\n")
    
    config_name = args.model_path
    
    print(f"Configuration:")
    print(f"  Model: {config_name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Prefill seq len: {args.prefill_seq_len}")
    print(f"  Decode steps: {args.decode_steps}")
    print(f"  Num warmup: {num_warmup_steps}")
    print(f"  Num bench: {num_bench_steps}")
    
    # Load model
    print(f"\n{'Loading model...':-^70}")
    try:
        with tqdm(total=3, desc="Model loading", unit="step") as pbar:
            pbar.set_description("Loading config")
            model = get_model_fp16(config_name)
            pbar.update(3)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print(f"\nNote: This benchmark requires a valid Llama model config.")
        print(f"You can use: 'meta-llama/Llama-2-7b-hf' or 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'")
        return
    
    # Run benchmarks
    time_prefill, time_decode, time_e2e, mem_prefill, mem_decode = run_all_for_model(
        model, args.batch_size, args.prefill_seq_len, args.decode_steps
    )
    
    del model
    _cleanup()
    
    # Print results
    print(f"\n{'='*70}")
    print(f"{'BENCHMARK RESULTS':^70}")
    print(f"{'='*70}\n")
    
    # Create results table
    print(f"{'Phase':<20} {'Metric':<25} {'Value':<25}")
    print(f"{'-'*70}")
    
    # Prefill results
    print(f"{'Prefill':<20} {'Time (ms)':<25} {np.mean(time_prefill):.3f} ± {1.96 * np.std(time_prefill):.3f}")
    print(f"{'':<20} {'Memory (GB)':<25} {np.mean(mem_prefill) / (1024**3):.3f}")
    throughput_prefill = args.batch_size * args.prefill_seq_len / (np.mean(time_prefill) / 1000)
    print(f"{'':<20} {'Throughput (tok/s)':<25} {throughput_prefill:.2f}")
    
    if args.decode_steps is not None:
        print(f"{'-'*70}")
        # Decode results
        print(f"{'Decode':<20} {'Time (ms)':<25} {np.mean(time_decode):.3f} ± {1.96 * np.std(time_decode):.3f}")
        print(f"{'':<20} {'Time/token (ms)':<25} {np.mean(time_decode) / args.decode_steps:.3f}")
        print(f"{'':<20} {'Memory (GB)':<25} {np.mean(mem_decode) / (1024**3):.3f}")
        throughput_decode = args.batch_size * args.decode_steps / (np.mean(time_decode) / 1000)
        print(f"{'':<20} {'Throughput (tok/s)':<25} {throughput_decode:.2f}")
        
        print(f"{'-'*70}")
        # E2E results
        print(f"{'End-to-End':<20} {'Time (ms)':<25} {np.mean(time_e2e):.3f} ± {1.96 * np.std(time_e2e):.3f}")
        total_tokens = args.prefill_seq_len + args.decode_steps
        print(f"{'':<20} {'Total tokens':<25} {total_tokens}")
        throughput_e2e = args.batch_size * total_tokens / (np.mean(time_e2e) / 1000)
        print(f"{'':<20} {'Throughput (tok/s)':<25} {throughput_e2e:.2f}")
    
    print(f"{'='*70}")
    
    # LaTeX table format output
    print(f"\n{'='*70}")
    print(f"{'LaTeX TABLE FORMAT':^70}")
    print(f"{'='*70}\n")
    
    print(f"% Prefill")
    print(f"Prefill & {config_name} & {args.batch_size} & {args.prefill_seq_len} & "
          f"{np.mean(time_prefill):.3f} & {np.mean(mem_prefill) / (1024**3):.3f}GB \\\\")
    
    if args.decode_steps is not None:
        print(f"\n% Decode")
        print(f"Decode & {config_name} & {args.batch_size} & {args.prefill_seq_len} & "
              f"{args.decode_steps} & {np.mean(time_decode):.3f} & "
              f"{np.mean(mem_decode) / (1024**3):.3f}GB \\\\")
        
        print(f"\n% E2E")
        print(f"E2E & {config_name} & {args.batch_size} & {args.prefill_seq_len} & "
              f"{args.decode_steps} & {np.mean(time_e2e):.3f} \\\\")
    
    print(f"\n{'='*70}")
    print("Benchmark completed!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark FLH_FP16LlamaForCausalLM')
    
    parser.add_argument(
        '--model_path', type=str,
        help='Model config name (e.g., meta-llama/Llama-2-7b-hf)',
        default='meta-llama/Llama-2-7b-hf',
    )
    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size',
        default=1,
    )
    parser.add_argument(
        '--prefill_seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--decode_steps', type=int,
        help='Decode steps (set to None to skip decode benchmark)',
        required=False,
        default=128,
    )
    
    args = parser.parse_args()
    benchmark(args)

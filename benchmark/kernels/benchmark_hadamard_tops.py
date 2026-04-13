"""
Hadamard Transform Throughput Benchmark Suite

This benchmark compares four implementations of Hadamard Transform on [n, 128] matrices:
1. FP16 PyTorch implementation (baseline)
2. fast-hadamard-transform library
3. flh CUDA implementation (in-place hadamard_transform_half)
4. flh fused Hadamard + int4 quantization (hadamard_and_quantize_i4)

All implementations are benchmarked in terms of throughput (elements/second).
"""

import torch
import time
import numpy as np
from fast_hadamard_transform import hadamard_transform
from typing import List, Tuple, Dict
import sys
import os

# Add current directory to path to import pytorch_had_trans
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pytorch_had_trans import pytorch_had_trans

# Import FLH implementations
try:
    from flh.cuda.hadamard import hadamard_transform_half as flh_had_trans_half
    HAS_FLH_HAD = True
except ImportError:
    HAS_FLH_HAD = False
    print("Warning: flh.cuda.hadamard_transform_half not available")

try:
    from flh.cuda.had_and_quant import hadamard_and_quantize_i4
    HAS_FLH_QUANT = True
except ImportError:
    HAS_FLH_QUANT = False
    print("Warning: flh.cuda.hadamard_and_quantize_i4 not available")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def calculate_throughput(batch_size: int, cols: int, time_seconds: float) -> float:
    """
    Calculate throughput (elements/second) for Hadamard transform

    Args:
        batch_size: Number of rows
        cols: Number of columns (128)
        time_seconds: Execution time in seconds

    Returns:
        Throughput in elements/second
    """
    total_elements = batch_size * cols
    throughput = total_elements / time_seconds
    return throughput

def benchmark_fp16_pytorch(
    batch_sizes: List[int] = [1, 10, 100, 1000, 10000],
    cols: int = 128,
    dtype: torch.dtype = torch.half,
    num_warmup: int = 30,
    num_runs: int = 100
) -> Tuple[List[int], List[float], List[float]]:
    """
    Benchmark FP16 PyTorch Hadamard Transform implementation

    Args:
        batch_sizes: List of batch sizes (number of rows) to test
        cols: Number of columns (fixed at 128)
        dtype: Data type (default torch.half for FP16)
        num_warmup: Number of warmup runs
        num_runs: Number of test runs

    Returns:
        (batch_sizes, avg_times_ms, throughput_values): Batch sizes, average execution times in ms, and throughput values
    """
    print(f"Starting FP16 PyTorch benchmark - Data type: {dtype}, Columns: {cols}")
    print(f"Warmup runs: {num_warmup}, Test runs: {num_runs}")
    print("-" * 60)

    avg_times_ms = []
    throughput_values = []

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")

        # Create input tensor
        x = torch.randn((batch_size, cols), dtype=dtype, device='cuda')

        # Warmup
        for _ in range(num_warmup):
            _ = pytorch_had_trans(x)

        # Synchronize GPU
        torch.cuda.synchronize()

        # Timing
        times = []
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = pytorch_had_trans(x)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
            times.append(elapsed_time)

        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        avg_times_ms.append(avg_time_ms)

        # Calculate throughput
        time_seconds = avg_time_ms / 1000.0
        throughput = calculate_throughput(batch_size, cols, time_seconds)
        throughput_values.append(throughput)

        print(f"  Average time: {avg_time_ms:.4f} ± {std_time_ms:.4f} ms")
        print(f"  Throughput: {throughput:.2e} elements/sec")
        print()

    return batch_sizes, avg_times_ms, throughput_values

def benchmark_fast_hadamard(
    batch_sizes: List[int] = [1, 10, 100, 1000, 10000],
    cols: int = 128,
    dtype: torch.dtype = torch.half,
    num_warmup: int = 30,
    num_runs: int = 100
) -> Tuple[List[int], List[float], List[float]]:
    """
    Benchmark fast-hadamard-transform library implementation

    Args:
        batch_sizes: List of batch sizes (number of rows) to test
        cols: Number of columns (fixed at 128)
        dtype: Data type (default torch.half for FP16)
        num_warmup: Number of warmup runs
        num_runs: Number of test runs

    Returns:
        (batch_sizes, avg_times_ms, throughput_values): Batch sizes, average execution times in ms, and throughput values
    """
    print(f"Starting fast-hadamard-transform benchmark - Data type: {dtype}, Columns: {cols}")
    print(f"Warmup runs: {num_warmup}, Test runs: {num_runs}")
    print("-" * 60)

    avg_times_ms = []
    throughput_values = []

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")

        # Create input tensor
        x = torch.randn((batch_size, cols), dtype=dtype, device='cuda')

        # Warmup
        for _ in range(num_warmup):
            _ = hadamard_transform(x)

        # Synchronize GPU
        torch.cuda.synchronize()

        # Timing
        times = []
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = hadamard_transform(x)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
            times.append(elapsed_time)

        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        avg_times_ms.append(avg_time_ms)

        # Calculate throughput
        time_seconds = avg_time_ms / 1000.0
        throughput = calculate_throughput(batch_size, cols, time_seconds)
        throughput_values.append(throughput)

        print(f"  Average time: {avg_time_ms:.4f} ± {std_time_ms:.4f} ms")
        print(f"  Throughput: {throughput:.2e} elements/sec")
        print()

    return batch_sizes, avg_times_ms, throughput_values

def benchmark_flh_hadamard(
    batch_sizes: List[int] = [1, 10, 100, 1000, 10000],
    cols: int = 128,
    dtype: torch.dtype = torch.half,
    num_warmup: int = 30,
    num_runs: int = 100
) -> Tuple[List[int], List[float], List[float]]:
    """
    Benchmark FLH Hadamard Transform implementation (in-place CUDA kernel)

    Args:
        batch_sizes: List of batch sizes (number of rows) to test
        cols: Number of columns (fixed at 128)
        dtype: Data type (default torch.half for FP16)
        num_warmup: Number of warmup runs
        num_runs: Number of test runs

    Returns:
        (batch_sizes, avg_times_ms, throughput_values): Batch sizes, average execution times in ms, and throughput values

    Note:
        Uses in-place operation, so input data is cloned for each test run
    """
    print(f"Starting FLH Hadamard benchmark - Data type: {dtype}, Columns: {cols}")
    print(f"Warmup runs: {num_warmup}, Test runs: {num_runs}")
    print("-" * 60)

    avg_times_ms = []
    throughput_values = []

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")

        # Create input tensor (in-place operation, so we clone for each run)
        x = torch.randn((batch_size, cols), dtype=dtype, device='cuda')

        # Warmup
        for _ in range(num_warmup):
            test_data = x.clone()
            flh_had_trans_half(test_data)

        # Synchronize GPU
        torch.cuda.synchronize()

        # Timing
        times = []
        for _ in range(num_runs):
            test_data = x.clone()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            flh_had_trans_half(test_data)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
            times.append(elapsed_time)

        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        avg_times_ms.append(avg_time_ms)

        # Calculate throughput
        time_seconds = avg_time_ms / 1000.0
        throughput = calculate_throughput(batch_size, cols, time_seconds)
        throughput_values.append(throughput)

        print(f"  Average time: {avg_time_ms:.4f} ± {std_time_ms:.4f} ms")
        print(f"  Throughput: {throughput:.2e} elements/sec")
        print()

    return batch_sizes, avg_times_ms, throughput_values

def benchmark_flh_fused(
    batch_sizes: List[int] = [1, 10, 100, 1000, 10000],
    cols: int = 128,
    dtype: torch.dtype = torch.half,
    num_warmup: int = 30,
    num_runs: int = 100
) -> Tuple[List[int], List[float], List[float]]:
    """
    Benchmark FLH fused Hadamard + int4 quantization (returns tuple, no in-place modification)

    Args:
        batch_sizes: List of batch sizes (number of rows) to test
        cols: Number of columns (fixed at 128)
        dtype: Data type (default torch.half for FP16)
        num_warmup: Number of warmup runs
        num_runs: Number of test runs

    Returns:
        (batch_sizes, avg_times_ms, throughput_values): Batch sizes, average execution times in ms, and throughput values

    Note:
        This benchmarks the fused operation: Hadamard transform + symmetric int4 quantization.
        The output is a tuple of (packed_uint8, scales).
    """
    print(f"Starting FLH fused (Hadamard+quant_i4) benchmark - Data type: {dtype}, Columns: {cols}")
    print(f"Warmup runs: {num_warmup}, Test runs: {num_runs}")
    print("-" * 60)

    avg_times_ms = []
    throughput_values = []

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")

        # Create input tensor
        x = torch.randn((batch_size, cols), dtype=dtype, device='cuda')

        # Warmup
        for _ in range(num_warmup):
            _ = hadamard_and_quantize_i4(x)

        # Synchronize GPU
        torch.cuda.synchronize()

        # Timing
        times = []
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = hadamard_and_quantize_i4(x)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
            times.append(elapsed_time)

        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        avg_times_ms.append(avg_time_ms)

        # Calculate throughput
        time_seconds = avg_time_ms / 1000.0
        throughput = calculate_throughput(batch_size, cols, time_seconds)
        throughput_values.append(throughput)

        print(f"  Average time: {avg_time_ms:.4f} ± {std_time_ms:.4f} ms")
        print(f"  Throughput: {throughput:.2e} elements/sec")
        print()

    return batch_sizes, avg_times_ms, throughput_values

def compare_all_implementations(
    batch_sizes: List[int] = [1, 10, 100, 1000, 10000],
    cols: int = 128,
    dtype: torch.dtype = torch.half,
    num_warmup: int = 10,
    num_runs: int = 50
) -> Dict[str, Tuple[List[int], List[float], List[float]]]:
    """
    Compare all implementations

    Args:
        batch_sizes: List of batch sizes to test
        cols: Number of columns
        dtype: Data type
        num_warmup: Number of warmup runs
        num_runs: Number of test runs

    Returns:
        Dictionary with results for each implementation
        Format: {name: (batch_sizes, avg_times_ms, throughput_values)}
    """
    print("=" * 80)
    print("Comparing FP16 PyTorch vs fast-hadamard-transform vs FLH Hadamard vs FLH Fused")
    print("=" * 80)

    results = {}

    # Test FP16 PyTorch implementation
    print("\n1. Testing FP16 PyTorch Implementation")
    print("-" * 40)
    try:
        batch_sizes_pt, times_pt, throughput_pt = benchmark_fp16_pytorch(
            batch_sizes, cols, dtype, num_warmup, num_runs
        )
        results["FP16_PyTorch"] = (batch_sizes_pt, times_pt, throughput_pt)
    except Exception as e:
        print(f"FP16 PyTorch implementation failed: {e}")
        import traceback
        traceback.print_exc()
        results["FP16_PyTorch"] = (batch_sizes, [0.0] * len(batch_sizes), [0.0] * len(batch_sizes))

    # Test fast-hadamard-transform implementation
    print("\n2. Testing fast-hadamard-transform Implementation")
    print("-" * 40)
    try:
        batch_sizes_fht, times_fht, throughput_fht = benchmark_fast_hadamard(
            batch_sizes, cols, dtype, num_warmup, num_runs
        )
        results["fast-hadamard-transform"] = (batch_sizes_fht, times_fht, throughput_fht)
    except Exception as e:
        print(f"fast-hadamard-transform implementation failed: {e}")
        import traceback
        traceback.print_exc()
        results["fast-hadamard-transform"] = (batch_sizes, [0.0] * len(batch_sizes), [0.0] * len(batch_sizes))

    # Test FLH Hadamard implementation
    if HAS_FLH_HAD:
        print("\n3. Testing FLH Hadamard (CUDA) Implementation")
        print("-" * 40)
        try:
            batch_sizes_flh_had, times_flh_had, throughput_flh_had = benchmark_flh_hadamard(
                batch_sizes, cols, dtype, num_warmup, num_runs
            )
            results["FLH_Hadamard"] = (batch_sizes_flh_had, times_flh_had, throughput_flh_had)
        except Exception as e:
            print(f"FLH Hadamard implementation failed: {e}")
            import traceback
            traceback.print_exc()
            results["FLH_Hadamard"] = (batch_sizes, [0.0] * len(batch_sizes), [0.0] * len(batch_sizes))

        # Test FLH fused implementation
        print("\n4. Testing FLH Fused (Hadamard+quant_i4) Implementation")
        print("-" * 40)
        try:
            batch_sizes_flh_fused, times_flh_fused, throughput_flh_fused = benchmark_flh_fused(
                batch_sizes, cols, dtype, num_warmup, num_runs
            )
            results["FLH_Fused"] = (batch_sizes_flh_fused, times_flh_fused, throughput_flh_fused)
        except Exception as e:
            print(f"FLH fused implementation failed: {e}")
            import traceback
            traceback.print_exc()
            results["FLH_Fused"] = (batch_sizes, [0.0] * len(batch_sizes), [0.0] * len(batch_sizes))
    else:
        print("\nSkipping FLH benchmarks - CUDA extension not available")

    return results

def test_correctness():
    """Test the correctness of all implementations"""
    print("Testing transform correctness...")

    torch.manual_seed(42)
    x = torch.randn((2, 128), dtype=torch.half, device='cuda')

    # Test FP16 PyTorch
    try:
        x_pt = x.clone()
        result_pt = pytorch_had_trans(x_pt)
        print(f"FP16 PyTorch: Output shape {result_pt.shape}, Range [{result_pt.min().item():.4f}, {result_pt.max().item():.4f}]")
    except Exception as e:
        print(f"FP16 PyTorch failed: {e}")

    # Test fast-hadamard-transform
    try:
        x_fht = x.clone()
        result_fht = hadamard_transform(x_fht)
        print(f"fast-hadamard-transform: Output shape {result_fht.shape}, Range [{result_fht.min().item():.4f}, {result_fht.max().item():.4f}]")
    except Exception as e:
        print(f"fast-hadamard-transform failed: {e}")

    # Test FLH Hadamard
    if HAS_FLH_HAD:
        try:
            x_flh = x.clone()
            flh_had_trans_half(x_flh)
            print(f"FLH Hadamard: Output shape {x_flh.shape}, Range [{x_flh.min().item():.4f}, {x_flh.max().item():.4f}]")
        except Exception as e:
            print(f"FLH Hadamard failed: {e}")

    # Test FLH fused
    if HAS_FLH_QUANT:
        try:
            x_fused = x.clone()
            q_packed, scales = hadamard_and_quantize_i4(x_fused)
            print(f"FLH Fused: q_packed shape {q_packed.shape}, scales shape {scales.shape}")
        except Exception as e:
            print(f"FLH Fused failed: {e}")

    print()

def plot_throughput_comparison(results: Dict[str, Tuple[List[int], List[float], List[float]]]):
    """Plot speedup comparison results for all implementations vs FP16 PyTorch"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping plot")
        return

    # Set global font sizes
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['font.family'] = 'Arial'

    fig = plt.figure(figsize=(10, 8))

    # Define colors and markers for different implementations
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'v', 'D', 'p']

    # Find FP16 PyTorch key
    fp16_key = None
    for key in results.keys():
        if "FP16_PyTorch" in key:
            fp16_key = key
            break

    ax = fig.add_subplot(111)

    # First, add FP16 PyTorch baseline (speedup = 1, dashed gray line only)
    if fp16_key and results[fp16_key][2][0] > 0:
        fp16_batch_sizes = results[fp16_key][0]
        speedup_baseline = [1.0] * len(fp16_batch_sizes)
        plt.semilogx(fp16_batch_sizes, speedup_baseline,
                    color='gray', linestyle='--', linewidth=2, label='FP16_PyTorch')

    # Collect legend handles
    all_handles = []
    all_labels = []

    # Plot speedup for all implementations relative to FP16 PyTorch
    if fp16_key and results[fp16_key][2][0] > 0:
        _, _, fp16_throughput = results[fp16_key]
        for i, (name, (batch_sizes, times, throughput)) in enumerate(results.items()):
            if throughput[0] > 0 and name != fp16_key:
                speedups = [t / ft for t, ft in zip(throughput, fp16_throughput)]
                line, = plt.semilogx(batch_sizes, speedups,
                            color=colors[i % len(colors)],
                            marker=markers[i % len(markers)],
                            linewidth=3, markersize=12, label=name)
                if name not in all_labels:
                    all_handles.append(line)
                    all_labels.append(name)

    # Add FP16 PyTorch to legend
    if fp16_key and results[fp16_key][2][0] > 0:
        line_baseline = plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2)
        all_handles.insert(0, line_baseline)
        all_labels.insert(0, 'FP16_PyTorch')

    plt.xlabel('Batch Size (Number of Rows)', fontsize=22, weight='bold')
    plt.ylabel('Speedup', fontsize=24, weight='bold')
    plt.title('Throughput Speedup vs FP16 PyTorch', fontsize=18, weight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=2)

    # Place legend at the top center
    fig.legend(all_handles, all_labels, loc='upper center',
              ncol=min(4, len(all_labels)), fontsize=18, frameon=True,
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.0)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('hadamard_throughput_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'hadamard_throughput_comparison.png'")
    plt.show()

def main():
    """Main function"""
    print("=" * 80)
    print("Hadamard Transform Throughput Benchmark Suite")
    print("Comparing FP16 PyTorch vs fast-hadamard-transform vs FLH")
    print("=" * 80)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return

    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()

    # Correctness test
    test_correctness()

    # Compare implementations
    batch_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    results = compare_all_implementations(batch_sizes, num_warmup=10, num_runs=30)

    # Plot comparison results
    plot_throughput_comparison(results)

    # Summary
    print("=" * 80)
    print("Benchmark Summary:")
    print("=" * 80)

    for name, (batch_sizes, times, throughput) in results.items():
        if throughput[0] > 0:  # Skip failed implementations
            max_throughput = max(throughput)
            min_time = min(times)
            max_throughput_idx = throughput.index(max_throughput)
            print(f"{name:25}: Max Throughput: {max_throughput:.2e} elements/sec (at batch_size={batch_sizes[max_throughput_idx]}), Min time: {min_time:.4f} ms")

    # Calculate speedups relative to FP16 PyTorch
    fp16_key = None
    for key in results.keys():
        if "FP16_PyTorch" in key:
            fp16_key = key
            break

    if fp16_key and results[fp16_key][2][0] > 0:
        print(f"\nThroughput Speedup vs {fp16_key}:")
        _, _, fp16_throughput = results[fp16_key]
        for name, (batch_sizes, times, throughput) in results.items():
            if throughput[0] > 0 and name != fp16_key:
                avg_speedup = np.mean([t / ft for t, ft in zip(throughput, fp16_throughput)])
                max_speedup = max([t / ft for t, ft in zip(throughput, fp16_throughput)])
                print(f"  {name:25}: Avg speedup: {avg_speedup:.2f}x, Max speedup: {max_speedup:.2f}x")

    print("=" * 80)

if __name__ == "__main__":
    main()

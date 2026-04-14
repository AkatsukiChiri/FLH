"""
FLH Hadamard Transform Speedup Benchmark

This benchmark compares FLH's Hadamard implementations (128, 64, 32 groups)
against FP16 PyTorch baseline and fast-hadamard-transform library.

All implementations are benchmarked in terms of throughput (elements/second)
and speedup relative to the FP16 PyTorch baseline.
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
    from flh.cuda.hadamard import (
        hadamard_transform_half,
        hadamard_transform_64_half,
        hadamard_transform_32_half,
    )
    HAS_FLH_ALL = True
except ImportError as e:
    HAS_FLH_ALL = False
    print(f"Warning: FLH CUDA extensions not available: {e}")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, plots will be skipped")


def calculate_throughput(batch_size: int, cols: int, time_seconds: float) -> float:
    """
    Calculate throughput (elements/second) for Hadamard transform

    Args:
        batch_size: Number of rows
        cols: Number of columns
        time_seconds: Execution time in seconds

    Returns:
        Throughput in elements/second
    """
    total_elements = batch_size * cols
    throughput = total_elements / time_seconds
    return throughput


# ==================== Baseline: FP16 PyTorch ====================

def benchmark_fp16_pytorch(
    batch_sizes: List[int],
    cols: int,
    dtype: torch.dtype = torch.half,
    num_warmup: int = 10,
    num_runs: int = 30
) -> Tuple[List[int], List[float], List[float]]:
    """
    Benchmark FP16 PyTorch Hadamard Transform implementation (baseline)

    Args:
        batch_sizes: List of batch sizes (number of rows) to test
        cols: Number of columns
        dtype: Data type (default torch.half for FP16)
        num_warmup: Number of warmup runs
        num_runs: Number of test runs

    Returns:
        (batch_sizes, avg_times_ms, throughput_values)
    """
    print(f"Starting FP16 PyTorch benchmark - Data type: {dtype}, Columns: {cols}")
    print(f"Warmup runs: {num_warmup}, Test runs: {num_runs}")
    print("-" * 60)

    avg_times_ms = []
    throughput_values = []

    for batch_size in batch_sizes:
        x = torch.randn((batch_size, cols), dtype=dtype, device='cuda')

        # Warmup
        for _ in range(num_warmup):
            _ = pytorch_had_trans(x)

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
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)

        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        avg_times_ms.append(avg_time_ms)

        time_seconds = avg_time_ms / 1000.0
        throughput = calculate_throughput(batch_size, cols, time_seconds)
        throughput_values.append(throughput)

        print(f"  batch_size={batch_size:6d}: {avg_time_ms:.4f} ± {std_time_ms:.4f} ms, "
              f"throughput={throughput:.2e} elem/s")

    return batch_sizes, avg_times_ms, throughput_values


# ==================== fast-hadamard-transform library ====================

def benchmark_fast_hadamard(
    batch_sizes: List[int],
    cols: int,
    dtype: torch.dtype = torch.half,
    num_warmup: int = 10,
    num_runs: int = 30
) -> Tuple[List[int], List[float], List[float]]:
    """
    Benchmark fast-hadamard-transform library implementation

    Args:
        batch_sizes: List of batch sizes to test
        cols: Number of columns
        dtype: Data type
        num_warmup: Number of warmup runs
        num_runs: Number of test runs

    Returns:
        (batch_sizes, avg_times_ms, throughput_values)
    """
    print(f"Starting fast-hadamard-transform benchmark - Columns: {cols}")
    print(f"Warmup runs: {num_warmup}, Test runs: {num_runs}")
    print("-" * 60)

    avg_times_ms = []
    throughput_values = []

    for batch_size in batch_sizes:
        x = torch.randn((batch_size, cols), dtype=dtype, device='cuda')

        # Warmup
        for _ in range(num_warmup):
            _ = hadamard_transform(x)

        torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = hadamard_transform(x)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)

        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        avg_times_ms.append(avg_time_ms)

        time_seconds = avg_time_ms / 1000.0
        throughput = calculate_throughput(batch_size, cols, time_seconds)
        throughput_values.append(throughput)

        print(f"  batch_size={batch_size:6d}: {avg_time_ms:.4f} ± {std_time_ms:.4f} ms, "
              f"throughput={throughput:.2e} elem/s")

    return batch_sizes, avg_times_ms, throughput_values


# ==================== FLH 128-group ====================

def benchmark_flh_hadamard_128(
    batch_sizes: List[int],
    cols: int = 128,
    dtype: torch.dtype = torch.half,
    num_warmup: int = 10,
    num_runs: int = 30
) -> Tuple[List[int], List[float], List[float]]:
    """
    Benchmark FLH Hadamard 128-group implementation

    Args:
        batch_sizes: List of batch sizes to test
        cols: Number of columns (must be 128)
        dtype: Data type
        num_warmup: Number of warmup runs
        num_runs: Number of test runs

    Returns:
        (batch_sizes, avg_times_ms, throughput_values)
    """
    assert cols == 128, "FLH 128-group requires cols=128"

    print(f"Starting FLH Hadamard 128-group benchmark - Columns: {cols}")
    print(f"Warmup runs: {num_warmup}, Test runs: {num_runs}")
    print("-" * 60)

    avg_times_ms = []
    throughput_values = []

    for batch_size in batch_sizes:
        x = torch.randn((batch_size, cols), dtype=dtype, device='cuda')

        # Warmup
        for _ in range(num_warmup):
            test_data = x.clone()
            hadamard_transform_half(test_data)

        torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            test_data = x.clone()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            hadamard_transform_half(test_data)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)

        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        avg_times_ms.append(avg_time_ms)

        time_seconds = avg_time_ms / 1000.0
        throughput = calculate_throughput(batch_size, cols, time_seconds)
        throughput_values.append(throughput)

        print(f"  batch_size={batch_size:6d}: {avg_time_ms:.4f} ± {std_time_ms:.4f} ms, "
              f"throughput={throughput:.2e} elem/s")

    return batch_sizes, avg_times_ms, throughput_values


# ==================== FLH 64-group ====================

def benchmark_flh_hadamard_64(
    batch_sizes: List[int],
    cols: int = 64,
    dtype: torch.dtype = torch.half,
    num_warmup: int = 10,
    num_runs: int = 30
) -> Tuple[List[int], List[float], List[float]]:
    """
    Benchmark FLH Hadamard 64-group implementation

    Args:
        batch_sizes: List of batch sizes to test
        cols: Number of columns (must be 64)
        dtype: Data type
        num_warmup: Number of warmup runs
        num_runs: Number of test runs

    Returns:
        (batch_sizes, avg_times_ms, throughput_values)
    """
    assert cols == 64, "FLH 64-group requires cols=64"

    print(f"Starting FLH Hadamard 64-group benchmark - Columns: {cols}")
    print(f"Warmup runs: {num_warmup}, Test runs: {num_runs}")
    print("-" * 60)

    avg_times_ms = []
    throughput_values = []

    for batch_size in batch_sizes:
        x = torch.randn((batch_size, cols), dtype=dtype, device='cuda')

        # Warmup
        for _ in range(num_warmup):
            test_data = x.clone()
            hadamard_transform_64_half(test_data)

        torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            test_data = x.clone()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            hadamard_transform_64_half(test_data)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)

        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        avg_times_ms.append(avg_time_ms)

        time_seconds = avg_time_ms / 1000.0
        throughput = calculate_throughput(batch_size, cols, time_seconds)
        throughput_values.append(throughput)

        print(f"  batch_size={batch_size:6d}: {avg_time_ms:.4f} ± {std_time_ms:.4f} ms, "
              f"throughput={throughput:.2e} elem/s")

    return batch_sizes, avg_times_ms, throughput_values


# ==================== FLH 32-group ====================

def benchmark_flh_hadamard_32(
    batch_sizes: List[int],
    cols: int = 32,
    dtype: torch.dtype = torch.half,
    num_warmup: int = 10,
    num_runs: int = 30
) -> Tuple[List[int], List[float], List[float]]:
    """
    Benchmark FLH Hadamard 32-group implementation

    Args:
        batch_sizes: List of batch sizes to test
        cols: Number of columns (must be 32)
        dtype: Data type
        num_warmup: Number of warmup runs
        num_runs: Number of test runs

    Returns:
        (batch_sizes, avg_times_ms, throughput_values)
    """
    assert cols == 32, "FLH 32-group requires cols=32"

    print(f"Starting FLH Hadamard 32-group benchmark - Columns: {cols}")
    print(f"Warmup runs: {num_warmup}, Test runs: {num_runs}")
    print("-" * 60)

    avg_times_ms = []
    throughput_values = []

    for batch_size in batch_sizes:
        x = torch.randn((batch_size, cols), dtype=dtype, device='cuda')

        # Warmup
        for _ in range(num_warmup):
            test_data = x.clone()
            hadamard_transform_32_half(test_data)

        torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            test_data = x.clone()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            hadamard_transform_32_half(test_data)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)

        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        avg_times_ms.append(avg_time_ms)

        time_seconds = avg_time_ms / 1000.0
        throughput = calculate_throughput(batch_size, cols, time_seconds)
        throughput_values.append(throughput)

        print(f"  batch_size={batch_size:6d}: {avg_time_ms:.4f} ± {std_time_ms:.4f} ms, "
              f"throughput={throughput:.2e} elem/s")

    return batch_sizes, avg_times_ms, throughput_values


# ==================== Compare All ====================

def compare_all_implementations(
    batch_sizes: List[int] = [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    cols_list: List[int] = [128, 64, 32],
    dtype: torch.dtype = torch.half,
    num_warmup: int = 10,
    num_runs: int = 30
) -> Dict[str, Dict[str, Tuple[List[int], List[float], List[float]]]]:
    """
    Compare all implementations for different column sizes

    Args:
        batch_sizes: List of batch sizes to test
        cols_list: List of column sizes to test (128, 64, 32)
        dtype: Data type
        num_warmup: Number of warmup runs
        num_runs: Number of test runs

    Returns:
        Nested dictionary: {cols: {name: (batch_sizes, times, throughput)}}
    """
    print("=" * 80)
    print("FLH Hadamard Transform Speedup Benchmark")
    print(f"Comparing FP16 PyTorch vs fast-hadamard-transform vs FLH (128/64/32)")
    print("=" * 80)

    all_results = {}

    for cols in cols_list:
        print(f"\n{'='*80}")
        print(f"Testing with cols={cols}")
        print(f"{'='*80}")

        results = {}

        # 1. FP16 PyTorch baseline
        print("\n1. FP16 PyTorch (baseline)")
        try:
            bs, times, tp = benchmark_fp16_pytorch(batch_sizes, cols, dtype, num_warmup, num_runs)
            results["FP16_PyTorch"] = (bs, times, tp)
        except Exception as e:
            print(f"  FAILED: {e}")
            results["FP16_PyTorch"] = (batch_sizes, [0.0]*len(batch_sizes), [0.0]*len(batch_sizes))

        # 2. fast-hadamard-transform
        print("\n2. fast-hadamard-transform")
        try:
            bs, times, tp = benchmark_fast_hadamard(batch_sizes, cols, dtype, num_warmup, num_runs)
            results["fast-hadamard-transform"] = (bs, times, tp)
        except Exception as e:
            print(f"  FAILED: {e}")
            results["fast-hadamard-transform"] = (batch_sizes, [0.0]*len(batch_sizes), [0.0]*len(batch_sizes))

        # 3. FLH implementations (if available)
        if HAS_FLH_ALL:
            if cols == 128:
                print("\n3. FLH Hadamard 128")
                try:
                    bs, times, tp = benchmark_flh_hadamard_128(batch_sizes, cols, dtype, num_warmup, num_runs)
                    results["FLH_128"] = (bs, times, tp)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    results["FLH_128"] = (batch_sizes, [0.0]*len(batch_sizes), [0.0]*len(batch_sizes))

            elif cols == 64:
                print("\n3. FLH Hadamard 64")
                try:
                    bs, times, tp = benchmark_flh_hadamard_64(batch_sizes, cols, dtype, num_warmup, num_runs)
                    results["FLH_64"] = (bs, times, tp)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    results["FLH_64"] = (batch_sizes, [0.0]*len(batch_sizes), [0.0]*len(batch_sizes))

            elif cols == 32:
                print("\n3. FLH Hadamard 32")
                try:
                    bs, times, tp = benchmark_flh_hadamard_32(batch_sizes, cols, dtype, num_warmup, num_runs)
                    results["FLH_32"] = (bs, times, tp)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    results["FLH_32"] = (batch_sizes, [0.0]*len(batch_sizes), [0.0]*len(batch_sizes))
        else:
            print("\nSkipping FLH benchmarks - CUDA extension not available")

        all_results[cols] = results

    return all_results


# ==================== Plotting ====================

def plot_throughput_comparison(all_results: Dict[str, Dict[str, Tuple[List[int], List[float], List[float]]]]):
    """
    Plot throughput comparison in 2x2 grid:
    - Top-left: cols=32 (all implementations)
    - Top-right: cols=64 (all implementations)
    - Bottom-left: cols=128 (all implementations)
    - Bottom-right: FLH comparison across all sizes (no PyTorch/fast-hadamard)
    
    All subplots share a unified legend at the top.

    Args:
        all_results: Nested dictionary from compare_all_implementations()
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping plots")
        return

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['font.family'] = 'Arial'

    cols_list = sorted(all_results.keys())

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Color and marker mapping for all implementations
    impl_colors = {
        'FP16_PyTorch': 'blue',
        'fast-hadamard-transform': 'red',
        'FLH_128': 'green',
        'FLH_64': 'orange',
        'FLH_32': 'purple',
    }
    impl_markers = {
        'FP16_PyTorch': 'o',
        'fast-hadamard-transform': 's',
        'FLH_128': '^',
        'FLH_64': 'D',
        'FLH_32': 'o',
    }

    # All implementation names in desired order
    all_impl_names = ['FP16_PyTorch', 'fast-hadamard-transform', 'FLH_128', 'FLH_64', 'FLH_32']

    # ========== Subplot 1: cols=32 (top-left) ==========
    ax = axes[0]
    if 32 in all_results:
        results = all_results[32]
        for name in all_impl_names:
            if name in results:
                bs, times, tp = results[name]
                if tp[0] > 0:
                    ax.loglog(bs, tp,
                             color=impl_colors.get(name, 'gray'),
                             marker=impl_markers.get(name, 'x'),
                             linewidth=2.5, markersize=10,
                             label=name)
    ax.set_xlabel('Batch Size', fontsize=18, weight='bold')
    ax.set_ylabel('Throughput (elements/sec)', fontsize=18, weight='bold')
    ax.set_title('cols=32', fontsize=20, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)

    # ========== Subplot 2: cols=64 (top-right) ==========
    ax = axes[1]
    if 64 in all_results:
        results = all_results[64]
        for name in all_impl_names:
            if name in results:
                bs, times, tp = results[name]
                if tp[0] > 0:
                    ax.loglog(bs, tp,
                             color=impl_colors.get(name, 'gray'),
                             marker=impl_markers.get(name, 'x'),
                             linewidth=2.5, markersize=10,
                             label=name)
    ax.set_xlabel('Batch Size', fontsize=18, weight='bold')
    ax.set_ylabel('Throughput (elements/sec)', fontsize=18, weight='bold')
    ax.set_title('cols=64', fontsize=20, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)

    # ========== Subplot 3: cols=128 (bottom-left) ==========
    ax = axes[2]
    if 128 in all_results:
        results = all_results[128]
        for name in all_impl_names:
            if name in results:
                bs, times, tp = results[name]
                if tp[0] > 0:
                    ax.loglog(bs, tp,
                             color=impl_colors.get(name, 'gray'),
                             marker=impl_markers.get(name, 'x'),
                             linewidth=2.5, markersize=10,
                             label=name)
    ax.set_xlabel('Batch Size', fontsize=18, weight='bold')
    ax.set_ylabel('Throughput (elements/sec)', fontsize=18, weight='bold')
    ax.set_title('cols=128', fontsize=20, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)

    # ========== Subplot 4: FLH Comparison across all sizes (bottom-right) ==========
    ax = axes[3]
    cols_to_label = {32: '32', 64: '64', 128: '128'}
    flh_colors = {
        'cols=32': 'purple',
        'cols=64': 'orange',
        'cols=128': 'green',
    }
    flh_markers = {
        'cols=32': 'o',
        'cols=64': 'D',
        'cols=128': '^',
    }
    for cols in cols_list:
        results = all_results[cols]
        for name in ['FLH_32', 'FLH_64', 'FLH_128']:
            if name in results:
                bs, times, tp = results[name]
                if tp[0] > 0:
                    ax.loglog(bs, tp,
                             color=flh_colors.get(f'cols={cols}', 'gray'),
                             marker=flh_markers.get(f'cols={cols}', 'x'),
                             linewidth=2.5, markersize=10,
                             label=f"{name} (cols={cols_to_label.get(cols, cols)})")
    ax.set_xlabel('Batch Size', fontsize=18, weight='bold')
    ax.set_ylabel('Throughput (elements/sec)', fontsize=18, weight='bold')
    ax.set_title('FLH Comparison', fontsize=20, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)

    # ========== Unified legend at the top ==========
    all_handles = []
    all_labels = []
    for ax in axes[:3]:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in all_labels:
                all_handles.append(h)
                all_labels.append(l)

    fig.legend(all_handles, all_labels,
              loc='upper center', bbox_to_anchor=(0.5, 1.03),
              ncol=5, fontsize=14, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('flh_hadamard_throughput_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'flh_hadamard_throughput_comparison.png'")
    plt.show()


def plot_speedup_comparison(all_results: Dict[str, Dict[str, Tuple[List[int], List[float], List[float]]]]):
    """
    Plot speedup comparison for all column sizes

    Args:
        all_results: Nested dictionary from compare_all_implementations()
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping plots")
        return

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['font.family'] = 'Arial'

    cols_list = sorted(all_results.keys())
    n_cols = len(cols_list)

    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    # ========== Step 1: 先收集所有子图中的 speedup 值，计算全局最大 speedup ==========
    all_speedup_values = []
    for cols in cols_list:
        results = all_results[cols]
        baseline_tp = results.get("FP16_PyTorch", ([], [], []))[2]
        if baseline_tp is None or baseline_tp[0] <= 0:
            continue
        for name, (bs, times, tp) in results.items():
            if name == "FP16_PyTorch" or tp[0] <= 0:
                continue
            speedups = [t / b for t, b in zip(tp, baseline_tp)]
            all_speedup_values.extend(speedups)

    # 计算全局最大 speedup，向上取整到最近的 0.5 的倍数，至少为 1.0
    if all_speedup_values:
        global_max_speedup = max(all_speedup_values)
        # 向上取整到最近的 0.5 的倍数
        ylim_max = max(1.0, np.ceil(global_max_speedup * 2) / 2)
    else:
        ylim_max = 1.0

    print(f"Global max speedup: {global_max_speedup:.2f}x, setting ylim_max={ylim_max:.1f}")

    # ========== Step 2: 绘制每个子图，使用统一的纵坐标范围 ==========
    for idx, cols in enumerate(cols_list):
        ax = axes[idx]
        results = all_results[cols]

        # Get baseline (FP16_PyTorch)
        baseline_tp = None
        if "FP16_PyTorch" in results and results["FP16_PyTorch"][2][0] > 0:
            baseline_tp = results["FP16_PyTorch"][2]

        if baseline_tp is None:
            ax.set_title(f"cols={cols} (no baseline)", fontsize=16, weight='bold')
            continue

        batch_sizes = results["FP16_PyTorch"][0]

        # Define colors and markers
        colors = {
            'fast-hadamard-transform': 'red',
            'FLH_128': 'green',
            'FLH_64': 'orange',
            'FLH_32': 'purple',
        }
        markers = {
            'fast-hadamard-transform': 's',
            'FLH_128': '^',
            'FLH_64': 'D',
            'FLH_32': 'o',
        }

        # Plot each implementation
        for name, (bs, times, tp) in results.items():
            if name == "FP16_PyTorch" or tp[0] <= 0:
                continue
            speedups = [t / b for t, b in zip(tp, baseline_tp)]
            ax.semilogx(bs, speedups,
                       color=colors.get(name, 'gray'),
                       marker=markers.get(name, 'x'),
                       linewidth=2.5, markersize=10,
                       label=name)

        ax.set_xlabel('Batch Size', fontsize=14, weight='bold')
        ax.set_ylabel('Speedup', fontsize=14, weight='bold')
        ax.set_title(f'Hadamard Speedup (cols={cols})', fontsize=16, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        ax.legend(fontsize=12, loc='best')

        # 统一设置纵坐标范围
        ax.set_ylim(0, ylim_max)

    plt.tight_layout()
    plt.savefig('flh_hadamard_speedup_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'flh_hadamard_speedup_comparison.png'")
    plt.show()


def print_summary_table(all_results: Dict[str, Dict[str, Tuple[List[int], List[float], List[float]]]]):
    """Print summary table of max speedups for each implementation"""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY - Max Speedup vs FP16 PyTorch")
    print("=" * 80)
    print(f"{'Columns':<10} {'Implementation':<30} {'Max Speedup':>12} {'@ batch_size':>12}")
    print("-" * 80)

    for cols in sorted(all_results.keys()):
        results = all_results[cols]
        baseline_tp = results.get("FP16_PyTorch", ([], [], []))[2]

        if not baseline_tp or baseline_tp[0] <= 0:
            continue

        for name, (bs, times, tp) in results.items():
            if name == "FP16_PyTorch" or tp[0] <= 0:
                continue

            speedups = [t / b for t, b in zip(tp, baseline_tp)]
            max_speedup = max(speedups)
            max_idx = speedups.index(max_speedup)
            max_batch = bs[max_idx]

            print(f"{cols:<10} {name:<30} {max_speedup:>12.2f}x {max_batch:>12}")

    print("=" * 80)


def main():
    """Main benchmark function"""
    print("=" * 80)
    print("FLH Hadamard Transform Speedup Benchmark")
    print("Comparing: FP16 PyTorch | fast-hadamard-transform | FLH 128/64/32")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"CUDA Device : {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Run benchmarks
    batch_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    cols_list = [128, 64, 32]

    all_results = compare_all_implementations(batch_sizes, cols_list, num_warmup=10, num_runs=30)

    # Print summary
    print_summary_table(all_results)

    # Plot results
    if HAS_MATPLOTLIB and all_results:
        try:
            print("\nGenerating speedup plot...")
            plot_speedup_comparison(all_results)
            print("\nGenerating throughput plot...")
            plot_throughput_comparison(all_results)
        except Exception as e:
            print(f"Plotting failed: {e}")

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()

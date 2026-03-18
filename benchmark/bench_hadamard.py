import os
import time
import math

import torch
import flh


def cuda_time_ms(fn, iters: int = 200, warmup: int = 50) -> float:
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


def get_hadamard_matrix(n: int, dtype=torch.float16, device="cuda"):
    """生成Hadamard矩阵 H_n"""
    h = torch.eye(n, dtype=dtype, device=device)
    for i in range(int(math.log2(n))):
        step = 2 ** (i + 1)
        for j in range(0, n, step):
            for k in range(j, j + step // 2):
                a = h[k, :].clone()
                b = h[k + step // 2, :].clone()
                h[k, :] = a + b
                h[k + step // 2, :] = a - b
    return h


def hadamard_transform_via_matmul(x: torch.Tensor) -> torch.Tensor:
    """
    通过矩阵乘法实现Hadamard变换: y = x @ H
    """
    n = x.shape[-1]
    h = get_hadamard_matrix(n, x.dtype, x.device)
    return x @ h


def main():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    assert torch.cuda.is_available()

    device = "cuda"
    n = 128  # Hadamard变换的维度

    # 测试不同的行数
    sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    # 预计算Hadamard矩阵（用于矩阵乘法方法）
    had_matrix = get_hadamard_matrix(n, dtype=torch.float16, device=device)

    print(f"{'M':>6} | {'Matmul':>10} | {'fast_hadamard':>14} | {'CUDA':>10} | {'matmul/fast':>10} | {'matmul/cuda':>10}")
    print("-" * 80)

    for m in sizes:
        # 创建输入
        x = torch.randn(m, n, dtype=torch.float16, device=device)

        # 方法1: 通过矩阵乘法实现
        def run_matmul():
            return x @ had_matrix

        # 方法2: flh.nn.fast_hadamard_transform
        def run_fast_hadamard():
            return flh.nn.fast_hadamard_transform(x, group_size=128, normalize=True)

        # 方法3: CUDA kernel (如果可用)
        def run_cuda():
            x_copy = x.clone().view(-1, 128)
            
            return flh._CUDA.hadamard_transform_half(x_copy).view(x.shape)

        # 预热
        for _ in range(10):
            _ = run_matmul()
            _ = run_fast_hadamard()
            if flh._CUDA is not None:
                _ = run_cuda()
        torch.cuda.synchronize()

        # 基准测试
        iters = 500 if m <= 512 else 200
        warmup = 50 if m <= 512 else 20

        ms_matmul = cuda_time_ms(run_matmul, iters=iters, warmup=warmup)
        ms_fast = cuda_time_ms(run_fast_hadamard, iters=iters, warmup=warmup)

        if flh._CUDA is not None:
            ms_cuda = cuda_time_ms(run_cuda, iters=iters, warmup=warmup)
            print(f"{m:>6} | {ms_matmul:>10.4f} | {ms_fast:>14.4f} | {ms_cuda:>10.4f} | {ms_matmul/ms_fast:>10.2f}x | {ms_matmul/ms_cuda:>10.2f}x")
        else:
            print(f"{m:>6} | {ms_matmul:>10.4f} | {ms_fast:>14.4f} | {'N/A':>10} | {ms_matmul/ms_fast:>10.2f}x | {'N/A':>10}")

        # 验证正确性（只验证小尺寸）
        if m <= 8:
            result_matmul = run_matmul()
            result_fast = run_fast_hadamard()
            result_fast_scaled = result_fast * math.sqrt(n)

            diff = (result_matmul - result_fast_scaled).abs()
            max_diff = diff.max().item()
            print(f"       验证: max_diff = {max_diff:.2e}")

            if flh._CUDA is not None:
                result_cuda = run_cuda()
                diff_cuda = (result_matmul - result_cuda).abs()
                max_diff_cuda = diff_cuda.max().item()
                print(f"       CUDA验证: max_diff = {max_diff_cuda:.2e}")


if __name__ == "__main__":
    main()

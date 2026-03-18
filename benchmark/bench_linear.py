import os
import time

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


def tflops(M: int, N: int, K: int, ms: float) -> float:
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)

    assert torch.cuda.is_available()
    assert flh._CUDA is not None

    device = "cuda"
    group_size = 128

    sizes = [
        (1, 4096, 4096),
        (8, 4096, 4096),
        (16, 4096, 4096),
        (32, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
        (256, 4096, 4096),
        (512, 4096, 4096),
        (1024, 4096, 4096),
    ]

    for (M, N, K) in sizes:
        print(f"\n{'='*60}")
        print(f"M={M} N={N} K={K}")

        # 创建原始 FP16 Linear 层
        fp16_layer = torch.nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
        fp16_layer.weight.data.copy_(torch.randint(-4, 4, (N, K), dtype=torch.float16, device=device))

        # 转换为 FLH Linear 层
        flh_layer = flh.nn.LinearFLH.from_float(
            fp16_layer,
            weight_bits=4,
            weight_group_size=group_size,
            in_group_size=group_size,
            out_group_size=group_size,
            weight_sym=True,
            no_hadamard=True,
            dual_hadamard=False
        )

        # 创建输入
        x = torch.randint(-4, 4, (M, K), dtype=torch.float16, device=device)

        # FP16 前向传播
        def run_fp16():
            _ = fp16_layer(x)

        # FLH 前向传播 (包含激活量化)
        act_quantizer = flh.nn.ActQuantizer(
            bits=4,
            group_size=group_size,
            sym=True,
            use_hadamard=False,
            packed_output=True
        )

        scale, zp, x_packed = act_quantizer(x)

        def run_flh():
            _ = flh_layer(x_packed, scale, zp, x_is_packed=True)

        ms_fp16 = cuda_time_ms(run_fp16)
        ms_flh = cuda_time_ms(run_flh)

        tflops_fp16 = tflops(M, N, K, ms_fp16)
        tflops_flh = tflops(M, N, K, ms_flh)

        print(f"FP16:   {ms_fp16:.4f} ms  {tflops_fp16:.3f} TFLOPS")
        print(f"FLH:    {ms_flh:.4f} ms  {tflops_flh:.3f} TFLOPS")
        print(f"加速比: {ms_fp16 / ms_flh:.3f}x")

        # 验证结果正确性
        y_fp16 = fp16_layer(x)
        y_flh = flh_layer(x_packed, scale, zp, x_is_packed=True)
        
        # 计算相对误差 - 使用更安全的方式
        diff = (y_flh - y_fp16).abs()
        # 只在 y_fp16.abs() > 1e-6 的位置计算相对误差
        valid_mask = y_fp16.abs() > 1e-6
        if valid_mask.any():
            rel_error = (diff[valid_mask] / y_fp16.abs()[valid_mask]).mean()
        else:
            rel_error = torch.tensor(float('nan'))
        max_error = diff.max()
        
        print(f"验证 - 相对误差: {rel_error:.4e}, 最大误差: {max_error:.4e}")


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        pass
    main()

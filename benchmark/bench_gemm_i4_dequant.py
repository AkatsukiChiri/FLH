import os
import time
 
import torch
import flh
 
 
def pack_s4(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int32)
    return ((x[:, 0::2] & 0xF) | ((x[:, 1::2] & 0xF) << 4)).to(torch.uint8).contiguous()
 
 
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
        (1, 1024, 2048),
        (8, 1024, 2048),
        (16, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
        (256, 4096, 4096),
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (2048, 4096, 4096),
        (4096, 4096, 4096),
    ]
 
    has_scaled_mm = hasattr(torch, "_scaled_mm") or hasattr(torch.ops.aten, "_scaled_mm")
 
    for (M, N, K) in sizes:
        assert K % group_size == 0
        Kg = K // group_size
 
        A_int = torch.randint(-8, 8, (M, K), device=device, dtype=torch.int32)
        B_int = torch.randint(-8, 8, (N, K), device=device, dtype=torch.int32)
        A_packed = pack_s4(A_int)
        B_packed = pack_s4(B_int)
        A_scale = (torch.rand((M, Kg), device=device, dtype=torch.float16) * 2).contiguous()
        B_scale = (torch.rand((N, Kg), device=device, dtype=torch.float16) * 2).contiguous()
 
        def run_flh():
            flh._CUDA.gemm_i4_dequant_o16(A_packed, B_packed, A_scale, B_scale)
 
        def run_fp16():
            A_deq = (A_int.view(M, Kg, group_size).to(torch.float16) * A_scale.view(M, Kg, 1)).view(M, K)
            B_deq = (B_int.view(N, Kg, group_size).to(torch.float16) * B_scale.view(N, Kg, 1)).view(N, K)
            A_deq @ B_deq.t()
 
        print(f"\nM={M} N={N} K={K}")
 
        ms_flh = cuda_time_ms(run_flh)
        ms_fp16 = cuda_time_ms(run_fp16)
        print(f"flh_i4_dequant_o16: {ms_flh:.4f} ms  {tflops(M,N,K,ms_flh):.3f} TFLOPS")
        print(f"torch_fp16_mm     : {ms_fp16:.4f} ms  {tflops(M,N,K,ms_fp16):.3f} TFLOPS")
        print(f"speedup_vs_fp16   : {ms_fp16 / ms_flh:.3f}x")
 
        if has_scaled_mm:
            A8 = torch.clamp(A_int, -128, 127).to(torch.int8).contiguous()
            B8 = torch.clamp(B_int, -128, 127).to(torch.int8).contiguous()
            sa = torch.full((M,), 1.0, device=device, dtype=torch.float16)
            sb = torch.full((N,), 1.0, device=device, dtype=torch.float16)
 
            def run_i8_scaled_mm():
                try:
                    torch._scaled_mm(A8, B8.t(), sa, sb, out_dtype=torch.float16)
                except Exception:
                    torch.ops.aten._scaled_mm(A8, B8.t(), sa, sb, out_dtype=torch.float16)
 
            try:
                ms_i8 = cuda_time_ms(run_i8_scaled_mm)
                print(f"torch_int8_scaled : {ms_i8:.4f} ms  {tflops(M,N,K,ms_i8):.3f} TFLOPS")
                print(f"speedup_vs_int8   : {ms_i8 / ms_flh:.3f}x")
            except Exception:
                print("torch_int8_scaled : skip")
        else:
            print("torch_int8_scaled : skip")
 
 
if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        pass
    main()


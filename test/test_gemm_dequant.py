import torch
import flh
import numpy as np


def main():
    
    torch.manual_seed(0)
    
    assert flh._CUDA is not None

    M, N, K = 1, 1024, 2048
    device = "cuda"

    # 生成有符号 4bit 数据
    A_int = torch.randint(-8, 8, (M, K), device=device, dtype=torch.int32)
    B_int = torch.randint(-8, 8, (N, K), device=device, dtype=torch.int32)

    # 打包成 s4（两数一字节），注意 & 0xF
    A_packed = ((A_int[:, 0::2] & 0xF) | ((A_int[:, 1::2] & 0xF) << 4)).to(torch.uint8)
    B_packed = ((B_int[:, 0::2] & 0xF) | ((B_int[:, 1::2] & 0xF) << 4)).to(torch.uint8)

    # scale = 1
    A_scale = torch.randn(M, K // 128, device=device, dtype=torch.float16)
    B_scale = torch.randn(N, K // 128, device=device, dtype=torch.float16)

    # kernel 输出（fp16）
    out = flh._CUDA.gemm_i4_dequant_o16(A_packed, B_packed, A_scale, B_scale)

    # 参考结果（直接用 int32 矩阵乘）
    A_int = A_int.reshape(M, K // 128, 128)
    A_scale = A_scale.reshape(M, K // 128, 1)
    A = A_int * A_scale
    A = A.reshape(M, K)
    B_int = B_int.reshape(N, K // 128, 128)
    B_scale = B_scale.reshape(N, K // 128, 1)
    B = B_int * B_scale
    B = B.reshape(N, K)
    ref = A @ B.t()

    print("out:", out)
    print(torch.isnan(out).any())
    print("ref:", ref)
    print("max_diff:", (out - ref).abs().max())
    print(torch.allclose(out, ref, atol=5))


if __name__ == "__main__":
    main()
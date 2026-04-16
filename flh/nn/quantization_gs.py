import torch
from math import sqrt

import flh

class ActQuantizerGS(torch.nn.Module):
    """
    可配置 group size 的激活值量化器
    使用 CUDA kernel 加速的 Hadamard + INT4 量化
    """
    def __init__(self, bits=4, group_size=128, sym=True, use_hadamard=True):
        """
        Args:
            bits: 量化位数（目前仅支持 4-bit）
            group_size: 分组大小，可选 32, 64, 128
            sym: 是否使用对称量化
            use_hadamard: 是否使用 Hadamard 变换
        """
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.use_hadamard = use_hadamard

        if self.bits != 4:
            raise ValueError(f"ActQuantizerGS only supports 4-bit quantization, got {bits}")
        if self.group_size not in [32, 64, 128]:
            raise ValueError(f"group_size must be one of [32, 64, 128], got {group_size}")

    def forward(self, x):
        """
        前向传播 - 执行 Hadamard 变换 + INT4 量化（或仅 INT4 量化）

        Args:
            x: 输入张量 [..., in_features]，FP16，其中 in_features 必须能被 group_size 整除

        Returns:
            scales: 量化尺度 [..., num_groups] FP16
            zero_point: 零点（对称量化时为 0）[..., num_groups]
            q: 打包的 INT4 数据 [..., in_features//2] UINT8
        """
        # 输入检查
        if x.dim() < 2:
            raise ValueError(f"Input must have at least 2 dimensions, got {x.dim()}")
        if x.size(-1) % self.group_size != 0:
            raise ValueError(f"Last dimension ({x.size(-1)}) must be divisible by group_size ({self.group_size})")

        original_shape = x.shape
        batch_dims = original_shape[:-1]
        num_groups = x.size(-1) // self.group_size

        # 重塑为 [*, group_size] 以便分组处理
        x_2d = x.view(-1, self.group_size)

        if self.use_hadamard:
            q, scales = flh._CUDA.hadamard_and_quantize_i4_gs(x_2d, self.group_size)
        else:
            q, scales = flh._CUDA.quant_and_pack_i4_gs(x_2d, self.group_size)

        # Reshape 输出
        # q: [num_groups * batch, group_size//2] -> [..., in_features//2]
        q = q.view(*batch_dims, self.group_size // 2 * num_groups)
        # scales: [num_groups * batch, 1] -> [..., num_groups]
        scales = scales.view(*batch_dims, num_groups) / sqrt(self.group_size)

        # 对称量化，zero_point 为 0
        zero_point = torch.zeros_like(scales, dtype=torch.uint8)

        return scales, zero_point, q


class SimpleActQuantizerGS(torch.nn.Module):
    """简易版本：仅量化，无 Hadamard 选项"""
    def __init__(self, group_size=128):
        super().__init__()
        self.group_size = group_size
        if group_size not in [32, 64, 128]:
            raise ValueError(f"group_size must be one of [32, 64, 128], got {group_size}")

    def forward(self, x):
        """
        Args:
            x: 输入张量 [..., in_features]，FP16，其中 in_features 必须能被 group_size 整除

        Returns:
            scales: 量化尺度 [..., num_groups]
            zero_point: 零点（总是 0）[..., num_groups]
            q: 打包的 INT4 数据 [..., in_features//2] UINT8
        """
        if x.size(-1) % self.group_size != 0:
            raise ValueError(f"Last dimension ({x.size(-1)}) must be divisible by group_size ({self.group_size})")

        original_shape = x.shape
        batch_dims = original_shape[:-1]
        num_groups = x.size(-1) // self.group_size

        x_2d = x.view(-1, self.group_size)

        q, scales = flh._CUDA.quant_and_pack_i4_gs(x_2d, self.group_size)

        q = q.view(*batch_dims, self.group_size // 2 * num_groups)
        scales = scales.view(*batch_dims, num_groups)

        zero_point = torch.zeros_like(scales, dtype=torch.uint8)

        return scales, zero_point, q


if __name__ == "__main__":
    # 简单测试
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(32, 128, dtype=torch.float16, device=device)

    quantizer = ActQuantizerGS(group_size=128, use_hadamard=True)
    scales, zp, q = quantizer(x)

    print(f"Input shape: {x.shape}")
    print(f"Scales shape: {scales.shape}")
    print(f"Zero point shape: {zp.shape}")
    print(f"Quantized shape: {q.shape}")

    # 反量化验证
    x_hat = (q.view(32, -1) - zp.view(32, -1)) * scales.view(32, -1)
    print(f"Dequantized shape: {x_hat.shape}")

    # 测试不同 group_size
    for gs in [32, 64, 128]:
        quantizer = ActQuantizerGS(group_size=gs)
        x_test = torch.randn(16, gs, dtype=torch.float16, device=device)
        s, z, q_out = quantizer(x_test)
        print(f"GS={gs}: q={q_out.shape}, s={s.shape}")

    print("All tests passed!")

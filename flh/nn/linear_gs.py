import torch
import flh


class LinearFLHGS(torch.nn.Module):
    """
    基于 group size 灵活配置的 FLH 线性层
    使用 CUDA kernel 加速的 INT4 GEMM + 同步反量化
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
        dtype=torch.float16,
        device='cuda'
    ):
        """
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            group_size: 量化分组大小（32, 64, 128）
            bias: 是否使用偏置
            dtype: 权重数据类型（FP16）
            device: 设备
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        if group_size not in [32, 64, 128]:
            raise ValueError(f"group_size must be one of [32, 64, 128], got {group_size}")

        # 验证 in_features 能被 group_size 整除
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size ({group_size})"
            )

        # 4bit 量化：每个 uint8 存储 2 个 4bit 权重
        packed_in_features = in_features // 2
        self.register_buffer(
            'w_packed',
            torch.zeros(out_features, packed_in_features, dtype=torch.uint8, device=device)
        )

        # Group-wise 量化参数
        num_groups = in_features // group_size
        self.register_buffer(
            'w_scale',
            torch.zeros(out_features, num_groups, dtype=dtype, device=device)
        )

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.bias = None

    def forward(self, x, a_scale=None):
        """
        前向传播：INT4 GEMM + 同步反量化

        Args:
            x: 激活值 [batch, in_features//2] UINT8（已打包的4bit量化值）
            a_scale: 激活值的 scale [batch, in_features/group_size] FP16
                     如果为 None，则使用全 1 scale

        Returns:
            output: [batch, out_features] FP16
        """
        batch_size = x.size(0)

        # 处理激活值 scale
        if a_scale is None:
            a_scale = torch.ones(
                batch_size,
                self.in_features // self.group_size,
                dtype=torch.float16,
                device=x.device
            )
        else:
            if a_scale.shape[0] != batch_size:
                raise ValueError(f"a_scale batch dim mismatch: {a_scale.shape[0]} vs {batch_size}")
            if a_scale.shape[1] != self.in_features // self.group_size:
                raise ValueError(
                    f"a_scale group dim mismatch: {a_scale.shape[1]} vs "
                    f"{self.in_features // self.group_size}"
                )

        # 调用 CUDA kernel
        output = flh._CUDA.gemm_i4_dequant_o16_gs(
            x,                                          # [batch, in_features//2] UINT8 packed
            self.w_packed,                              # [out_features, in_features//2]
            a_scale,                                    # [batch, num_groups] FP16
            self.w_scale,                               # [out_features, num_groups] FP16
            self.group_size
        )

        if self.bias is not None:
            output = output + self.bias

        return output

    @staticmethod
    def from_float(
        module: torch.nn.Linear,
        weight_group_size: int = 128,
        bias: bool = None
    ):
        """
        从 FP16 Linear 层转换

        Args:
            module: torch.nn.Linear 层
            weight_group_size: 权重分组大小
            bias: 是否保留偏置（默认与 module 相同）

        Returns:
            LinearFLHGS 层
        """
        if bias is None:
            bias = module.bias is not None

        device = module.weight.device
        dtype = module.weight.dtype

        linear_gs = LinearFLHGS(
            in_features=module.in_features,
            out_features=module.out_features,
            group_size=weight_group_size,
            bias=bias,
            dtype=dtype,
            device=device
        )

        W = module.weight.data.clone()

        # 量化权重
        from . import quantization as _quant
        weight_quantizer = _quant.WeightQuantizer(
            bits=4,
            group_size=weight_group_size,
            sym=True,
            channel_wise=(weight_group_size == -1),
            use_hadamard=False,  # 权重不应用 Hadamard
            clip_ratio=1.0
        )

        weight_quantizer.calibrate(W)
        scale, zp, w_int = weight_quantizer.quantize(W)

        # 调整 scale 形状
        if weight_group_size > 0 and scale.dim() == 3:
            scale = scale.squeeze(-1)
            zp = zp.squeeze(-1) if zp is not None else torch.zeros_like(scale, dtype=torch.int32)
        elif scale.dim() == 2:
            pass
        else:
            scale = scale.expand(linear_gs.out_features, 1)
            zp = zp.expand(linear_gs.out_features, 1) if zp is not None else torch.zeros(
                linear_gs.out_features, 1, dtype=torch.uint8, device=device
            )

        # 打包权重
        linear_gs._pack_weights(w_int, scale, zp)

        if bias and module.bias is not None:
            linear_gs.bias.copy_(module.bias.data)

        return linear_gs

    def _pack_weights(self, w_int, scale, zero):
        """
        打包量化权重

        Args:
            w_int: int8 权重 [out_features, in_features]
            scale: scale [out_features, num_groups]
            zero: zero point [out_features, num_groups]
        """
        out_features, in_features = w_int.shape

        # 存储量化参数
        self.w_scale.copy_(scale)

        # 权重限制在 [-8, 7] 并转换为 int32
        w_clamped = torch.clamp(w_int.flatten(), -8, 7).int()

        # 转换为无符号 4bit (0-15)
        w_unsigned = torch.where(w_clamped < 0, w_clamped + 16, w_clamped)

        # 打包
        total_elements = w_unsigned.size(0)
        if total_elements % 2 != 0:
            w_unsigned = torch.cat([w_unsigned, torch.zeros(1, dtype=w_unsigned.dtype, device=w_unsigned.device)])

        w_reshaped = w_unsigned.view(-1, 2)
        w_packed_vals = w_reshaped[:, 0] + (w_reshaped[:, 1] << 4)

        packed_size = self.w_packed.numel()
        if w_packed_vals.size(0) <= packed_size:
            self.w_packed.view(-1)[:w_packed_vals.size(0)].copy_(w_packed_vals.to(torch.uint8))
        else:
            self.w_packed.view(-1).copy_(w_packed_vals[:packed_size].to(torch.uint8))

    def get_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """
        解包权重为 FP16/FP32

        Args:
            dtype: 目标数据类型

        Returns:
            解包后的权重张量 [out_features, in_features]
        """
        out_features, in_features = self.out_features, self.in_features
        device = self.w_packed.device

        # 解包 4bit -> int8
        total_elements = out_features * in_features
        element_indices = torch.arange(total_elements, dtype=torch.int32, device=device)
        packed_indices = element_indices // 2
        bit_offsets = (element_indices % 2) * 4

        w_4bit = torch.zeros(total_elements, dtype=torch.int32, device=device)
        valid_mask = packed_indices < self.w_packed.numel()

        if valid_mask.any():
            packed_vals = self.w_packed.view(-1)[packed_indices[valid_mask]].int()
            bit_offs = bit_offsets[valid_mask]
            extracted_4bit = (packed_vals >> bit_offs) & 0xF
            w_4bit[valid_mask] = extracted_4bit

        # 转换为有符号 int8
        w_int = torch.where(w_4bit >= 8, w_4bit - 16, w_4bit).view(out_features, in_features).to(torch.int8)

        # Group-wise 反量化
        num_groups = in_features // self.group_size
        w_float = torch.zeros(out_features, in_features, dtype=self.w_scale.dtype, device=device)

        for g in range(num_groups):
            start_col = g * self.group_size
            end_col = start_col + self.group_size
            w_group = w_int[:, start_col:end_col].float()
            scale = self.w_scale[:, g:g+1]
            w_float[:, start_col:end_col] = w_group * scale

        if dtype is not None:
            w_float = w_float.to(dtype)

        return w_float


def linear_gs_demo():
    """演示如何使用 LinearFLHGS"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建随机输入和权重
    batch_size = 8
    in_features = 512
    out_features = 256
    group_size = 64

    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

    # 创建标准 Linear 层
    linear_fp16 = torch.nn.Linear(in_features, out_features, dtype=torch.float16, device=device)

    # 转换为 FLH-GS 版本
    linear_gs = LinearFLHGS.from_float(linear_fp16, weight_group_size=group_size)

    # 前向传播
    with torch.no_grad():
        y_fp16 = linear_fp16(x)
        y_gs = linear_gs(x)

    print(f"FP16 output: mean={y_fp16.mean():.6f}, std={y_fp16.std():.6f}")
    print(f"GS output:   mean={y_gs.mean():.6f}, std={y_gs.std():.6f}")
    print(f"Relative error: {(y_gs - y_fp16).abs().mean() / y_fp16.abs().mean():.6f}")


if __name__ == "__main__":
    linear_gs_demo()

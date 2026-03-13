import torch
import flh

from . import quantization as _quant


class LinearFLH(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16, device='cpu',
                 dual_hadamard=False, in_group_size=None, out_group_size=None, no_hadamard=False, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dual_hadamard = dual_hadamard
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.weight_group_size = group_size
        self.no_hadamard = no_hadamard
        
        # 4bit 量化：每个 int32 存储 8 个 4bit 权重
        packed_size = (out_features * in_features + 7) // 8
        self.register_buffer('w_packed', torch.zeros(packed_size, dtype=torch.int32, device=device))
        
        # Group-wise 量化参数
        if group_size > 0:
            num_groups = (in_features + group_size - 1) // group_size
            self.register_buffer('w_scale', torch.zeros(out_features, num_groups, dtype=dtype, device=device))
            self.register_buffer('w_zero', torch.zeros(out_features, num_groups, dtype=torch.int32, device=device))
        else:
            self.register_buffer('w_scale', torch.zeros(out_features, 1, dtype=dtype, device=device))
            self.register_buffer('w_zero', torch.zeros(out_features, 1, dtype=torch.int32, device=device))
        
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype, device=device))
        else:
            self.bias = None
    
    def _unpack_weights(self):
        """解包 4bit 权重到 float (GPU 加速版本)"""
        out_features, in_features = self.out_features, self.in_features
        device = self.w_packed.device
        
        # 使用 GPU 向量化操作解包 4bit 权重
        total_elements = out_features * in_features
        packed_size = self.w_packed.size(0)
        
        # 创建索引张量
        element_indices = torch.arange(total_elements, dtype=torch.int32, device=device)
        packed_indices = element_indices // 8
        bit_offsets = (element_indices % 8) * 4
        
        # 向量化提取 4bit 值
        w_4bit = torch.zeros(total_elements, dtype=torch.int32, device=device)
        valid_mask = packed_indices < packed_size
        
        if valid_mask.any():
            packed_vals = self.w_packed[packed_indices[valid_mask]]
            bit_offs = bit_offsets[valid_mask]
            extracted_4bit = (packed_vals >> bit_offs) & 0xF
            w_4bit[valid_mask] = extracted_4bit.int()
        
        # 转换为有符号 4bit (-8 到 7)
        w_4bit = torch.where(w_4bit >= 8, w_4bit - 16, w_4bit)
        
        # Reshape 到权重矩阵形状
        w_int = w_4bit.view(out_features, in_features).to(torch.int8)
        
        # 反量化
        if self.weight_group_size > 0:
            # Group-wise 反量化 (向量化版本)
            num_groups = (in_features + self.weight_group_size - 1) // self.weight_group_size
            w_float = torch.zeros(out_features, in_features, dtype=self.w_scale.dtype, device=device)
            
            for g in range(num_groups):
                start_col = g * self.weight_group_size
                end_col = min(start_col + self.weight_group_size, in_features)
                
                w_group = w_int[:, start_col:end_col].float()
                scale = self.w_scale[:, g:g+1]
                zero = self.w_zero[:, g:g+1].float()
                
                w_float[:, start_col:end_col] = (w_group - zero) * scale
        else:
            # Per-channel 反量化
            w_float = (w_int.float() - self.w_zero.float()) * self.w_scale
            
        return w_float
    
    def _pack_weights(self, w_int, scale, zero):
        """打包量化权重到 4bit (GPU 加速版本)"""
        out_features, in_features = w_int.shape
        device = w_int.device
        
        # 存储量化参数
        self.w_scale.copy_(scale)
        self.w_zero.copy_(zero.int())
        
        # 将权重限制在 4bit 有符号范围内 (-8 到 7)
        w_clamped = torch.clamp(w_int.flatten(), -8, 7).int()
        
        # 转换为无符号 4bit (0 到 15)
        w_unsigned = torch.where(w_clamped < 0, w_clamped + 16, w_clamped)
        
        # 计算需要的 packed 数量
        total_elements = w_unsigned.size(0)
        packed_size = (total_elements + 7) // 8
        
        # 填充到 8 的倍数
        if total_elements % 8 != 0:
            padding = 8 - (total_elements % 8)
            w_unsigned = torch.cat([w_unsigned, torch.zeros(padding, dtype=w_unsigned.dtype, device=device)])
        
        # Reshape 为 (packed_size, 8)
        w_reshaped = w_unsigned.view(-1, 8)
        
        # 向量化打包：每 8 个 4bit 值打包成一个 int32
        bit_shifts = torch.arange(8, dtype=torch.int32, device=device) * 4  # [0, 4, 8, 12, 16, 20, 24, 28]
        shift_powers = torch.pow(2, bit_shifts).int()  # [1, 16, 256, 4096, ...]
        w_shifted = w_reshaped.int() * shift_powers.unsqueeze(0)  # 广播乘法
        w_packed_vals = w_shifted.sum(dim=1)  # 按位或运算的等价操作
        
        # 更新 packed 权重
        self.w_packed.zero_()
        if w_packed_vals.size(0) <= self.w_packed.size(0):
            self.w_packed[:w_packed_vals.size(0)] = w_packed_vals
        else:
            self.w_packed.copy_(w_packed_vals[:self.w_packed.size(0)])
    
    def forward(self, x, a_scale=None, a_zero=None):
        if a_scale is not None:
            zp_a = a_zero if a_zero is not None else 0
            if a_scale.dim() == x.dim() + 1 and a_scale.size(-1) == 1:
                group_size = self.in_group_size
                n_groups = x.size(-1) // group_size
                x_view = x.view(*x.shape[:-1], n_groups, group_size)
                x = ((x_view - zp_a) * a_scale).view_as(x)
            else:
                x = (x - zp_a) * a_scale
        
        # 解包并反量化权重
        weight = self._unpack_weights()
            
        return torch.nn.functional.linear(x, weight.to(x.dtype), self.bias)
    
    def get_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        weight = self._unpack_weights()
        if dtype is not None:
            weight = weight.to(dtype)
        return weight
    
    def set_weight(self, weight: torch.Tensor):
        """直接设置浮点权重（用于 FP16 模型）"""
        # 对于 FP16 模型，我们需要量化后再存储
        weight_quantizer = _quant.WeightQuantizer(
            bits=4,
            group_size=self.weight_group_size,
            sym=True,
            channel_wise=(self.weight_group_size == -1),
            use_hadamard=False,
            clip_ratio=1.0
        )
        
        weight_quantizer.calibrate(weight)
        scale, zero, w_int = weight_quantizer.quantize(weight)
        
        # 调整 scale 和 zero 的形状以匹配我们的存储格式
        if self.weight_group_size > 0 and scale.dim() == 3:
            # Group-wise: (out_features, num_groups, 1) -> (out_features, num_groups)
            scale = scale.squeeze(-1)
            zero = zero.squeeze(-1) if zero is not None else torch.zeros_like(scale, dtype=torch.int32)
        elif scale.dim() == 2:
            # Per-channel: (out_features, 1) -> (out_features, 1)
            pass
        else:
            # Per-tensor: expand to (out_features, 1)
            scale = scale.expand(self.out_features, 1)
            zero = zero.expand(self.out_features, 1) if zero is not None else torch.zeros(self.out_features, 1, dtype=torch.int32, device=weight.device)
        
        self._pack_weights(w_int, scale, zero)
    
    @staticmethod
    def from_float(module: torch.nn.Linear, weight_bits=4, weight_group_size=128, weight_sym=True,
                   dual_hadamard=False, in_group_size=None, out_group_size=None, clip_ratio=1.0, no_hadamard=False):
        in_features = module.in_features
        out_features = module.out_features
        bias_flag = module.bias is not None
        dtype = module.weight.dtype
        device = module.weight.device

        flh_linear = LinearFLH(
            in_features, 
            out_features, 
            bias=bias_flag, 
            dtype=dtype,
            device=device,
            dual_hadamard=dual_hadamard,
            in_group_size=in_group_size,
            out_group_size=out_group_size,
            group_size=weight_group_size,
        )
        
        W = module.weight.data.clone()
        if bias_flag:
            flh_linear.bias.copy_(module.bias.data)
        
        # 应用 Hadamard 变换到权重和偏置
        if not no_hadamard:
            if dual_hadamard:
                W_temp = _quant.fast_hadamard_transform(
                    W,
                    group_size=in_group_size,
                    normalize=True,
                )
                W = _quant.fast_hadamard_transform(
                    W_temp.T,
                    group_size=out_group_size,
                    normalize=True,
                ).T
                if bias_flag:
                    b_dual = _quant.fast_hadamard_transform(
                        flh_linear.bias.unsqueeze(0),
                        group_size=out_group_size,
                        normalize=True,
                    ).squeeze(0)
                    flh_linear.bias.copy_(b_dual)
            else:
                W = _quant.fast_hadamard_transform(
                    W,
                    group_size=in_group_size,
                    normalize=True,
                )
        else:
            W = W
            if bias_flag:
                flh_linear.bias.copy_(module.bias.data)
        
        if torch.isnan(W).any():
            print(f"WARNING: NaN detected in weights after Hadamard for layer {out_features}x{in_features}")
            return flh_linear
        
        weight_quantizer = _quant.WeightQuantizer(
            bits=weight_bits,
            group_size=weight_group_size,
            sym=weight_sym,
            channel_wise=(weight_group_size == -1),
            use_hadamard=False,
            clip_ratio=clip_ratio
        )
        
        weight_quantizer.calibrate(W)
        
        if weight_bits < 16 and torch.isnan(weight_quantizer.scale).any():
            print(f"WARNING: NaN detected in quantization scale for layer {out_features}x{in_features}")
            print(f"  Weight stats: min={W.min():.6f}, max={W.max():.6f}, mean={W.mean():.6f}")
            return flh_linear
        
        scale, zp, w_int = weight_quantizer.quantize(W)
        
        # 调整 scale 和 zero 的形状以匹配我们的存储格式
        if weight_group_size > 0 and scale.dim() == 3:
            # Group-wise: (out_features, num_groups, 1) -> (out_features, num_groups)
            scale = scale.squeeze(-1)
            zp = zp.squeeze(-1) if zp is not None else torch.zeros_like(scale, dtype=torch.int32)
        elif scale.dim() == 2:
            # Per-channel: (out_features, 1) -> (out_features, 1)
            pass
        else:
            # Per-tensor: expand to (out_features, 1)
            scale = scale.expand(out_features, 1)
            zp = zp.expand(out_features, 1) if zp is not None else torch.zeros(out_features, 1, dtype=torch.int32, device=W.device)
        
        flh_linear._pack_weights(w_int, scale, zp)

        return flh_linear
    

if __name__ == "__main__":
    layer = torch.nn.Linear(1024, 1024)
    
    x = torch.randn(1024, 1024)
    y_ref = layer(x)
    
    layer_flh = LinearFLH.from_float(layer, weight_bits=4, weight_group_size=128, weight_sym=True)
    
    print("Original weight shape:", layer.weight.shape)
    print("Packed weight shape:", layer_flh.w_packed.shape)
    print("Scale shape:", layer_flh.w_scale.shape)
    print("Zero shape:", layer_flh.w_zero.shape)
    
    scale, zp, q = _quant.ActQuantizer(bits=15, group_size=-1, sym=True)(x)
    x_flh = q if scale is None else (q - (zp if zp is not None else 0)) * scale
    
    y_flh = layer_flh(x_flh)
    
    print("Output shapes match:", y_ref.shape == y_flh.shape)
    print("Results close:", torch.allclose(y_ref, y_flh, atol=1e-1)) 
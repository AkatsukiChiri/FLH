import torch
import math

# 缓存小Hadamard矩阵
_HADAMARD_CACHE = {}

def get_hadK(n: int, transpose=False):
    """
    获取小Hadamard矩阵用于加速
    对于小的n（通常K=1），直接返回None使用蝶形算法
    对于较大的n，预计算Hadamard矩阵
    
    :param n: 矩阵大小（必须是2的幂）
    :param transpose: 是否转置
    :return: (hadamard_matrix, K) 或 (None, n)
    """
    K = 1  # 使用蝶形算法处理到最小尺寸
    
    if K == 1:
        return None, K
    
    # 生成小Hadamard矩阵（如果需要）
    if n not in _HADAMARD_CACHE:
        def hadamard(k):
            if k == 1:
                return torch.ones(1, 1)
            else:
                H = hadamard(k // 2)
                return torch.cat([torch.cat([H, H], dim=1), 
                                 torch.cat([H, -H], dim=1)], dim=0)
        _HADAMARD_CACHE[n] = hadamard(n).float()
    
    hadK = _HADAMARD_CACHE[n]
    if transpose:
        hadK = hadK.t()
    
    return hadK, K


def had_transform(X: torch.Tensor, transpose=False):
    """
    全局Hadamard变换（用于非分组情况）
    
    :param X: 输入张量
    :param transpose: 是否转置（Hadamard矩阵是对称的，所以这个参数实际上不影响结果）
    :return: 变换后的张量
    """
    n = X.shape[-1]
    if (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2 for Hadamard transform, got {n}")
    
    hadK, K = get_hadK(n, transpose)
    
    input_shape = X.shape
    input = X.clone().to(torch.float64).view(-1, n, 1)
    output = input.clone()
    
    # 蝶形变换
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1]//2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        input, output = output, input
    
    del output
    
    if hadK is not None and K > 1:
        input = hadK.view(1, K, K).to(input) @ input
    
    return input.view(input_shape) / math.sqrt(float(n))


def had_transform_group(X: torch.Tensor, transpose=False, group_size: int = 128):
    """
    分组Hadamard变换（高效实现）
    
    :param X: 输入张量，shape为(..., n)
    :param transpose: 是否转置（Hadamard矩阵是对称的）
    :param group_size: 分组大小（必须是2的幂）
    :return: 变换后的张量
    """
    n = X.shape[-1]
    if n % group_size != 0:
        return had_transform(X, transpose)
    
    n = group_size
    hadK, K = get_hadK(n, transpose)
    group_num = X.shape[-1] // n
    input = X.clone().to(torch.float64).view(-1, group_num, n, 1)
    input = input.transpose(0, 1)
    output = input.clone()
    
    # 蝶形变换
    while input.shape[2] > K:
        input = input.view(group_num, input.shape[1], input.shape[2]//2, 2, input.shape[3])
        output = output.view(input.shape)
        output[:, :, :, 0, :] = input[:, :, :, 0, :] + input[:, :, :, 1, :]
        output[:, :, :, 1, :] = input[:, :, :, 0, :] - input[:, :, :, 1, :]
        output = output.view(group_num, input.shape[1], input.shape[2], -1)
        input, output = output, input
    
    del output
    
    if hadK is not None and K > 1:
        input = hadK.view(1, K, K).to(input) @ input
    
    input = input.transpose(0, 1)
    
    return input.view(X.shape) / math.sqrt(float(n))


def fast_hadamard_transform(x, group_size=None, normalize=True):
    """
    快速Walsh-Hadamard变换统一接口
    
    :param x: 输入张量，shape为(..., n)
    :param group_size: 分组大小（如果指定则使用分组变换）
    :param normalize: 是否归一化（已在内部实现，此参数保持兼容性）
    :return: 变换后的张量
    """
    orig_dtype = x.dtype
    x_f64 = x.to(torch.float64)
    if group_size is not None and group_size > 0:
        out = had_transform_group(x_f64, transpose=False, group_size=group_size)
    else:
        out = had_transform(x_f64, transpose=False)
    if orig_dtype == torch.float64:
        return out
    return out.to(orig_dtype)



class ActQuantizer(torch.nn.Module):
    def __init__(self, bits=8, group_size=-1, sym=True, input_clip_ratio=1.0, use_hadamard=True, packed_output=False):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.input_clip_ratio = input_clip_ratio
        self.use_hadamard = use_hadamard
        self.packed_output = packed_output
        
        if self.packed_output and self.bits != 4:
            raise ValueError("Packed output is only supported for 4-bit quantization")
        
    def forward(self, x):
        # 根据 use_hadamard 参数决定是否进行 Hadamard 变换
        if self.use_hadamard:
            x_transformed = fast_hadamard_transform(x, group_size=self.group_size, normalize=True)
        else:
            x_transformed = x
        
        if self.bits >= 16:
            return None, None, x_transformed
        
        if self.input_clip_ratio < 1.0:
            max_abs = x_transformed.abs().amax(dim=-1, keepdim=True)
            clip_val = max_abs * self.input_clip_ratio
            x_clipped = torch.clamp(x_transformed, -clip_val, clip_val)
        else:
            x_clipped = x_transformed
        orig_shape = x_clipped.shape

        if self.group_size is not None and self.group_size > 0 and x_clipped.size(-1) % self.group_size == 0:
            n_groups = x_clipped.size(-1) // self.group_size
            new_shape = x_clipped.shape[:-1] + (n_groups, self.group_size)
            x_grouped = x_clipped.view(new_shape)
            if self.sym:
                max_val = x_grouped.abs().amax(dim=-1, keepdim=True)
                qmin = -2 ** (self.bits - 1)
                qmax = 2 ** (self.bits - 1) - 1
                
                scale = max_val / qmax
                scale = torch.clamp(scale, min=1e-8)
                
                x_int = (x_grouped / scale).round().clamp(qmin, qmax)
                
                scale = scale.view(x_clipped.shape[:-1] + (n_groups,))
                zero_point = torch.zeros_like(scale)
            else:
                max_val = x_grouped.amax(dim=-1, keepdim=True)
                min_val = x_grouped.amin(dim=-1, keepdim=True)
                qmin = 0
                qmax = 2 ** self.bits - 1
                
                scale = (max_val - min_val) / (qmax - qmin + 1e-8)
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                zero_point = -min_val / scale
                
                x_int = ((x_grouped - min_val) / scale).round().clamp(qmin, qmax)
                scale = scale.view(x_clipped.shape[:-1] + (n_groups,))
                zero_point = zero_point.view(x_clipped.shape[:-1] + (n_groups,))
                
            x_int = x_int.view(orig_shape).to(x_clipped.dtype)
        else:
            if self.sym:
                max_val = x_clipped.abs().amax(dim=-1, keepdim=True)
                qmin = -2 ** (self.bits - 1)
                qmax = 2 ** (self.bits - 1) - 1
                
                scale = max_val / qmax
                scale = torch.clamp(scale, min=1e-8)
                zero_point = torch.zeros_like(scale)
                
                x_int = (x_clipped / scale).round().clamp(qmin, qmax)
            else:
                max_val = x_clipped.amax(dim=-1, keepdim=True)
                min_val = x_clipped.amin(dim=-1, keepdim=True)
                qmin = 0
                qmax = 2 ** self.bits - 1
                
                scale = (max_val - min_val) / (qmax - qmin + 1e-8)
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                zero_point = -min_val / scale
                
                x_int = ((x_clipped - min_val) / scale).round().clamp(qmin, qmax)
            x_int = x_int.to(x_clipped.dtype)
        
        # 如果需要 packed 输出，将 4bit 数据打包到 uint8
        if self.packed_output:
            x_packed = self._pack_4bit_to_uint8(x_int)
            return scale, zero_point, x_packed
        else:
            return scale, zero_point, x_int
    
    def _pack_4bit_to_uint8(self, x_int):
        """将 4bit 量化值打包到 uint8 (每个 uint8 存储 2 个 4bit 值)"""
        device = x_int.device
        orig_shape = x_int.shape
        
        # 展平
        x_flat = x_int.flatten().to(torch.int8)
        
        if self.sym:
            # 对称量化：限制在 4bit 有符号范围内 (-8 到 7)
            x_4bit = torch.clamp(x_flat, -8, 7)
        else:
            # 非对称量化：限制在 4bit 无符号范围内 (0 到 15)
            x_4bit = torch.clamp(x_flat, 0, 15)
        
        # 填充到 2 的倍数
        total_elements = x_4bit.size(0)
        if total_elements % 2 != 0:
            x_4bit = torch.cat([x_4bit, torch.zeros(1, dtype=x_4bit.dtype, device=device)])
        
        # Reshape 为 (packed_size, 2)
        x_reshaped = x_4bit.view(-1, 2)
        
        # 向量化打包：每 2 个 4bit 值打包成一个 uint8
        # 低4位存储第一个值，高4位存储第二个值
        x_packed_vals = (x_reshaped[:, 0] & 0x0F) | ((x_reshaped[:, 1] & 0x0F) << 4)
        
        # 计算 packed 后的形状
        packed_size = x_packed_vals.size(0)
        if len(orig_shape) == 1:
            # 1D 输入
            return x_packed_vals
        else:
            # 多维输入，计算正确的 packed 维度
            last_dim = orig_shape[-1]
            packed_last_dim = (last_dim + 1) // 2  # 每 2 个元素打包成 1 个
            new_shape = orig_shape[:-1] + (packed_last_dim,)
            
            # 确保 packed_vals 的大小与目标形状匹配
            target_size = torch.tensor(new_shape).prod().item()
            if packed_size != target_size:
                # 如果大小不匹配，截取或填充
                if packed_size > target_size:
                    x_packed_vals = x_packed_vals[:target_size]
                else:
                    padding = target_size - packed_size
                    x_packed_vals = torch.cat([x_packed_vals, torch.zeros(padding, dtype=x_packed_vals.dtype, device=x_packed_vals.device)])
            
            return x_packed_vals.view(new_shape).to(torch.uint8)


class WeightQuantizer(torch.nn.Module):
    def __init__(self, bits=4, group_size=-1, sym=True, channel_wise=True, use_hadamard=True, clip_ratio=1.0):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.channel_wise = channel_wise
        self.use_hadamard = use_hadamard
        self.clip_ratio = clip_ratio
        
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        self.calibrated = False
        
    def calibrate(self, weight):
        if self.bits >= 16:
            self.calibrated = True
            return
        
        # 根据 use_hadamard 参数决定是否应用 Hadamard 变换
        if self.use_hadamard:
            weight_transformed = fast_hadamard_transform(weight, group_size=self.group_size, normalize=True)
        else:
            weight_transformed = weight
        weight_transformed = self._clip_weight(weight_transformed)
        
        with torch.no_grad():
            if self.group_size > 0:
                self._calibrate_group_wise(weight_transformed)
            elif self.channel_wise:
                self._calibrate_channel_wise(weight_transformed)
            else:
                self._calibrate_tensor_wise(weight_transformed)
        
        self.calibrated = True
    
    def _calibrate_tensor_wise(self, weight):
        if self.sym:
            max_val = weight.abs().max()
            qmax = 2 ** (self.bits - 1) - 1
            qmin = -2 ** (self.bits - 1)
            
            scale = max_val / qmax
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.tensor(0.0, device=weight.device, dtype=weight.dtype)
        else:
            max_val = weight.max()
            min_val = weight.min()
            qmin = 0
            qmax = 2 ** self.bits - 1
            
            scale = (max_val - min_val) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = qmin - min_val / scale
        
        self.scale = scale.view(1)
        self.zero_point = zero_point.view(1)
    
    def _calibrate_channel_wise(self, weight):
        if self.sym:
            max_val = weight.abs().amax(dim=1, keepdim=True)
            qmax = 2 ** (self.bits - 1) - 1
            qmin = -2 ** (self.bits - 1)
            
            scale = max_val / qmax
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.zeros_like(scale)
        else:
            max_val = weight.amax(dim=1, keepdim=True)
            min_val = weight.amin(dim=1, keepdim=True)
            qmin = 0
            qmax = 2 ** self.bits - 1
            
            scale = (max_val - min_val) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = qmin - min_val / scale
        
        self.scale = scale.view(out_features, 1)
        self.zero_point = zero_point.view(out_features, 1)
    
    def _calibrate_group_wise(self, weight):
        out_features, in_features = weight.shape
        
        if in_features % self.group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"group_size ({self.group_size})"
            )
        
        num_groups = in_features // self.group_size
        
        weight_grouped = weight.view(out_features, num_groups, self.group_size)
        
        if self.sym:
            max_val = weight_grouped.abs().amax(dim=2, keepdim=True)
            qmax = 2 ** (self.bits - 1) - 1
            qmin = -2 ** (self.bits - 1)
            
            scale = max_val / qmax
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.zeros_like(scale)
        else:
            max_val = weight_grouped.amax(dim=2, keepdim=True)
            min_val = weight_grouped.amin(dim=2, keepdim=True)
            qmin = 0
            qmax = 2 ** self.bits - 1
            
            scale = (max_val - min_val) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = qmin - min_val / scale
        
        self.scale = scale.view(out_features, num_groups)
        self.zero_point = zero_point.view(out_features, num_groups)
    
    def _clip_weight(self, weight):
        if self.clip_ratio >= 1.0:
            return weight
        if self.group_size > 0 and weight.shape[1] % self.group_size == 0:
            num_groups = weight.shape[1] // self.group_size
            wg = weight.view(weight.shape[0], num_groups, self.group_size)
            max_abs = wg.abs().amax(dim=2, keepdim=True)
            clip_val = max_abs * self.clip_ratio
            wg = torch.clamp(wg, -clip_val, clip_val)
            return wg.view(weight.shape)
        if self.channel_wise:
            max_abs = weight.abs().amax(dim=1, keepdim=True)
            clip_val = max_abs * self.clip_ratio
            return torch.clamp(weight, -clip_val, clip_val)
        max_abs = weight.abs().max()
        clip_val = max_abs * self.clip_ratio
        return torch.clamp(weight, -clip_val, clip_val)
    
    def quantize(self, weight):
        # 根据 use_hadamard 参数决定是否应用 Hadamard 变换
        if self.use_hadamard:
            weight_transformed = fast_hadamard_transform(weight, group_size=self.group_size, normalize=True)
        else:
            weight_transformed = weight
        weight_transformed = self._clip_weight(weight_transformed)
        
        if self.bits >= 16:
            return None, None, weight_transformed
        
        if not self.calibrated:
            raise RuntimeError("WeightQuantizer must be calibrated before quantization")
        
        if self.sym:
            qmin = -2 ** (self.bits - 1)
            qmax = 2 ** (self.bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** self.bits - 1
        
        if self.group_size > 0:
            w_int = self._quantize_group_wise(weight_transformed, qmin, qmax)
        elif self.channel_wise:
            w_int = self._quantize_channel_wise(weight_transformed, qmin, qmax)
        else:
            w_int = self._quantize_tensor_wise(weight_transformed, qmin, qmax)
        
        return self.scale, self.zero_point, w_int
    
    def _quantize_tensor_wise(self, weight, qmin, qmax):
        if self.sym:
            w_int = (weight / self.scale).round().clamp(qmin, qmax)
        else:
            w_int = (weight / self.scale + self.zero_point).round().clamp(qmin, qmax)
        return w_int.to(weight.dtype)
    
    def _quantize_channel_wise(self, weight, qmin, qmax):
        if self.sym:
            w_int = (weight / self.scale).round().clamp(qmin, qmax)
        else:
            w_int = (weight / self.scale + self.zero_point).round().clamp(qmin, qmax)
        return w_int.to(weight.dtype)
    
    def _quantize_group_wise(self, weight, qmin, qmax):
        out_features, in_features = weight.shape
        num_groups = in_features // self.group_size
        weight_grouped = weight.view(out_features, num_groups, self.group_size)
        if self.sym:
            w_int = (weight_grouped / self.scale.view(out_features, num_groups, 1)).round().clamp(qmin, qmax)
        else:
            w_int = (weight_grouped / self.scale.view(out_features, num_groups, 1) + self.zero_point.view(out_features, num_groups, 1)).round().clamp(qmin, qmax)
        return w_int.view(out_features, in_features).to(weight.dtype)
    
    def forward(self, weight):
        if not self.calibrated:
            self.calibrate(weight)
        
        return self.quantize(weight)


Quantizer = ActQuantizer

if __name__ == '__main__':
    A = torch.randn(1024, 1024)
    A_had = fast_hadamard_transform(A, group_size=128)
    A_had_had = fast_hadamard_transform(A_had, group_size=128)
    print(A_had_had)
    print(A)
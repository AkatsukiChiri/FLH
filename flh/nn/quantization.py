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
    input = X.view(-1, n, 1)
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
    
    return input.view(input_shape) / torch.tensor(n).sqrt()


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
    input = X.clone().view(-1, group_num, n, 1)
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
    
    return input.view(X.shape) / torch.tensor(n).sqrt()


def fast_hadamard_transform(x, group_size=None, normalize=True):
    """
    快速Walsh-Hadamard变换统一接口
    
    :param x: 输入张量，shape为(..., n)
    :param group_size: 分组大小（如果指定则使用分组变换）
    :param normalize: 是否归一化（已在内部实现，此参数保持兼容性）
    :return: 变换后的张量
    """
    if group_size is not None and group_size > 0:
        return had_transform_group(x, transpose=False, group_size=group_size)
    else:
        return had_transform(x, transpose=False)



class ActQuantizer(torch.nn.Module):
    """
    Activation Quantizer for dynamic quantization of activations.
    
    This quantizer is designed for runtime activation quantization where
    quantization parameters are computed on-the-fly based on input statistics.
    """
    def __init__(self, bits=8, group_size=-1, sym=True, input_clip_ratio=1.0):
        """
        Args:
            bits: int, number of quantization bits (1~16)
            group_size: int, number of channels per group (if -1, disable group quant)
            sym: bool, whether to use symmetric quantization
            input_clip_ratio: float, clipping ratio for input activation
        """
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.input_clip_ratio = input_clip_ratio
        
    def forward(self, x):
        # 应用快速Hadamard变换（使用蝶形算法，无需矩阵）
        # 优势：O(n log n) 速度 + 无内存开销 + 无GPU传输
        x_transformed = fast_hadamard_transform(x, group_size=self.group_size, normalize=True)
        
        if self.bits >= 16:
            # 不量化时，直接返回Hadamard变换后的结果
            # 注意：权重也会做相同的Hadamard变换，所以 (X@H) @ (W@H)^T = X @ W^T
            return x_transformed
        
        # 基于实际分布进行clipping，而不是固定值
        # 使用percentile或者max_abs来确定clipping范围
        if self.input_clip_ratio < 1.0:
            # 如果设置了clip_ratio < 1.0，使用基于实际分布的clipping
            max_abs = x_transformed.abs().amax(dim=-1, keepdim=True)
            clip_val = max_abs * self.input_clip_ratio
            x_clipped = torch.clamp(x_transformed, -clip_val, clip_val)
        else:
            # clip_ratio >= 1.0 时不进行clipping
            x_clipped = x_transformed
        orig_shape = x_clipped.shape

        # For last-dim (usually channel/out feature dim)
        if self.group_size is not None and self.group_size > 0 and x_clipped.size(-1) % self.group_size == 0:
            # Reshape to (..., n_groups, group_size)
            n_groups = x_clipped.size(-1) // self.group_size
            new_shape = x_clipped.shape[:-1] + (n_groups, self.group_size)
            x_grouped = x_clipped.view(new_shape)

            if self.sym:
                # 对称量化：使用有符号范围 [-2^(bits-1), 2^(bits-1)-1]
                max_val = x_grouped.abs().amax(dim=-1, keepdim=True)
                qmin = -2 ** (self.bits - 1)
                qmax = 2 ** (self.bits - 1) - 1
                
                scale = max_val / qmax
                scale = torch.clamp(scale, min=1e-8)
                
                x_int = (x_grouped / scale).round().clamp(qmin, qmax)
                x_deq = x_int * scale
            else:
                # 非对称量化：使用无符号范围 [0, 2^bits-1]
                max_val = x_grouped.amax(dim=-1, keepdim=True)
                min_val = x_grouped.amin(dim=-1, keepdim=True)
                qmin = 0
                qmax = 2 ** self.bits - 1
                
                scale = (max_val - min_val) / (qmax - qmin + 1e-8)
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                zero_point = -min_val / scale
                
                x_int = ((x_grouped - min_val) / scale).round().clamp(qmin, qmax)
                x_deq = (x_int * scale) + min_val
            x_deq = x_deq.view(orig_shape)
        else:
            # Per-tensor or per-channel quantization
            if self.sym:
                # 对称量化：使用有符号范围 [-2^(bits-1), 2^(bits-1)-1]
                max_val = x_clipped.abs().amax(dim=-1, keepdim=True)
                qmin = -2 ** (self.bits - 1)
                qmax = 2 ** (self.bits - 1) - 1
                
                scale = max_val / qmax
                scale = torch.clamp(scale, min=1e-8)
                
                x_int = (x_clipped / scale).round().clamp(qmin, qmax)
                x_deq = x_int * scale
            else:
                # 非对称量化：使用无符号范围 [0, 2^bits-1]
                max_val = x_clipped.amax(dim=-1, keepdim=True)
                min_val = x_clipped.amin(dim=-1, keepdim=True)
                qmin = 0
                qmax = 2 ** self.bits - 1
                
                scale = (max_val - min_val) / (qmax - qmin + 1e-8)
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                zero_point = -min_val / scale
                
                x_int = ((x_clipped - min_val) / scale).round().clamp(qmin, qmax)
                x_deq = (x_int * scale) + min_val
        
        # 返回量化后的Hadamard域激活（不做逆变换）
        # 因为权重也在Hadamard域，利用 (X@H) @ (W@H)^T = X @ H @ H^T @ W^T = X @ W^T
        return x_deq


class WeightQuantizer(torch.nn.Module):
    """
    Weight Quantizer for static quantization of model weights.
    
    This quantizer is designed for weight quantization where quantization
    parameters are computed once and reused. Supports per-channel and
    per-group quantization with symmetric and asymmetric modes.
    """
    def __init__(self, bits=4, group_size=-1, sym=True, channel_wise=True):
        """
        Args:
            bits: int, number of quantization bits (1~16)
            group_size: int, size of groups for group quantization
                        -1: per-channel quantization (channel_wise must be True)
                        >0: group quantization with specified group size
            sym: bool, whether to use symmetric quantization
            channel_wise: bool, whether to use per-channel quantization
                         Only used when group_size=-1
        """
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.channel_wise = channel_wise
        
        # Quantization parameters (will be registered as buffers after calibration)
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        self.calibrated = False
        
    def calibrate(self, weight):
        """
        Calibrate quantization parameters based on weight statistics.
        使用快速Hadamard变换（蝶形算法）
        
        Args:
            weight: torch.Tensor, shape (out_features, in_features) or similar
        """
        if self.bits >= 16:
            self.calibrated = True
            return
        
        # 应用快速Hadamard变换（无需GPU传输和矩阵存储）
        weight_transformed = fast_hadamard_transform(weight, group_size=self.group_size, normalize=True)
        
        with torch.no_grad():
            if self.group_size > 0:
                # Group quantization
                self._calibrate_group_wise(weight_transformed)
            elif self.channel_wise:
                # Per-channel quantization
                self._calibrate_channel_wise(weight_transformed)
            else:
                # Per-tensor quantization
                self._calibrate_tensor_wise(weight_transformed)
        
        self.calibrated = True
    
    def _calibrate_tensor_wise(self, weight):
        """Per-tensor quantization: single scale and zero_point for entire tensor"""
        if self.sym:
            # Symmetric quantization: use signed range [-2^(bits-1), 2^(bits-1)-1]
            max_val = weight.abs().max()
            qmax = 2 ** (self.bits - 1) - 1
            qmin = -2 ** (self.bits - 1)
            
            scale = max_val / qmax
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.tensor(0.0, device=weight.device, dtype=weight.dtype)
        else:
            # Asymmetric quantization: use unsigned range [0, 2^bits-1]
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
        """Per-channel quantization: scale and zero_point per output channel"""
        # Assume weight shape is (out_features, in_features)
        # Quantize along in_features dimension (dim=1)
        if self.sym:
            # Symmetric quantization: use signed range
            max_val = weight.abs().amax(dim=1, keepdim=True)
            qmax = 2 ** (self.bits - 1) - 1
            qmin = -2 ** (self.bits - 1)
            
            scale = max_val / qmax
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.zeros_like(scale)
        else:
            # Asymmetric quantization: use unsigned range
            max_val = weight.amax(dim=1, keepdim=True)
            min_val = weight.amin(dim=1, keepdim=True)
            qmin = 0
            qmax = 2 ** self.bits - 1
            
            scale = (max_val - min_val) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = qmin - min_val / scale
        
        self.scale = scale
        self.zero_point = zero_point
    
    def _calibrate_group_wise(self, weight):
        """
        Group quantization: divide each channel into groups and quantize separately.
        
        For weight shape (out_features, in_features):
        - Divide in_features into groups of size group_size
        - Compute scale and zero_point for each group
        """
        out_features, in_features = weight.shape
        
        if in_features % self.group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"group_size ({self.group_size})"
            )
        
        num_groups = in_features // self.group_size
        
        # Reshape to (out_features, num_groups, group_size)
        weight_grouped = weight.view(out_features, num_groups, self.group_size)
        
        if self.sym:
            # Symmetric quantization: use signed range
            max_val = weight_grouped.abs().amax(dim=2, keepdim=True)
            qmax = 2 ** (self.bits - 1) - 1
            qmin = -2 ** (self.bits - 1)
            
            scale = max_val / qmax
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.zeros_like(scale)
        else:
            # Asymmetric quantization: use unsigned range
            max_val = weight_grouped.amax(dim=2, keepdim=True)
            min_val = weight_grouped.amin(dim=2, keepdim=True)
            qmin = 0
            qmax = 2 ** self.bits - 1
            
            scale = (max_val - min_val) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = qmin - min_val / scale
        
        # Store with shape (out_features, num_groups, 1)
        self.scale = scale
        self.zero_point = zero_point
    
    def quantize(self, weight):
        """
        Quantize weight using calibrated parameters.
        使用快速Hadamard变换（蝶形算法）
        
        Args:
            weight: torch.Tensor to quantize
            
        Returns:
            quantized and dequantized weight (in Hadamard domain)
        """
        # 应用快速Hadamard变换（无需GPU传输和矩阵存储）
        weight_transformed = fast_hadamard_transform(weight, group_size=self.group_size, normalize=True)
        
        if self.bits >= 16:
            # 不量化时，直接返回Hadamard变换后的权重
            return weight_transformed
        
        if not self.calibrated:
            raise RuntimeError("WeightQuantizer must be calibrated before quantization")
        
        # Determine quantization range based on symmetric/asymmetric mode
        if self.sym:
            qmin = -2 ** (self.bits - 1)
            qmax = 2 ** (self.bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** self.bits - 1
        
        if self.group_size > 0:
            # Group quantization
            weight_quantized = self._quantize_group_wise(weight_transformed, qmin, qmax)
        elif self.channel_wise:
            # Per-channel quantization
            weight_quantized = self._quantize_channel_wise(weight_transformed, qmin, qmax)
        else:
            # Per-tensor quantization
            weight_quantized = self._quantize_tensor_wise(weight_transformed, qmin, qmax)
        
        # 返回量化后的Hadamard域权重（不做逆变换）
        # 因为激活也在Hadamard域，利用 (X@H) @ (W@H)^T = X @ W^T
        return weight_quantized
    
    def _quantize_tensor_wise(self, weight, qmin, qmax):
        """Quantize using per-tensor parameters"""
        if self.sym:
            w_int = (weight / self.scale).round().clamp(qmin, qmax)
            w_dequant = w_int * self.scale
        else:
            w_int = (weight / self.scale + self.zero_point).round().clamp(qmin, qmax)
            w_dequant = (w_int - self.zero_point) * self.scale
        
        return w_dequant
    
    def _quantize_channel_wise(self, weight, qmin, qmax):
        """Quantize using per-channel parameters"""
        if self.sym:
            w_int = (weight / self.scale).round().clamp(qmin, qmax)
            w_dequant = w_int * self.scale
        else:
            w_int = (weight / self.scale + self.zero_point).round().clamp(qmin, qmax)
            w_dequant = (w_int - self.zero_point) * self.scale
        
        return w_dequant
    
    def _quantize_group_wise(self, weight, qmin, qmax):
        """Quantize using group-wise parameters"""
        out_features, in_features = weight.shape
        num_groups = in_features // self.group_size
        
        # Reshape to (out_features, num_groups, group_size)
        weight_grouped = weight.view(out_features, num_groups, self.group_size)
        
        if self.sym:
            w_int = (weight_grouped / self.scale).round().clamp(qmin, qmax)
            w_dequant = w_int * self.scale
        else:
            w_int = (weight_grouped / self.scale + self.zero_point).round().clamp(qmin, qmax)
            w_dequant = (w_int - self.zero_point) * self.scale
        
        # Reshape back to original shape
        w_dequant = w_dequant.view(out_features, in_features)
        
        return w_dequant
    
    def forward(self, weight):
        """
        Forward pass: calibrate if needed, then quantize.
        
        Args:
            weight: torch.Tensor to quantize
            
        Returns:
            quantized weight
        """
        if not self.calibrated:
            self.calibrate(weight)
        
        return self.quantize(weight)


# Backward compatibility
Quantizer = ActQuantizer

    
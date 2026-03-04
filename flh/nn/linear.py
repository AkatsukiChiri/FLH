import torch
import flh

from . import quantization as _quant


class LinearFLH(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16, device='cpu', 
                 dual_hadamard=False, in_group_size=None, out_group_size=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dual_hadamard = dual_hadamard
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        
        self.register_buffer('weight', 
                           torch.randn(self.out_features, self.in_features, dtype=dtype, device=device, requires_grad=False))
        
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype, device=device))
        else:
            self.bias = None
    
    def forward(self, x):
        # 推理过程保持简单，输入的 Hadamard 变换在外部处理
        # 无论单侧还是双侧，都是标准的线性变换
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
    def apply_single_hadamard(self):
        """
        将单侧 Hadamard 变换应用到权重
        W_single = W @ H, bias 不变
        """
        with torch.no_grad():
            # 权重单侧变换: W_single = W @ H
            W_single = _quant.fast_hadamard_transform(
                self.weight, 
                group_size=self.in_group_size, 
                normalize=True
            )
            self.weight.copy_(W_single)
            # bias 不变换（单侧应该恢复原始结果）
    
    def apply_dual_hadamard(self):
        """
        将双侧 Hadamard 变换应用到权重和偏置
        W_dual = H @ W @ H, b_dual = b @ H
        """
        with torch.no_grad():
            # 权重双侧变换: W_dual = H @ W @ H
            # 步骤1: W @ H (对输入维度应用 Hadamard)
            W_temp = _quant.fast_hadamard_transform(
                self.weight, 
                group_size=self.in_group_size, 
                normalize=True
            )
            
            # 步骤2: H @ (W @ H) (对输出维度应用 Hadamard)
            W_dual = _quant.fast_hadamard_transform(
                W_temp.T,  # 转置使列变成行
                group_size=self.out_group_size, 
                normalize=True
            ).T  # 转置回来
            
            self.weight.copy_(W_dual)
            
            # Bias 单侧变换: b_dual = b @ H
            if self.bias is not None:
                b_dual = _quant.fast_hadamard_transform(
                    self.bias.unsqueeze(0), 
                    group_size=self.out_group_size, 
                    normalize=True
                ).squeeze(0)
                self.bias.copy_(b_dual)
    
    @staticmethod    
    def from_float(module: torch.nn.Linear, weight_bits=4, weight_group_size=128, weight_sym=True,
                   dual_hadamard=False, in_group_size=None, out_group_size=None):
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
            out_group_size=out_group_size
        )

        flh_linear.weight.copy_(module.weight.data)
        if bias_flag:
            flh_linear.bias.copy_(module.bias.data)
        
        # 应用 Hadamard 变换到权重和偏置
        if dual_hadamard:
            flh_linear.apply_dual_hadamard()
        else:
            # 单侧 Hadamard：只变换权重，不变换 bias
            flh_linear.apply_single_hadamard()
        
        if torch.isnan(flh_linear.weight.data).any():
            print(f"WARNING: NaN detected in weights after Hadamard for layer {out_features}x{in_features}")
            return flh_linear
        
        weight_quantizer = _quant.WeightQuantizer(
            bits=weight_bits,
            group_size=weight_group_size,
            sym=weight_sym,
            channel_wise=(weight_group_size == -1),
            use_hadamard=False  # 禁用内部 Hadamard 变换，在外部处理
        )
        
        weight_quantizer.calibrate(flh_linear.weight)
        
        if weight_bits < 16 and torch.isnan(weight_quantizer.scale).any():
            print(f"WARNING: NaN detected in quantization scale for layer {out_features}x{in_features}")
            print(f"  Weight stats: min={flh_linear.weight.min():.6f}, max={flh_linear.weight.max():.6f}, mean={flh_linear.weight.mean():.6f}")
            return flh_linear
        
        quantized_weight = weight_quantizer.quantize(flh_linear.weight)
        
        if torch.isnan(quantized_weight).any():
            print(f"WARNING: NaN detected in quantized weights for layer {out_features}x{in_features}")
            if weight_bits < 16:
                print(f"  Scale stats: min={weight_quantizer.scale.min():.6f}, max={weight_quantizer.scale.max():.6f}")
            return flh_linear
        
        flh_linear.weight.copy_(quantized_weight)

        return flh_linear
    
    

if __name__ == "__main__":
    layer = torch.nn.Linear(1024, 1024)
    
    x = torch.randn(1024, 1024)
    y_ref = layer(x)
    
    layer_flh = LinearFLH.from_float(layer, weight_bits=15, weight_group_size=-1, weight_sym=True)
    x_flh = _quant.ActQuantizer(bits=15, group_size=-1, sym=True)(x)
    
    y_flh = layer_flh(x)
    
    print(y_ref)
    print(y_flh)
    
    print(torch.allclose(y_ref, y_flh, atol=1e-2))
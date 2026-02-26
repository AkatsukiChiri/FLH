import torch

from . import quantization as _quant


class LinearFLH(torch.nn.Module):
    """
    Quantized Linear layer with fake quantization.
    
    This layer performs forward pass with pre-quantized weights.
    Weight quantization is done once during from_float() conversion.
    """
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameter (will store quantized weights)
        self.register_buffer('weight', 
                           torch.randn(self.out_features, self.in_features, dtype=dtype, device=device, requires_grad=False))
        
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype, device=device))
        else:
            self.bias = None
    
    def forward(self, x):
        # Standard linear operation with pre-quantized weight
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        return x
    
    @staticmethod    
    def from_float(module: torch.nn.Linear, weight_bits=4, weight_group_size=128, weight_sym=True):
        """
        Convert a float Linear module to LinearFLH with weight quantization.

        Args:
            module (torch.nn.Linear): Source torch Linear layer
            weight_bits: Number of bits for weight quantization (default: 4)
            weight_group_size: Group size for weight quantization (default: 128, -1 for per-channel)
            weight_sym: Whether to use symmetric quantization (default: True)

        Returns:
            LinearFLH instance with quantized weights
        """
        in_features = module.in_features
        out_features = module.out_features
        bias_flag = module.bias is not None
        dtype = module.weight.dtype
        device = module.weight.device

        # Create LinearFLH instance on the same device
        flh_linear = LinearFLH(
            in_features, 
            out_features, 
            bias=bias_flag, 
            dtype=dtype,
            device=device
        )

        # Copy weights
        flh_linear.weight.copy_(module.weight.data)
        
        if bias_flag:
            flh_linear.bias.copy_(module.bias.data)
        
        # ⭐ 重要：无论weight_bits是多少，都要应用Hadamard变换和量化
        # 这样权重和激活才能在同一个域中进行矩阵乘法
        
        # Check for NaN in original weights
        if torch.isnan(module.weight.data).any():
            print(f"WARNING: NaN detected in original weights for layer {out_features}x{in_features}")
            return flh_linear
        
        # Create weight quantizer
        weight_quantizer = _quant.WeightQuantizer(
            bits=weight_bits,
            group_size=weight_group_size,
            sym=weight_sym,
            channel_wise=(weight_group_size == -1)
        )
        
        # Calibrate and quantize weights (即使bits >= 16也要执行，以应用Hadamard变换)
        weight_quantizer.calibrate(flh_linear.weight)
        
        # Check for NaN in scale (only if actually quantizing)
        if weight_bits < 16 and torch.isnan(weight_quantizer.scale).any():
            print(f"WARNING: NaN detected in quantization scale for layer {out_features}x{in_features}")
            print(f"  Weight stats: min={flh_linear.weight.min():.6f}, max={flh_linear.weight.max():.6f}, mean={flh_linear.weight.mean():.6f}")
            return flh_linear
        
        quantized_weight = weight_quantizer.quantize(flh_linear.weight)
        
        # Check for NaN in quantized weights
        if torch.isnan(quantized_weight).any():
            print(f"WARNING: NaN detected in quantized weights for layer {out_features}x{in_features}")
            if weight_bits < 16:
                print(f"  Scale stats: min={weight_quantizer.scale.min():.6f}, max={weight_quantizer.scale.max():.6f}")
            return flh_linear
        
        # Replace original weight with quantized (and Hadamard-transformed) weight
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
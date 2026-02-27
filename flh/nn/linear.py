import torch

from . import quantization as _quant


class LinearFLH(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer('weight', 
                           torch.randn(self.out_features, self.in_features, dtype=dtype, device=device, requires_grad=False))
        
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype, device=device))
        else:
            self.bias = None
    
    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        return x
    
    @staticmethod    
    def from_float(module: torch.nn.Linear, weight_bits=4, weight_group_size=128, weight_sym=True):
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
            device=device
        )

        flh_linear.weight.copy_(module.weight.data)
        
        if bias_flag:
            flh_linear.bias.copy_(module.bias.data)
        
        if torch.isnan(module.weight.data).any():
            print(f"WARNING: NaN detected in original weights for layer {out_features}x{in_features}")
            return flh_linear
        
        weight_quantizer = _quant.WeightQuantizer(
            bits=weight_bits,
            group_size=weight_group_size,
            sym=weight_sym,
            channel_wise=(weight_group_size == -1)
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
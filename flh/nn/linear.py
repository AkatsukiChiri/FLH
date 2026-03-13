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
        self.weight_group_size = None
        
        self.register_buffer('w_int', None)
        self.register_buffer('w_scale', None)
        self.register_buffer('w_zero', None)
        
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype, device=device))
        else:
            self.bias = None
    
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
        scale = self.w_scale
        zp = self.w_zero
        w_int = self.w_int
        if w_int is None:
            raise RuntimeError("LinearFLH: w_int is None, layer has not been quantized correctly.")
        if scale is not None:
            if self.weight_group_size is not None and self.weight_group_size > 0 and scale.dim() == 3:
                out_features, in_features = w_int.shape
                group_size = self.weight_group_size
                num_groups = in_features // group_size
                w_int_3d = w_int.view(out_features, num_groups, group_size)
                zp_eff = zp if zp is not None else 0
                w_deq_3d = (w_int_3d - zp_eff) * scale
                weight = w_deq_3d.view(out_features, in_features)
            else:
                zp_eff = zp if zp is not None else 0
                weight = (w_int - zp_eff) * scale
        else:
            weight = w_int
        return torch.nn.functional.linear(x, weight.to(x.dtype), self.bias)
    
    @staticmethod    
    def from_float(module: torch.nn.Linear, weight_bits=4, weight_group_size=128, weight_sym=True,
                   dual_hadamard=False, in_group_size=None, out_group_size=None, clip_ratio=1.0):
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
        
        W = module.weight.data.clone()
        if bias_flag:
            flh_linear.bias.copy_(module.bias.data)
        
        # 应用 Hadamard 变换到权重和偏置
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
        flh_linear.weight_group_size = weight_group_size
        flh_linear.w_int = w_int
        flh_linear.w_scale = scale
        flh_linear.w_zero = zp

        return flh_linear
    
    

if __name__ == "__main__":
    layer = torch.nn.Linear(1024, 1024)
    
    x = torch.randn(1024, 1024)
    y_ref = layer(x)
    
    layer_flh = LinearFLH.from_float(layer, weight_bits=15, weight_group_size=-1, weight_sym=True)
    scale, zp, q = _quant.ActQuantizer(bits=15, group_size=-1, sym=True)(x)
    x_flh = q if scale is None else (q - (zp if zp is not None else 0)) * scale
    
    y_flh = layer_flh(x_flh)
    
    print(y_ref)
    print(y_flh)
    
    print(torch.allclose(y_ref, y_flh, atol=1e-2))
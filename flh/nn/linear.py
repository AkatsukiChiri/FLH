import torch
import flh


class LinearFLH(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales',
                             torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', 
                             torch.randn(self.out_features, self.in_features, dtype=dtype, requires_grad=False))
        
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
    
    def forward(self, x):
        assert type(x) == flh.PackedQuantizedTensor
        x, scales_x = x.quantized_x, x.scales_x
        
        x_dequant = x * scales_x
        weight_dequant = self.weight * self.weight_scales   
        x = torch.matmul(x_dequant, weight_dequant.t())
        
        if self.bias is not None:
            return x + self.bias
        else:
            return x
    
    @staticmethod    
    def from_float(module: torch.nn.Linear, weight_scales=None):
        """
        根据浮点 Linear 模块以及权重量化尺度，构建 LinearFLH 量化线性层

        Args:
            module (torch.nn.Linear): 源 torch Linear 层
            weight_scales (Tensor or None): 权重量化尺度 (out_features, 1)

        Returns:
            LinearFLH实例
        """
        in_features = module.in_features
        out_features = module.out_features
        bias_flag = module.bias is not None
        dtype = module.weight.dtype

        flh_linear = LinearFLH(in_features, out_features, bias=bias_flag, dtype=dtype)

        # weight_scales: 若未提供则默认全1（无量化）
        if weight_scales is None:
            weight_scales = torch.ones((out_features, 1), dtype=module.weight.dtype, device=module.weight.device)

        # register weight_scales, weight
        flh_linear.weight_scales.copy_(weight_scales)
        flh_linear.weight.copy_(module.weight.data / weight_scales)  # 假定weight已量化: weight = orig_weight / scale

        if bias_flag:
            orig_bias = module.bias.data
            # 简单假定weight/activation都量化后，bias通常不用量化或只用量化scale校正
            # 此处假设权重scale已纳入，总保持float
            flh_linear.bias.copy_(orig_bias)

        return flh_linear
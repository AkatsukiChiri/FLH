import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        # Add weight parameter for compatibility with LlamaRMSNorm
        self.weight = torch.nn.Parameter(torch.ones(mean_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        # Apply learned weight
        x = x * self.weight
        return x.to(input_dtype)
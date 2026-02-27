import torch

class OnlineHadamard(torch.nn.Module):
    def __init__(self, hadamard_dim: int, group_size: int=-1, force_fp32=False, device=None):
        super().__init__()
        self.force_fp32 = force_fp32
        self.group_size = group_size
        self.hadamard_dim = hadamard_dim
        self.device = device
        self.hadamard_matrix = self.get_hadamard_matrix()
    
    def hadamard(self, n):
        if n == 1:
            return torch.ones(1, 1)
        else:
            H = self.hadamard(n // 2)
            top = torch.cat([H, H], dim=1)
            bottom = torch.cat([H, -H], dim=1)
            return torch.cat([top, bottom], dim=0)
       
    def get_hadamard_matrix(self):
        if self.group_size == -1:
            return self.hadamard(self.hadamard_dim)
        else:
            if (self.group_size & (self.group_size - 1)) != 0 or self.group_size == 0:
                raise ValueError("group_size must be a power of 2")
            if self.hadamard_dim % self.group_size != 0:
                raise ValueError("hadamard_dim must be a multiple of group_size")
            num_groups = self.hadamard_dim // self.group_size
            
            group_hadamard = self.hadamard(self.group_size)
            device = group_hadamard.device
            dtype = torch.float32 if self.force_fp32 else group_hadamard.dtype
            hadamard_mat = torch.zeros(
                self.hadamard_dim, self.hadamard_dim, dtype=dtype, device=device
            )
            for i in range(num_groups):
                start = i * self.group_size
                end = start + self.group_size
                hadamard_mat[start:end, start:end] = group_hadamard
            return hadamard_mat
        
    def forward(self, x):
        x_dtype = x.dtype
        if self.force_fp32:
            x = x.float()
        x = x @ self.hadamard_matrix
        if self.force_fp32:
            x = x.to(x_dtype)
        return x
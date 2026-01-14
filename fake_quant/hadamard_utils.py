import torch

import utils

def get_hadamard_matrix(size, device=utils.DEV):
    """
    根据size生成Hadamard矩阵
    """
    # Hadamard矩阵的size必须是2的幂
    def hadamard(n):
        """Recursive hadamard matrix construction."""
        if n == 1:
            return torch.tensor([[1.]], device=device)
        else:
            H = hadamard(n // 2)
            return torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1)
            ], dim=0)

    assert (size & (size - 1) == 0) and size > 0, "Hadamard size must be a positive power of 2"
    H = hadamard(size)
    # 归一化
    H = H / (size ** 0.5)
    return H.to(device)

def get_group_had_matrix(n, group_size, device=utils.DEV, had_dtype=torch.float64):
    if group_size > 0:
        assert group_size % 2 == 0
        assert n % group_size == 0
        
        grouphad_matrix = get_hadamard_matrix(group_size, device=device)
        
        had_matrix = torch.zeros((n, n), device=device)
        
        for i in range(n//group_size):
            start = i * group_size
            end = (i + 1) * group_size
            had_matrix[start:end, start:end] = grouphad_matrix
        return had_matrix.to(had_dtype)
    else:
        return get_hadamard_matrix(n).to(had_dtype).to(device)
    
def matmul_had_cuda(x, group_size):
    n = x.shape[-1]
    had_matrix = get_group_had_matrix(n, group_size, device=x.device, had_dtype=x.dtype)
    return x @ had_matrix


if __name__  == '__main__':
    x = torch.randn(1024, 1024)
    group_size = -1
    had_x = matmul_had_cuda(x, group_size)
    had_had_x = matmul_had_cuda(had_x, group_size)
    
    print(x)
    print(had_x)
    print(had_had_x)
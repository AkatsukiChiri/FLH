from math import sqrt
import torch
import flh

# 创建测试数据
x = torch.randn(5, 8, 1024, dtype=torch.float16, device='cuda')

x_had = flh.nn.fast_hadamard_transform(x, group_size=128, normalize=True)

x_copy = x.clone()
x_copy = x_copy.view(-1, 128)

print(x_copy)
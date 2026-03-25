import torch
import flh
from math import sqrt

x = torch.randn(1024, 2048, dtype=torch.float16, device='cuda')

actquantizer = flh.nn.ActQuantizer(bits=4, group_size=128, sym=True, packed_output=True, use_hadamard=True)

scale, zp, q = actquantizer(x)
scale_ref, zp_ref, q_ref = actquantizer.forward_origin(x)

print(q)
print(scale)
print(zp)

print(q_ref)
print(scale_ref)
print(zp_ref)

print(torch.allclose(q, q_ref, atol=1))
print(torch.allclose(scale, scale_ref, atol=1))

print(q - q_ref)
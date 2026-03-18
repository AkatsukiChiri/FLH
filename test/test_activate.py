import torch
import flh

# torch.manual_seed(0)

x = torch.randn(1, 1, 256).to(torch.float16)

quantizer_no_packed = flh.nn.ActQuantizer(bits=4, group_size=128, sym=True, packed_output=False)
quantizer_packed = flh.nn.ActQuantizer(bits=4, group_size=128, sym=True, packed_output=True)

scale, zp, q = quantizer_no_packed(x)
scale_packed, zp_packed, q_packed = quantizer_packed(x)

print(q)
print(q_packed)
print(scale)
print(scale_packed)
print(zp)
print(zp_packed)
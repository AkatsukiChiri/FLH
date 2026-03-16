import torch
import flh

torch.manual_seed(0)

layer = torch.nn.Linear(256, 256, bias=False).to(torch.float16)

x = torch.randn(1, 10, 256).to(torch.float16)

layer_flh = flh.nn.LinearFLH.from_float(layer, weight_bits=4, weight_group_size=128, in_group_size=128, out_group_size=128, weight_sym=True, no_hadamard=False, dual_hadamard=False)

scale, zp, q = flh.nn.ActQuantizer(bits=4, group_size=128, sym=False, packed_output=True)(x)

y_flh = layer_flh(q, scale, zp, x_is_packed=True)

print(y_flh)
print(layer(x))

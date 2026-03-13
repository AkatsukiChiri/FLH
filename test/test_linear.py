import torch
import flh

torch.manual_seed(0)

layer = torch.nn.Linear(256, 256)

x = torch.randn(1, 10, 256)

y = layer(x)

layer_flh = flh.nn.LinearFLH.from_float(layer, weight_bits=8, weight_group_size=128, weight_sym=True, in_group_size=128, out_group_size=128)

print(layer.weight.data[0][:4])
print(layer_flh.w_int.data[0][:4])
print(flh.nn.fast_hadamard_transform(layer.weight.data, group_size=128, normalize=True)[0][:4])

scale, zp, q = flh.nn.ActQuantizer(bits=8, group_size=128, sym=False)(x)

y_flh = layer_flh(q, scale, zp)

print(y)
print(y_flh)
print(layer_flh(flh.nn.fast_hadamard_transform(x, group_size=128, normalize=True)))
print(torch.allclose(y, y_flh, atol=1e-1))
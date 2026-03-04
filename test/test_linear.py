import torch
import flh

layer = torch.nn.Linear(128, 128)
layer.weight.data = torch.randn(128, 128)
layer.bias.data = torch.randn(128)

x = torch.randn(1, 128)

y = layer(x)
layer_flh = flh.nn.LinearFLH.from_float(layer, weight_bits=15, weight_group_size=128, weight_sym=True)
x_flh = flh.nn.ActQuantizer(bits=15, group_size=128, sym=True)(x)
y_flh = layer_flh(x_flh)

print(layer.weight.data[0][:4])
print(layer_flh.weight.data[0][:4])
print(flh.nn.fast_hadamard_transform(layer.weight.data, group_size=128, normalize=True)[0][:4])

print(y)
print(y_flh)
print(torch.allclose(y, y_flh, atol=1e-2))
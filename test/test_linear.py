import torch
import flh

torch.manual_seed(0)

layer = torch.nn.Linear(256, 256, bias=False).to(torch.float16)

x = torch.randn(1, 10, 256).to(torch.float16)

layer_flh = flh.nn.LinearFLH.from_float(layer, weight_bits=15, weight_group_size=128, in_group_size=128, out_group_size=128, weight_sym=True, no_hadamard=False, dual_hadamard=False)

print(layer_flh.get_weight())
print(flh.nn.fast_hadamard_transform(layer.weight.data, group_size=128, normalize=True))

print(layer_flh(flh.nn.fast_hadamard_transform(x, group_size=128, normalize=True)))
print(layer(x))
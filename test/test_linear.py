import torch
import flh

torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

layer = torch.nn.Linear(2048, 2048, bias=False, dtype=torch.float16).to(device)
# layer.weight.data.fill_(1)

x = torch.randn(1, 10, 2048, dtype=torch.float16).to(device)

layer_flh = flh.nn.LinearFLH.from_float(layer, weight_bits=4, weight_group_size=128, in_group_size=128, out_group_size=128, weight_sym=True, no_hadamard=False, dual_hadamard=False)

scale, zp, q = flh.nn.ActQuantizer(bits=4, group_size=128, sym=True, packed_output=True, use_hadamard=True)(x)

print(q)


y_flh = layer_flh(q, scale, zp, x_is_packed=True)

print(y_flh)
print(layer(x))

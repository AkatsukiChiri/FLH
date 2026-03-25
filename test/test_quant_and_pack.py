import torch
import flh
import tqdm

x = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')

actquantizer = flh.nn.ActQuantizer(bits=4, group_size=128, sym=True, packed_output=True, use_hadamard=False)

for i in tqdm.tqdm(range(100)):
    scale, zp, q = actquantizer(x)
for i in tqdm.tqdm(range(100)):
    scale_ref, zp_ref, q_ref = actquantizer.forward_origin(x)

print(q)
print(q_ref)
print(scale)
print(scale_ref)

print(torch.allclose(q, q_ref, atol=1e-4))
print(torch.allclose(scale, scale_ref, atol=1e-4))

print(q - q_ref)
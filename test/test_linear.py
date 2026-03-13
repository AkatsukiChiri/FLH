import torch
import flh

torch.manual_seed(0)

layer = torch.nn.Linear(256, 256, bias=False).to(torch.float16)

x = torch.randn(1, 10, 256).to(torch.float16)

layer_flh = flh.nn.LinearFLH.from_float(layer, weight_bits=4, weight_group_size=128, in_group_size=128, out_group_size=128, weight_sym=True, no_hadamard=False, dual_hadamard=False)

def int32_to_8int4(x):
    """将int32数据转换为8个int4"""
    # 每个int32包含8个int4，每个int4占用4位
    result = []
    for i in range(8):
        # 提取第i个int4（从低位开始）
        int4_val = (x >> (i * 4)) & 0xF
        # 处理符号位：如果最高位是1，则是负数
        if int4_val & 0x8:
            int4_val = int4_val - 16
        result.append(int4_val)
    return result

print(int32_to_8int4(layer_flh.w_packed[0].item()))
print(layer_flh.get_weight())
print(flh.nn.fast_hadamard_transform(layer.weight.data, group_size=128, normalize=True))

print(layer_flh(flh.nn.fast_hadamard_transform(x, group_size=128, normalize=True)))
print(layer(x))


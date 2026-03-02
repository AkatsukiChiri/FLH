#!/usr/bin/env python3

import torch
import numpy as np
from transformers import LlamaConfig, LlamaForCausalLM
from flh.quantized_model.modeling_llama import FLH_LlamaForCausalLM, get_calibration_dataloader

def create_small_llama_config():
    return LlamaConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        _attn_implementation='flash_attention_2'
    )

def test_gptq_vs_rtn():
    print("🔍 测试 GPTQ vs RTN 是否有区别...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建小模型
    config = create_small_llama_config()
    original_model = LlamaForCausalLM(config).to(device, dtype=torch.float16)
    
    # 创建简单的校准数据（避免网络下载）
    print("创建校准数据...")
    calibration_dataloader = []
    for _ in range(8):
        # 创建随机校准数据
        input_ids = torch.randint(0, config.vocab_size, (1, 64))
        calibration_dataloader.append([input_ids])
    
    # 测试输入
    input_ids = torch.randint(0, config.vocab_size, (1, 32)).to(device)
    
    print("\n1. 创建 RTN 量化模型...")
    rtn_model = FLH_LlamaForCausalLM.from_float(
        original_model,
        target_device=device,
        weight_bits=4,
        weight_group_size=128,
        act_bits=4,
        act_group_size=128,
        use_gptq=False  # RTN
    )
    
    print("2. 创建 GPTQ 量化模型...")
    gptq_model = FLH_LlamaForCausalLM.from_float(
        original_model,
        target_device=device,
        weight_bits=4,
        weight_group_size=128,
        act_bits=4,
        act_group_size=128,
        use_gptq=True,  # GPTQ
        calibration_dataloader=calibration_dataloader,
        gptq_nsamples=8
    )
    
    print("\n3. 比较权重...")
    # 比较第一层的 q_proj 权重
    rtn_weight = rtn_model.model.layers[0].self_attn.q_proj.weight.data
    gptq_weight = gptq_model.model.layers[0].self_attn.q_proj.weight.data
    
    weight_diff = torch.abs(rtn_weight - gptq_weight).mean().item()
    print(f"第一层 q_proj 权重平均绝对差异: {weight_diff:.6f}")
    
    if weight_diff < 1e-6:
        print("❌ 权重几乎相同，GPTQ 可能没有生效")
    else:
        print("✅ 权重有明显差异，GPTQ 正在工作")
    
    print("\n4. 比较输出...")
    with torch.no_grad():
        rtn_output = rtn_model(input_ids).logits
        gptq_output = gptq_model(input_ids).logits
    
    output_diff = torch.abs(rtn_output - gptq_output).mean().item()
    print(f"输出平均绝对差异: {output_diff:.6f}")
    
    if output_diff < 1e-6:
        print("❌ 输出几乎相同，GPTQ 可能没有生效")
    else:
        print("✅ 输出有差异，GPTQ 正在工作")
    
    # 检查校准数据是否被使用
    print("\n5. 检查校准数据收集...")
    from flh.quantized_model.modeling_llama import _collect_calibration_inputs_flh, FLH_FP16LlamaForCausalLM
    
    fp16_model = FLH_FP16LlamaForCausalLM.from_float(original_model, target_device="cpu", fuse_layernorm=True)
    cal_data = _collect_calibration_inputs_flh(fp16_model, calibration_dataloader, device, 8, 64)
    
    print(f"校准数据层数: {len(cal_data)}")
    if len(cal_data) > 0:
        layer_0_data = cal_data[0]
        print(f"第0层校准数据键: {list(layer_0_data.keys())}")
        if "self_attn.q_proj" in layer_0_data:
            q_proj_data = layer_0_data["self_attn.q_proj"]
            print(f"q_proj 校准数据数量: {len(q_proj_data)}")
            if len(q_proj_data) > 0:
                print(f"第一个校准张量形状: {q_proj_data[0].shape}")
                print("✅ 校准数据收集正常")
            else:
                print("❌ 校准数据为空")
        else:
            print("❌ 没有找到 q_proj 校准数据")
    else:
        print("❌ 没有收集到校准数据")

if __name__ == "__main__":
    test_gptq_vs_rtn()
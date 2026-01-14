#!/usr/bin/env python3
"""
测试FLH模型与原始transformers模型的一致性
"""

import torch
import numpy as np
from transformers import AutoConfig, LlamaForCausalLM
from flh.quantized_model.modeling_llama import FLH_FP16LlamaForCausalLM

def test_model_consistency():
    # 配置
    config = AutoConfig.from_pretrained("/home/mashaobo/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-1B/")
    config._attn_implementation = "flash_attention_2"
    
    print("=" * 80)
    print("测试FLH模型与原始模型的一致性")
    print("=" * 80)
    
    # 创建模型
    print("\n1. 创建模型...")
    
    # 原始模型
    print("  - 创建原始LlamaForCausalLM...")
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    
    with torch.no_grad():
        original_model = LlamaForCausalLM(config)
    
    torch.set_default_dtype(dtype_old)
    original_model = original_model.to(dtype=torch.float16, device='cuda')
    original_model.eval()
    
    # FLH模型
    print("  - 创建FLH_FP16LlamaForCausalLM...")
    torch.set_default_dtype(torch.float16)
    
    with torch.no_grad():
        flh_model = FLH_FP16LlamaForCausalLM(config)
    
    torch.set_default_dtype(dtype_old)
    flh_model = flh_model.to(dtype=torch.float16, device='cuda')
    flh_model.eval()
    
    # 复制权重确保相同
    print("\n2. 复制权重...")
    with torch.no_grad():
        flh_model.load_state_dict(original_model.state_dict(), strict=False)
    
    # 测试输入
    print("\n3. 准备测试输入...")
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device='cuda')
    
    print(f"  Input shape: {input_ids.shape}")
    
    # 前向传播
    print("\n4. 执行前向传播...")
    with torch.no_grad():
        # 原始模型
        original_output = original_model(input_ids, use_cache=False)
        original_logits = original_output.logits
        
        # FLH模型  
        flh_output = flh_model(input_ids, use_cache=False)
        flh_logits = flh_output.logits
    
    # 比较结果
    print("\n5. 比较输出...")
    print(f"  Original logits shape: {original_logits.shape}")
    print(f"  FLH logits shape: {flh_logits.shape}")
    
    # 计算差异
    abs_diff = torch.abs(original_logits - flh_logits)
    rel_diff = abs_diff / (torch.abs(original_logits) + 1e-8)
    
    print(f"\n  绝对差异统计:")
    print(f"    Mean: {abs_diff.mean().item():.6e}")
    print(f"    Max: {abs_diff.max().item():.6e}")
    print(f"    Min: {abs_diff.min().item():.6e}")
    print(f"    Std: {abs_diff.std().item():.6e}")
    
    print(f"\n  相对差异统计:")
    print(f"    Mean: {rel_diff.mean().item():.6e}")
    print(f"    Max: {rel_diff.max().item():.6e}")
    
    # 判断是否一致
    print(f"\n6. 一致性检查...")
    
    # 使用不同的tolerance
    tolerances = [1e-5, 1e-4, 1e-3, 1e-2]
    for tol in tolerances:
        is_close = torch.allclose(original_logits, flh_logits, rtol=tol, atol=tol)
        status = '✓' if is_close else '✗'
        print(f"  rtol={tol}, atol={tol}: {status}")
    
    # 详细比较attention输出
    print(f"\n7. 逐层比较attention...")
    
    with torch.no_grad():
        # Hook函数来捕获中间输出
        original_attn_outputs = []
        flh_attn_outputs = []
        
        def make_hook(output_list):
            def hook(module, input, output):
                output_list.append(output[0].detach().clone())
            return hook
        
        # 只比较第一层
        layer_idx = 0
        original_hook = original_model.model.layers[layer_idx].self_attn.register_forward_hook(
            make_hook(original_attn_outputs)
        )
        flh_hook = flh_model.model.layers[layer_idx].self_attn.register_forward_hook(
            make_hook(flh_attn_outputs)
        )
        
        # 再次前向传播
        original_model(input_ids, use_cache=False)
        flh_model(input_ids, use_cache=False)
        
        original_hook.remove()
        flh_hook.remove()
        
        # 比较attention输出
        if original_attn_outputs and flh_attn_outputs:
            orig_attn = original_attn_outputs[0]
            flh_attn = flh_attn_outputs[0]
            
            attn_diff = torch.abs(orig_attn - flh_attn)
            print(f"  Layer {layer_idx} attention output差异:")
            print(f"    Mean: {attn_diff.mean().item():.6e}")
            print(f"    Max: {attn_diff.max().item():.6e}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    # 返回差异信息
    return {
        'abs_diff_mean': abs_diff.mean().item(),
        'abs_diff_max': abs_diff.max().item(),
        'rel_diff_mean': rel_diff.mean().item(),
        'rel_diff_max': rel_diff.max().item(),
    }


if __name__ == "__main__":
    try:
        results = test_model_consistency()
        
        # 判断差异是否可接受
        if results['abs_diff_max'] < 1e-3:
            print("\n✓ 模型输出基本一致（差异在可接受范围内）")
        else:
            print("\n✗ 模型输出存在显著差异！")
            print("  可能的原因：")
            print("  1. 第112行的transpose操作")
            print("  2. KV cache的transpose操作")
            print("  3. Flash Attention的实现差异")
            
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

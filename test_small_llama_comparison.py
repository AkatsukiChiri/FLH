#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from flh.quantized_model.modeling_llama import FLH_LlamaForCausalLM, FLH_FP16LlamaForCausalLM
import numpy as np

def create_small_llama_config():
    """创建小型Llama模型配置"""
    return LlamaConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        _attn_implementation='flash_attention_2'
    )

def compare_outputs(original_logits, flh_logits, model_name):
    """比较两个模型的输出"""
    print(f"\n=== {model_name} 输出对比 ===")
    
    # 计算差异
    abs_diff = torch.abs(original_logits - flh_logits)
    rel_diff = abs_diff / (torch.abs(original_logits) + 1e-8)
    
    print(f"输出形状: {original_logits.shape}")
    print(f"最大绝对差异: {abs_diff.max().item():.6f}")
    print(f"平均绝对差异: {abs_diff.mean().item():.6f}")
    print(f"最大相对差异: {rel_diff.max().item():.6f}")
    print(f"平均相对差异: {rel_diff.mean().item():.6f}")
    
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(
        original_logits.flatten(), 
        flh_logits.flatten(), 
        dim=0
    )
    print(f"余弦相似度: {cos_sim.item():.6f}")
    
    # 检查概率分布差异
    original_probs = F.softmax(original_logits, dim=-1)
    flh_probs = F.softmax(flh_logits, dim=-1)
    kl_div = F.kl_div(
        F.log_softmax(flh_logits, dim=-1),
        original_probs,
        reduction='mean'
    )
    print(f"KL散度: {kl_div.item():.6f}")
    
    return {
        'max_abs_diff': abs_diff.max().item(),
        'mean_abs_diff': abs_diff.mean().item(),
        'max_rel_diff': rel_diff.max().item(),
        'mean_rel_diff': rel_diff.mean().item(),
        'cosine_sim': cos_sim.item(),
        'kl_div': kl_div.item()
    }

def test_generation_difference(original_model, flh_model, model_name, input_ids):
    """测试生成结果差异"""
    print(f"\n=== {model_name} 生成测试 ===")
    
    with torch.no_grad():
        # 原始模型生成
        original_outputs = original_model.generate(
            input_ids, 
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=0
        )
        
        # FLH模型生成
        flh_outputs = flh_model.generate(
            input_ids,
            max_new_tokens=1, 
            do_sample=False,
            pad_token_id=0
        )
    
    print(f"原始模型生成: {original_outputs[0].tolist()}")
    print(f"{model_name}生成: {flh_outputs[0].tolist()}")
    
    # 计算生成序列的匹配度
    matches = (original_outputs == flh_outputs).float().mean()
    print(f"生成序列匹配度: {matches.item():.4f}")
    
    return matches.item()

def main():
    torch.manual_seed(42)
    print("创建小型Llama模型进行对比测试...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建配置和原始模型
    config = create_small_llama_config()
    print(f"模型配置: {config.num_hidden_layers}层, {config.hidden_size}维")
    
    # 创建原始模型
    original_model = LlamaForCausalLM(config).to(device, dtype=torch.float16)
    print("✓ 原始Llama模型创建完成")
    
    # 创建FLH FP16模型
    fp16_model = FLH_FP16LlamaForCausalLM.from_float(original_model, target_device=device)
    print("✓ FLH FP16模型创建完成")
    
    # 创建FLH W8A8量化模型 (不使用GPTQ)
    w8a8_model = FLH_LlamaForCausalLM.from_float(
        original_model, 
        target_device=device,
        weight_bits=16,
        weight_group_size=128,
        act_bits=16,
        act_group_size=128,
        use_gptq=False  # 禁用GPTQ，使用RTN量化
    )
    print("✓ FLH W8A8量化模型创建完成")
    
    # 创建测试输入
    batch_size = 1
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    print(f"测试输入形状: {input_ids.shape}")
    
    # 测试前向传播
    with torch.no_grad():
        original_outputs = original_model(input_ids)
        fp16_outputs = fp16_model(input_ids)
        w8a8_outputs = w8a8_model(input_ids)
    
    # 对比输出
    results = {}
    results['fp16'] = compare_outputs(
        original_outputs.logits, 
        fp16_outputs.logits, 
        "FLH_FP16LlamaForCausalLM"
    )
    
    results['w8a8'] = compare_outputs(
        original_outputs.logits,
        w8a8_outputs.logits,
        "FLH_LlamaForCausalLM(W8A8)"
    )
    
    # 测试生成差异
    gen_results = {}
    gen_results['fp16'] = test_generation_difference(
        original_model, fp16_model, "FLH_FP16", input_ids
    )
    gen_results['w8a8'] = test_generation_difference(
        original_model, w8a8_model, "FLH_W8A8", input_ids
    )
    
    # 总结
    print("\n" + "="*50)
    print("测试总结")
    print("="*50)
    
    print("\n输出差异对比:")
    print(f"{'指标':<15} {'FP16':<12} {'W8A8':<12}")
    print("-" * 40)
    print(f"{'最大绝对差异':<15} {results['fp16']['max_abs_diff']:<12.6f} {results['w8a8']['max_abs_diff']:<12.6f}")
    print(f"{'平均绝对差异':<15} {results['fp16']['mean_abs_diff']:<12.6f} {results['w8a8']['mean_abs_diff']:<12.6f}")
    print(f"{'余弦相似度':<15} {results['fp16']['cosine_sim']:<12.6f} {results['w8a8']['cosine_sim']:<12.6f}")
    print(f"{'KL散度':<15} {results['fp16']['kl_div']:<12.6f} {results['w8a8']['kl_div']:<12.6f}")
    
    print(f"\n生成匹配度:")
    print(f"FP16: {gen_results['fp16']:.4f}")
    print(f"W8A8: {gen_results['w8a8']:.4f}")
    
    # 性能评估
    print(f"\n模型参数量:")
    original_params = sum(p.numel() for p in original_model.parameters())
    print(f"原始模型: {original_params:,}")
    print(f"FP16模型: {sum(p.numel() for p in fp16_model.parameters()):,}")
    print(f"W8A8模型: {sum(p.numel() for p in w8a8_model.parameters()):,}")

if __name__ == "__main__":
    main()
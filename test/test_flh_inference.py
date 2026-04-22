"""
FLH W4A4 量化模型 vs 原始 FP16 模型 性能对比分析
"""
import torch
import time
import gc
from collections import defaultdict

# 全局计时器
class Timer:
    def __init__(self):
        self.times = defaultdict(list)
        self.current_name = None
        self.start_time = None
    
    def reset(self):
        self.times.clear()
        self.current_name = None
        self.start_time = None
    
    def start(self, name):
        torch.cuda.synchronize()
        self.current_name = name
        self.start_time = time.perf_counter()
    
    def stop(self):
        if self.current_name is not None:
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - self.start_time) * 1000
            self.times[self.current_name].append(elapsed)
            self.current_name = None
    
    def report(self, prefix=""):
        print("\n" + "=" * 80)
        print(prefix + "各组件耗时分析")
        print(prefix + "=" * 80)
        
        if not self.times:
            print(prefix + "没有收集到计时数据")
            return
        
        total_time = sum(sum(v) for v in self.times.values())
        
        merged_times = defaultdict(float)
        merged_counts = defaultdict(int)
        
        for name, times_list in self.times.items():
            if name.startswith("LayerNorm_in_") or name.startswith("LayerNorm_post_"):
                base_name = "LayerNorm"
            elif name.startswith("Q_proj_"):
                base_name = "Q_proj"
            elif name.startswith("K_proj_"):
                base_name = "K_proj"
            elif name.startswith("V_proj_"):
                base_name = "V_proj"
            elif name.startswith("O_proj_"):
                base_name = "O_proj"
            elif name.startswith("Gate_proj_"):
                base_name = "Gate_proj"
            elif name.startswith("Up_proj_"):
                base_name = "Up_proj"
            elif name.startswith("Down_proj_"):
                base_name = "Down_proj"
            elif name.startswith("FlashAttention_"):
                base_name = "FlashAttention"
            elif name.startswith("Quant1_"):
                base_name = "Quant1"
            elif name.startswith("Quant2_"):
                base_name = "Quant2"
            elif name == "Embedding":
                base_name = "Embedding"
            elif name == "Final_norm":
                base_name = "Final_norm"
            elif name == "LM_head":
                base_name = "LM_head"
            elif name == "KV_cache":
                base_name = "KV_cache"
            elif name == "total":
                base_name = "total"
            else:
                base_name = name
            
            merged_times[base_name] += sum(times_list)
            merged_counts[base_name] += len(times_list)
        
        sorted_times = sorted(merged_times.items(), key=lambda x: x[1], reverse=True)
        
        print("\n" + prefix + "{:<25} {:<15} {:<15} {:<10}".format("组件类型", "总时间(ms)", "平均(ms)", "占比"))
        print(prefix + "-" * 70)
        
        for name, total in sorted_times:
            count = merged_counts[name]
            avg = total / count if count > 0 else 0
            pct = total / total_time * 100 if total_time > 0 else 0
            print(prefix + "{:<25} {:<15.2f} {:<15.4f} {:>6.1f}%".format(name, total, avg, pct))
        
        print(prefix + "-" * 70)
        print(prefix + "{:<25} {:<15.2f}".format("总计", total_time))
        
        return merged_times, total_time


def load_fp16_model(model_path, device="cuda"):
    """加载原始 FP16 模型"""
    import os
    from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
    
    # 展开路径中的 ~
    model_path = os.path.expanduser(model_path)
    
    print("加载原始 FP16 模型: " + model_path)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载配置
    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "flash_attention_2"
    
    # 加载原始模型到 CUDA
    print("  加载 LlamaForCausalLM 到 CUDA...")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    
    print("✓ FP16 模型加载完成")
    return model, tokenizer


def load_quantized_model(model_path, device="cuda"):
    """加载量化模型"""
    from transformers import AutoTokenizer
    from flh.quantized_model.modeling_llama import FLH_LlamaForCausalLM
    
    print("加载量化模型...")
    model = FLH_LlamaForCausalLM.load_quantized(model_path, target_device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    # # 开启 KV Cache 量化
    # print("开启 KV Cache 量化...")
    # model.kv_cache_quant = True
    # for layer in model.model.layers:
    #     layer.self_attn.kv_cache_quant = True
    
    print("✓ 量化模型加载完成")
    return model, tokenizer


def test_fp16_model(batch_size=8, seq_len=128, gen_tokens=8):
    """测试原始 FP16 模型的性能"""
    print("\n" + "=" * 80)
    print("原始 FP16 模型性能测试")
    print("=" * 80)
    
    # 使用与 evaluate_model.py 相同的方法加载
    fp16_model_path = "~/.cache/modelscope/hub/LLM-Research/llama-2-7b/"
    model, tokenizer = load_fp16_model(fp16_model_path)
    
    print("\n测试配置:")
    print("  batch_size: " + str(batch_size))
    print("  seq_len: " + str(seq_len))
    print("  gen_tokens: " + str(gen_tokens))
    
    # 创建 batch 输入
    prompt = "The quick brown fox jumps over the lazy dog. "
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :seq_len].cuda()
    input_ids = input_ids.expand(batch_size, -1)  # [batch_size, seq_len]
    
    print("  输入形状: " + str(input_ids.shape))
    
    # 预热
    print("\n预热中...")
    with torch.no_grad():
        for _ in range(5):
            out = model(input_ids, use_cache=True)
            for _ in range(3):
                next_token = out.logits[:, -1:].argmax(dim=-1)
                out = model(next_token, past_key_values=out.past_key_values, use_cache=True)
    torch.cuda.synchronize()
    
    # Timer
    timer = Timer()
    
    # Monkey-patch DynamicCache.update 来计时 KV cache 操作
    from transformers.cache_utils import DynamicCache
    original_dynamic_update = DynamicCache.update
    
    def timed_dynamic_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        timer.stop()
        timer.start("KV_cache")
        result = original_dynamic_update(self, key_states, value_states, layer_idx, cache_kwargs)
        timer.stop()
        return result
    
    DynamicCache.update = timed_dynamic_update
    
    # 注册 hooks 来计时各组件
    hooks = []
    
    def make_pre_hook(name):
        def hook(module, inputs):
            timer.stop()
            timer.start(name)
        return hook
    
    def make_post_hook(name):
        def hook(module, inputs, output):
            timer.stop()
        return hook
    
    # Embedding
    hooks.append(model.model.embed_tokens.register_forward_pre_hook(make_pre_hook("Embedding")))
    hooks.append(model.model.embed_tokens.register_forward_hook(make_post_hook("Embedding")))
    
    # 每层的组件
    for i, layer in enumerate(model.model.layers):
        # LayerNorm in
        hooks.append(layer.input_layernorm.register_forward_pre_hook(make_pre_hook("LayerNorm_in_" + str(i))))
        hooks.append(layer.input_layernorm.register_forward_hook(make_post_hook("LayerNorm_in_" + str(i))))
        
        # Q, K, V proj
        hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(make_pre_hook("Q_proj_" + str(i))))
        hooks.append(layer.self_attn.q_proj.register_forward_hook(make_post_hook("Q_proj_" + str(i))))
        hooks.append(layer.self_attn.k_proj.register_forward_pre_hook(make_pre_hook("K_proj_" + str(i))))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(make_post_hook("K_proj_" + str(i))))
        hooks.append(layer.self_attn.v_proj.register_forward_pre_hook(make_pre_hook("V_proj_" + str(i))))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make_post_hook("V_proj_" + str(i))))
        
        # O proj
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_pre_hook("O_proj_" + str(i))))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(make_post_hook("O_proj_" + str(i))))
        
        # LayerNorm post
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(make_pre_hook("LayerNorm_post_" + str(i))))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(make_post_hook("LayerNorm_post_" + str(i))))
        
        # MLP
        hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(make_pre_hook("Gate_proj_" + str(i))))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_post_hook("Gate_proj_" + str(i))))
        hooks.append(layer.mlp.up_proj.register_forward_pre_hook(make_pre_hook("Up_proj_" + str(i))))
        hooks.append(layer.mlp.up_proj.register_forward_hook(make_post_hook("Up_proj_" + str(i))))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_pre_hook("Down_proj_" + str(i))))
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_post_hook("Down_proj_" + str(i))))
    
    # Final norm 和 lm_head
    hooks.append(model.model.norm.register_forward_pre_hook(make_pre_hook("Final_norm")))
    hooks.append(model.model.norm.register_forward_hook(make_post_hook("Final_norm")))
    hooks.append(model.lm_head.register_forward_pre_hook(make_pre_hook("LM_head")))
    hooks.append(model.lm_head.register_forward_hook(make_post_hook("LM_head")))
    
    # 测试
    print("\n开始性能测试...")
    n_iter = 5
    all_times = []
    
    timer.reset()
    with torch.no_grad():
        for iter_idx in range(n_iter):
            iter_start = time.perf_counter()
            
            timer.start("total")
            out = model(input_ids, use_cache=True)
            past_kv = out.past_key_values
            
            for step in range(gen_tokens):
                next_token = out.logits[:, -1:].argmax(dim=-1)
                out = model(next_token, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
            
            timer.stop()
            iter_time = (time.perf_counter() - iter_start) * 1000
            all_times.append(iter_time)
            print("  第 " + str(iter_idx + 1) + " 次: {:.2f} ms".format(iter_time))
            
            torch.cuda.synchronize()
    
    print("\n平均生成 " + str(gen_tokens) + " token 时间: {:.2f} ms".format(sum(all_times)/len(all_times)))
    
    fp16_times, fp16_total = timer.report(prefix="[FP16] ")
    
    # 清理
    for h in hooks:
        h.remove()
    DynamicCache.update = original_dynamic_update
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return fp16_times, fp16_total


def test_quantized_model(batch_size=8, seq_len=128, gen_tokens=8):
    """测试量化模型的性能"""
    print("\n" + "=" * 80)
    print("量化模型 (FLH W4A4) 性能测试")
    print("=" * 80)
    
    # 加载量化模型
    quant_model_path = "./flh_model/llama2-7b"
    model, tokenizer = load_quantized_model(quant_model_path)
    
    print("\n测试配置:")
    print("  batch_size: " + str(batch_size))
    print("  seq_len: " + str(seq_len))
    print("  gen_tokens: " + str(gen_tokens))
    
    # 创建 batch 输入
    prompt = "The quick brown fox jumps over the lazy dog. "
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :seq_len].cuda()
    input_ids = input_ids.expand(batch_size, -1)  # [batch_size, seq_len]
    
    print("  输入形状: " + str(input_ids.shape))
    
    # 预热
    print("\n预热中...")
    with torch.no_grad():
        for _ in range(5):
            out = model(input_ids, use_cache=True)
            for _ in range(3):
                next_token = out.logits[:, -1:].argmax(dim=-1)
                out = model(next_token, past_key_values=out.past_key_values, use_cache=True)
    torch.cuda.synchronize()
    
    # Timer
    timer = Timer()
    
    # 注册 hooks 来计时各组件
    hooks = []
    
    def make_pre_hook(name):
        def hook(module, inputs):
            timer.stop()
            timer.start(name)
        return hook
    
    def make_post_hook(name):
        def hook(module, inputs, output):
            timer.stop()
        return hook
    
    # 注册 hooks 来计时各组件
    hooks = []
    
    def make_pre_hook(name):
        def hook(module, inputs):
            timer.stop()
            timer.start(name)
        return hook
    
    def make_post_hook(name):
        def hook(module, inputs, output):
            timer.stop()
        return hook
    
    # Embedding
    hooks.append(model.model.embed_tokens.register_forward_pre_hook(make_pre_hook("Embedding")))
    hooks.append(model.model.embed_tokens.register_forward_hook(make_post_hook("Embedding")))
    
    for i, layer in enumerate(model.model.layers):
        # LayerNorm in
        hooks.append(layer.input_layernorm.register_forward_pre_hook(make_pre_hook("LayerNorm_in_" + str(i))))
        hooks.append(layer.input_layernorm.register_forward_hook(make_post_hook("LayerNorm_in_" + str(i))))
        
        # Quantizer1
        if hasattr(layer.self_attn, 'quantizer1'):
            hooks.append(layer.self_attn.quantizer1.register_forward_pre_hook(make_pre_hook("Quant1_" + str(i))))
            hooks.append(layer.self_attn.quantizer1.register_forward_hook(make_post_hook("Quant1_" + str(i))))
        
        # Q, K, V proj
        hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(make_pre_hook("Q_proj_" + str(i))))
        hooks.append(layer.self_attn.q_proj.register_forward_hook(make_post_hook("Q_proj_" + str(i))))
        hooks.append(layer.self_attn.k_proj.register_forward_pre_hook(make_pre_hook("K_proj_" + str(i))))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(make_post_hook("K_proj_" + str(i))))
        hooks.append(layer.self_attn.v_proj.register_forward_pre_hook(make_pre_hook("V_proj_" + str(i))))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make_post_hook("V_proj_" + str(i))))
        
        # O proj
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_pre_hook("O_proj_" + str(i))))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(make_post_hook("O_proj_" + str(i))))
        
        # Quantizer2
        if hasattr(layer.self_attn, 'quantizer2'):
            hooks.append(layer.self_attn.quantizer2.register_forward_pre_hook(make_pre_hook("Quant2_" + str(i))))
            hooks.append(layer.self_attn.quantizer2.register_forward_hook(make_post_hook("Quant2_" + str(i))))
        
        # LayerNorm post
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(make_pre_hook("LayerNorm_post_" + str(i))))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(make_post_hook("LayerNorm_post_" + str(i))))
        
        # MLP
        hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(make_pre_hook("Gate_proj_" + str(i))))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_post_hook("Gate_proj_" + str(i))))
        hooks.append(layer.mlp.up_proj.register_forward_pre_hook(make_pre_hook("Up_proj_" + str(i))))
        hooks.append(layer.mlp.up_proj.register_forward_hook(make_post_hook("Up_proj_" + str(i))))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_pre_hook("Down_proj_" + str(i))))
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_post_hook("Down_proj_" + str(i))))
    
    # Final norm 和 lm_head
    hooks.append(model.model.norm.register_forward_pre_hook(make_pre_hook("Final_norm")))
    hooks.append(model.model.norm.register_forward_hook(make_post_hook("Final_norm")))
    hooks.append(model.lm_head.register_forward_pre_hook(make_pre_hook("LM_head")))
    hooks.append(model.lm_head.register_forward_hook(make_post_hook("LM_head")))
    
    # 测试
    print("\n开始性能测试...")
    n_iter = 5
    all_times = []
    
    timer.reset()
    with torch.no_grad():
        for iter_idx in range(n_iter):
            iter_start = time.perf_counter()
            
            timer.start("total")
            out = model(input_ids, use_cache=True)
            past_kv = out.past_key_values
            
            for step in range(gen_tokens):
                next_token = out.logits[:, -1:].argmax(dim=-1)
                out = model(next_token, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
            
            timer.stop()
            iter_time = (time.perf_counter() - iter_start) * 1000
            all_times.append(iter_time)
            print("  第 " + str(iter_idx + 1) + " 次: {:.2f} ms".format(iter_time))
            
            torch.cuda.synchronize()
    
    print("\n平均生成 " + str(gen_tokens) + " token 时间: {:.2f} ms".format(sum(all_times)/len(all_times)))
    
    quant_times, quant_total = timer.report(prefix="[量化] ")
    
    # 清理
    for h in hooks:
        h.remove()
    
    return quant_times, quant_total


def compare_results(fp16_times, fp16_total, quant_times, quant_total):
    """对比 FP16 和量化模型的结果"""
    print("\n" + "=" * 80)
    print("FP16 vs 量化模型 性能对比")
    print("=" * 80)
    
    # 所有组件名称
    all_components = set(fp16_times.keys()) | set(quant_times.keys())
    all_components.discard("total")
    
    print("\n{:<25} {:>15} {:>15} {:>15}".format(
        "组件类型", "FP16 (ms)", "量化 (ms)", "加速比"
    ))
    print("-" * 70)
    
    for comp in sorted(all_components):
        fp16_t = fp16_times.get(comp, 0)
        quant_t = quant_times.get(comp, 0)
        
        if fp16_t > 0 and quant_t > 0:
            speedup = fp16_t / quant_t
            print("{:<25} {:>15.2f} {:>15.2f} {:>15.2f}x".format(comp, fp16_t, quant_t, speedup))
        elif fp16_t > 0:
            print("{:<25} {:>15.2f} {:>15} {:>15}".format(comp, fp16_t, "N/A", "-"))
        else:
            print("{:<25} {:>15} {:>15.2f} {:>15}".format(comp, "N/A", quant_t, "-"))
    
    print("-" * 70)
    speedup_total = fp16_total / quant_total if quant_total > 0 else 0
    print("{:<25} {:>15.2f} {:>15.2f} {:>15.2f}x".format("总计", fp16_total, quant_total, speedup_total))
    
    print("\n" + "=" * 80)
    print("按占比对比")
    print("=" * 80)
    
    print("\n{:<25} {:>12} {:>12}".format("组件类型", "FP16 占比", "量化 占比"))
    print("-" * 50)
    
    for comp in sorted(all_components):
        fp16_t = fp16_times.get(comp, 0)
        quant_t = quant_times.get(comp, 0)
        
        fp16_pct = fp16_t / fp16_total * 100 if fp16_total > 0 else 0
        quant_pct = quant_t / quant_total * 100 if quant_total > 0 else 0
        
        print("{:<25} {:>11.1f}% {:>11.1f}%".format(comp, fp16_pct, quant_pct))
    
    print("-" * 50)
    print("{:<25} {:>11.1f}% {:>11.1f}%".format("总计", 100.0, 100.0))


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FLH LLaMA-2-7B W4A4 量化模型 vs 原始 FP16 模型 性能对比")
    print("=" * 80)
    
    # 测试配置
    batch_size = 8
    seq_len = 128
    gen_tokens = 8
    
    print("\n测试配置:")
    print("  batch_size: " + str(batch_size))
    print("  seq_len: " + str(seq_len))
    print("  gen_tokens: " + str(gen_tokens))
    
    # 测试 FP16 模型
    fp16_times, fp16_total = test_fp16_model(batch_size, seq_len, gen_tokens)
    
    # 测试量化模型
    quant_times, quant_total = test_quantized_model(batch_size, seq_len, gen_tokens)
    
    # 对比结果
    compare_results(fp16_times, fp16_total, quant_times, quant_total)
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)

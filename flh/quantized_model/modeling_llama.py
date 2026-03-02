import gc
import flh
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.utils import logging
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaFlashAttention2, apply_rotary_pos_emb, LlamaMLP
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Tuple, List, Dict

logger = logging.get_logger(__name__)

ALL_LAYERNORM_LAYERS = [flh.nn.RMSNorm]


def _collect_calibration_inputs(model, dataloader, device, nsamples, seqlen):
    """
    收集原始模型的校准输入（用于未融合的模型）
    """
    model.eval()
    model.config.use_cache = False
    if not hasattr(model, "seqlen"):
        model.seqlen = seqlen
    layers = model.model.layers
    dtype = next(model.parameters()).dtype

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    cache = {"i": 0, "attention_mask": None, "position_ids": None}
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            with torch.no_grad():
                model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    sequential = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]

    collected = {}
    outs = torch.zeros_like(inps)

    for i in range(len(layers)):
        layer = layers[i].to(device)

        def make_hook(name):
            inputs_list = []
            def hook(module, inp, out):
                inputs_list.append(inp[0].detach().cpu())
            return hook, inputs_list

        hooks = []
        inputs_dict = {}
        for names in sequential:
            for name in names:
                module = layer
                for part in name.split("."):
                    module = getattr(module, part)
                hook, inputs_list = make_hook(name)
                inputs_dict[name] = inputs_list
                hooks.append(module.register_forward_hook(hook))

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        for h in hooks:
            h.remove()

        collected[i] = {k: v for k, v in inputs_dict.items()}
        inps, outs = outs, inps
        layers[i] = layer.cpu()
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = True
    return collected


def _collect_calibration_inputs_fused(model, dataloader, device, nsamples, seqlen):
    """
    收集融合后模型的校准输入（考虑RMSNorm权重融合的影响）
    """
    model.eval()
    model.config.use_cache = False
    if not hasattr(model, "seqlen"):
        model.seqlen = seqlen
    layers = model.model.layers
    dtype = next(model.parameters()).dtype

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    cache = {"i": 0, "attention_mask": None, "position_ids": None}
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            with torch.no_grad():
                model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    # 对于融合后的模型，我们需要模拟融合后的计算流
    sequential = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],  # 这些已经融合了input_layernorm
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],  # 这些已经融合了post_attention_layernorm
        ["mlp.down_proj"],
    ]

    collected = {}
    outs = torch.zeros_like(inps)

    for i in range(len(layers)):
        layer = layers[i].to(device)
        
        # 获取原始的layernorm权重用于模拟融合后的输入
        input_norm_weight = layer.input_layernorm.weight.data
        post_norm_weight = layer.post_attention_layernorm.weight.data

        def make_fused_hook(name, norm_weight):
            """创建考虑权重融合和Hadamard变换的hook"""
            inputs_list = []
            def hook(module, inp, out):
                # 对于融合了norm权重的层，我们需要模拟融合后的输入
                original_inp = inp[0].detach()
                
                # 应用RMSNorm但权重为融合的权重
                if name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
                    # Hook获取的original_inp已经是 RMSNorm(x) * input_norm_weight
                    # 融合后的模型需要的是 RMSNorm(x)，所以要除以input_norm_weight
                    input_dtype = original_inp.dtype
                    x = original_inp.to(torch.float32) if original_inp.dtype == torch.float16 else original_inp
                    norm_w = norm_weight.to(torch.float32) if norm_weight.dtype == torch.float16 else norm_weight
                    
                    # 除以原始的norm权重，得到纯RMSNorm结果
                    normalized = x / norm_w.unsqueeze(0).unsqueeze(0)
                    
                    inputs_list.append(normalized.to(input_dtype).cpu())
                elif name in ["mlp.gate_proj", "mlp.up_proj"]:
                    # Hook获取的original_inp已经是 RMSNorm(x) * post_norm_weight
                    # 融合后的模型需要的是 RMSNorm(x)，所以要除以post_norm_weight
                    input_dtype = original_inp.dtype
                    x = original_inp.to(torch.float32) if original_inp.dtype == torch.float16 else original_inp
                    norm_w = norm_weight.to(torch.float32) if norm_weight.dtype == torch.float16 else norm_weight
                    
                    # 除以原始的norm权重，得到纯RMSNorm结果
                    normalized = x / norm_w.unsqueeze(0).unsqueeze(0)
                    
                    inputs_list.append(normalized.to(input_dtype).cpu())
                else:
                    # 其他层没有融合LayerNorm权重，直接使用原始输入
                    # Hadamard变换由FLHGPTQ内部处理
                    inputs_list.append(original_inp.cpu())
            return hook, inputs_list

        hooks = []
        inputs_dict = {}
        for names in sequential:
            for name in names:
                module = layer
                for part in name.split("."):
                    module = getattr(module, part)
                
                # 根据层的类型选择合适的norm权重
                if name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
                    hook, inputs_list = make_fused_hook(name, input_norm_weight)
                elif name in ["mlp.gate_proj", "mlp.up_proj"]:
                    hook, inputs_list = make_fused_hook(name, post_norm_weight)
                else:
                    hook, inputs_list = make_fused_hook(name, None)
                
                inputs_dict[name] = inputs_list
                hooks.append(module.register_forward_hook(hook))

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        for h in hooks:
            h.remove()

        collected[i] = {k: v for k, v in inputs_dict.items()}
        inps, outs = outs, inps
        layers[i] = layer.cpu()
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = True
    return collected


def get_calibration_dataloader(model_name_or_path, nsamples=32, seqlen=512, seed=0):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import random
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    random.seed(seed)
    dataloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        inp = trainenc.input_ids[:, i:i + seqlen]
        dataloader.append((inp,))
    return dataloader

class FLH_LlamaConfig(LlamaConfig):
    model_type = "flh_llama"

class FLH_FP16LlamaAttention(LlamaFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = torch.nn.Identity()
        self.o_proj_hadamard = torch.nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        hidden_states = self.quantizer(hidden_states)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update KV cache using transformers' standard interface
        if past_key_value is not None:
            # DynamicCache expects (batch, num_heads, seq_len, head_dim) format
            # but we have (batch, seq_len, num_heads, head_dim) from Flash Attention
            # So we need to transpose before and after cache update
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # # Transpose to cache format: (batch, num_heads, seq_len, head_dim)
            # key_states_cache = key_states.transpose(1, 2)
            # value_states_cache = value_states.transpose(1, 2)
            
            # # Update cache
            # key_states_cache, value_states_cache = past_key_value.update(
            #     key_states_cache, value_states_cache, self.layer_idx, cache_kwargs
            # )
            
            # # Transpose back to Flash Attention format: (batch, seq_len, num_heads, head_dim)
            # key_states = key_states_cache.transpose(1, 2)
            # value_states = value_states_cache.transpose(1, 2)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        dropout_rate = self.attention_dropout if self.training else 0.0

        # Use Flash Attention 2
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.quantizer(attn_output)
        attn_output = self.o_proj(attn_output)
        return attn_output, None if not output_attentions else None, past_key_value
    
class FLH_LlamaAttention(FLH_FP16LlamaAttention):
    def __init__(self, *args, act_bits=16, act_group_size=-1, act_sym=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = flh.nn.ActQuantizer(bits=act_bits, group_size=act_group_size, sym=act_sym)
        
class FLH_LlamaMLP(LlamaMLP):
    def __init__(self, *args, act_bits=16, act_group_size=-1, act_sym=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = flh.nn.ActQuantizer(bits=act_bits, group_size=act_group_size, sym=act_sym)
        
    def forward(self, x):
        x = self.quantizer(x)
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = self.quantizer(x)
        return self.down_proj(x)
    
class FLH_FP16LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = FLH_FP16LlamaAttention(config=config, layer_idx=layer_idx)
    
    @classmethod
    def from_float(cls, float_model, target_device="cuda"):
        config = float_model.config
        config._attn_implementation = "flash_attention_2"
        
        dtype_old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        
        print("  Creating FLH model structure on CPU...")
        flh_model = cls(config)
        
        torch.set_default_dtype(dtype_old)
        flh_model = flh_model.half().cpu()
        
        print("Copying weights from original model to FLH model (on CPU)...")
        
        print("  - Copying embedding and lm_head...")
        flh_model.model.embed_tokens.load_state_dict(float_model.model.embed_tokens.state_dict())
        flh_model.lm_head.load_state_dict(float_model.lm_head.state_dict())
        flh_model.model.norm.load_state_dict(float_model.model.norm.state_dict())
        
        num_layers = len(flh_model.model.layers)
        print(f"  - Copying {num_layers} transformer layers...")
        for layer_idx, (flh_layer, float_layer) in enumerate(tqdm(
            zip(flh_model.model.layers, float_model.model.layers),
            total=num_layers,
            desc="  Copying layers"
        )):
            flh_layer.self_attn.q_proj.load_state_dict(float_layer.self_attn.q_proj.state_dict())
            flh_layer.self_attn.k_proj.load_state_dict(float_layer.self_attn.k_proj.state_dict())
            flh_layer.self_attn.v_proj.load_state_dict(float_layer.self_attn.v_proj.state_dict())
            flh_layer.self_attn.o_proj.load_state_dict(float_layer.self_attn.o_proj.state_dict())
            
            flh_layer.mlp.gate_proj.load_state_dict(float_layer.mlp.gate_proj.state_dict())
            flh_layer.mlp.up_proj.load_state_dict(float_layer.mlp.up_proj.state_dict())
            flh_layer.mlp.down_proj.load_state_dict(float_layer.mlp.down_proj.state_dict())
            
            flh_layer.input_layernorm.load_state_dict(float_layer.input_layernorm.state_dict())
            flh_layer.post_attention_layernorm.load_state_dict(float_layer.post_attention_layernorm.state_dict())
        
        print("✓ Weight copying completed!")
        
        if target_device != "cpu":
            print(f"  Moving model to {target_device}...")
            flh_model = flh_model.to(device=target_device)
            print(f"✓ Model successfully moved to {target_device}!")
        
        return flh_model

class FLH_LlamaForCausalLM(FLH_FP16LlamaForCausalLM):
    def __init__(self, config, weight_bits=4, weight_group_size=128, act_bits=16, act_group_size=-1):
        LlamaForCausalLM.__init__(self, config)
        assert config._attn_implementation == "flash_attention_2"
        
        self.weight_bits = weight_bits
        self.weight_group_size = weight_group_size
        self.act_bits = act_bits
        self.act_group_size = act_group_size
        
        self.model.norm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        print(f"Initializing FLH_LlamaForCausalLM with:")
        print(f"  Weight quantization: {weight_bits}-bit, group_size={weight_group_size}")
        print(f"  Activation quantization: {act_bits}-bit, group_size={act_group_size}")
        
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = FLH_LlamaAttention(
                config=config, 
                layer_idx=layer_idx,
                act_bits=act_bits,
                act_group_size=act_group_size
            )
            layer.input_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = FLH_LlamaMLP(
                config=config,
                act_bits=act_bits,
                act_group_size=act_group_size
            )
    
    @classmethod
    def from_float(cls, float_model, target_device="cuda", weight_bits=4, weight_group_size=128, act_bits=4, act_group_size=128, weight_sym=False, act_sym=True, save_quantized_path=None, use_gptq=False, calibration_dataloader=None, gptq_nsamples=128, gptq_percdamp=0.01, gptq_actorder=False):
        """
        Convert a standard LlamaForCausalLM model to FLH_LlamaForCausalLM with quantization.
        
        Args:
            float_model: A LlamaForCausalLM model instance (can be on CPU or GPU)
            target_device: Target device for the converted model (default: "cuda")
            weight_bits: Number of bits for weight quantization (default: 4)
            weight_group_size: Group size for weight quantization (default: 128)
            act_bits: Number of bits for activation quantization (default: 16, no quant)
            act_group_size: Group size for activation quantization (default: -1, per-channel)
            weight_sym: Whether to use symmetric quantization for weights (default: False)
            act_sym: Whether to use symmetric quantization for activations (default: True)
            save_quantized_path: Optional path to save quantized model before moving to GPU (default: None)
            use_gptq: Use GPTQ for weight quantization (higher accuracy, requires calibration)
            calibration_dataloader: DataLoader for GPTQ calibration (required when use_gptq=True)
            gptq_nsamples: Number of calibration samples for GPTQ (default: 128)
            gptq_percdamp: Damping factor for GPTQ (default: 0.01)
            gptq_actorder: Use activation order for GPTQ (default: False)
            
        Returns:
            FLH_LlamaForCausalLM model with quantized weights in half precision
        """
        # Create a new FLH model with the same config on CPU first
        config = float_model.config
        config._attn_implementation = "flash_attention_2"
        
        # Set default dtype to half for model initialization
        dtype_old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        
        print("  Creating FLH quantized model structure on CPU...")
        print(f"  Quantization config: W{weight_bits}G{weight_group_size}{'_sym' if weight_sym else '_asym'} / A{act_bits}G{act_group_size}{'_sym' if act_sym else '_asym'}")
        
        # Create model WITHOUT calling __init__ to avoid premature quantization
        flh_model = cls.__new__(cls)
        LlamaForCausalLM.__init__(flh_model, config)
        flh_model.weight_bits = weight_bits
        flh_model.weight_group_size = weight_group_size
        flh_model.act_bits = act_bits
        flh_model.act_group_size = act_group_size
        flh_model.weight_sym = weight_sym
        flh_model.act_sym = act_sym
        
        torch.set_default_dtype(dtype_old)
        
        # Ensure model is in half precision (FP16) and on CPU
        flh_model = flh_model.half().cpu()
        flh_model.model.embed_tokens = flh_model.model.embed_tokens.half()
        flh_model.lm_head = flh_model.lm_head.half()
        
        print("Copying and quantizing weights from original model to FLH model (on CPU, FP16)...")
        
        # Copy embedding and lm_head (keep in half, no quantization)
        print("  - Copying embedding and lm_head...")
        flh_model.model.embed_tokens.load_state_dict(float_model.model.embed_tokens.state_dict())
        flh_model.lm_head.load_state_dict(float_model.lm_head.state_dict())
        
        # Replace and copy norm layer
        flh_model.model.norm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        flh_model.model.norm.weight.data.copy_(float_model.model.norm.weight.data)
        
        num_layers = len(flh_model.model.layers)
        seqlen = getattr(float_model, "seqlen", 2048)

        def _quantize_linear(float_linear, use_gptq_layer, gptq_data):
            in_f, out_f = float_linear.in_features, float_linear.out_features
            bias = float_linear.bias is not None
            dtype = float_linear.weight.dtype
            cal_device = target_device if target_device != "cpu" else "cuda" if torch.cuda.is_available() else "cpu"

            if use_gptq_layer and gptq_data is not None and len(gptq_data) > 0:
                tmp_linear = nn.Linear(in_f, out_f, bias=bias, device=cal_device, dtype=dtype)
                tmp_linear.weight.data.copy_(float_linear.weight.data.to(cal_device))
                if bias:
                    tmp_linear.bias.data.copy_(float_linear.bias.data.to(cal_device))

                # FLHGPTQ内部会正确处理Hadamard变换
                gptq = flh.nn.FLHGPTQ(tmp_linear)
                for inp in gptq_data:
                    if inp is not None and inp.numel() > 0:
                        gptq.add_batch(inp.to(cal_device), None, group_size=weight_group_size if weight_group_size > 0 else -1)
                gptq.fasterquant(
                    group_size=weight_group_size,
                    blocksize=128,
                    percdamp=gptq_percdamp,
                    actorder=gptq_actorder,
                    sym=weight_sym,
                    bits=weight_bits,
                )
                gptq.free()

                # 直接使用GPTQ的结果，权重已经被正确量化
                flh_linear = flh.nn.LinearFLH(in_f, out_f, bias=bias, dtype=dtype, device="cpu")
                flh_linear.weight.copy_(tmp_linear.weight.data.cpu())
                if bias:
                    flh_linear.bias.copy_(tmp_linear.bias.data.cpu())
                del tmp_linear, gptq
            else:
                flh_linear = flh.nn.LinearFLH.from_float(
                    float_linear,
                    weight_bits=weight_bits,
                    weight_group_size=weight_group_size,
                    weight_sym=weight_sym
                )
            return flh_linear

        calibration_data = None
        if use_gptq and calibration_dataloader is not None:
            print("  - GPTQ calibration (collecting activations with fused LayerNorm)...")
            cal_device = target_device if target_device != "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
            cal_seqlen = calibration_dataloader[0][0].shape[1] if calibration_dataloader else seqlen
            # 使用融合后的校准函数，考虑RMSNorm权重融合的影响
            calibration_data = _collect_calibration_inputs_fused(
                float_model, calibration_dataloader, cal_device, gptq_nsamples, cal_seqlen
            )
            torch.cuda.empty_cache()
            print(f"  - Fused calibration complete, quantizing {num_layers} layers with GPTQ...")
        elif use_gptq:
            print("  - Warning: use_gptq=True but no calibration_dataloader, falling back to RTN quantization")

        print(f"  - Copying and quantizing {num_layers} transformer layers...")
        for layer_idx, (flh_layer, float_layer) in enumerate(tqdm(
            zip(flh_model.model.layers, float_model.model.layers),
            total=num_layers,
            desc="  Quantizing layers"
        )):
            flh_layer.self_attn = FLH_LlamaAttention(
                config=config,
                layer_idx=layer_idx,
                act_bits=act_bits,
                act_group_size=act_group_size,
                act_sym=act_sym
            )

            cal = calibration_data.get(layer_idx, {}) if calibration_data else {}
            qkv_inps = cal.get("self_attn.q_proj", []) or cal.get("self_attn.k_proj", []) or cal.get("self_attn.v_proj", [])

            # 获取input_layernorm权重用于QKV权重预乘
            input_norm_weight = float_layer.input_layernorm.weight.data
            
            # 为QKV权重应用input_layernorm权重预乘
            def _quantize_qkv_linear(float_linear, use_gptq_layer, gptq_data, norm_weight):
                # 创建临时线性层，权重已预乘norm权重
                temp_linear = torch.nn.Linear(
                    float_linear.in_features, 
                    float_linear.out_features, 
                    bias=float_linear.bias is not None,
                    device=float_linear.weight.device,
                    dtype=float_linear.weight.dtype
                )
                
                # 权重预乘input_layernorm权重 (W * diag(norm_weight))
                temp_linear.weight.data = float_linear.weight.data * norm_weight.unsqueeze(0)
                if float_linear.bias is not None:
                    temp_linear.bias.data = float_layer.bias.data.clone()
                
                # 使用预乘后的权重进行量化
                return _quantize_linear(temp_linear, use_gptq_layer, gptq_data)

            flh_layer.self_attn.q_proj = _quantize_qkv_linear(float_layer.self_attn.q_proj, use_gptq, qkv_inps, input_norm_weight)
            flh_layer.self_attn.k_proj = _quantize_qkv_linear(float_layer.self_attn.k_proj, use_gptq, qkv_inps, input_norm_weight)
            flh_layer.self_attn.v_proj = _quantize_qkv_linear(float_layer.self_attn.v_proj, use_gptq, qkv_inps, input_norm_weight)
            flh_layer.self_attn.o_proj = _quantize_linear(float_layer.self_attn.o_proj, use_gptq, cal.get("self_attn.o_proj", []))

            # 获取post_attention_layernorm权重用于MLP gate/up权重预乘
            post_norm_weight = float_layer.post_attention_layernorm.weight.data
            
            # 为MLP gate/up权重应用post_attention_layernorm权重预乘
            def _quantize_mlp_linear(float_linear, use_gptq_layer, gptq_data, norm_weight):
                # 创建临时线性层，权重已预乘norm权重
                temp_linear = torch.nn.Linear(
                    float_linear.in_features, 
                    float_linear.out_features, 
                    bias=float_linear.bias is not None,
                    device=float_linear.weight.device,
                    dtype=float_linear.weight.dtype
                )
                
                # 权重预乘post_attention_layernorm权重 (W * diag(norm_weight))
                temp_linear.weight.data = float_linear.weight.data * norm_weight.unsqueeze(0)
                if float_linear.bias is not None:
                    temp_linear.bias.data = float_linear.bias.data.clone()
                
                # 使用预乘后的权重进行量化
                return _quantize_linear(temp_linear, use_gptq_layer, gptq_data)

            flh_layer.mlp = FLH_LlamaMLP(
                config=config,
                act_bits=act_bits,
                act_group_size=act_group_size,
                act_sym=act_sym
            )
            gate_up_inps = cal.get("mlp.gate_proj", []) or cal.get("mlp.up_proj", [])

            # 对gate和up投影应用post_attention_layernorm权重融合
            flh_layer.mlp.gate_proj = _quantize_mlp_linear(float_layer.mlp.gate_proj, use_gptq, gate_up_inps, post_norm_weight)
            flh_layer.mlp.up_proj = _quantize_mlp_linear(float_layer.mlp.up_proj, use_gptq, gate_up_inps, post_norm_weight)
            # down投影不需要融合，因为它接收的是激活后的输出
            flh_layer.mlp.down_proj = _quantize_linear(float_layer.mlp.down_proj, use_gptq, cal.get("mlp.down_proj", []))

            # 创建input_layernorm，保持归一化但权重设为1（因为已预乘到QKV权重中）
            flh_layer.input_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            flh_layer.input_layernorm.weight.data.fill_(1.0)  # 权重设为1
            
            # 创建post_attention_layernorm，保持归一化但权重设为1（因为已预乘到gate/up权重中）
            flh_layer.post_attention_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            flh_layer.post_attention_layernorm.weight.data.fill_(1.0)  # 权重设为1
        
        print("\n✓ Weight copying and quantization completed!")
        
        # 确保模型是FP16格式
        flh_model = flh_model.half()
        
        # ⭐ 优化：如果需要保存，先在CPU上保存（更快，节省GPU内存，FP16格式）
        if save_quantized_path:
            print(f"\n  Saving quantized model to {save_quantized_path} (on CPU, FP16 format)...")
            flh_model.save_quantized(save_quantized_path)
            print(f"  ✓ Model saved! You can load it next time with: --load-quantized {save_quantized_path}")
        
        # 然后再移到目标设备（如果需要）
        if target_device != "cpu":
            print(f"\n  Moving quantized model to {target_device}...")
            flh_model = flh_model.to(device=target_device)
            print(f"✓ Model successfully moved to {target_device} (FP16)!")
        
        return flh_model
    
    def save_quantized(self, save_directory):
        import os
        os.makedirs(save_directory, exist_ok=True)

        try:
            stat = os.statvfs(os.path.dirname(os.path.abspath(save_directory)))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            if free_gb < 15:
                print(f"  ⚠️ Warning: Low disk space ({free_gb:.1f} GB free). LLaMA-7B needs ~14GB. Save may fail.")
        except (OSError, AttributeError):
            pass

        print(f"Saving quantized model to {save_directory} (FP16 format)...")
        
        device = next(self.parameters()).device
        if device.type != "cpu":
            print(f"  ⚠️ Warning: Model is on {device}, moving to CPU for saving...")
            self.cpu()
        
        self.half()
        
        state_dict = self.state_dict()
        for key in state_dict:
            if state_dict[key].dtype == torch.float32:
                state_dict[key] = state_dict[key].half()
        
        model_path = os.path.join(save_directory, "model.pt")
        try:
            torch.save(state_dict, model_path, pickle_protocol=4)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if "file write failed" in err_msg or "unexpected pos" in err_msg:
                raise RuntimeError(
                    f"Failed to save model to {save_directory}. "
                    "Likely causes: disk full, quota exceeded, or permission issues. "
                    "Check disk space with 'df -h' and try a path with more space (LLaMA-7B needs ~14GB). "
                    f"Original error: {e}"
                ) from e
            raise

        print(f"  ✓ Weights saved (FP16, CPU)")
        
        self.config.save_pretrained(save_directory)
        
        quant_config = {
            "weight_bits": self.weight_bits,
            "weight_group_size": self.weight_group_size,
            "act_bits": self.act_bits,
            "act_group_size": self.act_group_size,
            "weight_sym": self.weight_sym,
            "act_sym": self.act_sym,
            "dtype": "float16",
        }
        import json
        with open(os.path.join(save_directory, "quantization_config.json"), "w") as f:
            json.dump(quant_config, f, indent=2)
        
        print(f"✓ Model saved to {save_directory} (FP16 format)")
    
    @classmethod
    def load_quantized(cls, load_directory, target_device="cuda"):
        import os
        import json
        from transformers import AutoConfig
        
        print(f"Loading quantized model from {load_directory} (FP16 format)...")
        
        config = AutoConfig.from_pretrained(load_directory)
        config._attn_implementation = "flash_attention_2"
        
        with open(os.path.join(load_directory, "quantization_config.json"), "r") as f:
            quant_config = json.load(f)
        
        weight_sym = quant_config.get("weight_sym", False)
        act_sym = quant_config.get("act_sym", True)
        
        print(f"  Quantization config: W{quant_config['weight_bits']}G{quant_config['weight_group_size']}{'_sym' if weight_sym else '_asym'} / "
              f"A{quant_config['act_bits']}G{quant_config['act_group_size']}{'_sym' if act_sym else '_asym'}")
        
        if quant_config.get("dtype", "float16") != "float16":
            print(f"  ⚠ Warning: Model was saved with dtype={quant_config.get('dtype')}, converting to float16")
        
        print(f"  Loading model weights to CPU...")
        
        load_kwargs = {"map_location": "cpu"}
        try:
            state_dict = torch.load(
                os.path.join(load_directory, "model.pt"), 
                mmap=True,
                **load_kwargs
            )
        except TypeError:
            state_dict = torch.load(
                os.path.join(load_directory, "model.pt"),
                **load_kwargs
            )
        
        for key in list(state_dict.keys()):
            if state_dict[key].dtype == torch.float32:
                print(f"  ⚠ Converting {key} from float32 to float16")
                state_dict[key] = state_dict[key].half()
        
        print("  Creating model structure...")
        dtype_old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        
        use_meta = hasattr(torch, 'device') and 'meta' in dir(torch)
        if use_meta:
            try:
                with torch.device("meta"):
                    flh_model = cls.__new__(cls)
                    LlamaForCausalLM.__init__(flh_model, config)
            except:
                use_meta = False
        
        if not use_meta:
            flh_model = cls.__new__(cls)
            LlamaForCausalLM.__init__(flh_model, config)
        
        flh_model.weight_bits = quant_config["weight_bits"]
        flh_model.weight_group_size = quant_config["weight_group_size"]
        flh_model.act_bits = quant_config["act_bits"]
        flh_model.act_group_size = quant_config["act_group_size"]
        flh_model.weight_sym = weight_sym
        flh_model.act_sym = act_sym
        
        torch.set_default_dtype(dtype_old)
        
        print("  Replacing with FLH components...")
        flh_model.model.norm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        num_layers = len(flh_model.model.layers)
        for layer_idx in range(num_layers):
            layer = flh_model.model.layers[layer_idx]
            layer.self_attn = FLH_LlamaAttention(
                config=config,
                layer_idx=layer_idx,
                act_bits=quant_config["act_bits"],
                act_group_size=quant_config["act_group_size"],
                act_sym=act_sym
            )
            layer.input_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = FLH_LlamaMLP(
                config=config,
                act_bits=quant_config["act_bits"],
                act_group_size=quant_config["act_group_size"],
                act_sym=act_sym
            )
        
        print("  Loading weights to model (on CPU)...")
        try:
            flh_model.load_state_dict(state_dict, assign=True, strict=False)
        except TypeError:
            flh_model.load_state_dict(state_dict, strict=False)
        
        flh_model.half()
        
        if target_device != "cpu":
            print(f"  Moving complete model to {target_device}...")
            flh_model = flh_model.to(device=target_device)
            print(f"  ✓ All modules moved to {target_device}")
        
        flh_model.eval()
        print(f"✓ Model loaded successfully on {target_device} (FP16)")
        
        return flh_model
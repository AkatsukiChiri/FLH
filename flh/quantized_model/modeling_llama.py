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


def _collect_calibration_inputs_flh(model, dataloader, device, nsamples, seqlen, act_group_size=128):
    """
    使用FLH模型收集校准输入（LayerNorm权重已融合，结构完全匹配）。
    对 o_proj 和 down_proj 额外收集线性层输出的 Hadamard 变换，供双侧 Hadamard GPTQ 使用。
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

    # FLH模型的结构：LayerNorm权重已融合，直接收集各层输入
    sequential = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],  # 已融合input_layernorm
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],  # 已融合post_attention_layernorm
        ["mlp.down_proj"],
    ]
    need_output_hadamard = {"self_attn.o_proj", "mlp.down_proj"}

    collected = {}  # 按层组织: {layer_idx: {name: [tensors]}}
    outs = torch.zeros_like(inps)
    gs = act_group_size if act_group_size and act_group_size > 0 else None

    for i in range(len(layers)):
        layer = layers[i].to(device)
        collected[i] = {}  # 为每一层创建字典

        def make_flh_hook(name, need_out_H):
            inputs_list = []
            out_H_list = [] if need_out_H else None
            def hook(module, inp, out):
                original_inp = inp[0].detach()
                inputs_list.append(original_inp.cpu())
                if need_out_H and out is not None:
                    out_detach = out.detach()
                    out_H = flh.nn.fast_hadamard_transform(out_detach, group_size=gs, normalize=True)
                    out_H_list.append(out_H.cpu())
            return hook, inputs_list, out_H_list

        hooks = []
        inputs_dict = {}
        out_H_dict = {}
        for names in sequential:
            for name in names:
                module = layer
                for part in name.split("."):
                    module = getattr(module, part)
                need_out_H = name in need_output_hadamard
                hook, inputs_list, out_H_list = make_flh_hook(name, need_out_H)
                inputs_dict[name] = inputs_list
                if need_out_H:
                    out_H_dict[name] = out_H_list
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

        for name in inputs_dict:
            collected[i][name] = inputs_dict[name]
        for name in out_H_dict:
            collected[i][name + "_out_H"] = out_H_dict[name]

        layers[i] = layer.cpu()
        inps, outs = outs, inps

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

    def _replace_with_flh_linear(self, weight_bits=4, weight_group_size=128, weight_sym=False,
                                   act_group_size=-1, dual_hadamard=False, no_hadamard=False, clip_ratio=1.0):
        """将普通 Linear 层替换为 LinearFLH（在 init 时调用）"""
        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        
        # 替换 q_proj (单侧 Hadamard)
        self.q_proj = flh.nn.LinearFLH(
            self.q_proj.in_features, self.q_proj.out_features,
            bias=self.q_proj.bias is not None,
            dtype=dtype, device=device,
            dual_hadamard=False,
            in_group_size=act_group_size,
            out_group_size=act_group_size,
            group_size=weight_group_size,
            no_hadamard=no_hadamard
        )
        
        # 替换 k_proj (单侧 Hadamard)
        self.k_proj = flh.nn.LinearFLH(
            self.k_proj.in_features, self.k_proj.out_features,
            bias=self.k_proj.bias is not None,
            dtype=dtype, device=device,
            dual_hadamard=False,
            in_group_size=act_group_size,
            out_group_size=act_group_size,
            group_size=weight_group_size,
            no_hadamard=no_hadamard
        )
        
        # 替换 v_proj (单侧 Hadamard)
        self.v_proj = flh.nn.LinearFLH(
            self.v_proj.in_features, self.v_proj.out_features,
            bias=self.v_proj.bias is not None,
            dtype=dtype, device=device,
            dual_hadamard=False,
            in_group_size=act_group_size,
            out_group_size=act_group_size,
            group_size=weight_group_size,
            no_hadamard=no_hadamard
        )
        
        # 替换 o_proj (双侧 Hadamard)
        self.o_proj = flh.nn.LinearFLH(
            self.o_proj.in_features, self.o_proj.out_features,
            bias=self.o_proj.bias is not None,
            dtype=dtype, device=device,
            dual_hadamard=dual_hadamard,
            in_group_size=act_group_size,
            out_group_size=act_group_size,
            group_size=weight_group_size,
            no_hadamard=no_hadamard
        )

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
    def __init__(self, *args, act_bits=16, act_group_size=-1, act_sym=True, 
                 weight_bits=4, weight_group_size=128, weight_sym=False, clip_ratio=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # 第一个量化器：仅量化，不进行 Hadamard 变换
        self.quantizer1 = flh.nn.ActQuantizer(bits=act_bits, group_size=act_group_size, sym=act_sym, use_hadamard=False, packed_output=True)
        # 第二个量化器：既量化又进行 Hadamard 变换
        self.quantizer2 = flh.nn.ActQuantizer(bits=act_bits, group_size=act_group_size, sym=act_sym, use_hadamard=False, packed_output=True)
        
        # 在 init 时将 q/k/v Linear 层替换为 LinearFLH（用于 save/load_quantized）
        # o_proj 单独创建，因为它使用双侧 Hadamard
        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        
        # 替换 q_proj (单侧 Hadamard)
        self.q_proj = flh.nn.LinearFLH(
            self.q_proj.in_features, self.q_proj.out_features,
            bias=self.q_proj.bias is not None,
            dtype=dtype, device=device,
            dual_hadamard=False,
            in_group_size=act_group_size,
            out_group_size=act_group_size,
            group_size=weight_group_size,
            no_hadamard=True
        )
        
        # 替换 k_proj (单侧 Hadamard)
        self.k_proj = flh.nn.LinearFLH(
            self.k_proj.in_features, self.k_proj.out_features,
            bias=self.k_proj.bias is not None,
            dtype=dtype, device=device,
            dual_hadamard=False,
            in_group_size=act_group_size,
            out_group_size=act_group_size,
            group_size=weight_group_size,
            no_hadamard=True
        )
        
        # 替换 v_proj (单侧 Hadamard)
        self.v_proj = flh.nn.LinearFLH(
            self.v_proj.in_features, self.v_proj.out_features,
            bias=self.v_proj.bias is not None,
            dtype=dtype, device=device,
            dual_hadamard=False,
            in_group_size=act_group_size,
            out_group_size=act_group_size,
            group_size=weight_group_size,
            no_hadamard=True
        )
        
        # o_proj 使用双侧 Hadamard
        self.o_proj = flh.nn.LinearFLH(
            self.o_proj.in_features, self.o_proj.out_features,
            bias=self.o_proj.bias is not None,
            dtype=dtype, device=device,
            dual_hadamard=True,
            in_group_size=act_group_size,
            out_group_size=act_group_size,
            group_size=weight_group_size,
            no_hadamard=True
        )
    
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

        scale, zp, q = self.quantizer1(hidden_states)
        
        query_states = self.q_proj(q, scale, zp, x_is_packed=True)
        key_states = self.k_proj(q, scale, zp, x_is_packed=True)
        value_states = self.v_proj(q, scale, zp, x_is_packed=True)

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
                target_dtype = self.q_proj.get_weight().dtype

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

        scale, zp, q = self.quantizer2(attn_output)
        attn_output = self.o_proj(q, scale, zp, x_is_packed=True)
        
        return attn_output, None if not output_attentions else None, past_key_value
    
    
class FLH_LlamaMLP(LlamaMLP):
    def __init__(self, *args, act_bits=16, act_group_size=-1, act_sym=True,
                 weight_bits=4, weight_group_size=128, weight_sym=False, clip_ratio=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # 第一个量化器：仅量化，不进行 Hadamard 变换
        self.quantizer1 = flh.nn.ActQuantizer(bits=act_bits, group_size=act_group_size, sym=act_sym, use_hadamard=False, packed_output=True)
        # 第二个量化器：既量化又进行 Hadamard 变换
        self.quantizer2 = flh.nn.ActQuantizer(bits=act_bits, group_size=act_group_size, sym=act_sym, use_hadamard=False, packed_output=True)
        
        # 在 init 时将 Linear 层替换为 LinearFLH（用于 save/load_quantized）
        device = self.gate_proj.weight.device
        dtype = self.gate_proj.weight.dtype
        
        # 替换 gate_proj (单侧 Hadamard)
        self.gate_proj = flh.nn.LinearFLH(
            self.gate_proj.in_features, self.gate_proj.out_features,
            bias=self.gate_proj.bias is not None,
            dtype=dtype, device=device,
            dual_hadamard=False,
            in_group_size=act_group_size,
            out_group_size=act_group_size,
            group_size=weight_group_size,
            no_hadamard=True
        )
        
        # 替换 up_proj (单侧 Hadamard)
        self.up_proj = flh.nn.LinearFLH(
            self.up_proj.in_features, self.up_proj.out_features,
            bias=self.up_proj.bias is not None,
            dtype=dtype, device=device,
            dual_hadamard=False,
            in_group_size=act_group_size,
            out_group_size=act_group_size,
            group_size=weight_group_size,
            no_hadamard=True
        )
        
        # 替换 down_proj (双侧 Hadamard)
        self.down_proj = flh.nn.LinearFLH(
            self.down_proj.in_features, self.down_proj.out_features,
            bias=self.down_proj.bias is not None,
            dtype=dtype, device=device,
            dual_hadamard=True,
            in_group_size=act_group_size,
            out_group_size=act_group_size,
            group_size=weight_group_size,
            no_hadamard=True
        )
        
    def forward(self, x):
        scale, zp, q = self.quantizer1(x)
        x = self.act_fn(self.gate_proj(q, scale, zp, x_is_packed=True)) * self.up_proj(q, scale, zp, x_is_packed=True)
        scale, zp, q = self.quantizer2(x)
        return self.down_proj(q, scale, zp, x_is_packed=True)
    
class FLH_FP16LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = FLH_FP16LlamaAttention(config=config, layer_idx=layer_idx)
    
    @classmethod
    def from_float(cls, float_model, target_device="cuda", fuse_layernorm=True):
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
        if fuse_layernorm:
            print(f"  - Copying and fusing LayerNorm weights in {num_layers} transformer layers...")
        else:
            print(f"  - Copying {num_layers} transformer layers...")
            
        for layer_idx, (flh_layer, float_layer) in enumerate(tqdm(
            zip(flh_model.model.layers, float_model.model.layers),
            total=num_layers,
            desc="  Copying layers"
        )):
            if fuse_layernorm:
                # 融合input_layernorm权重到QKV
                input_norm_weight = float_layer.input_layernorm.weight.data
                flh_layer.self_attn.q_proj.weight.data = float_layer.self_attn.q_proj.weight.data * input_norm_weight.unsqueeze(0)
                flh_layer.self_attn.k_proj.weight.data = float_layer.self_attn.k_proj.weight.data * input_norm_weight.unsqueeze(0)
                flh_layer.self_attn.v_proj.weight.data = float_layer.self_attn.v_proj.weight.data * input_norm_weight.unsqueeze(0)
                
                # 复制偏置（如果存在）
                if float_layer.self_attn.q_proj.bias is not None:
                    flh_layer.self_attn.q_proj.bias.data = float_layer.self_attn.q_proj.bias.data.clone()
                if float_layer.self_attn.k_proj.bias is not None:
                    flh_layer.self_attn.k_proj.bias.data = float_layer.self_attn.k_proj.bias.data.clone()
                if float_layer.self_attn.v_proj.bias is not None:
                    flh_layer.self_attn.v_proj.bias.data = float_layer.self_attn.v_proj.bias.data.clone()
                
                # o_proj不融合
                flh_layer.self_attn.o_proj.load_state_dict(float_layer.self_attn.o_proj.state_dict())
                
                # 融合post_attention_layernorm权重到MLP gate/up
                post_norm_weight = float_layer.post_attention_layernorm.weight.data
                flh_layer.mlp.gate_proj.weight.data = float_layer.mlp.gate_proj.weight.data * post_norm_weight.unsqueeze(0)
                flh_layer.mlp.up_proj.weight.data = float_layer.mlp.up_proj.weight.data * post_norm_weight.unsqueeze(0)
                
                # 复制偏置（如果存在）
                if float_layer.mlp.gate_proj.bias is not None:
                    flh_layer.mlp.gate_proj.bias.data = float_layer.mlp.gate_proj.bias.data.clone()
                if float_layer.mlp.up_proj.bias is not None:
                    flh_layer.mlp.up_proj.bias.data = float_layer.mlp.up_proj.bias.data.clone()
                
                # down_proj不融合
                flh_layer.mlp.down_proj.load_state_dict(float_layer.mlp.down_proj.state_dict())
                
                # 设置LayerNorm权重为1
                flh_layer.input_layernorm.weight.data.fill_(1.0)
                flh_layer.post_attention_layernorm.weight.data.fill_(1.0)
            else:
                # 不融合，直接复制
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
    def __init__(self, config, weight_bits=4, weight_group_size=128, act_bits=16, act_group_size=-1, weight_sym=False, clip_ratio=1.0):
        LlamaForCausalLM.__init__(self, config)
        assert config._attn_implementation == "flash_attention_2"
        
        self.weight_bits = weight_bits
        self.weight_group_size = weight_group_size
        self.act_bits = act_bits
        self.act_group_size = act_group_size
        self.weight_sym = weight_sym
        self.clip_ratio = clip_ratio
        
        self.model.norm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        print(f"Initializing FLH_LlamaForCausalLM with:")
        print(f"  Weight quantization: {weight_bits}-bit, group_size={weight_group_size}")
        print(f"  Activation quantization: {act_bits}-bit, group_size={act_group_size}")
        
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = FLH_LlamaAttention(
                config=config, 
                layer_idx=layer_idx,
                act_bits=act_bits,
                act_group_size=act_group_size,
                act_sym=True,
                weight_bits=weight_bits,
                weight_group_size=weight_group_size,
                weight_sym=weight_sym,
                clip_ratio=clip_ratio
            )
            layer.input_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = FLH_LlamaMLP(
                config=config,
                act_bits=act_bits,
                act_group_size=act_group_size,
                act_sym=True,
                weight_bits=weight_bits,
                weight_group_size=weight_group_size,
                weight_sym=weight_sym,
                clip_ratio=clip_ratio
            )
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, cache_position=None):
        # 1. embedding层之后进行分组hadamard变换
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        inputs_embeds = flh.nn.fast_hadamard_transform(inputs_embeds, group_size=self.act_group_size, normalize=True)
        
        # 调用父类的 forward 方法，但传入变换后的 inputs_embeds
        outputs = super().forward(
            input_ids=None,  # 不使用 input_ids，直接使用 inputs_embeds
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )
        
        # 4. 在输出之前进行分组hadamard变换
        # if hasattr(outputs, 'logits'):
        #     outputs.logits = flh.nn.fast_hadamard_transform(outputs.logits, group_size=self.act_group_size, normalize=True)
        
        return outputs
    
    @classmethod
    def from_float(cls, float_model, target_device="cuda", weight_bits=4, weight_group_size=128, act_bits=4, act_group_size=128, weight_sym=False, act_sym=True, save_quantized_path=None, use_gptq=False, calibration_dataloader=None, gptq_nsamples=128, gptq_percdamp=0.01, gptq_actorder=True, clip_ratio=0.99):
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
        print(f"  Quantization config: W{weight_bits}G{weight_group_size}{'_sym' if weight_sym else '_asym'} / A{act_bits}G{act_group_size}{'_sym' if act_sym else '_asym'} clip_ratio={clip_ratio}")
        
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

        calibration_data = None
        if use_gptq and calibration_dataloader is not None:
            print("  - GPTQ calibration (using FP16 FLH model as reference)...")
            cal_device = target_device if target_device != "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
            cal_seqlen = calibration_dataloader[0][0].shape[1] if calibration_dataloader else seqlen
            
            # 创建FP16 FLH模型作为校准参考（LayerNorm已融合，但未量化）
            print("    Creating FP16 FLH reference model for calibration...")
            fp16_flh_model = FLH_FP16LlamaForCausalLM.from_float(float_model, target_device="cpu", fuse_layernorm=True)
            
            # 使用FP16 FLH模型收集校准数据（结构完全匹配）
            calibration_data = _collect_calibration_inputs_flh(
                fp16_flh_model, calibration_dataloader, cal_device, gptq_nsamples, cal_seqlen,
                act_group_size=act_group_size
            )
            
            # 清理FP16模型
            del fp16_flh_model
            torch.cuda.empty_cache()
            print(f"  - FLH-based calibration complete, quantizing {num_layers} layers with GPTQ...")
        elif use_gptq:
            print("  - Warning: use_gptq=True but no calibration_dataloader, falling back to RTN quantization")

        print(f"  - Copying and quantizing {num_layers} transformer layers...")
        for layer_idx, (flh_layer, float_layer) in enumerate(tqdm(
            zip(flh_model.model.layers, float_model.model.layers),
            total=num_layers,
            desc="  Quantizing layers"
        )):
            cal_device = target_device if target_device != "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
            flh_layer.self_attn = FLH_LlamaAttention(
                config=config,
                layer_idx=layer_idx,
                act_bits=act_bits,
                act_group_size=act_group_size,
                act_sym=act_sym,
                weight_bits=weight_bits,
                weight_group_size=weight_group_size,
                weight_sym=weight_sym,
                clip_ratio=clip_ratio
            )

            cal = calibration_data.get(layer_idx, {}) if calibration_data else {}
            qkv_inps = cal.get("self_attn.q_proj", []) or cal.get("self_attn.k_proj", []) or cal.get("self_attn.v_proj", [])

            # 获取input_layernorm权重用于QKV权重预乘
            input_norm_weight = float_layer.input_layernorm.weight.data
            
            # 为QKV权重应用input_layernorm权重预乘
            def _quantize_qkv_linear_flh(float_linear, norm_weight, qkv_inps_here, use_gptq_here=False):
                if use_gptq_here:                    
                    tmp_linear = nn.Linear(float_linear.in_features, float_linear.out_features, bias=float_linear.bias is not None, device=cal_device, dtype=float_linear.weight.dtype)
                    tmp_linear.weight.data.copy_(float_linear.weight.data.to(cal_device) * norm_weight.unsqueeze(0).to(cal_device))
                    if tmp_linear.bias is not None:
                        tmp_linear.bias.data.copy_(float_linear.bias.data.to(cal_device))
                    
                    gptq = flh.nn.FLHGPTQ(tmp_linear, dual_hadamard=False)
                    
                    for inp in qkv_inps_here:
                        if inp is not None and inp.numel() > 0:
                            gptq.add_batch(inp.to(cal_device), None, group_size=weight_group_size if weight_group_size > 0 else -1)
                    gptq.fasterquant(group_size=weight_group_size, blocksize=128, percdamp=gptq_percdamp, actorder=gptq_actorder, sym=weight_sym, bits=weight_bits, dual_hadamard=False, clip_ratio=clip_ratio)
                    
                    gptq.free()
                    flh_linear = flh.nn.LinearFLH.from_float(
                        tmp_linear.cpu(),
                        weight_bits=weight_bits,
                        weight_group_size=weight_group_size,
                        weight_sym=weight_sym,
                        dual_hadamard=False,
                        in_group_size=act_group_size,
                        out_group_size=act_group_size,
                        clip_ratio=clip_ratio,
                        no_hadamard=True
                    )
                    del tmp_linear, gptq
                    return flh_linear
                else:
                    # 创建临时线性层，权重已预乘norm权重
                    temp_linear = torch.nn.Linear(
                        float_linear.in_features, 
                        float_linear.out_features, 
                        bias=float_linear.bias is not None,
                        device=float_linear.weight.device,
                        dtype=float_linear.weight.dtype
                    )
                    
                    # 步骤1: 权重预乘input_layernorm权重 (W * diag(norm_weight))
                    temp_linear.weight.data = float_linear.weight.data * norm_weight.unsqueeze(0)
                    if float_linear.bias is not None:
                        temp_linear.bias.data = float_linear.bias.data.clone()
                    
                    # 步骤2: 使用LinearFLH进行量化（单侧Hadamard）
                    return flh.nn.LinearFLH.from_float(
                        temp_linear,
                        weight_bits=weight_bits,
                        weight_group_size=weight_group_size,
                        weight_sym=weight_sym,
                        dual_hadamard=False,
                        in_group_size=act_group_size,
                        out_group_size=act_group_size,
                        clip_ratio=clip_ratio
                    )
            
            flh_layer.self_attn.q_proj = _quantize_qkv_linear_flh(float_layer.self_attn.q_proj, input_norm_weight, qkv_inps, use_gptq_here=use_gptq)
            flh_layer.self_attn.k_proj = _quantize_qkv_linear_flh(float_layer.self_attn.k_proj, input_norm_weight, qkv_inps, use_gptq_here=use_gptq)
            flh_layer.self_attn.v_proj = _quantize_qkv_linear_flh(float_layer.self_attn.v_proj, input_norm_weight, qkv_inps, use_gptq_here=use_gptq)
            
            # o_proj 使用双侧 Hadamard；做 GPTQ 时校准需用 H(线性层输出)
            o_proj_inps = cal.get("self_attn.o_proj", [])
            o_proj_out_H = cal.get("self_attn.o_proj_out_H", [])
            
            if use_gptq and o_proj_inps and len(o_proj_inps) > 0:
                tmp_linear = nn.Linear(float_layer.self_attn.o_proj.in_features, float_layer.self_attn.o_proj.out_features, bias=float_layer.self_attn.o_proj.bias is not None, device=cal_device, dtype=float_layer.self_attn.o_proj.weight.dtype)
                tmp_linear.weight.data.copy_(float_layer.self_attn.o_proj.weight.data.to(cal_device))
                if tmp_linear.bias is not None:
                    tmp_linear.bias.data.copy_(float_layer.self_attn.o_proj.bias.data.to(cal_device))
                gptq = flh.nn.FLHGPTQ(tmp_linear, dual_hadamard=True)
                for k, inp in enumerate(o_proj_inps):
                    out_H = o_proj_out_H[k] if k < len(o_proj_out_H) else None
                    if inp is not None and inp.numel() > 0:
                        # gptq.add_batch(inp.to(cal_device), out_H.to(cal_device) if out_H is not None else None, group_size=weight_group_size if weight_group_size > 0 else -1)
                        gptq.add_batch(inp.to(cal_device), None, group_size=weight_group_size if weight_group_size > 0 else -1)
                gptq.fasterquant(group_size=weight_group_size, blocksize=128, percdamp=gptq_percdamp, actorder=gptq_actorder, sym=weight_sym, bits=weight_bits, dual_hadamard=True, clip_ratio=clip_ratio)
                gptq.free()
                flh_layer.self_attn.o_proj = flh.nn.LinearFLH.from_float(
                    tmp_linear.cpu(),
                    weight_bits=weight_bits,
                    weight_group_size=weight_group_size,
                    weight_sym=weight_sym,
                    dual_hadamard=True,
                    in_group_size=act_group_size,
                    out_group_size=act_group_size,
                    clip_ratio=clip_ratio,
                    no_hadamard=True
                )
                del tmp_linear, gptq
            else:
                flh_layer.self_attn.o_proj = flh.nn.LinearFLH.from_float(
                    float_layer.self_attn.o_proj,
                    weight_bits=weight_bits,
                    weight_group_size=weight_group_size,
                    weight_sym=weight_sym,
                    dual_hadamard=True,
                    in_group_size=act_group_size,
                    out_group_size=act_group_size,
                    clip_ratio=clip_ratio
                )

            # 获取post_attention_layernorm权重用于MLP gate/up权重预乘
            post_norm_weight = float_layer.post_attention_layernorm.weight.data
            
            # 为MLP gate/up权重应用post_attention_layernorm权重预乘
            def _quantize_mlp_linear_flh(float_linear, norm_weight, mlp_inps_here, use_gptq_here=False):
                if use_gptq_here:
                    tmp_linear = nn.Linear(float_linear.in_features, float_linear.out_features, bias=float_linear.bias is not None, device=cal_device, dtype=float_linear.weight.dtype)
                    tmp_linear.weight.data.copy_(float_linear.weight.data.to(cal_device) * norm_weight.unsqueeze(0).to(cal_device))
                    if tmp_linear.bias is not None:
                        tmp_linear.bias.data.copy_(float_linear.bias.data.to(cal_device))
                    gptq = flh.nn.FLHGPTQ(tmp_linear, dual_hadamard=False)
                    for k, inp in enumerate(mlp_inps_here):
                        if inp is not None and inp.numel() > 0:
                            gptq.add_batch(inp.to(cal_device), None, group_size=weight_group_size if weight_group_size > 0 else -1)
                    gptq.fasterquant(group_size=weight_group_size, blocksize=128, percdamp=gptq_percdamp, actorder=gptq_actorder, sym=weight_sym, bits=weight_bits, dual_hadamard=False, clip_ratio=clip_ratio)
                    gptq.free()
                    flh_linear = flh.nn.LinearFLH.from_float(
                        tmp_linear.cpu(),
                        weight_bits=weight_bits,
                        weight_group_size=weight_group_size,
                        weight_sym=weight_sym,
                        dual_hadamard=False,
                        in_group_size=act_group_size,
                        out_group_size=act_group_size,
                        clip_ratio=clip_ratio,
                        no_hadamard=True
                    )
                    del tmp_linear, gptq
                    return flh_linear
                # 创建临时线性层，权重已预乘norm权重
                else:
                    temp_linear = torch.nn.Linear(
                        float_linear.in_features, 
                        float_linear.out_features, 
                        bias=float_linear.bias is not None,
                        device=float_linear.weight.device,
                        dtype=float_linear.weight.dtype
                    )
                    
                    # 步骤1: 权重预乘post_attention_layernorm权重 (W * diag(norm_weight))
                    temp_linear.weight.data = float_linear.weight.data * norm_weight.unsqueeze(0)
                    if float_linear.bias is not None:
                        temp_linear.bias.data = float_linear.bias.data.clone()
                    
                    # 步骤2: 使用LinearFLH进行量化（单侧Hadamard）
                    return flh.nn.LinearFLH.from_float(
                        temp_linear,
                        weight_bits=weight_bits,
                        weight_group_size=weight_group_size,
                        weight_sym=weight_sym,
                        dual_hadamard=False,
                        in_group_size=act_group_size,
                        out_group_size=act_group_size,
                        clip_ratio=clip_ratio
                    )

            flh_layer.mlp = FLH_LlamaMLP(
                config=config,
                act_bits=act_bits,
                act_group_size=act_group_size,
                act_sym=act_sym,
                weight_bits=weight_bits,
                weight_group_size=weight_group_size,
                weight_sym=weight_sym,
                clip_ratio=clip_ratio
            )
            gate_up_inps = cal.get("mlp.gate_proj", []) or cal.get("mlp.up_proj", [])

            # 对gate和up投影应用post_attention_layernorm权重融合
            flh_layer.mlp.gate_proj = _quantize_mlp_linear_flh(float_layer.mlp.gate_proj, post_norm_weight, gate_up_inps, use_gptq_here=use_gptq)
            flh_layer.mlp.up_proj = _quantize_mlp_linear_flh(float_layer.mlp.up_proj, post_norm_weight, gate_up_inps, use_gptq_here=use_gptq)
            
            # down_proj 使用双侧 Hadamard；做 GPTQ 时校准需用 H(线性层输出)
            down_inps = cal.get("mlp.down_proj", [])
            down_out_H = cal.get("mlp.down_proj_out_H", [])
            
            if use_gptq and down_inps and len(down_inps) > 0:
                tmp_linear = nn.Linear(float_layer.mlp.down_proj.in_features, float_layer.mlp.down_proj.out_features, bias=float_layer.mlp.down_proj.bias is not None, device=cal_device, dtype=float_layer.mlp.down_proj.weight.dtype)
                tmp_linear.weight.data.copy_(float_layer.mlp.down_proj.weight.data.to(cal_device))
                if tmp_linear.bias is not None:
                    tmp_linear.bias.data.copy_(float_layer.mlp.down_proj.bias.data.to(cal_device))
                gptq = flh.nn.FLHGPTQ(tmp_linear, dual_hadamard=True)
                for k, inp in enumerate(down_inps):
                    out_H = down_out_H[k] if k < len(down_out_H) else None
                    if inp is not None and inp.numel() > 0:
                        gptq.add_batch(inp.to(cal_device), out_H.to(cal_device) if out_H is not None else None, group_size=weight_group_size if weight_group_size > 0 else -1)
                gptq.fasterquant(group_size=weight_group_size, blocksize=128, percdamp=gptq_percdamp, actorder=gptq_actorder, sym=weight_sym, bits=weight_bits, dual_hadamard=True, clip_ratio=clip_ratio)
                gptq.free()
                flh_layer.mlp.down_proj = flh.nn.LinearFLH.from_float(
                    tmp_linear.cpu(),
                    weight_bits=weight_bits,
                    weight_group_size=weight_group_size,
                    weight_sym=weight_sym,
                    dual_hadamard=True,
                    in_group_size=act_group_size,
                    out_group_size=act_group_size,
                    clip_ratio=clip_ratio,
                    no_hadamard=True
                )
                del tmp_linear, gptq
            else:
                flh_layer.mlp.down_proj = flh.nn.LinearFLH.from_float(
                    float_layer.mlp.down_proj,
                    weight_bits=weight_bits,
                    weight_group_size=weight_group_size,
                    weight_sym=weight_sym,
                    dual_hadamard=True,
                    in_group_size=act_group_size,
                    out_group_size=act_group_size,
                    clip_ratio=clip_ratio
                )

            # 创建input_layernorm，保持归一化但权重设为1（因为已预乘到QKV权重中）
            flh_layer.input_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            flh_layer.input_layernorm.weight.data.fill_(1.0)  # 权重设为1
            
            # 创建post_attention_layernorm，保持归一化但权重设为1（因为已预乘到gate/up权重中）
            flh_layer.post_attention_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            flh_layer.post_attention_layernorm.weight.data.fill_(1.0)  # 权重设为1
        
        print("\n✓ Weight copying and quantization completed!")
        
        # 将最终 norm 权重融合到 lm_head 上：W_fused = W * diag(norm_weight)
        final_norm_weight = float_model.model.norm.weight.data
        lm_head_float = float_model.lm_head
        fused_linear = nn.Linear(
            lm_head_float.in_features,
            lm_head_float.out_features,
            bias=lm_head_float.bias is not None,
            device=lm_head_float.weight.device,
            dtype=lm_head_float.weight.dtype,
        )
        fused_linear.weight.data = lm_head_float.weight.data * final_norm_weight.unsqueeze(0)
        if lm_head_float.bias is not None:
            fused_linear.bias.data.copy_(lm_head_float.bias.data)
        fused_linear.weight.data = flh.nn.fast_hadamard_transform(fused_linear.weight.data, group_size=weight_group_size, normalize=True)
        flh_model.lm_head = fused_linear
        
        # # 对融合后的 lm_head 做单侧 Hadamard + 量化
        # flh_model.lm_head = flh.nn.LinearFLH.from_float(
        #     fused_linear,
        #     weight_bits=16,
        #     weight_group_size=weight_group_size,
        #     weight_sym=weight_sym,
        #     dual_hadamard=False,
        #     in_group_size=act_group_size,
        #     out_group_size=act_group_size,
        #     clip_ratio=clip_ratio
        # )
        # 将顶层 norm 的权重设为 1（权重已融合进 lm_head）
        flh_model.model.norm.weight.data.fill_(1.0)
        
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
            "clip_ratio": getattr(self, 'clip_ratio', 1.0),
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
        flh_model.clip_ratio = quant_config.get("clip_ratio", 1.0)
        
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
                act_sym=act_sym,
                weight_bits=quant_config["weight_bits"],
                weight_group_size=quant_config["weight_group_size"],
                weight_sym=weight_sym,
                clip_ratio=flh_model.clip_ratio
            )
            layer.input_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = FLH_LlamaMLP(
                config=config,
                act_bits=quant_config["act_bits"],
                act_group_size=quant_config["act_group_size"],
                act_sym=act_sym,
                weight_bits=quant_config["weight_bits"],
                weight_group_size=quant_config["weight_group_size"],
                weight_sym=weight_sym,
                clip_ratio=flh_model.clip_ratio
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
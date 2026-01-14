import flh
import torch
from transformers.utils import logging
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaFlashAttention2, apply_rotary_pos_emb, LlamaMLP
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Tuple

logger = logging.get_logger(__name__)

ALL_LAYERNORM_LAYERS = [flh.nn.RMSNorm]

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
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
class FLH_LlamaAttention(FLH_FP16LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = flh.nn.Quantizer(bits=16, group_size=-1, sym=True)
        self.q_proj = flh.nn.LinearFLH.from_float(self.q_proj)
        self.k_proj = flh.nn.LinearFLH.from_float(self.k_proj)
        self.v_proj = flh.nn.LinearFLH.from_float(self.v_proj)
        self.o_proj = flh.nn.LinearFLH.from_float(self.o_proj)
        
class FLH_LlamaMLP(LlamaMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = flh.nn.Quantizer(bits=16, group_size=-1, sym=True)
        self.up_proj = flh.nn.LinearFLH.from_float(self.up_proj)
        self.gate_proj = flh.nn.LinearFLH.from_float(self.gate_proj)
        self.down_proj = flh.nn.LinearFLH.from_float(self.down_proj)
        
    def forward(self, x):
        x = self.quantizer(x)
        return super().forward(x)
    
class FLH_FP16LlamaForCausalLM(LlamaForCausalLM):
    """
    FLH FP16 Llama model using transformers' native KV cache.
    This version uses DynamicCache instead of custom paged cache.
    """
    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        # Replace attention modules with FLH versions
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = FLH_FP16LlamaAttention(config=config, layer_idx=layer_idx)

class FLH_LlamaForCausalLM(FLH_FP16LlamaForCausalLM):
    """
    Fully quantized FLH Llama model with INT4 quantization.
    Uses transformers' native KV cache.
    """
    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        # Replace with quantized versions
        self.norm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = FLH_LlamaAttention(config=config, layer_idx=layer_idx)
            layer.input_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = flh.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = FLH_LlamaMLP(config=config)
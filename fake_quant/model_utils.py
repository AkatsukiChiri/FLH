import torch
import transformers
import logging

LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer


def get_rope_function_name(model):
    if isinstance(model, LLAMA_MODEL):
        return "apply_rotary_pos_emb"
    raise NotImplementedError

def get_layers(model):
    if isinstance(model, LLAMA_MODEL):
        return model.model.layers
    raise NotImplementedError

def get_model(model_name, hf_token=None):
    # TODO: Only llama model supported
    return get_llama(model_name, hf_token)

def get_model_type(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
    
def get_embeddings(model, model_type) -> list[torch.nn.Module]:
    if model_type == LLAMA_MODEL:
        return [model.model.embed_tokens]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL:
        return [layer for layer in model.model.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass

def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model

def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm, transformers.models.llama.modeling_llama.LlamaRMSNorm)
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm

def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)
    
def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
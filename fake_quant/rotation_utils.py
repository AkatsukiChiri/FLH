import torch
import typing
import transformers
import tqdm

import utils
import model_utils
import hadamard_utils

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        
    # Calculating new weight and bias
    W_ = linear.weight.data.double()
    linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
    
    if hasattr(layernorm, 'bias'):
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
        linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
        linear.bias.data = linear.bias.data.to(linear_dtype)
    
    
def fuse_layer_norms(model):
    model_type = model_utils.get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}

    # Embedding fusion
    for W in model_utils.get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = model_utils.get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        else:
            raise ValueError(f'Unknown model type {model_type}')
    
    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)]) 
    
    model_utils.replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == model_utils.LLAMA_MODEL else torch.nn.LayerNorm,
        lambda _: model_utils.RMSN(model.config.hidden_size),
        replace_layers=False,
    )

def rotate_embeddings(model, group_size) -> None:
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        n = W_.shape[-1]
        H_ = hadamard_utils.get_group_had_matrix(n, group_size)
        W.weight.data = torch.matmul(W_, H_).to(device='cpu', dtype=dtype)
    
def rotate_head(model, group_size) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    n = W_.shape[-1]
    H_ = hadamard_utils.get_group_had_matrix(n, group_size)
    W.weight.data = torch.matmul(W_, H_).to(device='cpu', dtype=dtype)
    
def rotate_attention_inputs(layer, group_size, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        n = W_.shape[-1]
        H_ = hadamard_utils.get_group_had_matrix(n, group_size)
        W.weight.data = torch.matmul(W_, H_).to(device='cpu', dtype=dtype)

def rotate_attention_output(layer, group_size, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    n = W_.shape[0]
    H_ = hadamard_utils.get_group_had_matrix(n, group_size)
    W.weight.data = torch.matmul(H_.T, W_).to(device='cpu', dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(H_.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_input(layer, group_size, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        n = W_.shape[-1]
        H_ = hadamard_utils.get_group_had_matrix(n, group_size)
        W.weight.data = torch.matmul(W_, H_).to(device='cpu', dtype=dtype)
        
def rotate_mlp_output(layer, group_size, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    n1 = W_.shape[-1]
    H1_ = hadamard_utils.get_group_had_matrix(n1, group_size)
    W_ = torch.matmul(W_, H1_)
    
    n2 = W_.shape[0]
    H2_ = hadamard_utils.get_group_had_matrix(n2, group_size)
    W.weight.data = torch.matmul(H2_.T, W_).to(device='cpu', dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(H2_.T, b).to(device="cpu", dtype=dtype)

@torch.inference_mode()
def rotate_model(model, args):
    group_size = args.groupsize
    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, group_size)
    rotate_head(model, group_size)
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    
    for idx, layer in enumerate(tqdm.tqdm(layers, unit='layer', desc='Rotating')):
        rotate_attention_inputs(layers[idx], group_size, model_type)
        rotate_attention_output(layers[idx], group_size, model_type)
        rotate_mlp_input(layers[idx], group_size, model_type)
        rotate_mlp_output(layers[idx], group_size, model_type)
        #TODO: HERE 是否需要ov_rotate?
        
        
# def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
#     '''
#     This function adds a rotation wrapper after the output of a function call in forward. 
#     Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
#     '''
#     import monkeypatch
#     import functools
#     attr_name = f"{function_name}_qk_rotation_wrapper"
#     assert not hasattr(module, attr_name)
#     wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
#                                                                     function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
#     setattr(module, attr_name, wrapper)
"""
Pure Python/PyTorch fallback implementation for CUDA extensions.
This provides a simple reference implementation when CUDA extensions are not available.
"""

import torch
import torch.nn.functional as F


def init_kv_i4(kv_data, kv_param, kv_indptr, kv_indices, 
               last_page_offset, k, v, k_param, v_param,
               seqlen_indptr, layer_idx):
    """
    Initialize KV cache with int4 quantized data
    
    Args:
        kv_data: [num_pages, num_layers, 2(k/v), num_heads, page_size, head_dim//2] uint8
        kv_param: [num_pages, num_layers, 2, num_heads, page_size, 2(scale/zero)] float16
        kv_indptr: [batch_size + 1] indices into kv_indices
        kv_indices: [num_pages_used] page indices
        last_page_offset: [batch_size] offset in last page
        k, v: [total_tokens, num_heads, head_dim//2] uint8
        k_param, v_param: [total_tokens, num_heads, 2] float16
        seqlen_indptr: [batch_size + 1] cumulative sequence lengths
        layer_idx: int
    """
    batch_size = len(last_page_offset)
    num_heads = k.shape[1]
    
    for batch_idx in range(batch_size):
        seq_start = seqlen_indptr[batch_idx].item()
        seq_end = seqlen_indptr[batch_idx + 1].item()
        seq_len = seq_end - seq_start
        
        page_start = kv_indptr[batch_idx].item()
        page_end = kv_indptr[batch_idx + 1].item()
        
        token_idx = 0
        for page_offset in range(page_start, page_end):
            page_idx = kv_indices[page_offset].item()
            
            # Calculate tokens in this page
            remaining_tokens = seq_len - token_idx
            if page_offset == page_end - 1:
                # Last page, use last_page_offset
                tokens_in_page = last_page_offset[batch_idx].item()
            else:
                # Use full page size
                tokens_in_page = kv_data.shape[4]  # page_size
            
            # Ensure we don't exceed available tokens
            tokens_in_page = min(tokens_in_page, remaining_tokens)
            
            if tokens_in_page <= 0:
                break
            
            # Get slices with proper bounds checking
            k_slice = k[seq_start + token_idx:seq_start + token_idx + tokens_in_page, :, :]
            actual_tokens = k_slice.shape[0]  # Actual number of tokens in the slice
            
            if actual_tokens != tokens_in_page:
                # Adjust if slice is shorter than expected
                tokens_in_page = actual_tokens
            
            if tokens_in_page <= 0:
                break
            
            # Copy K
            kv_data[page_idx, layer_idx, 0, :, :tokens_in_page, :] = k_slice.transpose(0, 1)
            k_param_slice = k_param[seq_start + token_idx:seq_start + token_idx + tokens_in_page, :, :]
            kv_param[page_idx, layer_idx, 0, :, :tokens_in_page, :] = k_param_slice.transpose(0, 1)
            
            # Copy V
            v_slice = v[seq_start + token_idx:seq_start + token_idx + tokens_in_page, :, :]
            kv_data[page_idx, layer_idx, 1, :, :tokens_in_page, :] = v_slice.transpose(0, 1)
            v_param_slice = v_param[seq_start + token_idx:seq_start + token_idx + tokens_in_page, :, :]
            kv_param[page_idx, layer_idx, 1, :, :tokens_in_page, :] = v_param_slice.transpose(0, 1)
            
            token_idx += tokens_in_page


def init_kv_f16(kv_data, kv_param, kv_indptr, kv_indices,
                last_page_offset, k, v, k_param, v_param,
                seqlen_indptr, layer_idx):
    """
    Initialize KV cache with float16 data
    
    Args:
        kv_data: [num_pages, num_layers, 2(k/v), num_heads, page_size, head_dim] float16
        kv_param: [num_pages, num_layers, 2, num_heads, page_size, 2(scale/zero)] float16
        kv_indptr: [batch_size + 1] indices into kv_indices
        kv_indices: [num_pages_used] page indices
        last_page_offset: [batch_size] offset in last page
        k, v: [total_tokens, num_heads, head_dim] float16
        k_param, v_param: [total_tokens, num_heads, 2] float16
        seqlen_indptr: [batch_size + 1] cumulative sequence lengths
        layer_idx: int
    """
    batch_size = len(last_page_offset)
    num_heads = k.shape[1]
    
    for batch_idx in range(batch_size):
        seq_start = seqlen_indptr[batch_idx].item()
        seq_end = seqlen_indptr[batch_idx + 1].item()
        seq_len = seq_end - seq_start
        
        page_start = kv_indptr[batch_idx].item()
        page_end = kv_indptr[batch_idx + 1].item()
        
        token_idx = 0
        for page_offset in range(page_start, page_end):
            page_idx = kv_indices[page_offset].item()
            
            # Calculate tokens in this page
            remaining_tokens = seq_len - token_idx
            if page_offset == page_end - 1:
                # Last page, use last_page_offset
                tokens_in_page = last_page_offset[batch_idx].item()
            else:
                # Use full page size
                tokens_in_page = kv_data.shape[4]  # page_size
            
            # Ensure we don't exceed available tokens
            tokens_in_page = min(tokens_in_page, remaining_tokens)
            
            if tokens_in_page <= 0:
                break
            
            # Get slices with proper bounds checking
            k_slice = k[seq_start + token_idx:seq_start + token_idx + tokens_in_page, :, :]
            actual_tokens = k_slice.shape[0]  # Actual number of tokens in the slice
            
            if actual_tokens != tokens_in_page:
                # Adjust if slice is shorter than expected
                tokens_in_page = actual_tokens
            
            if tokens_in_page <= 0:
                break
            
            # Copy K
            kv_data[page_idx, layer_idx, 0, :, :tokens_in_page, :] = k_slice.transpose(0, 1)
            k_param_slice = k_param[seq_start + token_idx:seq_start + token_idx + tokens_in_page, :, :]
            kv_param[page_idx, layer_idx, 0, :, :tokens_in_page, :] = k_param_slice.transpose(0, 1)
            
            # Copy V
            v_slice = v[seq_start + token_idx:seq_start + token_idx + tokens_in_page, :, :]
            kv_data[page_idx, layer_idx, 1, :, :tokens_in_page, :] = v_slice.transpose(0, 1)
            v_param_slice = v_param[seq_start + token_idx:seq_start + token_idx + tokens_in_page, :, :]
            kv_param[page_idx, layer_idx, 1, :, :tokens_in_page, :] = v_param_slice.transpose(0, 1)
            
            token_idx += tokens_in_page


def append_kv_i4(kv_data, kv_param, kv_indptr, kv_indices,
                 last_page_offset, k, v, k_param, v_param,
                 layer_idx):
    """
    Append one token to KV cache with int4 quantization
    
    Args:
        kv_data: [num_pages, num_layers, 2(k/v), num_heads, page_size, head_dim//2] uint8
        kv_param: [num_pages, num_layers, 2, num_heads, page_size, 2] float16
        kv_indptr: [batch_size + 1]
        kv_indices: [num_pages_used]
        last_page_offset: [batch_size]
        k, v: [batch_size, num_heads, head_dim//2] uint8
        k_param, v_param: [batch_size, num_heads, 2] float16
        layer_idx: int
    """
    batch_size = k.shape[0]
    num_heads = k.shape[1]
    
    for batch_idx in range(batch_size):
        page_start = kv_indptr[batch_idx].item()
        page_end = kv_indptr[batch_idx + 1].item()
        
        # Get the last page for this batch
        last_page_idx = kv_indices[page_end - 1].item()
        offset = last_page_offset[batch_idx].item()
        
        # Append K and V to the last page at the offset position
        kv_data[last_page_idx, layer_idx, 0, :, offset - 1, :] = k[batch_idx, :, :]
        kv_param[last_page_idx, layer_idx, 0, :, offset - 1, :] = k_param[batch_idx, :, :]
        
        kv_data[last_page_idx, layer_idx, 1, :, offset - 1, :] = v[batch_idx, :, :]
        kv_param[last_page_idx, layer_idx, 1, :, offset - 1, :] = v_param[batch_idx, :, :]


def append_kv_f16(kv_data, kv_param, kv_indptr, kv_indices,
                  last_page_offset, k, v, k_param, v_param,
                  layer_idx):
    """
    Append one token to KV cache with float16
    
    Args:
        kv_data: [num_pages, num_layers, 2(k/v), num_heads, page_size, head_dim] float16
        kv_param: [num_pages, num_layers, 2, num_heads, page_size, 2] float16
        kv_indptr: [batch_size + 1]
        kv_indices: [num_pages_used]
        last_page_offset: [batch_size]
        k, v: [batch_size, num_heads, head_dim] float16
        k_param, v_param: [batch_size, num_heads, 2] float16
        layer_idx: int
    """
    batch_size = k.shape[0]
    num_heads = k.shape[1]
    
    for batch_idx in range(batch_size):
        page_start = kv_indptr[batch_idx].item()
        page_end = kv_indptr[batch_idx + 1].item()
        
        # Get the last page for this batch
        last_page_idx = kv_indices[page_end - 1].item()
        offset = last_page_offset[batch_idx].item()
        
        # Append K and V to the last page at the offset position
        kv_data[last_page_idx, layer_idx, 0, :, offset - 1, :] = k[batch_idx, :, :]
        kv_param[last_page_idx, layer_idx, 0, :, offset - 1, :] = k_param[batch_idx, :, :]
        
        kv_data[last_page_idx, layer_idx, 1, :, offset - 1, :] = v[batch_idx, :, :]
        kv_param[last_page_idx, layer_idx, 1, :, offset - 1, :] = v_param[batch_idx, :, :]


def _dequantize_i4(data, param):
    """
    Dequantize int4 data
    
    Args:
        data: [..., head_dim//2] uint8
        param: [..., 2] float16, where param[..., 0] is scale and param[..., 1] is zero
    
    Returns:
        dequantized: [..., head_dim] float16
    """
    # Unpack int4: each uint8 contains 2 int4 values
    low = (data & 0x0f).to(torch.float16)
    high = ((data >> 4) & 0x0f).to(torch.float16)
    unpacked = torch.stack([low, high], dim=-1).view(*data.shape[:-1], data.shape[-1] * 2)
    
    # Dequantize: value = scale * quantized - zero
    scale = param[..., 0:1]
    zero = param[..., 1:2]
    return unpacked * scale - zero


def batch_decode_i4(o, q, kv_data, kv_param, kv_indptr, kv_indices,
                    last_page_offset, layer_idx):
    """
    Optimized batch decode with int4 quantized KV cache
    
    Args:
        o: [batch_size, num_qo_heads, head_dim] output
        q: [batch_size, num_qo_heads, head_dim] query
        kv_data: [num_pages, num_layers, 2, num_kv_heads, page_size, head_dim//2] uint8
        kv_param: [num_pages, num_layers, 2, num_kv_heads, page_size, 2] float16
        kv_indptr: [batch_size + 1]
        kv_indices: [num_pages_used]
        last_page_offset: [batch_size]
        layer_idx: int
    """
    batch_size = q.shape[0]
    num_qo_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = kv_data.shape[3]
    page_size = kv_data.shape[4]
    
    # Handle grouped query attention
    num_queries_per_kv = num_qo_heads // num_kv_heads
    
    for batch_idx in range(batch_size):
        page_start = kv_indptr[batch_idx].item()
        page_end = kv_indptr[batch_idx + 1].item()
        last_offset = last_page_offset[batch_idx].item()
        
        # Pre-calculate total tokens
        num_pages = page_end - page_start
        total_tokens = (num_pages - 1) * page_size + last_offset
        
        # Pre-allocate output tensors (dequantized)
        k_all = torch.empty(num_kv_heads, total_tokens, head_dim, 
                           dtype=torch.float16, device=kv_data.device)
        v_all = torch.empty(num_kv_heads, total_tokens, head_dim,
                           dtype=torch.float16, device=kv_data.device)
        
        # Gather and dequantize efficiently
        token_offset = 0
        for page_offset in range(page_start, page_end):
            page_idx = kv_indices[page_offset].item()
            tokens_in_page = last_offset if page_offset == page_end - 1 else page_size
            
            # Get K page and dequantize in-place
            k_page = kv_data[page_idx, layer_idx, 0, :, :tokens_in_page, :]
            k_param_page = kv_param[page_idx, layer_idx, 0, :, :tokens_in_page, :]
            k_all[:, token_offset:token_offset + tokens_in_page, :] = _dequantize_i4(k_page, k_param_page)
            
            # Get V page and dequantize in-place
            v_page = kv_data[page_idx, layer_idx, 1, :, :tokens_in_page, :]
            v_param_page = kv_param[page_idx, layer_idx, 1, :, :tokens_in_page, :]
            v_all[:, token_offset:token_offset + tokens_in_page, :] = _dequantize_i4(v_page, v_param_page)
            
            token_offset += tokens_in_page
        
        # Expand for GQA using broadcasting
        if num_queries_per_kv > 1:
            k_all = k_all.unsqueeze(1).expand(num_kv_heads, num_queries_per_kv, total_tokens, head_dim)
            k_all = k_all.reshape(num_qo_heads, total_tokens, head_dim)
            v_all = v_all.unsqueeze(1).expand(num_kv_heads, num_queries_per_kv, total_tokens, head_dim)
            v_all = v_all.reshape(num_qo_heads, total_tokens, head_dim)
        
        # Optimized attention computation
        q_batch = q[batch_idx]
        
        # Use scaled_dot_product_attention if available
        try:
            q_reshaped = q_batch.unsqueeze(0).unsqueeze(2)
            k_reshaped = k_all.unsqueeze(0)
            v_reshaped = v_all.unsqueeze(0)
            
            attn_out = F.scaled_dot_product_attention(
                q_reshaped, k_reshaped, v_reshaped,
                attn_mask=None, dropout_p=0.0, is_causal=False
            )
            o[batch_idx] = attn_out.squeeze(0).squeeze(1)
        except:
            # Fallback
            attn_scores = torch.matmul(q_batch.unsqueeze(1), k_all.transpose(1, 2)).squeeze(1)
            attn_scores = attn_scores / (head_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            o[batch_idx] = torch.matmul(attn_weights.unsqueeze(1), v_all).squeeze(1)


def batch_decode_f16(o, q, kv_data, kv_param, kv_indptr, kv_indices,
                     last_page_offset, layer_idx):
    """
    Optimized batch decode with float16 KV cache
    
    Args:
        o: [batch_size, num_qo_heads, head_dim] output
        q: [batch_size, num_qo_heads, head_dim] query
        kv_data: [num_pages, num_layers, 2, num_kv_heads, page_size, head_dim] float16
        kv_param: [num_pages, num_layers, 2, num_kv_heads, page_size, 2] float16
        kv_indptr: [batch_size + 1]
        kv_indices: [num_pages_used]
        last_page_offset: [batch_size]
        layer_idx: int
    """
    batch_size = q.shape[0]
    num_qo_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = kv_data.shape[3]
    page_size = kv_data.shape[4]
    
    # Handle grouped query attention
    num_queries_per_kv = num_qo_heads // num_kv_heads
    
    # Optimization: Process all batches together when possible
    # For simple case (all batches have same number of pages and same last_page_offset)
    if batch_size == 1 or (kv_indptr[1] - kv_indptr[0] == kv_indptr[-1] - kv_indptr[-2]):
        # Fast path for uniform batches
        for batch_idx in range(batch_size):
            page_start = kv_indptr[batch_idx].item()
            page_end = kv_indptr[batch_idx + 1].item()
            last_offset = last_page_offset[batch_idx].item()
            
            # Pre-calculate total tokens to avoid multiple .item() calls
            num_pages = page_end - page_start
            total_tokens = (num_pages - 1) * page_size + last_offset
            
            # Pre-allocate tensors instead of using lists
            k_all = torch.empty(num_kv_heads, total_tokens, head_dim, 
                               dtype=kv_data.dtype, device=kv_data.device)
            v_all = torch.empty(num_kv_heads, total_tokens, head_dim,
                               dtype=kv_data.dtype, device=kv_data.device)
            
            # Gather KV efficiently
            token_offset = 0
            for page_offset in range(page_start, page_end):
                page_idx = kv_indices[page_offset].item()
                tokens_in_page = last_offset if page_offset == page_end - 1 else page_size
                
                # Direct slice copy (faster than append + cat)
                k_all[:, token_offset:token_offset + tokens_in_page, :] = \
                    kv_data[page_idx, layer_idx, 0, :, :tokens_in_page, :]
                v_all[:, token_offset:token_offset + tokens_in_page, :] = \
                    kv_data[page_idx, layer_idx, 1, :, :tokens_in_page, :]
                token_offset += tokens_in_page
            
            # Expand for GQA using broadcasting instead of repeat_interleave
            if num_queries_per_kv > 1:
                # [num_kv_heads, seq, head_dim] -> [num_kv_heads, 1, seq, head_dim] -> [num_kv_heads, num_queries_per_kv, seq, head_dim]
                k_all = k_all.unsqueeze(1).expand(num_kv_heads, num_queries_per_kv, total_tokens, head_dim)
                k_all = k_all.reshape(num_qo_heads, total_tokens, head_dim)
                v_all = v_all.unsqueeze(1).expand(num_kv_heads, num_queries_per_kv, total_tokens, head_dim)
                v_all = v_all.reshape(num_qo_heads, total_tokens, head_dim)
            
            # Optimized attention computation
            q_batch = q[batch_idx]  # [num_qo_heads, head_dim]
            
            # Use scaled_dot_product_attention if available (faster)
            # Q: [num_qo_heads, head_dim] -> [1, num_qo_heads, 1, head_dim]
            # K: [num_qo_heads, total_tokens, head_dim] -> [1, num_qo_heads, total_tokens, head_dim]
            # V: [num_qo_heads, total_tokens, head_dim] -> [1, num_qo_heads, total_tokens, head_dim]
            q_reshaped = q_batch.unsqueeze(0).unsqueeze(2)  # [1, num_qo_heads, 1, head_dim]
            k_reshaped = k_all.unsqueeze(0)  # [1, num_qo_heads, total_tokens, head_dim]
            v_reshaped = v_all.unsqueeze(0)  # [1, num_qo_heads, total_tokens, head_dim]
            
            # Use PyTorch's optimized attention
            try:
                attn_out = F.scaled_dot_product_attention(
                    q_reshaped, k_reshaped, v_reshaped, 
                    attn_mask=None, dropout_p=0.0, is_causal=False
                )
                o[batch_idx] = attn_out.squeeze(0).squeeze(1)  # [num_qo_heads, head_dim]
            except:
                # Fallback to manual computation
                attn_scores = torch.matmul(q_batch.unsqueeze(1), k_all.transpose(1, 2)).squeeze(1)
                attn_scores = attn_scores / (head_dim ** 0.5)
                attn_weights = F.softmax(attn_scores, dim=-1)
                o[batch_idx] = torch.matmul(attn_weights.unsqueeze(1), v_all).squeeze(1)
    else:
        # Fallback to original implementation for non-uniform batches
        for batch_idx in range(batch_size):
            page_start = kv_indptr[batch_idx].item()
            page_end = kv_indptr[batch_idx + 1].item()
            
            k_list = []
            v_list = []
            
            for page_offset in range(page_start, page_end):
                page_idx = kv_indices[page_offset].item()
                tokens_in_page = last_page_offset[batch_idx].item() if page_offset == page_end - 1 else page_size
                
                k_list.append(kv_data[page_idx, layer_idx, 0, :, :tokens_in_page, :])
                v_list.append(kv_data[page_idx, layer_idx, 1, :, :tokens_in_page, :])
            
            k_all = torch.cat(k_list, dim=1)
            v_all = torch.cat(v_list, dim=1)
            
            if num_queries_per_kv > 1:
                k_all = k_all.repeat_interleave(num_queries_per_kv, dim=0)
                v_all = v_all.repeat_interleave(num_queries_per_kv, dim=0)
            
            q_batch = q[batch_idx]
            attn_scores = torch.matmul(q_batch.unsqueeze(1), k_all.transpose(1, 2)).squeeze(1)
            attn_scores = attn_scores / (head_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            o[batch_idx] = torch.matmul(attn_weights.unsqueeze(1), v_all).squeeze(1)

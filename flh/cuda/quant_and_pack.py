import torch


def quant_and_pack_i4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA kernel: symmetric int4 quantization + packing (no Hadamard transform).

    Args:
        x: Input tensor of shape (M, 128), dtype float16, on CUDA.

    Returns:
        A tuple (q_packed, scales):
          - q_packed: uint8 tensor of shape (M, 64), each byte packs two int4 values.
          - scales: float16 tensor of shape (M,), per-row symmetric quantization scales.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA")
    if x.dtype != torch.float16:
        raise ValueError("Input tensor must be float16 (half)")
    if x.dim() != 2:
        raise ValueError("Input tensor must be 2D")
    if x.shape[-1] != 128:
        raise ValueError("Last dimension must be 128")

    from flh import _CUDA

    x_contiguous = x.contiguous()
    q_packed, scales = _CUDA.quant_and_pack_i4(x_contiguous)
    return q_packed, scales

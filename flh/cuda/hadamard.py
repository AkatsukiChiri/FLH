import torch


def hadamard_transform_half(x: torch.Tensor) -> torch.Tensor:
    """
    In-place Fast Hadamard Transform for (M, 128) half matrix.

    This is a CUDA-optimized implementation that transforms each row
    of the input matrix using the Hadamard transform.

    Args:
        x: Input tensor of shape (M, 128) with dtype float16 on CUDA.
            The transform is performed in-place.

    Returns:
        The same tensor after Hadamard transform (in-place).

    Raises:
        RuntimeError: If input is not on CUDA, not float16, not 2D,
                      or last dimension is not 128.
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
    return _CUDA.hadamard_transform_half(x)


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Fast Hadamard Transform with automatic backend selection.

    For (M, 128) half tensors on CUDA, uses optimized CUDA kernel.
    Otherwise falls back to pure Python implementation.

    Args:
        x: Input tensor with last dimension being a power of 2 (preferably 128)
        scale: Scaling factor to apply after transform (default: 1.0)

    Returns:
        Transformed tensor scaled by the scale factor
    """
    # Use CUDA implementation if possible
    if (x.is_cuda and
        x.dtype == torch.float16 and
        x.dim() == 2 and
        x.shape[-1] == 128):
        result = hadamard_transform_half(x.clone())
        return result * scale

    # Fallback to pure Python implementation
    from ..functional.hadamard import hadamard_transform as python_hadamard
    return python_hadamard(x, scale=scale)

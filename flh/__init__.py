import torch
from . import nn

# Try to import CUDA extension, fallback to pure Python implementation
_CUDA = None
_cuda_extension_available = False

try:
    # Try to import compiled CUDA extension
    import importlib.util
    spec = importlib.util.find_spec('flh._CUDA')
    if spec is not None:
        from . import _CUDA as _CUDA_MODULE
        _CUDA = _CUDA_MODULE
        _cuda_extension_available = True
except (ImportError, AttributeError):
    pass

if not _cuda_extension_available:
    # Use pure Python fallback implementation
    from . import _cuda_fallback
    _CUDA = _cuda_fallback
    print("Warning: Using pure Python fallback for CUDA operations (slower). "
          "Consider compiling CUDA extensions for better performance.")


class PackedQuantizedTensor:
    def __init__(self,
                 quantized_x: torch.Tensor,
                 scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x
    
    def size(self):
        return self.quantized_x.size()
    
    @property
    def device(self):
        return self.quantized_x.device
    
    @property
    def dtype(self):
        return self.quantized_x.dtype
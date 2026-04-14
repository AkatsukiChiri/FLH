import os
from pathlib import Path
 
import torch
from torch.utils.cpp_extension import load
 
 
_EXT = None
 
 
def load_flh_cuda_ext():
  global _EXT
  if _EXT is not None:
    return _EXT
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")
 
  this_dir = Path(__file__).resolve().parent
  root = this_dir.parent.parent
 
  sources = [
    str(this_dir / "flh_cuda_bindings.cpp"),
    str(root / "flh" / "kernels" / "gemm_and_dequant_i4.cu"),
    str(root / "flh" / "kernels" / "cuda_had_kernel.cu"),
    str(root / "flh" / "kernels" / "cuda_had_64_kernel.cu"),
    str(root / "flh" / "kernels" / "cuda_had_32_kernel.cu"),
    str(root / "flh" / "kernels" / "had_and_quant.cu"),
    str(root / "flh" / "kernels" / "quant_and_pack.cu"),
  ]
 
  extra_cuda_cflags = [
    "-lineinfo",
    "--use_fast_math",
  ]
 
  _EXT = load(
    name="flh__CUDA",
    sources=sources,
    extra_cuda_cflags=extra_cuda_cflags,
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
  )
  return _EXT


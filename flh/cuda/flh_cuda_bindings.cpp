#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../kernels/had_and_quant.h"
 
extern "C" void flhDenseLayerGEMM_i4_o16(
  const uint8_t* A,
  const uint8_t* B,
  const half* A_scale,
  const half* B_scale,
  half* D,
  size_t M_GLOBAL,
  size_t N_GLOBAL,
  size_t K_GLOBAL
);

// Hadamard transform kernel (implemented in cuda_had_kernel.cu)
void had_trans_half_cuda(torch::Tensor& data);
 
// Fused hadamard transform + int4 quantization kernel (implemented in had_and_quant.cu)
void had_and_quant_host(
  const half* data,
  Int4Storage* quantized_data,
  half* scales,
  uint32_t M
);

// Quantization + packing kernel (no Hadamard, implemented in quant_and_pack.cu)
void quant_and_pack_host(
  const half* data,
  Int4Storage* quantized_data,
  half* scales,
  uint32_t M
);

static torch::Tensor gemm_i4_dequant_o16(
  const torch::Tensor& A,
  const torch::Tensor& B,
  const torch::Tensor& A_scale,
  const torch::Tensor& B_scale
){
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A_scale.is_cuda() && B_scale.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(A.scalar_type() == torch::kUInt8, "A must be uint8 packed int4");
  TORCH_CHECK(B.scalar_type() == torch::kUInt8, "B must be uint8 packed int4");
  TORCH_CHECK(A_scale.scalar_type() == torch::kFloat16, "A_scale must be float16");
  TORCH_CHECK(B_scale.scalar_type() == torch::kFloat16, "B_scale must be float16");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && A_scale.is_contiguous() && B_scale.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B must be 2D");
  TORCH_CHECK(A_scale.dim() == 2 && B_scale.dim() == 2, "scale must be 2D");
 
  const int64_t M = A.size(0);
  const int64_t K_packed = A.size(1);
  const int64_t K = K_packed * 2;
  const int64_t N = B.size(0);
  TORCH_CHECK(B.size(1) == K_packed, "B.shape[1] must equal A.shape[1]");
 
  TORCH_CHECK(A_scale.size(0) == M, "A_scale.shape[0] must equal M");
  TORCH_CHECK(B_scale.size(0) == N, "B_scale.shape[0] must equal N");
 
  auto D = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));
 
  flhDenseLayerGEMM_i4_o16(
    (const uint8_t*)A.data_ptr<uint8_t>(),
    (const uint8_t*)B.data_ptr<uint8_t>(),
    (const half*)A_scale.data_ptr<at::Half>(),
    (const half*)B_scale.data_ptr<at::Half>(),
    (half*)D.data_ptr<at::Half>(),
    (size_t)M,
    (size_t)N,
    (size_t)K
  );
 
  return D;
}

static torch::Tensor hadamard_transform_half(
  torch::Tensor& input
) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "Input tensor must be float16 (half)");
  TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
  TORCH_CHECK(input.size(1) == 128, "Last dimension must be 128");

  had_trans_half_cuda(input);

  return input;
}

static std::vector<torch::Tensor> hadamard_and_quantize_i4(
  const torch::Tensor& input
) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "Input tensor must be float16 (half)");
  TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
  TORCH_CHECK(input.size(1) == 128, "Last dimension must be 128");
  TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

  const int64_t M = input.size(0);
  auto q = torch::empty({M, 64}, torch::TensorOptions().device(input.device()).dtype(torch::kUInt8));
  auto scales = torch::empty({M}, torch::TensorOptions().device(input.device()).dtype(torch::kFloat16));

  had_and_quant_host(
    (const half*)input.data_ptr<at::Half>(),
    (Int4Storage*)q.data_ptr<uint8_t>(),
    (half*)scales.data_ptr<at::Half>(),
    (uint32_t)M
  );

  return {q, scales};
}

static std::vector<torch::Tensor> quant_and_pack_i4(
  const torch::Tensor& input
) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "Input tensor must be float16 (half)");
  TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
  TORCH_CHECK(input.size(1) == 128, "Last dimension must be 128");
  TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

  const int64_t M = input.size(0);
  auto q = torch::empty({M, 64}, torch::TensorOptions().device(input.device()).dtype(torch::kUInt8));
  auto scales = torch::empty({M}, torch::TensorOptions().device(input.device()).dtype(torch::kFloat16));

  quant_and_pack_host(
    (const half*)input.data_ptr<at::Half>(),
    (Int4Storage*)q.data_ptr<uint8_t>(),
    (half*)scales.data_ptr<at::Half>(),
    (uint32_t)M
  );

  return {q, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_i4_dequant_o16", &gemm_i4_dequant_o16, "int4 GEMM with sync dequant (fp16 out)");
  m.def("hadamard_transform_half", &hadamard_transform_half, "In-place Hadamard transform for (M, 128) half matrix");
  m.def("hadamard_and_quantize_i4", &hadamard_and_quantize_i4, "Fused Hadamard transform + int4 quantization (returns packed uint8 and scales)");
  m.def("quant_and_pack_i4", &quant_and_pack_i4, "Symmetric int4 quantization + packing (no Hadamard, returns packed uint8 and scales)");
}


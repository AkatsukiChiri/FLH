#ifndef GEMM_AND_DEQUANT_I4_GS_H
#define GEMM_AND_DEQUANT_I4_GS_H

#include <cstdint>
#include <cuda_fp16.h>

/**
 * @brief INT4 GEMM with synchronous dequantization to FP16 (configurable group size)
 *
 * @param group_size Quantization group size (32, 64, or 128)
 * @param A Input activations, packed INT4 [M, K/2] UINT8
 * @param B Input weights, packed INT4 [N, K/2] UINT8
 * @param D Output matrix [M, N] FP16
 * @param M_GLOBAL Number of rows
 * @param N_GLOBAL Number of columns
 * @param K_GLOBAL Number of input features (must be divisible by group_size)
 * @param A_scale Activation scales [M, K/group_size] FP16
 * @param B_scale Weight scales [N, K/group_size] FP16
 */
void flhDenseLayerGEMM_i4_o16_gs(
    int group_size,
    const uint8_t* A,
    const uint8_t* B,
    half* D,
    size_t M_GLOBAL,
    size_t N_GLOBAL,
    size_t K_GLOBAL,
    const half* A_scale,
    const half* B_scale
);

#endif // GEMM_AND_DEQUANT_I4_GS_H
#ifndef HAD_AND_QUANT_GS_H
#define HAD_AND_QUANT_GS_H

#include <cstdint>
#include <cuda_fp16.h>

using Int4Storage = uint8_t;

/**
 * @brief 融合的 Hadamard 变换 + INT4 量化（支持不同 group size）
 *
 * @param group_size 分组大小，支持 32, 64, 128
 * @param data 输入数据 [M, group_size] FP16
 * @param quantized_data 输出量化数据 [M, group_size/2] UINT8 (packed INT4)
 * @param scales 输出量化 scales [M] FP16
 * @param M 行数
 */
void had_and_quant_host_gs(
    int group_size,
    const half* data,
    Int4Storage* quantized_data,
    half* scales,
    uint32_t M
);

#endif // HAD_AND_QUANT_GS_H
#ifndef HAD_AND_QUANT_H
#define HAD_AND_QUANT_H

#include <cstdint>
#include <cuda_fp16.h>

using Int4Storage = uint8_t;

/**
 * @brief 融合的 Hadamard 变换 + INT4 量化
 * 
 * 对输入数据执行 Hadamard 变换，然后进行 INT4 对称量化并打包到 UINT8
 * 
 * @param data 输入数据 [M, 128] FP16
 * @param quantized_data 输出量化数据 [M, 64] UINT8 (packed INT4)
 * @param scales 输出量化 scales [M] FP16
 * @param M 行数
 */
void had_and_quant_host(
    const half* data,
    Int4Storage* quantized_data,
    half* scales,
    uint32_t M
);

#endif // HAD_AND_QUANT_H


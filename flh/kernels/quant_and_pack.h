#ifndef QUANT_AND_PACK_H
#define QUANT_AND_PACK_H

#include <cstdint>
#include <cuda_fp16.h>

using Int4Storage = uint8_t;

/**
 * @brief INT4 对称量化与打包（无 Hadamard 变换）
 *
 * 对输入数据执行 INT4 对称量化 (scale = max_abs / 7) 并打包到 UINT8
 *
 * @param data 输入数据 [M, 128] FP16
 * @param quantized_data 输出量化数据 [M, 64] UINT8 (packed INT4)
 * @param scales 输出量化 scales [M] FP16
 * @param M 行数
 */
void quant_and_pack_host(
    const half* data,
    Int4Storage* quantized_data,
    half* scales,
    uint32_t M
);

#endif // QUANT_AND_PACK_H

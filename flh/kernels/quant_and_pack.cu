#include "quant_and_pack.h"

// ==================== INT4 Quantization + Packing Kernel (no Hadamard) ====================

// Quantization + Packing Kernel
// Processes rows of 128 elements each
// Each warp processes one row completely
__global__ void quant_and_pack_kernel(
    const __half* __restrict__ data,
    Int4Storage* __restrict__ quantized_data,
    __half* __restrict__ scales,
    uint32_t M
) {
    constexpr int kNElts = 4;               // Each thread handles 4 elements
    constexpr int kNThreadsPerRow = 32;     // 32 threads per row (1 warp)
    constexpr int kN = 128;                 // Fixed dimension size
    constexpr int kLogThreadsPerRow = 5;    // log2(32)

    const int warp_id = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    const int tid_in_warp = threadIdx.x % 32;

    if (warp_id >= M) return;

    // Each warp processes one row
    const int row = warp_id;
    const __half* x_row = data + row * kN;

    // 1. Load data into registers (vectorized)
    __half x_vals[kNElts];
    const int col_start = tid_in_warp * kNElts;

    #pragma unroll
    for (int i = 0; i < kNElts; ++i) {
        x_vals[i] = x_row[col_start + i];
    }

    // 2. Compute scale for quantization (warp-level reduction)
    // Find max absolute value across the warp
    __half max_val = __float2half(0.0f);
    #pragma unroll
    for (int i = 0; i < kNElts; ++i) {
        __half abs_val = __habs(x_vals[i]);
        max_val = __hmax(max_val, abs_val);
    }

    // Warp reduction to find global max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        __half other = __shfl_down_sync(0xffffffff, max_val, offset);
        max_val = __hmax(max_val, other);
    }

    // Broadcast max_val to all threads in warp
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // Compute scale: max / 7.0 (INT4 range is [-8, 7])
    __half scale = __hdiv(max_val, __float2half(7.0f));

    // Avoid division by zero
    if (__hle(scale, __float2half(1e-6f))) {
        scale = __float2half(1.0f);
    }

    // Store scale (only first thread in warp)
    if (tid_in_warp == 0) {
        scales[row] = scale;
    }

    // 3. Quantize to INT4 and pack
    // Each thread handles 4 elements -> produces 2 packed bytes
    Int4Storage storage[2];
    storage[0] = 0;
    storage[1] = 0;

    #pragma unroll
    for (int i = 0; i < kNElts; ++i) {
        // Quantize: round(x / scale) and clamp to [-8, 7]
        __half normalized = __hdiv(x_vals[i], scale);
        int qval = __half2int_rn(normalized);
        qval = max(-8, min(7, qval));

        // Pack two 4-bit values per byte
        int pack_idx = i / 2;
        int bit_pos = (i % 2) * 4;

        // Store signed INT4 directly
        int8_t sval = qval & 0x0F;
        storage[pack_idx] |= (sval << bit_pos);
    }

    // 4. Write packed quantized data
    // Each thread writes 2 bytes (4 INT4 values)
    const int out_col = tid_in_warp * 2;
    quantized_data[row * (kN / 2) + out_col] = storage[0];
    quantized_data[row * (kN / 2) + out_col + 1] = storage[1];
}

// Host function
void quant_and_pack_host(
    const half* data,
    Int4Storage* quantized_data,
    half* scales,
    uint32_t M
) {
    // Launch configuration
    // Each warp processes one row, so we need M warps total
    const int kThreadsPerBlock = 256;
    const int kWarpsPerBlock = kThreadsPerBlock / 32;
    const int num_blocks = (M + kWarpsPerBlock - 1) / kWarpsPerBlock;

    quant_and_pack_kernel<<<num_blocks, kThreadsPerBlock>>>(
        data,
        quantized_data,
        scales,
        M
    );
}

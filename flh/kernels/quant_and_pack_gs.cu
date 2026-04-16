#include "quant_and_pack_gs.h"

// ==================== INT4 Quantization + Packing Kernel (no Hadamard, Group Size Flexible) ====================

// Helper to select threads per row based on GROUP_SIZE
template<int GROUP_SIZE>
struct ThreadsPerRowHelper;

template<>
struct ThreadsPerRowHelper<128> {
    static constexpr int value = 32;
};

template<>
struct ThreadsPerRowHelper<64> {
    static constexpr int value = 16;
};

template<>
struct ThreadsPerRowHelper<32> {
    static constexpr int value = 8;
};

// Quantization + Packing Kernel (Template Version)
template<int GROUP_SIZE>
__global__ void quant_and_pack_kernel_gs(
    const __half* __restrict__ data,
    Int4Storage* __restrict__ quantized_data,
    __half* __restrict__ scales,
    uint32_t M
) {
    constexpr int kNThreadsPerRow = ThreadsPerRowHelper<GROUP_SIZE>::value;
    constexpr int kN = GROUP_SIZE;
    constexpr int kElemsPerThread = GROUP_SIZE / kNThreadsPerRow;
    
    const int warp_id = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    const int tid_in_warp = threadIdx.x % 32;
    
    if (warp_id >= M) return;
    
    // Only threads that participate in this row process it
    if (tid_in_warp >= kNThreadsPerRow) return;
    
    // Each warp processes one row
    const int row = warp_id;
    const __half* x_row = data + row * kN;
    
    // 1. Load data into registers (each thread handles kElemsPerThread elements)
    __half x_vals[kElemsPerThread];
    const int col_start = tid_in_warp * kElemsPerThread;
    
    #pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
        x_vals[i] = x_row[col_start + i];
    }
    
    // 2. Compute scale for quantization (warp-level reduction)
    __half max_val = __float2half(0.0f);
    #pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
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
    Int4Storage storage[2];
    storage[0] = 0;
    storage[1] = 0;
    
    constexpr int kStoragePerThread = kElemsPerThread / 2;
    
    #pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
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
    const int out_col = tid_in_warp * kStoragePerThread;
    #pragma unroll
    for (int i = 0; i < kStoragePerThread; ++i) {
        quantized_data[row * (kN / 2) + out_col + i] = storage[i];
    }
}

// ==================== Template Instantiations ====================

template __global__ void quant_and_pack_kernel_gs<32>(
    const __half* __restrict__ data,
    Int4Storage* __restrict__ quantized_data,
    __half* __restrict__ scales,
    uint32_t M
);

template __global__ void quant_and_pack_kernel_gs<64>(
    const __half* __restrict__ data,
    Int4Storage* __restrict__ quantized_data,
    __half* __restrict__ scales,
    uint32_t M
);

template __global__ void quant_and_pack_kernel_gs<128>(
    const __half* __restrict__ data,
    Int4Storage* __restrict__ quantized_data,
    __half* __restrict__ scales,
    uint32_t M
);

// ==================== Host Functions ====================

template<int GROUP_SIZE>
void quant_and_pack_host_impl(
    const half* data,
    Int4Storage* quantized_data,
    half* scales,
    uint32_t M
) {
    constexpr int kNThreadsPerRow = ThreadsPerRowHelper<GROUP_SIZE>::value;
    const int kThreadsPerBlock = 256;
    const int kWarpsPerBlock = kThreadsPerBlock / 32;
    const int num_blocks = (M + kWarpsPerBlock - 1) / kWarpsPerBlock;
    
    quant_and_pack_kernel_gs<GROUP_SIZE><<<num_blocks, kThreadsPerBlock>>>(
        data,
        quantized_data,
        scales,
        M
    );
}

// Explicit template instantiations for host functions
template void quant_and_pack_host_impl<32>(
    const half* data,
    Int4Storage* quantized_data,
    half* scales,
    uint32_t M
);

template void quant_and_pack_host_impl<64>(
    const half* data,
    Int4Storage* quantized_data,
    half* scales,
    uint32_t M
);

template void quant_and_pack_host_impl<128>(
    const half* data,
    Int4Storage* quantized_data,
    half* scales,
    uint32_t M
);

// Unified host function with switch
void quant_and_pack_host_gs(
    int group_size,
    const half* data,
    Int4Storage* quantized_data,
    half* scales,
    uint32_t M
) {
    switch (group_size) {
        case 32:
            quant_and_pack_host_impl<32>(data, quantized_data, scales, M);
            break;
        case 64:
            quant_and_pack_host_impl<64>(data, quantized_data, scales, M);
            break;
        case 128:
        default:
            quant_and_pack_host_impl<128>(data, quantized_data, scales, M);
            break;
    }
}

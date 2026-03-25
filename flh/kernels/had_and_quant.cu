#include "had_and_quant.h"
#include <cuda_fp16.h>

// ==================== Hadamard Transform + Quantization Fused Kernel ====================

// Compile-time log2
constexpr int cilog2_local(int n) {
    return (n <= 1) ? 0 : 1 + cilog2_local(n / 2);
}

// Thread-level Hadamard transform
template<int kNElts>
__device__ __forceinline__ void hadamard_mult_thread(__half x[kNElts]) {
    constexpr int kLogNElts = cilog2_local(kNElts);
    static_assert(1 << kLogNElts == kNElts, "kNElts must be power of 2");
    
    #pragma unroll
    for (int s = 0; s < kLogNElts; ++s) {
        const int stride = 1 << s;
        #pragma unroll
        for (int j = 0; j < kNElts / 2; ++j) {
            const int lo = j & (stride - 1);
            const int idx = (j - lo) * 2 + lo;
            
            const __half a = x[idx];
            const __half b = x[idx + stride];
            x[idx] = __hadd(a, b);
            x[idx + stride] = __hsub(a, b);
        }
    }
}

// Warp-level Hadamard transform using shuffle
template<int kLogThreadsPerRow, int kNElts>
__device__ __forceinline__ void hadamard_mult_warp(__half x[kNElts], int tid_in_row) {
    static_assert(kLogThreadsPerRow <= 5, "Threads per row can't exceed 32");
    
    constexpr int kNThreadsPerRow = 1 << kLogThreadsPerRow;
    
    #pragma unroll
    for (int step = 0; step < kLogThreadsPerRow; ++step) {
        const int lane_mask = 1 << step;
        const bool should_negate = (tid_in_row & lane_mask);
        
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            const __half x_other = __shfl_xor_sync(0xffffffff, x[i], lane_mask);
            
            if (should_negate) {
                x[i] = __hsub(x_other, x[i]);
            } else {
                x[i] = __hadd(x[i], x_other);
            }
        }
    }
}

// Fused Hadamard Transform + Quantization Kernel
// Processes rows of 128 elements each
// Each warp processes one row completely
__global__ void had_and_quant_kernel(
    const __half* __restrict__ data,
    Int4Storage* __restrict__ quantized_data,
    __half* __restrict__ scales,
    uint32_t M
) {
    constexpr int kNElts = 4;           // Each thread handles 4 elements
    constexpr int kNThreadsPerRow = 32; // 32 threads per row (1 warp)
    constexpr int kN = 128;             // Fixed dimension size
    constexpr int kLogThreadsPerRow = 5; // log2(32)
    
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
    
    // 2. Thread-level Hadamard transform
    hadamard_mult_thread<kNElts>(x_vals);
    
    // 3. Warp-level Hadamard transform (using shuffle)
    hadamard_mult_warp<kLogThreadsPerRow, kNElts>(x_vals, tid_in_warp);
    
    // 4. Compute scale for quantization (warp-level reduction)
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
    
    // 5. Quantize to INT4 and pack
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
    
    // 6. Write packed quantized data
    // Each thread writes 2 bytes (4 INT4 values)
    const int out_col = tid_in_warp * 2;
    quantized_data[row * (kN / 2) + out_col] = storage[0];
    quantized_data[row * (kN / 2) + out_col + 1] = storage[1];
}

// Host function
void had_and_quant_host(
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
    
    had_and_quant_kernel<<<num_blocks, kThreadsPerBlock>>>(
        data,
        quantized_data,
        scales,
        M
    );
}


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>

// ==================== 编译时工具函数 ====================

constexpr int cilog2(int n) {
    return (n <= 1) ? 0 : 1 + cilog2(n / 2);
}

template<int N>
struct BytesToType {};

template<> struct BytesToType<16> { using Type = uint4; };
template<> struct BytesToType<8> { using Type = uint2; };
template<> struct BytesToType<4> { using Type = uint; };
template<> struct BytesToType<2> { using Type = unsigned short; };

// ==================== Kernel Traits (32分组) ====================

template<int kNElts_, int kNThreadsPerRow_, int kRowsPerWarp_, int kRowsPerBlock_, int kLogN_, typename input_t_>
struct hadamard_32_kernel_traits {
    using input_t = input_t_;
    
    // 基本配置参数
    static constexpr int kNElts = kNElts_;  // 一个线程处理的元素数（默认4）
    static constexpr int kNThreadsPerRow = kNThreadsPerRow_;  // 每行使用的线程数（默认8）
    static constexpr int kRowsPerWarp = kRowsPerWarp_;  // 一个warp处理的行数（默认4）
    static constexpr int kRowsPerBlock = kRowsPerBlock_;  // 一个block处理的行数（默认16）
    
    // 导出参数
    static constexpr int kLogN = kLogN_;  // log2(32) = 5
    static constexpr int N = 1 << kLogN;  // 32 for log2(32) = 5
    static constexpr int kNBytes = sizeof(input_t);  // 2 bytes for FP16
    static_assert(kNBytes == 2, "Only FP16 supported");
    
    // 验证配置的合理性
    static_assert(kNElts * kNThreadsPerRow == N, "kNElts * kNThreadsPerRow must equal N");
    static_assert(kRowsPerWarp * kNThreadsPerRow <= 32, "Rows per warp exceed warp size");
    static_assert(kRowsPerBlock % kRowsPerWarp == 0, "Rows per block must be multiple of rows per warp");
    
    // 计算线程和warp配置
    static constexpr int kNThreads = kRowsPerBlock * kNThreadsPerRow;  // 一个block的总线程数
    static constexpr int kWarpsPerBlock = kRowsPerBlock / kRowsPerWarp;  // block中的warp数
    static constexpr int kWarpSize = kRowsPerWarp * kNThreadsPerRow;  // warp大小
    
    // 向量化参数
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);  // 2
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;  // 向量化类型
    
    // 每个线程处理的块数（当前配置下为1）
    static constexpr int kNChunks = N / (kNElts * kNThreadsPerRow);
    
    // 共享内存大小（用于同一行内的跨warp交换，如果需要）
    static constexpr int kSmemExchangeSize = std::min(N * 4 * kRowsPerBlock, 48 * 1024);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

// ==================== 向量化加载/存储 ====================

template<int kNChunks, int kNElts, int kNThreadsPerRow, typename input_t, typename vec_t>
__device__ __forceinline__ void load_input_vectorized(
    const input_t* __restrict__ data,
    __half x_vals[kNChunks][kNElts],
    int row_id,
    int cols,
    int tid_in_row
) {
    const input_t* x = data + row_id * cols;
    
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        const int idx = (c * kNThreadsPerRow + tid_in_row) * kNElts;
        
        vec_t vec_data = *reinterpret_cast<const vec_t*>(x + idx);
        input_t* data_ptr = reinterpret_cast<input_t*>(&vec_data);
        
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            x_vals[c][i] = data_ptr[i];
        }
    }
}

template<int kNChunks, int kNElts, int kNThreadsPerRow, typename input_t, typename vec_t>
__device__ __forceinline__ void store_output_vectorized(
    input_t* __restrict__ data,
    const __half x_vals[kNChunks][kNElts],
    int row_id,
    int cols,
    int tid_in_row
) {
    input_t* out = data + row_id * cols;
    
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        const int idx = (c * kNThreadsPerRow + tid_in_row) * kNElts;
        
        vec_t temp;
        input_t* temp_ptr = reinterpret_cast<input_t*>(&temp);
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            temp_ptr[i] = x_vals[c][i];
        }
        
        *reinterpret_cast<vec_t*>(out + idx) = temp;
    }
}

// ==================== 线程内Hadamard变换 ====================

template<int kNElts>
__device__ __forceinline__ void hadamard_mult_thread(__half x[kNElts]) {
    constexpr int kLogNElts = cilog2(kNElts);
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

template<int kNElts, int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk(__half x[kNChunks][kNElts]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        hadamard_mult_thread<kNElts>(x[c]);
    }
}

// ==================== Warp内Hadamard变换（使用shuffle） ====================

template<int kLogThreadsPerRow, int kNChunks, int kNElts>
__device__ __forceinline__ void hadamard_mult_warp(__half x[kNChunks][kNElts], int tid_in_row) {
    static_assert(kLogThreadsPerRow <= 5, "Threads per row can't exceed 32");
    
    constexpr int kNThreadsPerRow = 1 << kLogThreadsPerRow;
    
    #pragma unroll
    for (int step = 0; step < kLogThreadsPerRow; ++step) {
        const int lane_mask = 1 << step;
        const bool should_negate = (tid_in_row & lane_mask);
        
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                const __half x_other = __shfl_xor_sync(0xffffffff, x[c][i], lane_mask);
                
                if (should_negate) {
                    x[c][i] = __hsub(x_other, x[c][i]);
                } else {
                    x[c][i] = __hadd(x[c][i], x_other);
                }
            }
        }
    }
}

// ==================== 主Kernel (32分组) ====================

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void hadamard_32_transform_kernel_optimized(
    __half* __restrict__ data,
    int rows,
    int cols
) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNChunks = Ktraits::kNChunks;
    constexpr int kNThreadsPerRow = Ktraits::kNThreadsPerRow;
    constexpr int kRowsPerBlock = Ktraits::kRowsPerBlock;
    constexpr int N = Ktraits::N;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    
    const int tid = threadIdx.x;
    const int tid_in_row = tid % kNThreadsPerRow;
    const int row_in_block = tid / kNThreadsPerRow;
    
    const int global_row = blockIdx.x * kRowsPerBlock + row_in_block;
    if (global_row >= rows) return;
    
    __half x_vals[kNChunks][kNElts];
    
    load_input_vectorized<kNChunks, kNElts, kNThreadsPerRow, input_t, vec_t>(
        data, x_vals, global_row, cols, tid_in_row);
    
    constexpr int kLogNElts = cilog2(kNElts);
    static_assert(1 << kLogNElts == kNElts);
    hadamard_mult_thread_chunk<kNElts, kNChunks>(x_vals);
    
    constexpr int kLogThreadsPerRow = cilog2(kNThreadsPerRow);
    static_assert(1 << kLogThreadsPerRow == kNThreadsPerRow);
    
    if constexpr (kLogThreadsPerRow > 0) {
        hadamard_mult_warp<kLogThreadsPerRow, kNChunks, kNElts>(x_vals, tid_in_row);
    }
    
    if constexpr (kNChunks > 1) {
        __half x_vals_transposed[kNElts][kNChunks];
        
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                x_vals_transposed[i][c] = x_vals[c][i];
            }
        }
        
        constexpr int kLogNChunks = cilog2(kNChunks);
        static_assert(1 << kLogNChunks == kNChunks);
        
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            hadamard_mult_thread<kNChunks>(x_vals_transposed[i]);
        }
        
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                x_vals[c][i] = x_vals_transposed[i][c];
            }
        }
    }
    
    store_output_vectorized<kNChunks, kNElts, kNThreadsPerRow, input_t, vec_t>(
        data, x_vals, global_row, cols, tid_in_row);
}

// ==================== 启动函数 ====================

template<int kNElts, int kNThreadsPerRow, int kRowsPerWarp, int kRowsPerBlock, int kLogN>
void launch_hadamard_32_kernel(torch::Tensor& data, int rows, int cols) {
    using Ktraits = hadamard_32_kernel_traits<kNElts, kNThreadsPerRow, kRowsPerWarp, kRowsPerBlock, kLogN, __half>;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kSmemSize = 0;
    
    const int num_blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    
    dim3 grid(num_blocks);
    dim3 block(kNThreads);
    
    auto kernel = &hadamard_32_transform_kernel_optimized<Ktraits>;
    
    kernel<<<grid, block, kSmemSize>>>(
        reinterpret_cast<__half*>(data.data_ptr<c10::Half>()),
        rows,
        cols
    );
}

// ==================== 主入口函数 ====================

void had_trans_32_half_cuda(torch::Tensor& data) {
    if (data.dim() != 2) {
        throw std::runtime_error(
            "Error: Input tensor must be 2D. Please reshape to 2D tensor with last dimension = 32. "
            "Current dimensions: " + std::to_string(data.dim())
        );
    }

    int rows = data.size(0);
    int cols = data.size(1);

    if (cols != 32) {
        throw std::runtime_error(
            "Error: The last dimension must be 32, but got " + std::to_string(cols)
        );
    }

    // 32分组配置：每个线程处理4个元素，共8线程处理一行
    constexpr int kNElts = 4;            // 每个线程处理4个元素
    constexpr int kNThreadsPerRow = 8;    // 8线程处理一行（1/4 warp）
    constexpr int kRowsPerWarp = 4;       // 每个warp处理4行
    constexpr int kLogN = 5;             // log2(32)
    
    // 根据行数选择每个block处理的行数
    if (rows <= 512) {
        constexpr int kRowsPerBlock = 4;
        launch_hadamard_32_kernel<kNElts, kNThreadsPerRow, kRowsPerWarp, kRowsPerBlock, kLogN>(data, rows, cols);
    } else if (rows <= 1024) {
        constexpr int kRowsPerBlock = 8;
        launch_hadamard_32_kernel<kNElts, kNThreadsPerRow, kRowsPerWarp, kRowsPerBlock, kLogN>(data, rows, cols);
    } else if (rows <= 2048) {
        constexpr int kRowsPerBlock = 16;
        launch_hadamard_32_kernel<kNElts, kNThreadsPerRow, kRowsPerWarp, kRowsPerBlock, kLogN>(data, rows, cols);
    } else if (rows <= 4096) {
        constexpr int kRowsPerBlock = 32;
        launch_hadamard_32_kernel<kNElts, kNThreadsPerRow, kRowsPerWarp, kRowsPerBlock, kLogN>(data, rows, cols);
    } else {
        constexpr int kRowsPerBlock = 64;
        launch_hadamard_32_kernel<kNElts, kNThreadsPerRow, kRowsPerWarp, kRowsPerBlock, kLogN>(data, rows, cols);
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "CUDA kernel launch failed: " + std::string(cudaGetErrorString(err))
        );
    }
}

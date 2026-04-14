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

// 32字节向量类型（用于16个FP16元素）
struct uint8 {
    uint4 x;
    uint4 y;
};

template<> struct BytesToType<32> { using Type = uint8; };
template<> struct BytesToType<16> { using Type = uint4; };
template<> struct BytesToType<8> { using Type = uint2; };
template<> struct BytesToType<4> { using Type = uint; };
template<> struct BytesToType<2> { using Type = unsigned short; };

// ==================== Kernel Traits (64分组) ====================

template<int kNElts_, int kNThreadsPerRow_, int kRowsPerWarp_, int kRowsPerBlock_, int kLogN_, typename input_t_>
struct hadamard_64_kernel_traits {
    using input_t = input_t_;
    
    // 基本配置参数
    static constexpr int kNElts = kNElts_;  // 一个线程处理的元素数（默认8）
    static constexpr int kNThreadsPerRow = kNThreadsPerRow_;  // 每行使用的线程数（默认8）
    static constexpr int kRowsPerWarp = kRowsPerWarp_;  // 一个warp处理的行数（默认4）
    static constexpr int kRowsPerBlock = kRowsPerBlock_;  // 一个block处理的行数（默认16）
    
    // 导出参数
    static constexpr int kLogN = kLogN_;  // log2(64) = 6
    static constexpr int N = 1 << kLogN;  // 64 for log2(64) = 6
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
    // tid_in_row: 该线程在行内的位置 (0 到 kNThreadsPerRow-1)
    const input_t* x = data + row_id * cols;
    
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        const int idx = (c * kNThreadsPerRow + tid_in_row) * kNElts;
        
        // 向量化加载，直接读取为 half 类型
        vec_t vec_data = *reinterpret_cast<const vec_t*>(x + idx);
        input_t* data_ptr = reinterpret_cast<input_t*>(&vec_data);
        
        // 直接复制到 half 寄存器（无需类型转换）
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
        
        // 直接向量化存储（无需类型转换）
        vec_t temp;
        input_t* temp_ptr = reinterpret_cast<input_t*>(&temp);
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            temp_ptr[i] = x_vals[c][i];
        }
        
        // 向量化存储
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
            x[idx] = __hadd(a, b);        // half 加法
            x[idx + stride] = __hsub(a, b); // half 减法
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

// 在行内进行 Hadamard 变换（同一行的线程之间交换数据）
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
                // 使用 half shuffle 交换数据（只在同一行的线程之间）
                const __half x_other = __shfl_xor_sync(0xffffffff, x[c][i], lane_mask);
                
                if (should_negate) {
                    x[c][i] = __hsub(x_other, x[c][i]);  // x_other - x = -(x - x_other)
                } else {
                    x[c][i] = __hadd(x[c][i], x_other);  // x + x_other
                }
            }
        }
    }
}

// ==================== 主Kernel (64分组) ====================

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void hadamard_64_transform_kernel_optimized(
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
    
    // 计算线程在 block 内处理的行号和在行内的位置
    const int tid = threadIdx.x;
    const int tid_in_row = tid % kNThreadsPerRow;  // 在行内的位置 [0, kNThreadsPerRow)
    const int row_in_block = tid / kNThreadsPerRow;  // 在 block 内处理的行号 [0, kRowsPerBlock)
    
    // 计算全局行号
    const int global_row = blockIdx.x * kRowsPerBlock + row_in_block;
    if (global_row >= rows) return;
    
    // 数据保持在 half 寄存器中（减少寄存器使用，提高占用率）
    __half x_vals[kNChunks][kNElts];
    
    // 1. 向量化加载
    load_input_vectorized<kNChunks, kNElts, kNThreadsPerRow, input_t, vec_t>(
        data, x_vals, global_row, cols, tid_in_row);
    
    // 2. 线程内Hadamard变换
    constexpr int kLogNElts = cilog2(kNElts);
    static_assert(1 << kLogNElts == kNElts);
    hadamard_mult_thread_chunk<kNElts, kNChunks>(x_vals);
    
    // 3. 行内线程间的Hadamard变换（使用shuffle）
    constexpr int kLogThreadsPerRow = cilog2(kNThreadsPerRow);
    static_assert(1 << kLogThreadsPerRow == kNThreadsPerRow);
    
    if constexpr (kLogThreadsPerRow > 0) {
        hadamard_mult_warp<kLogThreadsPerRow, kNChunks, kNElts>(x_vals, tid_in_row);
    }
    
    // 4. 跨Chunk的Hadamard变换（如果有多个chunks）
    if constexpr (kNChunks > 1) {
        __half x_vals_transposed[kNElts][kNChunks];
        
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                x_vals_transposed[i][c] = x_vals[c][i];
            }
        }
        
        // 对转置后的每一行（原来的列）执行Hadamard变换
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
    
    // 5. 向量化存储
    store_output_vectorized<kNChunks, kNElts, kNThreadsPerRow, input_t, vec_t>(
        data, x_vals, global_row, cols, tid_in_row);
}

// ==================== 启动函数 ====================

template<int kNElts, int kNThreadsPerRow, int kRowsPerWarp, int kRowsPerBlock, int kLogN>
void launch_hadamard_64_kernel(torch::Tensor& data, int rows, int cols) {
    using Ktraits = hadamard_64_kernel_traits<kNElts, kNThreadsPerRow, kRowsPerWarp, kRowsPerBlock, kLogN, __half>;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kSmemSize = 0;  // 当前配置下不需要共享内存
    
    // 计算需要的 block 数量
    const int num_blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    
    dim3 grid(num_blocks);
    dim3 block(kNThreads);
    
    auto kernel = &hadamard_64_transform_kernel_optimized<Ktraits>;
    
    kernel<<<grid, block, kSmemSize>>>(
        reinterpret_cast<__half*>(data.data_ptr<c10::Half>()),
        rows,
        cols
    );
}

// ==================== 主入口函数 ====================

void had_trans_64_half_cuda(torch::Tensor& data) {
    // 检查输入
    if (data.dim() != 2) {
        throw std::runtime_error(
            "Error: Input tensor must be 2D. Please reshape to 2D tensor with last dimension = 64. "
            "Current dimensions: " + std::to_string(data.dim())
        );
    }

    int rows = data.size(0);
    int cols = data.size(1);

    if (cols != 64) {
        throw std::runtime_error(
            "Error: The last dimension must be 64, but got " + std::to_string(cols)
        );
    }

    // 64分组配置：每个线程处理4个元素，共16线程处理一行
    constexpr int kNElts = 4;            // 每个线程处理4个元素
    constexpr int kNThreadsPerRow = 16;  // 16线程处理一行（半个warp）
    constexpr int kRowsPerWarp = 2;      // 每个warp处理2行
    constexpr int kLogN = 6;             // log2(64)
    
    // 根据行数选择每个block处理的行数
    if (rows <= 512) {
        constexpr int kRowsPerBlock = 2;
        launch_hadamard_64_kernel<kNElts, kNThreadsPerRow, kRowsPerWarp, kRowsPerBlock, kLogN>(data, rows, cols);
    } else if (rows <= 1024) {
        constexpr int kRowsPerBlock = 4;
        launch_hadamard_64_kernel<kNElts, kNThreadsPerRow, kRowsPerWarp, kRowsPerBlock, kLogN>(data, rows, cols);
    } else if (rows <= 2048) {
        constexpr int kRowsPerBlock = 8;
        launch_hadamard_64_kernel<kNElts, kNThreadsPerRow, kRowsPerWarp, kRowsPerBlock, kLogN>(data, rows, cols);
    } else if (rows <= 4096) {
        constexpr int kRowsPerBlock = 16;
        launch_hadamard_64_kernel<kNElts, kNThreadsPerRow, kRowsPerWarp, kRowsPerBlock, kLogN>(data, rows, cols);
    } else {
        constexpr int kRowsPerBlock = 32;
        launch_hadamard_64_kernel<kNElts, kNThreadsPerRow, kRowsPerWarp, kRowsPerBlock, kLogN>(data, rows, cols);
    }
    
    // 同步检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "CUDA kernel launch failed: " + std::string(cudaGetErrorString(err))
        );
    }
}

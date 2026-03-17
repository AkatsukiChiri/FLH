#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <random>

// Accumulator: 128 * 128 * sizeof(int32_t) = 64KB
// Block A + B: 128 * 128 * sizeof(int8_t) * 0.5 * 2 = 16KB
// Seems that loading to registers is bottleneck. Since
// with same data size, we have more computation in one block.
// But has less computation in one mma(8x8x32). Maybe we 2-stage
// pipeline in (gmem, smem) and 2-stage pipeline in (smem, reg)
// is best.
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 128

#define BLOCK_WARPS 8
#define BLOCK_ROW_WARPS 4
#define BLOCK_COL_WARPS 2

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 4

#define M 16
#define N 8
#define K 64

#define STAGE 4

// GPU configuration.
#define WARP_SIZE 32

// Quantization configuration
#define GROUP_SIZE 128

// 简化的 scale 布局：不再使用复杂的 packing
// A_scale: [M, K/GROUP_SIZE]
// B_scale: [N, K/GROUP_SIZE]

// 16 Bytes = 128 bits = 32 * sizeof(u4) -> actually per row
// Chunk means per row loading
typedef int4 copy_t;
#define CHUNK_LOAD_BYTES (BLOCK_K * sizeof(int8_t) / 2)
#define CHUNK_LOAD_LANES_PER (CHUNK_LOAD_BYTES / sizeof(copy_t))
#define CHUNK_LOAD_PER_WARP (WARP_SIZE / CHUNK_LOAD_LANES_PER)

#define E2S(x) ((x) >> 1)

// 不再需要 INT8 keeper 相关定义

// Load BLOCK_M * BLOCK_K elements from global memory to shared memory
// Cooperative loading within same block with all 8 warps (Block-level)
__device__ __forceinline__ void loadASMem(
  uint8_t *smem,       // Start address of loaded space
  const uint8_t *gmem, // Start address of this block address
  const int max_m_dimension,  // M_GLOBAL
  const int gmem_ldm,  // K_GLOBAL
  const int k,         // Current k offset
  bool predGuard       // To resolve the extra access
){
  const int warpId = threadIdx.y + threadIdx.z * blockDim.y;
  const int laneId = threadIdx.x;
  // Note: col index is counted at the granularity of copy_t
  // Row is determined by warp idx & lane idx
  // Col is determined by lane idx 
  int gmem_row = warpId * BLOCK_M / BLOCK_WARPS + laneId / CHUNK_LOAD_LANES_PER;
  int gmem_col = laneId % CHUNK_LOAD_LANES_PER;
  int smem_row = gmem_row;
  int smem_col = gmem_col ^ ((smem_row / 2) & 3);

  // Deal with M tail block: avoid illegal memory access
  // Check each lane's M-loading dimension to determine whether illegal
  predGuard = predGuard && ((gmem_row + blockIdx.y * BLOCK_M) < max_m_dimension);

  // Deal with K tail block: avoid illegal memory access
  // Check the k dimension pointer address.
  predGuard = predGuard && ((k + gmem_col * 2 * sizeof(copy_t)) < gmem_ldm);
  
  // @!p st.shared.v4.u32 is bottleneck for 20 Tops drop.
#pragma unroll
  for(int i = 0; i < BLOCK_M / BLOCK_WARPS / CHUNK_LOAD_PER_WARP; ++i){
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %0, 0;\n"
      "@!p st.shared.v4.u32 [%1], {0, 0, 0, 0};\n"
      "@p cp.async.cg.shared.global [%1], [%2], 16;\n"
      "}\n"
      ::
        "r"((int) predGuard),  
        "l"(__cvta_generic_to_shared((void*)smem) + E2S(smem_row * BLOCK_K) + sizeof(copy_t) * smem_col),
        "l"((copy_t*)(&gmem[E2S(gmem_row * gmem_ldm)]) + gmem_col)
    );
    gmem_row += CHUNK_LOAD_PER_WARP;
    smem_row += CHUNK_LOAD_PER_WARP;
    predGuard = predGuard && ((gmem_row + blockIdx.y * BLOCK_M) < max_m_dimension);
  }
}

__device__ __forceinline__ void loadBSMem(
  uint8_t *smem,       // Start address of loaded space
  const uint8_t *gmem, // Start address of this block address
  const int gmem_ldm,  // K_GLOBAL
  const int k,         // Current k offset
  bool predGuard       // To resolve the extra access
){
  const int warpId = threadIdx.y + threadIdx.z * blockDim.y;
  const int laneId = threadIdx.x;
  // Note: col index is counted at the granularity of copy_t
  // Row is determined by warp idx & lane idx
  // Col is determined by lane idx 
  int gmem_row = warpId * BLOCK_N / BLOCK_WARPS + laneId / CHUNK_LOAD_LANES_PER;
  int gmem_col = laneId % CHUNK_LOAD_LANES_PER;
  int smem_row = gmem_row;
  int smem_col = gmem_col ^ ((smem_row / 2) & 3);

  // Deal with K tail block: avoid illegal memory access
  // Note: B Matrix is always weight matrix. So we don't consider the tail block of N dimension.
  predGuard = predGuard && ((k + gmem_col * 2 * sizeof(copy_t)) < gmem_ldm);

#pragma unroll
  for(int i = 0; i < BLOCK_N / BLOCK_WARPS / CHUNK_LOAD_PER_WARP; ++i){
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %0, 0;\n"
      "@!p st.shared.v4.u32 [%1], {0, 0, 0, 0};\n"
      "@p cp.async.cg.shared.global [%1], [%2], 16;\n" 
      "}\n"
      ::
        "r"((int) predGuard),  
        "l"(__cvta_generic_to_shared((void*)smem) + E2S(smem_row * BLOCK_K) + sizeof(copy_t) * smem_col),
        "l"((copy_t*)(&gmem[E2S(gmem_row * gmem_ldm)]) + gmem_col)
    );
    gmem_row += CHUNK_LOAD_PER_WARP;
    smem_row += CHUNK_LOAD_PER_WARP;
  }
}

// Block-level function
// Note: need to output [M, N], take care of the M tail blocks
__device__ __forceinline__ void storeSMem(
  const half *smem,           // Start address of loaded space
  half *gmem,                 // Start address of this block address
  const int smem_ldm,
  const int max_m_dimension,  // M_GLOBAL
  const int gmem_ldm          // N_GLOBAL
){
  const int warpId = threadIdx.y + threadIdx.z * blockDim.y;
  const int laneId = threadIdx.x;
  // 128 * 16 / 8 = 16 * 16
  // One warp for two chunk
  int gmem_row = warpId * BLOCK_M / BLOCK_WARPS + laneId / (CHUNK_LOAD_LANES_PER * 4);
  int gmem_col = laneId % (CHUNK_LOAD_LANES_PER * 4);
  int smem_row = gmem_row;
  int smem_col = gmem_col;

#pragma unroll
  for(int i = 0; i < BLOCK_M / BLOCK_WARPS / (CHUNK_LOAD_PER_WARP / 4); ++i){
    if(gmem_row + blockIdx.y * BLOCK_M < max_m_dimension){
      *((copy_t*)(&gmem[gmem_row * gmem_ldm]) + gmem_col) =
        *((copy_t*)(smem + smem_row * smem_ldm) + smem_col);
        
      gmem_row += (CHUNK_LOAD_PER_WARP / 4);
      smem_row += (CHUNK_LOAD_PER_WARP / 4);
    }
  }
}

// Warp-level function
// Input address is warp-level specific
__device__ __forceinline__ void loadAFrag(
  int32_t *a_frag,
  const uint8_t *smem,
  const int smem_ldm,
  const int k
){
  const int tid = threadIdx.x;
  // Since each ldmatrix can load 4x8x8, we want to reduce the instruction number.
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; i += 1){
    int smem_row = i * M + tid % 16; // 16 x 64
    int smem_col = (k * 2 + tid / 16) ^ ((smem_row / 2) & 3);
    copy_t *ptr = (copy_t*)(&smem[E2S(smem_row * smem_ldm)]) + smem_col;
    asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      :  "=r"(a_frag[i * 4]), "=r"(a_frag[i * 4 + 1]), "=r"(a_frag[i * 4 + 2]), "=r"(a_frag[i * 4 + 3])
      :  "l"(__cvta_generic_to_shared(ptr))
    );
  }
}

__device__ __forceinline__ void loadBFrag(
  int32_t *b_frag,
  const uint8_t *smem,
  const int smem_ldm,
  const int k
){
  const int tid = threadIdx.x;
#pragma unroll
  for(int i = 0;i < WARP_ROW_TILES; i += 2){
    int smem_row = i * N + tid % 16;
    int smem_col = (k * 2 + tid / 16) ^ ((smem_row / 2) & 3);
    copy_t *ptr = (copy_t*)(&smem[E2S(smem_row * smem_ldm)]) + smem_col;
    asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      :  "=r"(b_frag[i * 2 + 0]), "=r"(b_frag[i * 2 + 2]), "=r"(b_frag[i * 2 + 1]), "=r"(b_frag[i * 2 + 3])
      :  "l"(__cvta_generic_to_shared(ptr))
    );    
  }
}

// Warp-level function
// Input address is warp-level specific: which means we only cares about single warp.
__device__ __forceinline__ void storeAccumulator(
  float *c_frag,  // [col, row, 2]
  half *smem,        // cast int8_t* to int*
  const int smem_ldm
){
  // According to fragment layout
  const int ti = threadIdx.x % 4;
  const int tj = threadIdx.x / 4;
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; ++i){
#pragma unroll
    for(int j = 0;j < WARP_ROW_TILES; ++j){
      half *ptr = &smem[i * smem_ldm * M + j * N];
      ptr[tj * smem_ldm + ti * 2 + 0] = __float2half(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]);
      ptr[tj * smem_ldm + ti * 2 + 1] = __float2half(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]);
      ptr[(tj+8) * smem_ldm + ti * 2 + 0] = __float2half(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]);
      ptr[(tj+8) * smem_ldm + ti * 2 + 1] = __float2half(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3]);
    }
  }
}

template <bool initZero>
__device__ __forceinline__ void mma_calculate(
  int32_t* __restrict__ c_frag,
  int32_t* __restrict__ a_frag,
  int32_t* __restrict__ b_frag
){
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; ++i){
#pragma unroll
    for(int j = 0;j < WARP_ROW_TILES; ++j){
      if constexpr (initZero){
        asm volatile(
          "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10,  %11,  %12,  %13};\n"
          : "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]),
            "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3])
          : "r"(a_frag[i * 4]), "r"(a_frag[i * 4 + 1]), "r"(a_frag[i * 4 + 2]), "r"(a_frag[i * 4 + 3]),
            "r"(b_frag[j * 2]), "r"(b_frag[j * 2 + 1]),
            "r"(0), "r"(0),
            "r"(0), "r"(0)
        );
      }else{
        asm volatile(
          "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10,  %11,  %12,  %13};\n"
          : "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]),
            "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]), "=r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3])
          : "r"(a_frag[i * 4]), "r"(a_frag[i * 4 + 1]), "r"(a_frag[i * 4 + 2]), "r"(a_frag[i * 4 + 3]),
            "r"(b_frag[j * 2]), "r"(b_frag[j * 2 + 1]),
            "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]), "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]),
            "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]), "r"(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3])
        );
      }
    }
  }
}

// 移除 INT8 keeper 的 MMA 计算函数

/*
  简化的 scale 加载函数
  A_scale: [M, K/GROUP_SIZE] FP16, row-major
  B_scale: [N, K/GROUP_SIZE] FP16, row-major
  
  每个 block 需要加载：
  - A_scale[block_m_start:block_m_end, k_group]  (BLOCK_M 个元素)
  - B_scale[block_n_start:block_n_end, k_group]  (BLOCK_N 个元素)
*/
__device__ __forceinline__ void loadScale(
  half *smem_A_scale,           // [BLOCK_M]
  half *smem_B_scale,           // [BLOCK_N]
  const half *gmem_A_scale,     // [M, K/GROUP_SIZE]
  const half *gmem_B_scale,     // [N, K/GROUP_SIZE]
  const int M_GLOBAL,
  const int N_GLOBAL,
  const int K_groups,           // K_GLOBAL / GROUP_SIZE
  const int block_m_start,      // blockIdx.y * BLOCK_M
  const int block_n_start,      // blockIdx.x * BLOCK_N
  const int k_group_idx,        // 当前处理的 K group 索引
  bool predGuard
){
  const int laneId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  
  // 加载 A_scale[block_m_start:block_m_start+BLOCK_M, k_group_idx]
  // 每个线程加载一个 FP16
  if (laneId < BLOCK_M && predGuard) {
    int m_idx = block_m_start + laneId;
    if (m_idx < M_GLOBAL) {
      smem_A_scale[laneId] = gmem_A_scale[m_idx * K_groups + k_group_idx];
    } else {
      smem_A_scale[laneId] = __float2half(1.0f);  // 填充值
    }
  }
  
  // 加载 B_scale[block_n_start:block_n_start+BLOCK_N, k_group_idx]
  if (laneId < BLOCK_N && predGuard) {
    int n_idx = block_n_start + laneId;
    if (n_idx < N_GLOBAL) {
      smem_B_scale[laneId] = gmem_B_scale[n_idx * K_groups + k_group_idx];
    } else {
      smem_B_scale[laneId] = __float2half(1.0f);  // 填充值
    }
  }
}

// Warp-level function
// 从 shared memory 加载 scale 到寄存器
// 简化版本：直接加载需要的 scale 值
__device__ __forceinline__ void loadScaleReg(
  half *reg_a,          // [WARP_COL_TILES] = 4
  half *reg_b,          // [WARP_ROW_TILES] = 4
  const half *smem_A_scale,  // [BLOCK_M]
  const half *smem_B_scale,  // [BLOCK_N]
  const int warp_m_offset,   // wj * WARP_COL_TILES * M
  const int warp_n_offset    // wi * WARP_ROW_TILES * N
){
  // 加载 A scale: 对应 warp 处理的 M 行
  #pragma unroll
  for (int i = 0; i < WARP_COL_TILES; ++i) {
    int m_idx = warp_m_offset + i * M;  // M = 16
    reg_a[i] = smem_A_scale[m_idx];
  }
  
  // 加载 B scale: 对应 warp 处理的 N 列
  #pragma unroll
  for (int j = 0; j < WARP_ROW_TILES; ++j) {
    int n_idx = warp_n_offset + j * N;  // N = 8
    reg_b[j] = smem_B_scale[n_idx];
  }
}

__device__ __forceinline__ void dequant(
  int32_t *c_frag,
  half *reg_a,      // [WARP_COL_TILES]
  half *reg_b,      // [WARP_ROW_TILES]
  float *accu
){
  // 反量化: accu[i,j] += c_frag[i,j] * scale_a[i] * scale_b[j]
#pragma unroll
  for(int i = 0; i < WARP_COL_TILES; ++i){
    float scale_a = __half2float(reg_a[i]);
#pragma unroll
    for(int j = 0; j < WARP_ROW_TILES; ++j){
      float scale_b = __half2float(reg_b[j]);
      float scale = scale_a * scale_b;
      
      // 每个 (i, j) tile 有 4 个元素
      // mma 指令产生的布局: 每个 tile 对应 2x2 的结果
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 0] += 
        (float)(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]) * scale;
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 1] += 
        (float)(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]) * scale;
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 2] +=
        (float)(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]) * scale;
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 3] +=
        (float)(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3]) * scale;
    }
  }
}

__global__ void __launch_bounds__(256) compute_gemm_imma(
  const uint8_t *A,
  const uint8_t *B,
  half *D,
  const int M_GLOBAL,
  const int N_GLOBAL,
  const int K_GLOBAL,
  const uint8_t *A_scale,
  const uint8_t *B_scale
){
  extern __shared__ uint8_t shmem[];

  // Shared memory 布局简化
  const size_t shmem_B_offset = E2S(BLOCK_M * BLOCK_K * sizeof(int8_t));
  const size_t shmem_stage_offset = E2S(BLOCK_K * (BLOCK_M + BLOCK_N) * sizeof(int8_t));
  const size_t shmem_scale_offset = STAGE * shmem_stage_offset;
  
  // Scale shared memory: 简单的 1D 数组
  // A_scale: [BLOCK_M] FP16
  // B_scale: [BLOCK_N] FP16
  half *smem_A_scale = (half*)(shmem + shmem_scale_offset);
  half *smem_B_scale = smem_A_scale + BLOCK_M;
  
  const int K_groups = K_GLOBAL / GROUP_SIZE;

  const int bi = blockIdx.x; // N dimension
  const int bj = blockIdx.y; // M dimension
  const int wi = threadIdx.y;
  const int wj = threadIdx.z;

  // m16n8k32 -> 16*32 accumulator
  int32_t c[WARP_COL_TILES * WARP_ROW_TILES * 4] = {0};
  int32_t a[2][WARP_COL_TILES * 4] = {0};
  int32_t b[2][WARP_ROW_TILES * 2] = {0};
  // real accumulator
  float c_fp[WARP_COL_TILES * WARP_ROW_TILES * 4] = {0.0f};
  half a_s[WARP_COL_TILES] = {__float2half(1.0f)};
  half b_s[WARP_ROW_TILES] = {__float2half(1.0f)};

  // Each time (writePtr + 1) % STAGE is consumed, and writePtr is produced.
  size_t writePtr = STAGE - 1;
  // Keep one unused stage for producing
#pragma unroll
  for(int i = 0; i < STAGE - 1;++i){
    loadASMem(
      shmem + i * shmem_stage_offset,
      A + E2S(bj * BLOCK_M * K_GLOBAL + i * BLOCK_K),
      M_GLOBAL,
      K_GLOBAL,
      (i * BLOCK_K),
      true
    );
    loadBSMem(
      shmem + shmem_B_offset + i * shmem_stage_offset,
      B + E2S(bi * BLOCK_N * K_GLOBAL + i * BLOCK_K),
      K_GLOBAL,
      (i * BLOCK_K),
      true
    );
    // 注意：scale 在 K 循环中每个 group 需要重新加载
    // 但在 prologue 中我们跳过 scale 加载，在主循环中加载
    asm volatile("cp.async.commit_group;\n" ::);
  }
  asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGE - 2));
  // __syncthreads();

  loadAFrag(
    a[0],
    shmem + E2S(wj * WARP_COL_TILES * M * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
    BLOCK_K,
    0
  );
  loadBFrag(
    b[0],
    shmem + shmem_B_offset + E2S(wi * WARP_ROW_TILES * N * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
    BLOCK_K,
    0
  );
  // Main loop is calculated at the unit of calculation
  // Use predicate reg to avoid unnecessary if instruction
  for(int k = 0; k < K_GLOBAL; k += BLOCK_K){
    // 在每个 K 迭代开始时加载对应的 scale
    int k_group_idx = k / GROUP_SIZE;
    loadScale(
      smem_A_scale,
      smem_B_scale,
      (const half*)A_scale,
      (const half*)B_scale,
      M_GLOBAL,
      N_GLOBAL,
      K_groups,
      bj * BLOCK_M,
      bi * BLOCK_N,
      k_group_idx,
      true
    );
    __syncthreads();  // 确保所有线程都加载完 scale
    
    // 加载 scale 到寄存器
    loadScaleReg(
      a_s,
      b_s,
      smem_A_scale,
      smem_B_scale,
      wj * WARP_COL_TILES * M,
      wi * WARP_ROW_TILES * N
    );
    
    loadAFrag(
      a[1],
      shmem + E2S(wj * WARP_COL_TILES * M * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      1
    );
    loadBFrag(
      b[1],
      shmem + shmem_B_offset + E2S(wi * WARP_ROW_TILES * N * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      1
    );
    
    mma_calculate<true>(c, a[0], b[0]);
    // Pipeline load
    bool predGuard = (k + (STAGE - 1) * BLOCK_K) < K_GLOBAL;
    loadASMem(
      shmem + writePtr * shmem_stage_offset,
      A + E2S(bj * BLOCK_M * K_GLOBAL + k + (STAGE - 1) * BLOCK_K),
      M_GLOBAL,
      K_GLOBAL,
      (k + (STAGE - 1) * BLOCK_K),
      predGuard
    );
    loadBSMem(
      shmem + shmem_B_offset + writePtr * shmem_stage_offset,
      B + E2S(bi * BLOCK_N * K_GLOBAL + k + (STAGE - 1) * BLOCK_K),
      K_GLOBAL,
      (k + (STAGE - 1) * BLOCK_K),
      predGuard
    );
    asm volatile("cp.async.commit_group;\n" ::);
    mma_calculate<false>(c, a[1], b[1]);
    asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGE - 2));
    writePtr = (writePtr + 1) % STAGE;
    // __syncthreads();
    loadAFrag(
      a[0],
      shmem + E2S(wj * WARP_COL_TILES * M * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      0
    );
    loadBFrag(
      b[0],
      shmem + shmem_B_offset + E2S(wi * WARP_ROW_TILES * N * BLOCK_K) + (writePtr + 1) % STAGE * shmem_stage_offset,
      BLOCK_K,
      0
    );
    dequant(
      c,
      a_s,
      b_s,
      c_fp
    );
  }

  // Offload accumulator from registers to shared memory
  // Extra stage is for avoiding random access to shared memory
  storeAccumulator(
    c_fp,
    (half *)shmem + wj * WARP_COL_TILES * M * BLOCK_N + wi * WARP_ROW_TILES * N,
    BLOCK_N
  );
  // __syncthreads();

  // Write back to global memory
  storeSMem(
    (half *)shmem,
    (half *)D + bj * BLOCK_M * N_GLOBAL + bi * BLOCK_N,
    BLOCK_N,
    M_GLOBAL,
    N_GLOBAL
  );
}

/*!
 * \brief Pure INT4 GEMM kernel with group-wise quantization
 * \brief Assume quantization group size = 128
 * \param A INT4 matrix in global memory. Packed in uint8_t. [M, K] row-major, stored as [M, K/2]
 * \param B INT4 matrix in global memory. Packed in uint8_t. [N, K] row-major, stored as [N, K/2]
 * \param A_scale Scale for A. FP16 (half). [M, K/128] row-major
 * \param B_scale Scale for B. FP16 (half). [N, K/128] row-major
 * \param D Output matrix in global memory. [M, N] row-major. FP16
 * \param M_GLOBAL Number of rows of matrix A
 * \param N_GLOBAL Number of rows of matrix B (output columns)
 * \param K_GLOBAL K dimension (hidden dimension)
 * 
 * Note: group_size固定为128
 * Note: Scale 布局已简化为简单的 2D row-major 张量
*/
void DenseLayerGEMM_i4_o16(
  const uint8_t *A,
  const uint8_t *B,
  const uint8_t *A_scale,
  const uint8_t *B_scale,
  half *D,
  const size_t M_GLOBAL,
  const size_t N_GLOBAL,
  const size_t K_GLOBAL
){
  dim3 gridDim(
    (N_GLOBAL + BLOCK_N - 1) / BLOCK_N,
    (M_GLOBAL + BLOCK_M - 1) / BLOCK_M
  );
  dim3 blockDim(
    WARP_SIZE,
    BLOCK_ROW_WARPS,
    BLOCK_COL_WARPS
  );

  // Shared memory 大小计算 (简化版)
  // Part 1: A + B 数据 (带 stage)
  constexpr size_t shmem_data_size = sizeof(uint8_t) * BLOCK_K * (BLOCK_M + BLOCK_N) / 2 * STAGE;
  // Part 2: Scale 数据 (简单的 1D 数组)
  constexpr size_t shmem_scale_size = sizeof(half) * (BLOCK_M + BLOCK_N);
  // Part 3: 存储输出时需要的 shared memory
  constexpr size_t shmem_output_size = BLOCK_M * BLOCK_N * sizeof(half);
  
  constexpr size_t shmem_size1 = shmem_data_size + shmem_scale_size;
  constexpr size_t SHMEM_SZ = shmem_size1 > shmem_output_size ? shmem_size1 : shmem_output_size;

  cudaFuncSetAttribute(
    compute_gemm_imma,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    SHMEM_SZ
  );
  
  compute_gemm_imma<<<gridDim, blockDim, SHMEM_SZ>>>(
    A, B, D,
    M_GLOBAL, N_GLOBAL, K_GLOBAL,
    A_scale, B_scale
  );
}

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// GPU configuration
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

// Quantization configuration - parameterized
template<int GROUP_SIZE>
struct GemmConfig;

template<>
struct GemmConfig<128> {
    static constexpr int GROUP_SIZE = 128;
    static constexpr int K_GROUPS = BLOCK_K / 128; // always 1 for BLOCK_K=128
};

template<>
struct GemmConfig<64> {
    static constexpr int GROUP_SIZE = 64;
    static constexpr int K_GROUPS = BLOCK_K / 64; // 2
};

template<>
struct GemmConfig<32> {
    static constexpr int GROUP_SIZE = 32;
    static constexpr int K_GROUPS = BLOCK_K / 32; // 4
};

// 16 Bytes = 128 bits = 32 * sizeof(u4)
typedef int4 copy_t;
#define CHUNK_LOAD_BYTES (BLOCK_K * sizeof(int8_t) / 2)
#define CHUNK_LOAD_LANES_PER (CHUNK_LOAD_BYTES / sizeof(copy_t))
#define CHUNK_LOAD_PER_WARP (WARP_SIZE / CHUNK_LOAD_LANES_PER)

#define E2S(x) ((x) >> 1)

__device__ __forceinline__ void loadASMem(
  uint8_t *smem,
  const uint8_t *gmem,
  const int max_m_dimension,
  const int gmem_ldm,
  const int k,
  bool predGuard
){
  const int warpId = threadIdx.y + threadIdx.z * blockDim.y;
  const int laneId = threadIdx.x;
  int gmem_row = warpId * BLOCK_M / BLOCK_WARPS + laneId / CHUNK_LOAD_LANES_PER;
  int gmem_col = laneId % CHUNK_LOAD_LANES_PER;
  int smem_row = gmem_row;
  int smem_col = gmem_col ^ ((smem_row / 2) & 3);

  predGuard = predGuard && ((gmem_row + blockIdx.y * BLOCK_M) < max_m_dimension);
  predGuard = predGuard && ((k + gmem_col * 2 * sizeof(copy_t)) < gmem_ldm);

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
  uint8_t *smem,
  const uint8_t *gmem,
  const int gmem_ldm,
  const int k,
  bool predGuard
){
  const int warpId = threadIdx.y + threadIdx.z * blockDim.y;
  const int laneId = threadIdx.x;
  int gmem_row = warpId * BLOCK_N / BLOCK_WARPS + laneId / CHUNK_LOAD_LANES_PER;
  int gmem_col = laneId % CHUNK_LOAD_LANES_PER;
  int smem_row = gmem_row;
  int smem_col = gmem_col ^ ((smem_row / 2) & 3);

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

__device__ __forceinline__ void storeSMem(
  const half *smem,
  half *gmem,
  const int smem_ldm,
  const int max_m_dimension,
  const int gmem_ldm
){
  const int warpId = threadIdx.y + threadIdx.z * blockDim.y;
  const int laneId = threadIdx.x;
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

__device__ __forceinline__ void loadAFrag(
  int32_t *a_frag,
  const uint8_t *smem,
  const int smem_ldm,
  const int k
){
  const int tid = threadIdx.x;
#pragma unroll
  for(int i = 0;i < WARP_COL_TILES; i += 1){
    int smem_row = i * M + tid % 16;
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

__device__ __forceinline__ void storeAccumulator(
  float *c_frag,
  half *smem,
  const int smem_ldm
){
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

__device__ __forceinline__ void loadScale(
  half *smem_A_scale,
  half *smem_B_scale,
  const half *gmem_A_scale,
  const half *gmem_B_scale,
  const int M_GLOBAL,
  const int N_GLOBAL,
  const int K_groups,
  const int block_m_start,
  const int block_n_start,
  const int k_group_idx,
  bool predGuard
){
  const int laneId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

  if (laneId < BLOCK_M && predGuard) {
    int m_idx = block_m_start + laneId;
    smem_A_scale[laneId] = (m_idx < M_GLOBAL) ? gmem_A_scale[m_idx * K_groups + k_group_idx] : __float2half(1.0f);
  }
  if (laneId < BLOCK_N && predGuard) {
    int n_idx = block_n_start + laneId;
    smem_B_scale[laneId] = (n_idx < N_GLOBAL) ? gmem_B_scale[n_idx * K_groups + k_group_idx] : __float2half(1.0f);
  }
}

__device__ __forceinline__ void loadScaleReg(
  half *reg_a,
  half *reg_b,
  const half *smem_A_scale,
  const half *smem_B_scale,
  const int warp_m_offset,
  const int warp_n_offset
){
#pragma unroll
  for (int i = 0; i < WARP_COL_TILES; ++i) {
    int m_idx = warp_m_offset + i * M;
    reg_a[i] = smem_A_scale[m_idx];
  }
#pragma unroll
  for (int j = 0; j < WARP_ROW_TILES; ++j) {
    int n_idx = warp_n_offset + j * N;
    reg_b[j] = smem_B_scale[n_idx];
  }
}

__device__ __forceinline__ void dequant_accumulate(
  int32_t *c_frag,
  const half *smem_A_scale,
  const half *smem_B_scale,
  int warp_m_base,
  int warp_n_base,
  float *accu
){
  const int ti = threadIdx.x & 3;
  const int tj = threadIdx.x >> 2;
#pragma unroll
  for(int i = 0; i < WARP_COL_TILES; ++i){
    int row_base = warp_m_base + i * M;
#pragma unroll
    for(int j = 0; j < WARP_ROW_TILES; ++j){
      int col_base = warp_n_base + j * N;
      float a0 = __half2float(smem_A_scale[row_base + tj]);
      float a1 = __half2float(smem_A_scale[row_base + tj + 8]);
      float b0 = __half2float(smem_B_scale[col_base + ti * 2 + 0]);
      float b1 = __half2float(smem_B_scale[col_base + ti * 2 + 1]);
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 0] += (float)(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]) * (a0 * b0);
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 1] += (float)(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]) * (a0 * b1);
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 2] += (float)(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]) * (a1 * b0);
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 3] += (float)(c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3]) * (a1 * b1);
    }
  }
}

// Template kernel with configurable GROUP_SIZE
template<int GROUP_SIZE>
__global__ void flh_gemm_i4_dequant_o16_kernel_gs(
  const uint8_t *A,
  const uint8_t *B,
  half *D,
  const int M_GLOBAL,
  const int N_GLOBAL,
  const int K_GLOBAL,
  const half *A_scale,
  const half *B_scale
){
  extern __shared__ uint8_t shmem[];

  const size_t shmem_B_offset = E2S(BLOCK_M * BLOCK_K * sizeof(int8_t));
  const size_t shmem_stage_offset = E2S(BLOCK_K * (BLOCK_M + BLOCK_N) * sizeof(int8_t));
  const size_t shmem_scale_offset = STAGE * shmem_stage_offset;

  half *smem_A_scale = (half*)(shmem + shmem_scale_offset);
  half *smem_B_scale = smem_A_scale + BLOCK_M;

  constexpr int GROUP_SIZE_VAL = GemmConfig<GROUP_SIZE>::GROUP_SIZE;
  constexpr int K_GROUPS_PER_BLOCK = GemmConfig<GROUP_SIZE>::K_GROUPS;
  const int K_groups = K_GLOBAL / GROUP_SIZE_VAL;

  const int bi = blockIdx.x;
  const int bj = blockIdx.y;
  const int wi = threadIdx.y;
  const int wj = threadIdx.z;

  int32_t c[WARP_COL_TILES * WARP_ROW_TILES * 4] = {0};
  int32_t a[2][WARP_COL_TILES * 4] = {0};
  int32_t b[2][WARP_ROW_TILES * 2] = {0};
  float c_fp[WARP_COL_TILES * WARP_ROW_TILES * 4] = {0.0f};
  const int warp_m_base = wj * WARP_COL_TILES * M;
  const int warp_n_base = wi * WARP_ROW_TILES * N;

  size_t writePtr = STAGE - 1;
#pragma unroll
  for(int i = 0; i < STAGE - 1; ++i){
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
    asm volatile("cp.async.commit_group;\n" ::);
  }
  asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGE - 2));
  __syncthreads();

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

  for(int k = 0; k < K_GLOBAL; k += BLOCK_K){
    int k_group_idx = k / GROUP_SIZE_VAL;
    loadScale(
      smem_A_scale,
      smem_B_scale,
      A_scale,
      B_scale,
      M_GLOBAL,
      N_GLOBAL,
      K_groups,
      bj * BLOCK_M,
      bi * BLOCK_N,
      k_group_idx,
      true
    );
    __syncthreads();

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
    __syncthreads();
    writePtr = (writePtr + 1) % STAGE;

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

    dequant_accumulate(c, smem_A_scale, smem_B_scale, warp_m_base, warp_n_base, c_fp);
  }

  storeAccumulator(
    c_fp,
    (half *)shmem + wj * WARP_COL_TILES * M * BLOCK_N + wi * WARP_ROW_TILES * N,
    BLOCK_N
  );
  __syncthreads();

  storeSMem(
    (half *)shmem,
    (half *)D + bj * BLOCK_M * N_GLOBAL + bi * BLOCK_N,
    BLOCK_N,
    M_GLOBAL,
    N_GLOBAL
  );
}

// ==================== Template Instantiations ====================

template __global__ void flh_gemm_i4_dequant_o16_kernel_gs<32>(
  const uint8_t *A,
  const uint8_t *B,
  half *D,
  const int M_GLOBAL,
  const int N_GLOBAL,
  const int K_GLOBAL,
  const half *A_scale,
  const half *B_scale
);

template __global__ void flh_gemm_i4_dequant_o16_kernel_gs<64>(
  const uint8_t *A,
  const uint8_t *B,
  half *D,
  const int M_GLOBAL,
  const int N_GLOBAL,
  const int K_GLOBAL,
  const half *A_scale,
  const half *B_scale
);

template __global__ void flh_gemm_i4_dequant_o16_kernel_gs<128>(
  const uint8_t *A,
  const uint8_t *B,
  half *D,
  const int M_GLOBAL,
  const int N_GLOBAL,
  const int K_GLOBAL,
  const half *A_scale,
  const half *B_scale
);

// ==================== Host Functions ====================

template<int GROUP_SIZE>
void flhDenseLayerGEMM_i4_o16_gs_impl(
    const uint8_t *A,
    const uint8_t *B,
    half *D,
    size_t M_GLOBAL,
    size_t N_GLOBAL,
    size_t K_GLOBAL,
    const half *A_scale,
    const half *B_scale
) {
    // Verify K_GLOBAL is divisible by GROUP_SIZE
    assert(K_GLOBAL % GROUP_SIZE == 0);
    
    constexpr int GROUP_SIZE_VAL = GemmConfig<GROUP_SIZE>::GROUP_SIZE;
    
    dim3 gridDim(
        (N_GLOBAL + BLOCK_N - 1) / BLOCK_N,
        (M_GLOBAL + BLOCK_M - 1) / BLOCK_M
    );
    dim3 blockDim(
        WARP_SIZE,
        BLOCK_ROW_WARPS,
        BLOCK_COL_WARPS
    );

    constexpr size_t shmem_data_size = sizeof(uint8_t) * BLOCK_K * (BLOCK_M + BLOCK_N) / 2 * STAGE;
    constexpr size_t shmem_scale_size = sizeof(half) * (BLOCK_M + BLOCK_N);
    constexpr size_t shmem_output_size = BLOCK_M * BLOCK_N * sizeof(half);
    constexpr size_t shmem_size1 = shmem_data_size + shmem_scale_size;
    constexpr size_t SHMEM_SZ = shmem_size1 > shmem_output_size ? shmem_size1 : shmem_output_size;

    cudaFuncSetAttribute(
        flh_gemm_i4_dequant_o16_kernel_gs<GROUP_SIZE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SHMEM_SZ
    );

    flh_gemm_i4_dequant_o16_kernel_gs<GROUP_SIZE><<<gridDim, blockDim, SHMEM_SZ>>>(
        A, B, D,
        (int)M_GLOBAL, (int)N_GLOBAL, (int)K_GLOBAL,
        A_scale, B_scale
    );
}

// Explicit template instantiations for host functions
template void flhDenseLayerGEMM_i4_o16_gs_impl<32>(
    const uint8_t *A,
    const uint8_t *B,
    half *D,
    size_t M_GLOBAL,
    size_t N_GLOBAL,
    size_t K_GLOBAL,
    const half *A_scale,
    const half *B_scale
);

template void flhDenseLayerGEMM_i4_o16_gs_impl<64>(
    const uint8_t *A,
    const uint8_t *B,
    half *D,
    size_t M_GLOBAL,
    size_t N_GLOBAL,
    size_t K_GLOBAL,
    const half *A_scale,
    const half *B_scale
);

template void flhDenseLayerGEMM_i4_o16_gs_impl<128>(
    const uint8_t *A,
    const uint8_t *B,
    half *D,
    size_t M_GLOBAL,
    size_t N_GLOBAL,
    size_t K_GLOBAL,
    const half *A_scale,
    const half *B_scale
);

// Unified host function with switch
void flhDenseLayerGEMM_i4_o16_gs(
    int group_size,
    const uint8_t *A,
    const uint8_t *B,
    half *D,
    size_t M_GLOBAL,
    size_t N_GLOBAL,
    size_t K_GLOBAL,
    const half *A_scale,
    const half *B_scale
) {
    assert(K_GLOBAL % group_size == 0);
    
    switch (group_size) {
        case 32:
            flhDenseLayerGEMM_i4_o16_gs_impl<32>(A, B, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, A_scale, B_scale);
            break;
        case 64:
            flhDenseLayerGEMM_i4_o16_gs_impl<64>(A, B, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, A_scale, B_scale);
            break;
        case 128:
        default:
            flhDenseLayerGEMM_i4_o16_gs_impl<128>(A, B, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, A_scale, B_scale);
            break;
    }
}

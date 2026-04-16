#!/bin/bash

# FLH CUDA扩展编译脚本
# 用法: ./build.sh [clean]

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 清理旧的编译文件
clean() {
    echo_info "清理torch扩展缓存..."
    rm -rf /home/mashaobo/.cache/torch_extensions/py311_cu124/flh__CUDA
    echo_info "清理完成"
}

# 编译CUDA扩展
build() {
    echo_info "开始编译CUDA扩展..."

    # 检测CUDA架构（可选）
    if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
        echo_warn "TORCH_CUDA_ARCH_LIST未设置，将编译所有可见CUDA架构"
        echo_warn "建议设置: export TORCH_CUDA_ARCH_LIST=8.6  (对于RTX 30/40系列)"
    fi

    # 编译扩展
    python -c "from flh.cuda import load_flh_cuda_ext; load_flh_cuda_ext()"

    if [ $? -eq 0 ]; then
        echo_info "编译成功！"
    else
        echo_error "编译失败！"
        exit 1
    fi
}

# 测试编译结果
test_build() {
    echo_info "测试编译结果..."
    python -c "
import torch
from flh.cuda import hadamard_transform_64_half, hadamard_transform_32_half

# 测试64分组
x = torch.randn(128, 64, dtype=torch.float16, device='cuda')
hadamard_transform_64_half(x)
print('64分组: OK')

# 测试32分组
x = torch.randn(128, 32, dtype=torch.float16, device='cuda')
hadamard_transform_32_half(x)
print('32分组: OK')

print('所有测试通过!')
"
}

# 测试 group size flexible 函数
test_gs() {
    echo_info "测试 group size flexible 函数..."
    python -c "
import torch
from flh.cuda import (
    hadamard_and_quantize_i4_gs,
    quant_and_pack_i4_gs,
    gemm_i4_dequant_o16_gs
)

# 测试 hadamard_and_quantize_i4_gs
for gs in [32, 64, 128]:
    x = torch.randn(64, gs, dtype=torch.float16, device='cuda')
    q, s = hadamard_and_quantize_i4_gs(x, gs)
    assert q.shape == (64, gs // 2), f'hadamard_and_quantize_i4_gs {gs} quant shape'
    assert s.shape == (64,), f'hadamard_and_quantize_i4_gs {gs} scale shape'
    print(f'hadamard_and_quantize_i4_gs (gs={gs}): OK')

# 测试 quant_and_pack_i4_gs
for gs in [32, 64, 128]:
    x = torch.randn(64, gs, dtype=torch.float16, device='cuda')
    q, s = quant_and_pack_i4_gs(x, gs)
    assert q.shape == (64, gs // 2), f'quant_and_pack_i4_gs {gs} quant shape'
    assert s.shape == (64,), f'quant_and_pack_i4_gs {gs} scale shape'
    print(f'quant_and_pack_i4_gs (gs={gs}): OK')

# 测试 gemm_i4_dequant_o16_gs
for gs in [32, 64, 128]:
    M, N, K = 128, 256, gs * 4  # 确保 K 可被 group_size 整除
    A = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device='cuda')
    B = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device='cuda')
    A_scale = torch.randn(M, K // gs, dtype=torch.float16, device='cuda').abs() + 0.1
    B_scale = torch.randn(N, K // gs, dtype=torch.float16, device='cuda').abs() + 0.1
    D = gemm_i4_dequant_o16_gs(A, B, A_scale, B_scale, gs)
    assert D.shape == (M, N), f'gemm_i4_dequant_o16_gs {gs} output shape'
    print(f'gemm_i4_dequant_o16_gs (gs={gs}): OK')

print('所有 group size 测试通过!')
"
}

# 主程序
case "${1:-}" in
    clean)
        clean
        ;;
    rebuild)
        clean
        build
        ;;
    test)
        test_build
        ;;
    test_gs)
        test_gs
        ;;
    *)
        echo "用法: $0 {clean|rebuild|test|test_gs}"
        echo "  clean   - 清理编译缓存"
        echo "  rebuild - 清理并重新编译"
        echo "  test    - 测试基础编译结果"
        echo "  test_gs - 测试 group size flexible 函数"
        exit 1
        ;;
esac

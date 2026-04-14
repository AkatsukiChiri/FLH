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
    *)
        echo "用法: $0 {clean|rebuild|test}"
        echo "  clean   - 清理编译缓存"
        echo "  rebuild - 清理并重新编译"
        echo "  test    - 测试编译结果"
        exit 1
        ;;
esac

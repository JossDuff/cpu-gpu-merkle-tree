#!/bin/bash

# Compilation script for all tests
# Usage: ./compile_tests.sh

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Compiling Merkle Tree Tests                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check for CUDA
if command -v nvcc &> /dev/null; then
    HAS_CUDA=true
    echo "✓ CUDA found: $(nvcc --version | grep release | cut -d' ' -f5)"
else
    HAS_CUDA=false
    echo "⚠  CUDA not found - GPU tests will be disabled"
fi

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "Compiling: test_all_correctness (comprehensive verification)"
echo "─────────────────────────────────────────────────────────────"

if [ "$HAS_CUDA" = true ]; then
    # Compile with CUDA support
    g++ -std=c++17 -DHAS_CUDA -I/usr/local/cuda/include \
        test_all_correctness.cpp \
        src/merkle_tree_multithread.cpp \
        src/merkle_tree_gpu.cu \
        src/merkle_tree_gpu_adaptive.cpp \
        src/merkle_tree_gpu_streams.cpp \
        src/adaptive_chunk_optimizer.cpp \
        -o test_all_correctness \
        -L/usr/local/cuda/lib64 -lcudart -lssl -lcrypto -lpthread \
        -w  # Suppress warnings for cleaner output

    echo "✓ Compiled with CUDA support"
else
    # Compile without CUDA
    g++ -std=c++17 \
        test_all_correctness.cpp \
        src/merkle_tree_multithread.cpp \
        -o test_all_correctness \
        -lssl -lcrypto -lpthread \
        -w

    echo "✓ Compiled (CPU only - no CUDA)"
fi

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "Compiling: test_correctness (original 2-implementation test)"
echo "─────────────────────────────────────────────────────────────"

if [ "$HAS_CUDA" = true ]; then
    g++ -std=c++17 -DHAS_CUDA -I/usr/local/cuda/include \
        test_correctness.cpp \
        src/merkle_tree_multithread.cpp \
        src/merkle_tree_gpu.cu \
        -o test_correctness \
        -L/usr/local/cuda/lib64 -lcudart -lssl -lcrypto -lpthread \
        -w

    echo "✓ Compiled with CUDA support"
else
    g++ -std=c++17 \
        test_correctness.cpp \
        src/merkle_tree_multithread.cpp \
        -o test_correctness \
        -lssl -lcrypto -lpthread \
        -w

    echo "✓ Compiled (CPU only)"
fi

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "Compiling: demo_streams (CUDA streams demonstration)"
echo "─────────────────────────────────────────────────────────────"

if [ "$HAS_CUDA" = true ]; then
    g++ -std=c++17 -DHAS_CUDA -I/usr/local/cuda/include \
        demo_streams.cpp \
        src/merkle_tree_multithread.cpp \
        src/merkle_tree_gpu.cu \
        src/merkle_tree_gpu_streams.cpp \
        -o demo_streams \
        -L/usr/local/cuda/lib64 -lcudart -lssl -lcrypto -lpthread \
        -w

    echo "✓ Compiled"
else
    echo "⊘ Skipped (requires CUDA)"
fi

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "Compiling: demo_adaptive (adaptive optimizer demonstration)"
echo "─────────────────────────────────────────────────────────────"

if [ "$HAS_CUDA" = true ]; then
    g++ -std=c++17 -DHAS_CUDA -I/usr/local/cuda/include \
        demo_adaptive.cpp \
        src/merkle_tree_multithread.cpp \
        src/merkle_tree_gpu.cu \
        src/merkle_tree_gpu_adaptive.cpp \
        src/adaptive_chunk_optimizer.cpp \
        -o demo_adaptive \
        -L/usr/local/cuda/lib64 -lcudart -lssl -lcrypto -lpthread \
        -w

    echo "✓ Compiled"
else
    echo "⊘ Skipped (requires CUDA)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ Compilation Complete                                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Available executables:"
echo "  ./test_all_correctness   - Comprehensive verification (all implementations)"
echo "  ./test_correctness       - Original 2-implementation test"
if [ "$HAS_CUDA" = true ]; then
    echo "  ./demo_streams           - CUDA streams demonstration"
    echo "  ./demo_adaptive          - Adaptive optimizer demonstration"
fi
echo ""
echo "To verify correctness, run:"
echo "  $ ./test_all_correctness"
echo ""

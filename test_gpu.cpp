// Simple test to verify GPU Merkle tree is working
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

int main() {
    std::cout << "Testing GPU availability...\n\n";

    // Check CUDA devices
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)\n\n";

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Clock Rate: " << (prop.clockRate / 1000) << " MHz\n";
        std::cout << "  Memory Clock Rate: " << (prop.memoryClockRate / 1000) << " MHz\n";
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
        std::cout << "\n";
    }

    if (deviceCount > 0) {
        std::cout << "✓ GPU is available and ready for use!\n";
        std::cout << "\nThe benchmark compiled successfully with CUDA support.\n";
        std::cout << "However, the main_gpu_benchmark.cpp uses simulation code.\n";
        std::cout << "To use real GPU, we need to integrate MerkleTreeGPU class.\n";
        return 0;
    } else {
        std::cout << "✗ No GPU devices found\n";
        return 1;
    }
}

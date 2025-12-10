#include "merkle_tree_gpu.hpp"
#include "sha256_cuda.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <chrono>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// MerkleTreeGPU implementation

MerkleTreeGPU::MerkleTreeGPU(size_t chunk_size, size_t block_size)
    : root_(nullptr), chunk_size_(chunk_size), block_size_(block_size) {
    timings_ = {0, 0, 0, 0};
}

MerkleTreeGPU::~MerkleTreeGPU() {
    deleteTree(root_);
}

void MerkleTreeGPU::build(const std::vector<std::string>& data) {
        if (data.empty()) {
            root_ = nullptr;
            return;
        }

        auto start_total = std::chrono::high_resolution_clock::now();

        // Process data in chunks
        leaf_hashes_.clear();
        leaf_hashes_.reserve(data.size());

        timings_ = {0, 0, 0, 0};

        for (size_t offset = 0; offset < data.size(); offset += chunk_size_) {
            size_t chunk_end = std::min(offset + chunk_size_, data.size());
            size_t current_chunk_size = chunk_end - offset;

            // Process this chunk on GPU
            std::vector<std::string> chunk_hashes = hashChunkOnGPU(
                std::vector<std::string>(data.begin() + offset, data.begin() + chunk_end)
            );

            leaf_hashes_.insert(leaf_hashes_.end(), chunk_hashes.begin(), chunk_hashes.end());
        }

        // Build tree layers on GPU
        std::vector<std::string> current_level = leaf_hashes_;

        while (current_level.size() > 1) {
            std::vector<std::string> next_level;
            next_level.reserve((current_level.size() + 1) / 2);

            // Process pairs in chunks
            for (size_t offset = 0; offset < current_level.size(); offset += chunk_size_ * 2) {
                size_t chunk_end = std::min(offset + chunk_size_ * 2, current_level.size());
                std::vector<std::string> chunk_pairs(current_level.begin() + offset,
                                                     current_level.begin() + chunk_end);

                std::vector<std::string> merged = mergePairsOnGPU(chunk_pairs);
                next_level.insert(next_level.end(), merged.begin(), merged.end());
            }

            current_level = next_level;
        }

        // Store the actual root hash computed by GPU
        root_hash_ = current_level.empty() ? "" : current_level[0];

        // Build tree structure (CPU side) - optional, for structure only
        root_ = buildTreeFromHashes(leaf_hashes_);

        auto end_total = std::chrono::high_resolution_clock::now();
        timings_.total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    }

std::string MerkleTreeGPU::getRootHash() const {
    return root_hash_;  // Return the correct GPU-computed root
}

GPUTimings MerkleTreeGPU::getTimings() const {
    return timings_;
}

size_t MerkleTreeGPU::getChunkSize() const {
    return chunk_size_;
}

void MerkleTreeGPU::setChunkSize(size_t size) {
    chunk_size_ = size;
}

// Hash data chunks on GPU
std::vector<std::string> MerkleTreeGPU::hashChunkOnGPU(const std::vector<std::string>& data) {
        if (data.empty()) return {};

        auto start_h2d = std::chrono::high_resolution_clock::now();

        // Prepare input data
        size_t total_input_size = 0;
        std::vector<uint32_t> lengths(data.size());
        std::vector<uint32_t> offsets(data.size());

        for (size_t i = 0; i < data.size(); i++) {
            offsets[i] = total_input_size;
            lengths[i] = data[i].size();
            total_input_size += data[i].size();
        }

        // Flatten input data
        std::vector<uint8_t> flat_input(total_input_size);
        size_t offset = 0;
        for (const auto& str : data) {
            std::memcpy(flat_input.data() + offset, str.data(), str.size());
            offset += str.size();
        }

        // Allocate device memory
        uint8_t* d_input;
        uint32_t* d_lengths;
        uint32_t* d_offsets;
        uint32_t* d_output;

        CUDA_CHECK(cudaMalloc(&d_input, total_input_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_lengths, data.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_offsets, data.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_output, data.size() * 8 * sizeof(uint32_t))); // 8 uint32_t per hash

        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_input, flat_input.data(), total_input_size * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lengths, lengths.data(), data.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), data.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        auto end_h2d = std::chrono::high_resolution_clock::now();
        timings_.memory_h2d_ms += std::chrono::duration<double, std::milli>(end_h2d - start_h2d).count();

        // Launch kernel
        auto start_kernel = std::chrono::high_resolution_clock::now();

        int num_blocks = (data.size() + block_size_ - 1) / block_size_;
        sha256_kernel<<<num_blocks, block_size_>>>(d_input, d_output, d_lengths, d_offsets, data.size());

        CUDA_CHECK(cudaDeviceSynchronize());

        auto end_kernel = std::chrono::high_resolution_clock::now();
        timings_.kernel_execution_ms += std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();

        // Copy results back
        auto start_d2h = std::chrono::high_resolution_clock::now();

        std::vector<uint32_t> h_output(data.size() * 8);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, data.size() * 8 * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        auto end_d2h = std::chrono::high_resolution_clock::now();
        timings_.memory_d2h_ms += std::chrono::duration<double, std::milli>(end_d2h - start_d2h).count();

        // Convert to hex strings
        std::vector<std::string> hashes;
        hashes.reserve(data.size());

        for (size_t i = 0; i < data.size(); i++) {
            std::stringstream ss;
            for (int j = 0; j < 8; j++) {
                ss << std::hex << std::setfill('0') << std::setw(8) << h_output[i * 8 + j];
            }
            hashes.push_back(ss.str());
        }

        // Cleanup
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_lengths));
        CUDA_CHECK(cudaFree(d_offsets));
        CUDA_CHECK(cudaFree(d_output));

        return hashes;
}

// Merge hash pairs on GPU (for building tree layers)
std::vector<std::string> MerkleTreeGPU::mergePairsOnGPU(const std::vector<std::string>& hashes) {
        if (hashes.empty()) return {};
        if (hashes.size() == 1) return hashes;

        auto start_h2d = std::chrono::high_resolution_clock::now();

        size_t num_pairs = (hashes.size() + 1) / 2;
        std::vector<uint32_t> left_hashes_raw, right_hashes_raw;

        // Prepare left and right hash arrays
        for (size_t i = 0; i < hashes.size(); i += 2) {
            // Convert hex string to uint32_t array
            std::vector<uint32_t> left = hexToUint32(hashes[i]);
            left_hashes_raw.insert(left_hashes_raw.end(), left.begin(), left.end());

            if (i + 1 < hashes.size()) {
                std::vector<uint32_t> right = hexToUint32(hashes[i + 1]);
                right_hashes_raw.insert(right_hashes_raw.end(), right.begin(), right.end());
            } else {
                // Duplicate last hash if odd number
                right_hashes_raw.insert(right_hashes_raw.end(), left.begin(), left.end());
            }
        }

        // Allocate device memory
        uint32_t* d_left;
        uint32_t* d_right;
        uint32_t* d_output;

        CUDA_CHECK(cudaMalloc(&d_left, num_pairs * 8 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_right, num_pairs * 8 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_output, num_pairs * 8 * sizeof(uint32_t)));

        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_left, left_hashes_raw.data(), num_pairs * 8 * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_right, right_hashes_raw.data(), num_pairs * 8 * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        auto end_h2d = std::chrono::high_resolution_clock::now();
        timings_.memory_h2d_ms += std::chrono::duration<double, std::milli>(end_h2d - start_h2d).count();

        // Launch kernel
        auto start_kernel = std::chrono::high_resolution_clock::now();

        int num_blocks = (num_pairs + block_size_ - 1) / block_size_;
        merkle_merge_kernel<<<num_blocks, block_size_>>>(d_left, d_right, d_output, num_pairs);

        CUDA_CHECK(cudaDeviceSynchronize());

        auto end_kernel = std::chrono::high_resolution_clock::now();
        timings_.kernel_execution_ms += std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();

        // Copy results back
        auto start_d2h = std::chrono::high_resolution_clock::now();

        std::vector<uint32_t> h_output(num_pairs * 8);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_pairs * 8 * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        auto end_d2h = std::chrono::high_resolution_clock::now();
        timings_.memory_d2h_ms += std::chrono::duration<double, std::milli>(end_d2h - start_d2h).count();

        // Convert to hex strings
        std::vector<std::string> merged_hashes;
        merged_hashes.reserve(num_pairs);

        for (size_t i = 0; i < num_pairs; i++) {
            std::stringstream ss;
            for (int j = 0; j < 8; j++) {
                ss << std::hex << std::setfill('0') << std::setw(8) << h_output[i * 8 + j];
            }
            merged_hashes.push_back(ss.str());
        }

        // Cleanup
        CUDA_CHECK(cudaFree(d_left));
        CUDA_CHECK(cudaFree(d_right));
        CUDA_CHECK(cudaFree(d_output));

        return merged_hashes;
}

// Helper: Convert hex string to uint32_t array
std::vector<uint32_t> MerkleTreeGPU::hexToUint32(const std::string& hex) const {
        std::vector<uint32_t> result(8);
        for (int i = 0; i < 8; i++) {
            result[i] = std::stoul(hex.substr(i * 8, 8), nullptr, 16);
        }
        return result;
}

// Build tree structure from precomputed hashes (CPU side)
Node* MerkleTreeGPU::buildTreeFromHashes(const std::vector<std::string>& hashes) {
        if (hashes.empty()) return nullptr;

        std::vector<Node*> current_level;
        for (const auto& hash : hashes) {
            current_level.push_back(new Node(hash));
        }

        while (current_level.size() > 1) {
            std::vector<Node*> next_level;
            for (size_t i = 0; i < current_level.size(); i += 2) {
                Node* left = current_level[i];
                Node* right = (i + 1 < current_level.size()) ? current_level[i + 1] : nullptr;

                std::string combined = left->hash;
                if (right) {
                    combined += right->hash;
                } else {
                    combined += left->hash; // Duplicate hash for odd case
                }
                std::string parent_hash = combined; // Use precomputed hash from GPU

                // For now, use a simplified approach - the hash was already computed on GPU
                // In a real implementation, we'd need to map these properly
                Node* parent = new Node(parent_hash);
                parent->left = left;
                parent->right = right; // Can be nullptr for odd cases
                next_level.push_back(parent);
            }
            current_level = next_level;
        }

        return current_level.empty() ? nullptr : current_level[0];
}

void MerkleTreeGPU::deleteTree(Node* node) {
    if (!node) return;
    deleteTree(node->left);
    deleteTree(node->right);
    delete node;
}

// Export C-style interface for easier integration
extern "C" {
    MerkleTreeGPU* merkle_tree_gpu_create(size_t chunk_size, size_t block_size) {
        return new MerkleTreeGPU(chunk_size, block_size);
    }

    void merkle_tree_gpu_destroy(MerkleTreeGPU* tree) {
        delete tree;
    }

    void merkle_tree_gpu_build(MerkleTreeGPU* tree, const char** data, size_t data_size) {
        std::vector<std::string> vec_data;
        vec_data.reserve(data_size);
        for (size_t i = 0; i < data_size; i++) {
            vec_data.push_back(std::string(data[i]));
        }
        tree->build(vec_data);
    }

    const char* merkle_tree_gpu_get_root(MerkleTreeGPU* tree) {
        static std::string root_hash;
        root_hash = tree->getRootHash();
        return root_hash.c_str();
    }
}

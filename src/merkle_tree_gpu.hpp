#ifndef MERKLE_TREE_GPU_HPP
#define MERKLE_TREE_GPU_HPP

#include "header_merkle_tree.hpp"
#include <vector>
#include <string>
#include <cstddef>

// GPU Timing metrics structure
struct GPUTimings {
    double memory_h2d_ms;     // Host to device transfer
    double kernel_execution_ms; // Kernel execution
    double memory_d2h_ms;     // Device to host transfer
    double total_ms;          // Total GPU time
};

// GPU Merkle Tree class declaration
// Implementation in merkle_tree_gpu.cu (requires CUDA)
class MerkleTreeGPU {
public:
    MerkleTreeGPU(size_t chunk_size = 1024, size_t block_size = 256);
    virtual ~MerkleTreeGPU();

    virtual void build(const std::vector<std::string>& data);
    std::string getRootHash() const;
    GPUTimings getTimings() const;
    size_t getChunkSize() const;
    void setChunkSize(size_t size);
    const std::vector<std::string>& getLeafHashes() const { return leaf_hashes_; }

protected:
    Node* root_;
    std::string root_hash_;  // Actual root hash computed by GPU
    std::vector<std::string> leaf_hashes_;
    size_t chunk_size_;
    size_t block_size_;
    GPUTimings timings_;

    // Private methods - implemented in .cu file
    std::vector<std::string> hashChunkOnGPU(const std::vector<std::string>& data);
    std::vector<std::string> mergePairsOnGPU(const std::vector<std::string>& hashes);
    std::vector<uint32_t> hexToUint32(const std::string& hex) const;
    Node* buildTreeFromHashes(const std::vector<std::string>& hashes);
    void deleteTree(Node* node);
};

#endif // MERKLE_TREE_GPU_HPP

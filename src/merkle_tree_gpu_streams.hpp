#ifndef MERKLE_TREE_GPU_STREAMS_HPP
#define MERKLE_TREE_GPU_STREAMS_HPP

#include "merkle_tree_gpu.hpp"
#include <cuda_runtime.h>
#include <vector>

/**
 * GPU Merkle Tree with CUDA Streams for Overlapped Execution
 *
 * Key improvements over base MerkleTreeGPU:
 * - Overlaps H2D transfer, kernel execution, and D2H transfer
 * - Uses pinned memory for async transfers
 * - Pipelines chunks across multiple streams
 * - Expected speedup: 20-30% over sequential version
 */
class MerkleTreeGPUStreams : public MerkleTreeGPU {
public:
    /**
     * Constructor
     * @param chunk_size Number of items per GPU chunk
     * @param block_size CUDA block size (threads per block)
     * @param num_streams Number of concurrent streams (default: 4)
     */
    MerkleTreeGPUStreams(size_t chunk_size, size_t block_size, int num_streams = 4);

    /**
     * Destructor - cleanup streams and pinned memory
     */
    ~MerkleTreeGPUStreams();

    /**
     * Build Merkle tree with overlapped execution
     * Overrides base class method
     */
    void build(const std::vector<std::string>& data) override;

    /**
     * Get detailed timing breakdown showing overlap benefit
     */
    struct StreamTimings {
        double total_ms;
        double h2d_total_ms;
        double kernel_total_ms;
        double d2h_total_ms;
        double overlap_benefit_ms;  // Time saved by overlapping
        double speedup_vs_sequential;
    };

    StreamTimings getStreamTimings() const { return stream_timings_; }

private:
    // Stream management
    std::vector<cudaStream_t> streams_;
    int num_streams_;

    // Pinned host memory pools for async transfers
    struct PinnedBuffer {
        char* data;
        size_t* lengths;
        size_t* offsets;
        size_t capacity;
    };
    std::vector<PinnedBuffer> pinned_buffers_;

    // GPU memory pools (one per stream)
    struct StreamGPUBuffers {
        char* d_data;
        size_t* d_lengths;
        size_t* d_offsets;
        char* d_output;
        size_t capacity;
    };
    std::vector<StreamGPUBuffers> gpu_buffers_;

    // Timing
    StreamTimings stream_timings_;

    // Helper methods
    void initializeStreams();
    void cleanupStreams();
    void allocatePinnedBuffers(size_t max_chunk_bytes);
    void allocateGPUBuffers(size_t max_chunk_bytes);
    void freePinnedBuffers();
    void freeGPUBuffers();

    /**
     * Process all chunks with stream overlap
     */
    std::vector<std::string> processChunksWithStreams(
        const std::vector<std::string>& data);

    /**
     * Process a single chunk on a specific stream
     */
    void processChunkAsync(
        const std::vector<std::string>& chunk_data,
        int stream_idx,
        std::vector<std::string>& output_hashes,
        size_t output_offset);
};

#endif // MERKLE_TREE_GPU_STREAMS_HPP

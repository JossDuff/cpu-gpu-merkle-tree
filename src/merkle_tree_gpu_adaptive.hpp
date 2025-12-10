#ifndef MERKLE_TREE_GPU_ADAPTIVE_HPP
#define MERKLE_TREE_GPU_ADAPTIVE_HPP

#include "merkle_tree_gpu.hpp"
#include "adaptive_chunk_optimizer.hpp"
#include <memory>

/**
 * Adaptive GPU Merkle Tree - Automatically optimizes chunk size based on:
 * - GPU hardware capabilities
 * - Workload characteristics
 * - Optional runtime calibration
 */
class MerkleTreeGPUAdaptive : public MerkleTreeGPU {
public:
    /**
     * Constructor with automatic hardware detection
     * @param enable_calibration If true, runs micro-benchmarks on first use
     */
    explicit MerkleTreeGPUAdaptive(bool enable_calibration = false);

    /**
     * Build Merkle tree with automatic chunk size selection
     * Hides parent build() with adaptive optimization
     */
    void build(const std::vector<std::string>& data);

    /**
     * Build with explicit workload profile (for advanced users)
     */
    void buildWithProfile(const std::vector<std::string>& data,
                         const WorkloadProfile& profile);

    /**
     * Get the optimizer instance (for inspection/configuration)
     */
    AdaptiveChunkOptimizer& getOptimizer() { return *optimizer_; }
    const AdaptiveChunkOptimizer& getOptimizer() const { return *optimizer_; }

    /**
     * Get the configuration used for last build
     */
    OptimalChunkConfig getLastConfig() const { return last_config_; }

    /**
     * Force re-calibration
     */
    void recalibrate() { optimizer_->calibrate(true); }

private:
    std::unique_ptr<AdaptiveChunkOptimizer> optimizer_;
    OptimalChunkConfig last_config_;
    bool calibration_enabled_;
};

#endif // MERKLE_TREE_GPU_ADAPTIVE_HPP

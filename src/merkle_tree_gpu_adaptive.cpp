#include "merkle_tree_gpu_adaptive.hpp"
#include <iostream>
#include <iomanip>

MerkleTreeGPUAdaptive::MerkleTreeGPUAdaptive(bool enable_calibration)
    : MerkleTreeGPU(8192, 256), // Initial defaults (will be overridden)
      calibration_enabled_(enable_calibration) {

    optimizer_ = std::make_unique<AdaptiveChunkOptimizer>();

    // Load cached config if available
    optimizer_->loadConfig();

    // Run calibration if requested
    if (calibration_enabled_ && !optimizer_->isCalibrated()) {
        std::cout << "First run detected. Running calibration...\n";
        optimizer_->calibrate();
    }
}

void MerkleTreeGPUAdaptive::build(const std::vector<std::string>& data) {
    if (data.empty()) {
        root_ = nullptr;
        root_hash_ = "";
        return;
    }

    // Create workload profile
    WorkloadProfile profile = createWorkloadProfile(data);

    buildWithProfile(data, profile);
}

void MerkleTreeGPUAdaptive::buildWithProfile(const std::vector<std::string>& data,
                                             const WorkloadProfile& profile) {
    if (data.empty()) {
        root_ = nullptr;
        root_hash_ = "";
        return;
    }

    // Get optimal configuration for this workload
    last_config_ = optimizer_->getOptimalConfig(profile);

    // Apply optimal configuration
    chunk_size_ = last_config_.chunk_size;
    block_size_ = last_config_.block_size;

    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Adaptive GPU Configuration                                ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Workload: " << profile.item_count << " items, "
              << (profile.total_bytes / (1024 * 1024)) << " MB"
              << std::string(30, ' ') << "║\n";
    std::cout << "║  Uniformity: " << std::fixed << std::setprecision(2)
              << profile.uniformity_score << " | Avg item: " << profile.avg_item_size << " bytes"
              << std::string(20, ' ') << "║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Selected Config:                                          ║\n";
    std::cout << "║    Chunk size: " << last_config_.chunk_size << " items"
              << std::string(40 - std::to_string(last_config_.chunk_size).length(), ' ') << "║\n";
    std::cout << "║    Block size: " << last_config_.block_size << " threads"
              << std::string(38 - std::to_string(last_config_.block_size).length(), ' ') << "║\n";
    std::cout << "║    Confidence: " << std::fixed << std::setprecision(1)
              << (last_config_.confidence_score * 100) << "%"
              << std::string(40, ' ') << "║\n";
    std::cout << "║    Reasoning: " << last_config_.reasoning.substr(0, 45)
              << std::string(46 - std::min(last_config_.reasoning.length(), size_t(45)), ' ') << "║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // Call parent build with optimized settings
    MerkleTreeGPU::build(data);

    std::cout << "✓ Build complete with adaptive configuration\n";
    std::cout << "  Total time: " << std::fixed << std::setprecision(2)
              << timings_.total_ms << " ms\n";
    std::cout << "  Chunk size: " << chunk_size_ << " items\n\n";
}

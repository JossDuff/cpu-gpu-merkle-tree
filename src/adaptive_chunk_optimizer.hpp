#ifndef ADAPTIVE_CHUNK_OPTIMIZER_HPP
#define ADAPTIVE_CHUNK_OPTIMIZER_HPP

#include <cstddef>
#include <string>
#include <map>
#include <vector>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

/**
 * Hardware-Aware Adaptive Chunk Size Optimizer
 *
 * Automatically determines optimal GPU chunk size based on:
 * - GPU hardware capabilities (memory, SMs, compute capability)
 * - Workload characteristics (data size, item size variance)
 * - Optional micro-benchmarking for fine-tuning
 */

struct GPUHardwareProfile {
    std::string device_name;
    size_t total_memory_mb;
    size_t available_memory_mb;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    size_t shared_memory_per_block;
    int warp_size;

    // Derived metrics
    size_t total_cores;           // Estimated CUDA cores
    double memory_bandwidth_gbps; // Peak memory bandwidth
};

struct WorkloadProfile {
    size_t item_count;
    size_t total_bytes;
    size_t avg_item_size;
    size_t min_item_size;
    size_t max_item_size;
    double size_variance;
    bool is_uniform;              // Low variance

    // Computed properties
    double uniformity_score;      // 0-1, higher = more uniform
    size_t estimated_leaves;      // Number of leaf hashes
};

struct OptimalChunkConfig {
    size_t chunk_size;
    size_t block_size;            // Threads per block
    double confidence_score;      // 0-1, based on calibration
    std::string reasoning;        // Why this config was chosen
};

class AdaptiveChunkOptimizer {
public:
    AdaptiveChunkOptimizer();
    ~AdaptiveChunkOptimizer();

    // Main API
    OptimalChunkConfig getOptimalConfig(const WorkloadProfile& workload);
    OptimalChunkConfig getOptimalConfig(size_t item_count, size_t total_bytes);

    // Hardware detection
    bool detectGPUHardware();
    GPUHardwareProfile getHardwareProfile() const { return hardware_profile_; }

    // Calibration (optional, runs on first use)
    void calibrate(bool force_recalibrate = false);
    bool isCalibrated() const { return calibrated_; }

    // Configuration management
    void saveConfig(const std::string& filename = ".merkle_gpu_config");
    bool loadConfig(const std::string& filename = ".merkle_gpu_config");

    // Manual override
    void setChunkSize(size_t size) { manual_override_chunk_ = size; }
    void clearOverride() { manual_override_chunk_ = 0; }

private:
    GPUHardwareProfile hardware_profile_;
    bool hardware_detected_;
    bool calibrated_;
    size_t manual_override_chunk_;

    // Calibration results cache
    std::map<std::string, OptimalChunkConfig> config_cache_;

    // Heuristic-based optimization (no benchmarking)
    OptimalChunkConfig estimateFromHardware(const WorkloadProfile& workload);

    // Micro-benchmark-based optimization
    OptimalChunkConfig calibrateForWorkload(const WorkloadProfile& workload);

    // Helper functions
    WorkloadProfile analyzeWorkload(size_t item_count, size_t total_bytes) const;
    std::string getWorkloadSignature(const WorkloadProfile& workload) const;
    size_t estimateGPUMemoryUsage(size_t chunk_size, size_t avg_item_size) const;
    size_t calculateMaxSafeChunkSize(const WorkloadProfile& workload) const;

    // GPU detection helpers
#ifdef HAS_CUDA
    void queryGPUProperties();
    size_t getAvailableGPUMemory();
#endif
};

// Helper function: Create workload profile from data
WorkloadProfile createWorkloadProfile(const std::vector<std::string>& data);

#endif // ADAPTIVE_CHUNK_OPTIMIZER_HPP

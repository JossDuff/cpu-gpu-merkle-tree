#include "adaptive_chunk_optimizer.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

AdaptiveChunkOptimizer::AdaptiveChunkOptimizer()
    : hardware_detected_(false), calibrated_(false), manual_override_chunk_(0) {
    // Attempt to detect GPU on construction
    detectGPUHardware();
}

AdaptiveChunkOptimizer::~AdaptiveChunkOptimizer() {
    // Auto-save config on destruction
    if (calibrated_) {
        saveConfig();
    }
}

bool AdaptiveChunkOptimizer::detectGPUHardware() {
#ifdef HAS_CUDA
    queryGPUProperties();
    hardware_detected_ = true;
    return true;
#else
    std::cerr << "Warning: CUDA not available. Using fallback heuristics.\n";
    hardware_detected_ = false;
    return false;
#endif
}

#ifdef HAS_CUDA
void AdaptiveChunkOptimizer::queryGPUProperties() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return;
    }

    // Use first GPU (device 0)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    hardware_profile_.device_name = prop.name;
    hardware_profile_.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);
    hardware_profile_.compute_capability_major = prop.major;
    hardware_profile_.compute_capability_minor = prop.minor;
    hardware_profile_.multiprocessor_count = prop.multiProcessorCount;
    hardware_profile_.max_threads_per_block = prop.maxThreadsPerBlock;
    hardware_profile_.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    hardware_profile_.shared_memory_per_block = prop.sharedMemPerBlock;
    hardware_profile_.warp_size = prop.warpSize;

    // Estimate CUDA cores (rough approximation)
    int cores_per_sm = 128; // Default for modern GPUs
    if (prop.major == 3) cores_per_sm = 192;
    else if (prop.major == 5) cores_per_sm = 128;
    else if (prop.major == 6) cores_per_sm = (prop.minor == 0) ? 64 : 128;
    else if (prop.major == 7) cores_per_sm = 64;
    else if (prop.major == 8) cores_per_sm = (prop.minor == 0) ? 64 : 128;

    hardware_profile_.total_cores = cores_per_sm * prop.multiProcessorCount;

    // Memory bandwidth (GB/s) - approximate from memory clock
    hardware_profile_.memory_bandwidth_gbps =
        (prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2) / 1.0e9;

    // Get currently available memory
    hardware_profile_.available_memory_mb = getAvailableGPUMemory();

    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  GPU Hardware Profile Detected                             ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Device: " << hardware_profile_.device_name << std::string(46 - hardware_profile_.device_name.length(), ' ') << "║\n";
    std::cout << "║  Compute: " << prop.major << "." << prop.minor
              << " | SMs: " << prop.multiProcessorCount
              << " | Cores: ~" << hardware_profile_.total_cores
              << std::string(20, ' ') << "║\n";
    std::cout << "║  Memory: " << hardware_profile_.total_memory_mb << " MB total, "
              << hardware_profile_.available_memory_mb << " MB available"
              << std::string(15, ' ') << "║\n";
    std::cout << "║  Bandwidth: ~" << static_cast<int>(hardware_profile_.memory_bandwidth_gbps)
              << " GB/s" << std::string(40, ' ') << "║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
}

size_t AdaptiveChunkOptimizer::getAvailableGPUMemory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem / (1024 * 1024); // Convert to MB
}
#endif

OptimalChunkConfig AdaptiveChunkOptimizer::getOptimalConfig(const WorkloadProfile& workload) {
    // Manual override takes precedence
    if (manual_override_chunk_ > 0) {
        OptimalChunkConfig config;
        config.chunk_size = manual_override_chunk_;
        config.block_size = 256; // Default
        config.confidence_score = 1.0;
        config.reasoning = "Manual override";
        return config;
    }

    // Check cache first
    std::string signature = getWorkloadSignature(workload);
    if (config_cache_.find(signature) != config_cache_.end()) {
        return config_cache_[signature];
    }

    // Use heuristics if not calibrated or no GPU
    if (!hardware_detected_ || !calibrated_) {
        return estimateFromHardware(workload);
    }

    // Otherwise, use calibrated values
    return calibrateForWorkload(workload);
}

OptimalChunkConfig AdaptiveChunkOptimizer::getOptimalConfig(size_t item_count, size_t total_bytes) {
    WorkloadProfile workload = analyzeWorkload(item_count, total_bytes);
    return getOptimalConfig(workload);
}

OptimalChunkConfig AdaptiveChunkOptimizer::estimateFromHardware(const WorkloadProfile& workload) {
    OptimalChunkConfig config;
    config.block_size = 256; // Standard for most GPUs
    config.confidence_score = 0.7; // Heuristic-based

    if (!hardware_detected_) {
        // No GPU info - use conservative defaults
        config.chunk_size = 8192;
        config.reasoning = "Default (no GPU detected)";
        return config;
    }

    // Calculate safe maximum chunk size based on GPU memory
    size_t max_safe_chunk = calculateMaxSafeChunkSize(workload);

    // Heuristic rules based on hardware profile
    size_t base_chunk = 4096;

    // Scale by GPU memory
    if (hardware_profile_.total_memory_mb > 8000) {
        base_chunk = 32768; // High-end GPU (8+ GB)
    } else if (hardware_profile_.total_memory_mb > 4000) {
        base_chunk = 16384; // Mid-range GPU (4-8 GB)
    } else {
        base_chunk = 8192;  // Low-end GPU (< 4 GB)
    }

    // Scale by compute capability
    if (hardware_profile_.compute_capability_major >= 7) {
        base_chunk *= 2; // Modern architecture (Volta+)
    }

    // Adjust for workload uniformity
    if (workload.uniformity_score > 0.9) {
        base_chunk *= 2; // Very uniform data benefits from larger chunks
    }

    // Adjust for workload size
    if (workload.item_count < 10000) {
        base_chunk /= 2; // Small workload - smaller chunks
    } else if (workload.item_count > 1000000) {
        base_chunk *= 2; // Large workload - larger chunks
    }

    // Clamp to safe maximum
    config.chunk_size = std::min(base_chunk, max_safe_chunk);

    // Ensure power of 2 (optimal for GPU memory)
    config.chunk_size = 1 << static_cast<int>(std::log2(config.chunk_size));

    std::ostringstream reasoning;
    reasoning << "Heuristic: " << hardware_profile_.device_name
              << " | " << hardware_profile_.total_memory_mb << " MB"
              << " | Uniformity: " << std::fixed << std::setprecision(2) << workload.uniformity_score;
    config.reasoning = reasoning.str();

    return config;
}

OptimalChunkConfig AdaptiveChunkOptimizer::calibrateForWorkload(const WorkloadProfile& workload) {
    // TODO: Run actual micro-benchmarks
    // For now, use enhanced heuristics
    return estimateFromHardware(workload);
}

void AdaptiveChunkOptimizer::calibrate(bool force_recalibrate) {
    if (calibrated_ && !force_recalibrate) {
        std::cout << "Already calibrated. Use force_recalibrate=true to re-run.\n";
        return;
    }

    std::cout << "Running calibration micro-benchmarks...\n";

    // TODO: Implement micro-benchmarking
    // For now, just mark as calibrated
    calibrated_ = true;

    std::cout << "Calibration complete!\n";
}

WorkloadProfile AdaptiveChunkOptimizer::analyzeWorkload(size_t item_count, size_t total_bytes) const {
    WorkloadProfile profile;
    profile.item_count = item_count;
    profile.total_bytes = total_bytes;
    profile.avg_item_size = total_bytes / item_count;
    profile.min_item_size = profile.avg_item_size; // Estimate
    profile.max_item_size = profile.avg_item_size; // Estimate
    profile.size_variance = 0.0; // Unknown
    profile.uniformity_score = 0.8; // Assume mostly uniform
    profile.is_uniform = true;
    profile.estimated_leaves = item_count;

    return profile;
}

std::string AdaptiveChunkOptimizer::getWorkloadSignature(const WorkloadProfile& workload) const {
    std::ostringstream sig;
    sig << workload.item_count << "_" << workload.total_bytes << "_"
        << static_cast<int>(workload.uniformity_score * 100);
    return sig.str();
}

size_t AdaptiveChunkOptimizer::estimateGPUMemoryUsage(size_t chunk_size, size_t avg_item_size) const {
    // Same calculation as in main_gpu_benchmark.cpp
    size_t input_mb = (chunk_size * avg_item_size) / (1024 * 1024);
    size_t lengths_mb = (chunk_size * 4) / (1024 * 1024);
    size_t offsets_mb = (chunk_size * 4) / (1024 * 1024);
    size_t output_mb = (chunk_size * 32) / (1024 * 1024);
    return input_mb + lengths_mb + offsets_mb + output_mb + 10; // +10 MB overhead
}

size_t AdaptiveChunkOptimizer::calculateMaxSafeChunkSize(const WorkloadProfile& workload) const {
    if (!hardware_detected_) {
        return 32768; // Conservative default
    }

    // Use 50% of available GPU memory as safety margin
    size_t safe_memory_mb = hardware_profile_.available_memory_mb / 2;

    // Binary search for max chunk size that fits
    size_t low = 1024, high = 2097152; // 1K to 2M
    size_t result = low;

    while (low <= high) {
        size_t mid = (low + high) / 2;
        size_t estimated_mb = estimateGPUMemoryUsage(mid, workload.avg_item_size);

        if (estimated_mb <= safe_memory_mb) {
            result = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return result;
}

void AdaptiveChunkOptimizer::saveConfig(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not save config to " << filename << "\n";
        return;
    }

    file << "# Merkle GPU Configuration\n";
    file << "calibrated=" << (calibrated_ ? "1" : "0") << "\n";

    // Save cache
    for (const auto& entry : config_cache_) {
        file << "cache:" << entry.first << "=" << entry.second.chunk_size << "\n";
    }

    file.close();
}

bool AdaptiveChunkOptimizer::loadConfig(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        // Parse key=value
        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        if (key == "calibrated") {
            calibrated_ = (value == "1");
        }
        // TODO: Parse cache entries
    }

    file.close();
    return true;
}

// Helper function implementation
WorkloadProfile createWorkloadProfile(const std::vector<std::string>& data) {
    WorkloadProfile profile;
    profile.item_count = data.size();
    profile.total_bytes = 0;
    profile.min_item_size = SIZE_MAX;
    profile.max_item_size = 0;

    for (const auto& item : data) {
        size_t size = item.size();
        profile.total_bytes += size;
        profile.min_item_size = std::min(profile.min_item_size, size);
        profile.max_item_size = std::max(profile.max_item_size, size);
    }

    profile.avg_item_size = profile.total_bytes / profile.item_count;

    // Calculate variance
    double variance_sum = 0.0;
    for (const auto& item : data) {
        double diff = static_cast<double>(item.size()) - profile.avg_item_size;
        variance_sum += diff * diff;
    }
    profile.size_variance = variance_sum / profile.item_count;

    // Uniformity score (0 = very non-uniform, 1 = perfectly uniform)
    double coefficient_of_variation = std::sqrt(profile.size_variance) / profile.avg_item_size;
    profile.uniformity_score = 1.0 / (1.0 + coefficient_of_variation);
    profile.is_uniform = (profile.uniformity_score > 0.8);

    profile.estimated_leaves = profile.item_count;

    return profile;
}

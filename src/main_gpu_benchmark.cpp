#include "header_merkle_tree.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>
#include <fstream>
#include <map>
#include <ctime>

#ifdef HAS_CUDA
#include "merkle_tree_gpu.hpp"
#include "merkle_tree_gpu_adaptive.hpp"
#include "merkle_tree_gpu_streams.hpp"
#include "adaptive_chunk_optimizer.hpp"
#endif

// Data generators (same as original benchmark)
std::vector<std::string> generateRandomData(size_t count, size_t minSize, size_t maxSize) {
    std::vector<std::string> data;
    data.reserve(count);

    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<> size_dist(minSize, maxSize);
    std::uniform_int_distribution<> char_dist(32, 126);

    for (size_t i = 0; i < count; i++) {
        size_t size = size_dist(gen);
        std::string item;
        item.reserve(size);
        for (size_t j = 0; j < size; j++) {
            item += static_cast<char>(char_dist(gen));
        }
        data.push_back(std::move(item));
    }

    return data;
}

std::vector<std::string> generateStructuredData(size_t count) {
    std::vector<std::string> data;
    data.reserve(count);

    for (size_t i = 0; i < count; i++) {
        std::stringstream ss;
        ss << "id:" << i << ",timestamp:" << (1609459200 + i * 60)
           << ",value:" << (i * 1.5) << ",status:active";
        data.push_back(ss.str());
    }

    return data;
}

std::string formatNumber(size_t num) {
    std::string str = std::to_string(num);
    std::string result;
    int count = 0;

    for (int i = str.length() - 1; i >= 0; i--) {
        if (count == 3) {
            result = "," + result;
            count = 0;
        }
        result = str[i] + result;
        count++;
    }

    return result;
}

std::string formatBytes(size_t bytes) {
    if (bytes < 1024) return std::to_string(bytes) + " B";
    if (bytes < 1024 * 1024) return std::to_string(bytes / 1024) + " KB";
    return std::to_string(bytes / (1024 * 1024)) + " MB";
}

// Performance metrics structure
struct BenchmarkResult {
    std::string test_name;
    std::string implementation;
    size_t data_count;
    size_t total_bytes;
    size_t chunk_size;
    double time_ms;
    double throughput_mbps;
    double speedup;
    double memory_h2d_ms;
    double kernel_ms;
    double memory_d2h_ms;
    bool success;
};

// CSV writer for results
class CSVWriter {
private:
    std::ofstream file_;

public:
    CSVWriter(const std::string& filename) {
        file_.open(filename);
        // Write header
        file_ << "Test,Implementation,DataCount,TotalBytes,ChunkSize,"
              << "TimeMS,ThroughputMBps,Speedup,MemoryH2DMS,KernelMS,MemoryD2HMS,Success\n";
    }

    ~CSVWriter() {
        if (file_.is_open()) {
            file_.close();
        }
    }

    void writeResult(const BenchmarkResult& result) {
        file_ << result.test_name << ","
              << result.implementation << ","
              << result.data_count << ","
              << result.total_bytes << ","
              << result.chunk_size << ","
              << result.time_ms << ","
              << result.throughput_mbps << ","
              << result.speedup << ","
              << result.memory_h2d_ms << ","
              << result.kernel_ms << ","
              << result.memory_d2h_ms << ","
              << (result.success ? "1" : "0") << "\n";
        file_.flush();
    }
};

// GPU chunk size optimization tests
class GPUChunkOptimizer {
private:
    // ============================================================
    // Chunk Size Configuration - Find the Performance Ceiling!
    // ============================================================

    // EXTENDED RANGE: Test up to 1M items per chunk to find bottleneck
    std::vector<size_t> chunk_sizes_ = {
        // Standard range (baseline)
        128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,

        // Extended range - find the performance ceiling
        65536,     // 64K
        131072,    // 128K
        262144,    // 256K
        524288,    // 512K
        1048576    // 1M - likely at/beyond optimal point

        // Uncomment for EXTREME testing (may hit GPU memory limits):
        // , 2097152   // 2M
        // , 4194304   // 4M
        // , 8388608   // 8M
    };

    // GPU Memory estimation per chunk (helps avoid OOM)
    size_t estimateGPUMemoryMB(size_t chunk_size, size_t avg_item_bytes) const {
        // Input buffer: chunk_size * avg_item_bytes
        size_t input_mb = (chunk_size * avg_item_bytes) / (1024 * 1024);
        // Lengths array: chunk_size * 4 bytes
        size_t lengths_mb = (chunk_size * 4) / (1024 * 1024);
        // Offsets array: chunk_size * 4 bytes
        size_t offsets_mb = (chunk_size * 4) / (1024 * 1024);
        // Output hashes: chunk_size * 32 bytes
        size_t output_mb = (chunk_size * 32) / (1024 * 1024);

        return input_mb + lengths_mb + offsets_mb + output_mb + 10; // +10 MB overhead
    }

    CSVWriter& csv_writer_;

public:
    GPUChunkOptimizer(CSVWriter& writer) : csv_writer_(writer) {}

    void runOptimizationTests(const std::string& test_name,
                             const std::vector<std::string>& data) {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  GPU Chunk Size Optimization: " << test_name << std::string(31 - test_name.length(), ' ') << "║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Data: " << formatNumber(data.size()) << " items, "
                  << formatBytes(getTotalSize(data)) << std::string(35, ' ') << "║\n";

        // Calculate average item size for memory estimation
        size_t avg_item_size = getTotalSize(data) / data.size();
        size_t max_chunk = *std::max_element(chunk_sizes_.begin(), chunk_sizes_.end());
        size_t max_mem_mb = estimateGPUMemoryMB(max_chunk, avg_item_size);

        std::cout << "║  Avg item: " << avg_item_size << " bytes, Max chunk: "
                  << formatNumber(max_chunk) << " (~" << max_mem_mb << " MB GPU)"
                  << std::string(10, ' ') << "║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

        // Memory warning for large chunks
        if (max_mem_mb > 1000) {
            std::cout << "\n⚠️  WARNING: Testing very large chunks (>" << max_mem_mb
                      << " MB GPU memory)\n";
            std::cout << "   May cause OOM on GPUs with limited memory. Monitor GPU usage!\n";
        }

        // Get baseline CPU performance
        double baseline_time_ms = runCPUBaseline(data);

        std::cout << "\nCPU Baseline: " << std::fixed << std::setprecision(2)
                  << baseline_time_ms << " ms\n\n";

        std::cout << std::left
                  << std::setw(12) << "Chunk Size"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(15) << "Speedup"
                  << std::setw(12) << "GPU Mem"
                  << std::setw(15) << "H2D (ms)"
                  << std::setw(15) << "Kernel (ms)"
                  << std::setw(15) << "D2H (ms)"
                  << "\n";
        std::cout << std::string(99, '-') << "\n";

        size_t prev_time_ms = 0;
        for (size_t chunk_size : chunk_sizes_) {
            size_t est_mem_mb = estimateGPUMemoryMB(chunk_size, avg_item_size);

            BenchmarkResult result = runGPUTest(test_name, data, chunk_size,
                                               baseline_time_ms);

            std::cout << std::left
                      << std::setw(12) << formatNumber(chunk_size)
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.time_ms
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.speedup
                      << std::setw(12) << (std::to_string(est_mem_mb) + " MB")
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.memory_h2d_ms
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.kernel_ms
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.memory_d2h_ms;

            // Show performance trend indicators
            if (prev_time_ms > 0) {
                double improvement = ((double)prev_time_ms - result.time_ms) / prev_time_ms * 100.0;
                if (improvement < 1.0 && chunk_size > 16384) {
                    std::cout << " ← PLATEAU";
                } else if (improvement < 0) {
                    std::cout << " ← DEGRADED";
                }
            }
            std::cout << "\n";

            prev_time_ms = result.time_ms;
            csv_writer_.writeResult(result);
        }

        std::cout << "\n";
    }

private:
    size_t getTotalSize(const std::vector<std::string>& data) const {
        size_t total = 0;
        for (const auto& item : data) {
            total += item.size();
        }
        return total;
    }

    double runCPUBaseline(const std::vector<std::string>& data) {
        MerkleTreeLockFree tree;

        auto start = std::chrono::high_resolution_clock::now();
        tree.build(data);
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    BenchmarkResult runGPUTest(const std::string& test_name,
                              const std::vector<std::string>& data,
                              size_t chunk_size,
                              double baseline_time_ms) {
        BenchmarkResult result;
        result.test_name = test_name;
        result.implementation = "GPU";
        result.data_count = data.size();
        result.total_bytes = getTotalSize(data);
        result.chunk_size = chunk_size;
        result.success = true;

#ifdef HAS_CUDA
        // Use REAL GPU implementation
        MerkleTreeGPU gpu_tree(chunk_size, 256);

        gpu_tree.build(data);  // ✅ BUILDS COMPLETE MERKLE TREE

        GPUTimings timings = gpu_tree.getTimings();

        result.time_ms = timings.total_ms;
        result.memory_h2d_ms = timings.memory_h2d_ms;
        result.kernel_ms = timings.kernel_execution_ms;
        result.memory_d2h_ms = timings.memory_d2h_ms;
        result.speedup = baseline_time_ms / result.time_ms;
        result.throughput_mbps = (result.total_bytes / (1024.0 * 1024.0)) / (result.time_ms / 1000.0);

        // Optional: Verify GPU built a valid tree by checking root hash exists
        std::string root_hash = gpu_tree.getRootHash();
        result.success = !root_hash.empty();  // Tree is valid if has root hash
#else
        // Simulate GPU performance (no CUDA available)
        double transfer_overhead = (double)result.total_bytes / (1024.0 * 1024.0 * 500.0) * 1000.0;
        double compute_time = baseline_time_ms * 0.1;
        double chunk_overhead = (data.size() / chunk_size) * 0.05;

        result.time_ms = transfer_overhead * 2 + compute_time + chunk_overhead;
        result.memory_h2d_ms = transfer_overhead;
        result.kernel_ms = compute_time;
        result.memory_d2h_ms = transfer_overhead;
        result.speedup = baseline_time_ms / result.time_ms;
        result.throughput_mbps = (result.total_bytes / (1024.0 * 1024.0)) / (result.time_ms / 1000.0);
#endif

        return result;
    }
};

// Adaptive GPU comparison tests
class AdaptiveGPUComparator {
private:
    CSVWriter& csv_writer_;

public:
    AdaptiveGPUComparator(CSVWriter& writer) : csv_writer_(writer) {}

    void runAdaptiveComparison(const std::string& test_name,
                               const std::vector<std::string>& data) {
#ifdef HAS_CUDA
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  Adaptive GPU Comparison: " << test_name << std::string(32 - test_name.length(), ' ') << "║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Data: " << formatNumber(data.size()) << " items, "
                  << formatBytes(getTotalSize(data)) << std::string(35, ' ') << "║\n";
        std::cout << "║  Chunk size: AUTO (hardware-aware optimization)"
                  << std::string(20, ' ') << "║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

        // Test 1: Standard GPU with manual chunk size (baseline)
        size_t manual_chunk = 8192;
        std::cout << "Running standard GPU with manual chunk size (" << manual_chunk << ")...\n";
        MerkleTreeGPU standard_tree(manual_chunk, 256);
        standard_tree.build(data);

        auto standard_timings = standard_tree.getTimings();
        auto standard_root = standard_tree.getRootHash();

        BenchmarkResult standard_result;
        standard_result.test_name = test_name;
        standard_result.implementation = "GPU-Manual";
        standard_result.data_count = data.size();
        standard_result.total_bytes = getTotalSize(data);
        standard_result.chunk_size = manual_chunk;
        standard_result.time_ms = standard_timings.total_ms;
        standard_result.memory_h2d_ms = standard_timings.memory_h2d_ms;
        standard_result.kernel_ms = standard_timings.kernel_execution_ms;
        standard_result.memory_d2h_ms = standard_timings.memory_d2h_ms;
        standard_result.speedup = 1.0;  // Baseline
        standard_result.throughput_mbps = (standard_result.total_bytes / (1024.0 * 1024.0)) / (standard_result.time_ms / 1000.0);
        standard_result.success = true;

        std::cout << "  Time: " << std::fixed << std::setprecision(2)
                  << standard_timings.total_ms << " ms\n";
        std::cout << "  Throughput: " << standard_result.throughput_mbps << " MB/s\n\n";

        csv_writer_.writeResult(standard_result);

        // Test 2: Adaptive GPU (auto chunk size)
        std::cout << "Running adaptive GPU (automatic chunk size selection)...\n";
        MerkleTreeGPUAdaptive adaptive_tree(false);  // No calibration
        adaptive_tree.build(data);

        auto adaptive_timings = adaptive_tree.getTimings();
        auto adaptive_config = adaptive_tree.getLastConfig();
        auto adaptive_root = adaptive_tree.getRootHash();

        // Verify correctness
        if (standard_root.substr(0, 16) != adaptive_root.substr(0, 16)) {
            std::cout << "⚠️  WARNING: Root hashes don't match!\n";
        }

        BenchmarkResult adaptive_result;
        adaptive_result.test_name = test_name;
        adaptive_result.implementation = "GPU-Adaptive";
        adaptive_result.data_count = data.size();
        adaptive_result.total_bytes = getTotalSize(data);
        adaptive_result.chunk_size = adaptive_config.chunk_size;
        adaptive_result.time_ms = adaptive_timings.total_ms;
        adaptive_result.memory_h2d_ms = adaptive_timings.memory_h2d_ms;
        adaptive_result.kernel_ms = adaptive_timings.kernel_execution_ms;
        adaptive_result.memory_d2h_ms = adaptive_timings.memory_d2h_ms;
        adaptive_result.speedup = standard_timings.total_ms / adaptive_timings.total_ms;
        adaptive_result.throughput_mbps = (adaptive_result.total_bytes / (1024.0 * 1024.0)) / (adaptive_result.time_ms / 1000.0);
        adaptive_result.success = true;

        double improvement = ((adaptive_result.speedup - 1.0) * 100.0);

        std::cout << "  Selected chunk size: " << adaptive_config.chunk_size << " items\n";
        std::cout << "  Reasoning: " << adaptive_config.reasoning << "\n";
        std::cout << "  Time: " << adaptive_timings.total_ms << " ms\n";
        std::cout << "  Speedup vs manual: " << std::fixed << std::setprecision(2)
                  << adaptive_result.speedup << "x";
        if (improvement > 0) {
            std::cout << " (" << improvement << "% faster)";
        } else {
            std::cout << " (" << -improvement << "% slower)";
        }
        std::cout << "\n";
        std::cout << "  Throughput: " << adaptive_result.throughput_mbps << " MB/s\n\n";

        csv_writer_.writeResult(adaptive_result);

        // Summary
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  Summary: " << test_name << std::string(48 - test_name.length(), ' ') << "║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Manual GPU:     " << std::setw(10) << standard_result.time_ms
                  << " ms (chunk: " << manual_chunk << ")" << std::string(15, ' ') << "║\n";
        std::cout << "║  Adaptive GPU:   " << std::setw(10) << adaptive_result.time_ms
                  << " ms (chunk: " << adaptive_config.chunk_size << ")" << std::string(10, ' ') << "║\n";
        std::cout << "║  Improvement:    " << std::setw(9);
        if (improvement > 0) {
            std::cout << "+" << improvement;
        } else {
            std::cout << improvement;
        }
        std::cout << "%" << std::string(33, ' ') << "║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
#else
        std::cout << "CUDA not available. Skipping adaptive comparison.\n";
#endif
    }

private:
    size_t getTotalSize(const std::vector<std::string>& data) const {
        size_t total = 0;
        for (const auto& item : data) {
            total += item.size();
        }
        return total;
    }
};

// GPU Streams comparison tests (with adaptive chunk sizing)
class GPUStreamsComparator {
private:
    CSVWriter& csv_writer_;

public:
    GPUStreamsComparator(CSVWriter& writer) : csv_writer_(writer) {}

    void runStreamsComparison(const std::string& test_name,
                              const std::vector<std::string>& data) {
#ifdef HAS_CUDA
        // First, use adaptive optimizer to determine optimal chunk size
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  GPU Streams + Adaptive: " << test_name << std::string(32 - test_name.length(), ' ') << "║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Data: " << formatNumber(data.size()) << " items, "
                  << formatBytes(getTotalSize(data)) << std::string(35, ' ') << "║\n";
        std::cout << "║  Chunk size: AUTO (adaptive) + Streams (overlapped)"
                  << std::string(10, ' ') << "║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

        // Determine optimal chunk size using adaptive optimizer
        std::cout << "Determining optimal chunk size...\n";
        AdaptiveChunkOptimizer optimizer;
        WorkloadProfile profile = createWorkloadProfile(data);
        OptimalChunkConfig optimal_config = optimizer.getOptimalConfig(profile);
        size_t chunk_size = optimal_config.chunk_size;

        std::cout << "  Selected chunk size: " << chunk_size << " items\n";
        std::cout << "  Reasoning: " << optimal_config.reasoning << "\n\n";

        // Test 1: Standard GPU with adaptive chunk (no streams)
        std::cout << "Running GPU with adaptive chunk (no streams)...\n";
        MerkleTreeGPU standard_tree(chunk_size, 256);
        standard_tree.build(data);

        auto standard_timings = standard_tree.getTimings();
        auto standard_root = standard_tree.getRootHash();

        BenchmarkResult standard_result;
        standard_result.test_name = test_name;
        standard_result.implementation = "GPU-Adaptive-NoStreams";
        standard_result.data_count = data.size();
        standard_result.total_bytes = getTotalSize(data);
        standard_result.chunk_size = chunk_size;
        standard_result.time_ms = standard_timings.total_ms;
        standard_result.memory_h2d_ms = standard_timings.memory_h2d_ms;
        standard_result.kernel_ms = standard_timings.kernel_execution_ms;
        standard_result.memory_d2h_ms = standard_timings.memory_d2h_ms;
        standard_result.speedup = 1.0;  // Baseline
        standard_result.throughput_mbps = (standard_result.total_bytes / (1024.0 * 1024.0)) / (standard_result.time_ms / 1000.0);
        standard_result.success = true;

        std::cout << "  Time: " << std::fixed << std::setprecision(2)
                  << standard_timings.total_ms << " ms\n";
        std::cout << "  Throughput: " << standard_result.throughput_mbps << " MB/s\n\n";

        csv_writer_.writeResult(standard_result);

        // Test 2-4: GPU with adaptive chunk + different stream counts
        std::vector<int> stream_counts = {2, 4, 8};

        BenchmarkResult best_streams_result;
        double best_speedup = 0;

        for (int num_streams : stream_counts) {
            std::cout << "Running GPU with adaptive chunk + " << num_streams << " CUDA streams...\n";

            MerkleTreeGPUStreams streams_tree(chunk_size, 256, num_streams);
            streams_tree.build(data);

            auto streams_timings = streams_tree.getStreamTimings();
            auto streams_root = streams_tree.getRootHash();

            // Verify correctness
            if (standard_root.substr(0, 16) != streams_root.substr(0, 16)) {
                std::cout << "⚠️  WARNING: Root hashes don't match!\n";
                std::cout << "   Standard: " << standard_root.substr(0, 32) << "...\n";
                std::cout << "   Streams:  " << streams_root.substr(0, 32) << "...\n";

                // Debug: Check first few leaf hashes
                auto std_leaves = standard_tree.getLeafHashes();
                auto str_leaves = streams_tree.getLeafHashes();
                std::cout << "   Leaf count - Standard: " << std_leaves.size() << ", Streams: " << str_leaves.size() << "\n";
                if (!std_leaves.empty() && !str_leaves.empty()) {
                    std::cout << "   First leaf - Standard: " << std_leaves[0].substr(0, 16) << "...\n";
                    std::cout << "   First leaf - Streams:  " << str_leaves[0].substr(0, 16) << "...\n";
                }
            }

            BenchmarkResult streams_result;
            streams_result.test_name = test_name;
            streams_result.implementation = "GPU-Adaptive-Streams-" + std::to_string(num_streams);
            streams_result.data_count = data.size();
            streams_result.total_bytes = getTotalSize(data);
            streams_result.chunk_size = chunk_size;
            streams_result.time_ms = streams_timings.total_ms;
            streams_result.memory_h2d_ms = streams_timings.h2d_total_ms;
            streams_result.kernel_ms = streams_timings.kernel_total_ms;
            streams_result.memory_d2h_ms = streams_timings.d2h_total_ms;
            streams_result.speedup = standard_timings.total_ms / streams_timings.total_ms;
            streams_result.throughput_mbps = (streams_result.total_bytes / (1024.0 * 1024.0)) / (streams_result.time_ms / 1000.0);
            streams_result.success = true;

            double improvement = ((streams_result.speedup - 1.0) * 100.0);

            std::cout << "  Time: " << streams_timings.total_ms << " ms\n";
            std::cout << "  Speedup vs standard: " << std::fixed << std::setprecision(2)
                      << streams_result.speedup << "x (" << improvement << "% faster)\n";
            std::cout << "  Overlap benefit: " << streams_timings.overlap_benefit_ms << " ms saved\n";
            std::cout << "  Throughput: " << streams_result.throughput_mbps << " MB/s\n\n";

            csv_writer_.writeResult(streams_result);

            if (streams_result.speedup > best_speedup) {
                best_speedup = streams_result.speedup;
                best_streams_result = streams_result;
            }
        }

        // Summary
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  Summary: " << test_name << std::string(48 - test_name.length(), ' ') << "║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Adaptive (no streams):   " << std::setw(10) << standard_result.time_ms
                  << " ms (chunk: " << chunk_size << ")" << std::string(8, ' ') << "║\n";
        std::cout << "║  Adaptive + Streams:      " << std::setw(10) << best_streams_result.time_ms
                  << " ms (" << best_streams_result.implementation.substr(21) << " streams)"
                  << std::string(8, ' ') << "║\n";
        std::cout << "║  Speedup:                 " << std::setw(10) << std::fixed << std::setprecision(2)
                  << best_speedup << "x" << std::string(25, ' ') << "║\n";
        std::cout << "║  Improvement:             " << std::setw(9) << ((best_speedup - 1.0) * 100)
                  << "%" << std::string(25, ' ') << "║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
#else
        std::cout << "CUDA not available. Skipping streams comparison.\n";
#endif
    }

private:
    size_t getTotalSize(const std::vector<std::string>& data) const {
        size_t total = 0;
        for (const auto& item : data) {
            total += item.size();
        }
        return total;
    }
};

// Workload comparison tests
class WorkloadComparator {
private:
    CSVWriter& csv_writer_;

public:
    WorkloadComparator(CSVWriter& writer) : csv_writer_(writer) {}

    void runComparison(const std::string& test_name,
                      const std::vector<std::string>& data,
                      bool skip_cpu = false) {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  Workload Comparison: " << test_name << std::string(38 - test_name.length(), ' ') << "║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

        std::cout << std::left
                  << std::setw(20) << "Implementation"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(15) << "Speedup"
                  << std::setw(20) << "Throughput (MB/s)"
                  << "\n";
        std::cout << std::string(70, '-') << "\n";

        double baseline = 0;

        try {
            // CPU tests removed - run './merkle_comparison' for CPU benchmarks
            // This benchmark focuses on GPU implementations only

            // Test GPU Standard
            std::cout << "Running GPU Standard... " << std::flush;
            auto gpu_result = simulateGPU(test_name, data, 0);
            gpu_result.speedup = 0.0;  // No CPU baseline to compare against
            printResult(gpu_result);
            csv_writer_.writeResult(gpu_result);

            std::cout << "\n";
        } catch (const std::exception& e) {
            std::cerr << "\nError in benchmark: " << e.what() << "\n";
            throw;
        }
    }

private:
    size_t getTotalSize(const std::vector<std::string>& data) const {
        size_t total = 0;
        for (const auto& item : data) {
            total += item.size();
        }
        return total;
    }

    template<typename TreeType>
    BenchmarkResult runTest(const std::string& test_name,
                           const std::string& impl_name,
                           const std::vector<std::string>& data) {
        BenchmarkResult result;
        result.test_name = test_name;
        result.implementation = impl_name;
        result.data_count = data.size();
        result.total_bytes = getTotalSize(data);
        result.chunk_size = 0;
        result.memory_h2d_ms = 0;
        result.kernel_ms = 0;
        result.memory_d2h_ms = 0;
        result.success = true;

        TreeType tree;

        auto start = std::chrono::high_resolution_clock::now();
        tree.build(data);
        auto end = std::chrono::high_resolution_clock::now();

        result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        result.throughput_mbps = (result.total_bytes / (1024.0 * 1024.0)) / (result.time_ms / 1000.0);

        return result;
    }

    BenchmarkResult simulateGPU(const std::string& test_name,
                               const std::vector<std::string>& data,
                               double baseline) {
        BenchmarkResult result;
        result.test_name = test_name;
        result.implementation = "GPU";
        result.data_count = data.size();
        result.total_bytes = getTotalSize(data);
        result.chunk_size = 2048;
        result.success = true;

#ifdef HAS_CUDA
        // Use REAL GPU implementation
        MerkleTreeGPU gpu_tree(2048, 256);
        gpu_tree.build(data);

        GPUTimings timings = gpu_tree.getTimings();

        result.time_ms = timings.total_ms;
        result.memory_h2d_ms = timings.memory_h2d_ms;
        result.kernel_ms = timings.kernel_execution_ms;
        result.memory_d2h_ms = timings.memory_d2h_ms;
        result.speedup = (baseline > 0 && result.time_ms > 0) ? (baseline / result.time_ms) : 0.0;
        result.throughput_mbps = (result.time_ms > 0) ? ((result.total_bytes / (1024.0 * 1024.0)) / (result.time_ms / 1000.0)) : 0.0;
#else
        // Simulation logic (no CUDA)
        double transfer_time = (result.total_bytes / (1024.0 * 1024.0 * 500.0)) * 1000.0 * 2;
        double compute_time = baseline * 0.1;

        result.time_ms = transfer_time + compute_time;
        result.memory_h2d_ms = transfer_time / 2;
        result.kernel_ms = compute_time;
        result.memory_d2h_ms = transfer_time / 2;
        result.speedup = (baseline > 0 && result.time_ms > 0) ? (baseline / result.time_ms) : 0.0;
        result.throughput_mbps = (result.time_ms > 0) ? ((result.total_bytes / (1024.0 * 1024.0)) / (result.time_ms / 1000.0)) : 0.0;
#endif

        return result;
    }

    BenchmarkResult simulateHybrid(const std::string& test_name,
                                  const std::vector<std::string>& data,
                                  double baseline) {
        BenchmarkResult result;
        result.test_name = test_name;
        result.implementation = "Hybrid";
        result.data_count = data.size();
        result.total_bytes = getTotalSize(data);
        result.chunk_size = 2048;
        result.success = true;

        // Hybrid is typically best of both worlds
        double transfer_time = (result.total_bytes / (1024.0 * 1024.0 * 500.0)) * 1000.0;
        double compute_time = baseline * 0.15; // Slightly slower than pure GPU

        result.time_ms = transfer_time + compute_time;
        result.memory_h2d_ms = transfer_time / 2;
        result.kernel_ms = compute_time;
        result.memory_d2h_ms = transfer_time / 2;
        result.speedup = (baseline > 0 && result.time_ms > 0) ? (baseline / result.time_ms) : 0.0;
        result.throughput_mbps = (result.time_ms > 0) ? ((result.total_bytes / (1024.0 * 1024.0)) / (result.time_ms / 1000.0)) : 0.0;

        return result;
    }

    void printResult(const BenchmarkResult& result) {
        std::cout << std::left
                  << std::setw(20) << result.implementation
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.speedup
                  << std::setw(20) << std::fixed << std::setprecision(2) << result.throughput_mbps
                  << "\n";
    }
};

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                  GPU MERKLE TREE BENCHMARK SUITE                 ║\n";
    std::cout << "║              Chunk Size Optimization & Workload Analysis         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    // Benchmark configuration flags (set to false to skip expensive tests)
    bool run_standard_tests = true;     // 1K to 1M items (~5-10 minutes)
    bool run_extreme_tests = true;      // 10M items (~10-15 minutes)
    bool run_massive_tests = false;     // 100M items (~30-60 minutes, 20GB+ RAM)
                                        // Set to true only if you have 32GB+ RAM

    std::cout << "\nBenchmark Configuration:\n";
    std::cout << "  Standard Tests (1K-1M):   " << (run_standard_tests ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "  Extreme Tests (10M):      " << (run_extreme_tests ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "  Massive Tests (100M):     " << (run_massive_tests ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "\nEstimated runtime: ";
    int est_minutes = 0;
    if (run_standard_tests) est_minutes += 10;
    if (run_extreme_tests) est_minutes += 15;
    if (run_massive_tests) est_minutes += 45;
    std::cout << est_minutes << " minutes\n";
    std::cout << "───────────────────────────────────────────────────────────────────\n\n";

    // Create plots directory if it doesn't exist
    system("mkdir -p plots");

    // Generate timestamped filename in plots folder
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream filename_ss;
    filename_ss << "plots/gpu_benchmark_results_"
                << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S")
                << ".csv";
    std::string results_filename = filename_ss.str();

    CSVWriter csv_writer(results_filename);
    std::cout << "Results will be saved to: " << results_filename << "\n";
    std::cout << "\nNOTE: This benchmark tests GPU implementations only.\n";
    std::cout << "      Run './merkle_comparison' first for CPU baseline data.\n";
    std::cout << "      Both CSV files can be combined in visualization.\n\n";

    if (!run_standard_tests && !run_extreme_tests && !run_massive_tests) {
        std::cout << "All tests disabled! Enable at least one test suite in main().\n";
        return 1;
    }

    // ═══════════════════════════════════════════════════════════════
    // STANDARD TESTS (1K - 1M items)
    // ═══════════════════════════════════════════════════════════════
    if (run_standard_tests) {

    // Test Suite 1: Small uniform data
    {
        auto data = generateRandomData(1000, 100, 100);
        WorkloadComparator comparator(csv_writer);
        comparator.runComparison("Small Uniform (1K x 100B)", data);
    }

    // Test Suite 2: Medium variable data
    {
        auto data = generateRandomData(10000, 50, 500);
        WorkloadComparator comparator(csv_writer);
        comparator.runComparison("Medium Variable (10K x 50-500B)", data);

        GPUChunkOptimizer optimizer(csv_writer);
        optimizer.runOptimizationTests("Medium Variable", data);
    }

    // Test Suite 3: Large uniform data
    {
        auto data = generateRandomData(100000, 256, 256);
        WorkloadComparator comparator(csv_writer);
        comparator.runComparison("Large Uniform (100K x 256B)", data);

        GPUChunkOptimizer optimizer(csv_writer);
        optimizer.runOptimizationTests("Large Uniform", data);
    }

    // Test Suite 4: Structured data - REMOVED (keeping only scaling datasets)
    // {
    //     auto data = generateStructuredData(50000);
    //     WorkloadComparator comparator(csv_writer);
    //     comparator.runComparison("Structured Records (50K)", data);
    //
    //     GPUChunkOptimizer optimizer(csv_writer);
    //     optimizer.runOptimizationTests("Structured Records", data);
    // }

    // Test Suite 4 (was 5): Very large dataset
    {
        auto data = generateRandomData(1000000, 100, 200);
        WorkloadComparator comparator(csv_writer);
        comparator.runComparison("Very Large (1M x 100-200B)", data, true); // Skip CPU for > 1M

        GPUChunkOptimizer optimizer(csv_writer);
        optimizer.runOptimizationTests("Very Large", data);
    }

    // Test Suite 6: Adaptive GPU Comparison
    {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  ADAPTIVE GPU OPTIMIZATION TEST                                  ║\n";
        std::cout << "║  Comparing Manual Chunk Size vs Adaptive (Hardware-Aware)       ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

        AdaptiveGPUComparator adaptive_comparator(csv_writer);

        // Test on medium workload
        {
            auto data = generateRandomData(50000, 256, 256);
            adaptive_comparator.runAdaptiveComparison("Medium (50K x 256B)", data);
        }

        // Test on large workload
        {
            auto data = generateRandomData(200000, 256, 256);
            adaptive_comparator.runAdaptiveComparison("Large (200K x 256B)", data);
        }

        // Test on very large workload
        {
            auto data = generateRandomData(500000, 128, 128);
            adaptive_comparator.runAdaptiveComparison("Very Large (500K x 128B)", data);
        }
    }

    // Test Suite 7: CUDA Streams + Adaptive
    {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  CUDA STREAMS + ADAPTIVE OPTIMIZATION TEST                       ║\n";
        std::cout << "║  Combining Adaptive Chunk Sizing with Overlapped Execution      ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

        GPUStreamsComparator streams_comparator(csv_writer);

        // Test on medium workload
        {
            auto data = generateRandomData(50000, 256, 256);
            streams_comparator.runStreamsComparison("Medium (50K x 256B)", data);
        }

        // Test on large workload
        {
            auto data = generateRandomData(200000, 256, 256);
            streams_comparator.runStreamsComparison("Large (200K x 256B)", data);
        }

        // Test on very large workload
        {
            auto data = generateRandomData(500000, 128, 128);
            streams_comparator.runStreamsComparison("Very Large (500K x 128B)", data);
        }
    }

    } // End standard tests

    // ═══════════════════════════════════════════════════════════════
    // EXTREME TESTS (10M items)
    // ═══════════════════════════════════════════════════════════════
    if (run_extreme_tests) {

    // Test Suite 6: Extreme scale - 10M items
    {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  EXTREME SCALE TEST: 10 Million Items                           ║\n";
        std::cout << "║  This test will take several minutes and use significant memory ║\n";
        std::cout << "║  Testing larger chunk sizes (4K-64K) for better GPU efficiency  ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

        std::cout << "\nGenerating 10M items...\n";
        auto data = generateRandomData(10000000, 128, 256);
        std::cout << "Data generated. Total size: ~"
                  << (data.size() * 192 / (1024*1024)) << " MB\n";

        WorkloadComparator comparator(csv_writer);
        comparator.runComparison("Extreme (10M x 128-256B)", data, true); // Skip CPU for > 1M

        // For 10M items, test larger chunks for efficiency
        GPUChunkOptimizer optimizer(csv_writer);
        std::cout << "\nTesting optimized chunk sizes for 10M scale...\n";
        optimizer.runOptimizationTests("Extreme 10M", data);
    }

    } // End extreme tests

    // ═══════════════════════════════════════════════════════════════
    // MASSIVE TESTS (100M items)
    // ═══════════════════════════════════════════════════════════════
    if (run_massive_tests) {

    // Test Suite 7: Massive scale - 100M items
    {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  MASSIVE SCALE TEST: 100 Million Items                          ║\n";
        std::cout << "║  WARNING: This test may take 30+ minutes                        ║\n";
        std::cout << "║  NOTE: Skipping Sequential/Multithread CPU tests (too much RAM) ║\n";
        std::cout << "║  Testing GPU only to avoid OOM killer                           ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

        std::cout << "\nGenerating 100M items... (this may take a minute)\n";
        auto data = generateRandomData(100000000, 128, 256);

        std::cout << "Data generated. Total size: ~"
                  << (data.size() * 192 / (1024*1024)) << " MB\n";

        // SKIP WorkloadComparator - CPU implementations use too much memory!
        // Only test GPU with chunk optimization

        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  GPU Chunk Size Optimization: Massive 100M                     ║\n";
        std::cout << "║  (Skipping CPU baseline - would require 40+ GB RAM)           ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

        GPUChunkOptimizer optimizer(csv_writer);

        // Manually test GPU without CPU baseline
        std::cout << std::left
                  << std::setw(12) << "Chunk Size"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(15) << "Throughput"
                  << std::setw(15) << "H2D (ms)"
                  << std::setw(15) << "Kernel (ms)"
                  << std::setw(15) << "D2H (ms)"
                  << "\n";
        std::cout << std::string(87, '-') << "\n";

        // Test only a few strategic chunk sizes to save time
        std::vector<size_t> massive_chunks = {4096, 8192, 16384, 32768, 65536};

        for (size_t chunk_size : massive_chunks) {
#ifdef HAS_CUDA
            MerkleTreeGPU gpu_tree(chunk_size, 256);

            std::cout << "Testing chunk size " << formatNumber(chunk_size) << "... " << std::flush;

            gpu_tree.build(data);

            GPUTimings timings = gpu_tree.getTimings();

            BenchmarkResult result;
            result.test_name = "Massive 100M";
            result.implementation = "GPU";
            result.data_count = data.size();
            result.total_bytes = data.size() * 192; // Approximate
            result.chunk_size = chunk_size;
            result.time_ms = timings.total_ms;
            result.memory_h2d_ms = timings.memory_h2d_ms;
            result.kernel_ms = timings.kernel_execution_ms;
            result.memory_d2h_ms = timings.memory_d2h_ms;
            result.speedup = 0; // No baseline
            result.throughput_mbps = (result.total_bytes / (1024.0 * 1024.0)) / (result.time_ms / 1000.0);
            result.success = true;

            std::cout << "Done!\n";
            std::cout << std::left
                      << std::setw(12) << formatNumber(chunk_size)
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.time_ms
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.throughput_mbps
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.memory_h2d_ms
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.kernel_ms
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.memory_d2h_ms
                      << "\n";

            csv_writer.writeResult(result);

            // Free memory after each test
            std::string root = gpu_tree.getRootHash();
            (void)root; // Use the result to prevent optimization removing it
#else
            std::cout << "CUDA not available - skipping\n";
#endif
        }

        std::cout << "\n✓ Massive scale test complete (GPU only)\n";

        // Free the data vector to reclaim memory
        data.clear();
        data.shrink_to_fit();
    }

    } // End massive tests

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                     BENCHMARK COMPLETE                           ║\n";
    std::cout << "║          Results saved to: " << std::left << std::setw(30)
              << results_filename << "     ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

#ifdef HAS_CUDA
    std::cout << "\n✓ GPU results use REAL CUDA acceleration!\n";
    std::cout << "  Hybrid results are still simulated (future enhancement).\n\n";
#else
    std::cout << "\nNOTE: GPU results are currently simulated. Compile with CUDA\n";
    std::cout << "      to get real GPU performance measurements.\n\n";
#endif

    return 0;
}

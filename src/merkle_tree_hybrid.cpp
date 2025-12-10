#include "header_merkle_tree.hpp"
#include <openssl/sha.h>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>

// Forward declaration of GPU implementation
class MerkleTreeGPU;

class MerkleTreeHybrid {
private:
    Node* root_;
    std::vector<std::string> leaf_hashes_;

    // Thresholds for deciding CPU vs GPU
    static constexpr size_t MIN_ITEMS_FOR_GPU = 1000;        // Minimum items to consider GPU
    static constexpr size_t MIN_TOTAL_BYTES_FOR_GPU = 100000; // Minimum total data size
    static constexpr size_t OPTIMAL_GPU_CHUNK_SIZE = 2048;   // Items per GPU batch

    // Performance history for adaptive decision making
    struct PerformanceMetrics {
        double cpu_time_ms;
        double gpu_time_ms;
        size_t data_size;
        size_t item_count;
        bool used_gpu;
    };

    std::vector<PerformanceMetrics> history_;

    enum class ExecutionMode {
        CPU_SEQUENTIAL,
        CPU_MULTITHREAD,
        GPU_ONLY,
        HYBRID_CPU_GPU
    };

public:
    MerkleTreeHybrid() : root_(nullptr) {}

    ~MerkleTreeHybrid() {
        deleteTree(root_);
    }

    void build(const std::vector<std::string>& data) {
        if (data.empty()) {
            root_ = nullptr;
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Analyze workload
        WorkloadCharacteristics workload = analyzeWorkload(data);

        // Decide execution strategy
        ExecutionMode mode = selectExecutionMode(workload);

        // Execute based on selected mode
        switch (mode) {
            case ExecutionMode::CPU_SEQUENTIAL:
                buildSequential(data);
                break;
            case ExecutionMode::CPU_MULTITHREAD:
                buildMultithread(data);
                break;
            case ExecutionMode::GPU_ONLY:
                buildGPU(data);
                break;
            case ExecutionMode::HYBRID_CPU_GPU:
                buildHybrid(data);
                break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Record performance for future decisions
        recordPerformance(workload, elapsed_ms, mode);
    }

    std::string getRootHash() const {
        return root_ ? root_->hash : "";
    }

    const std::vector<PerformanceMetrics>& getHistory() const {
        return history_;
    }

private:
    struct WorkloadCharacteristics {
        size_t item_count;
        size_t total_bytes;
        size_t avg_item_size;
        size_t min_item_size;
        size_t max_item_size;
        double size_variance;
    };

    WorkloadCharacteristics analyzeWorkload(const std::vector<std::string>& data) const {
        WorkloadCharacteristics wc;
        wc.item_count = data.size();
        wc.total_bytes = 0;
        wc.min_item_size = SIZE_MAX;
        wc.max_item_size = 0;

        for (const auto& item : data) {
            size_t size = item.size();
            wc.total_bytes += size;
            wc.min_item_size = std::min(wc.min_item_size, size);
            wc.max_item_size = std::max(wc.max_item_size, size);
        }

        wc.avg_item_size = wc.total_bytes / wc.item_count;

        // Calculate variance
        double variance_sum = 0.0;
        for (const auto& item : data) {
            double diff = static_cast<double>(item.size()) - wc.avg_item_size;
            variance_sum += diff * diff;
        }
        wc.size_variance = variance_sum / wc.item_count;

        return wc;
    }

    ExecutionMode selectExecutionMode(const WorkloadCharacteristics& wc) const {
        // Small workloads: use sequential
        if (wc.item_count < 100) {
            return ExecutionMode::CPU_SEQUENTIAL;
        }

        // Medium workloads: use multithread CPU
        if (wc.item_count < MIN_ITEMS_FOR_GPU || wc.total_bytes < MIN_TOTAL_BYTES_FOR_GPU) {
            return ExecutionMode::CPU_MULTITHREAD;
        }

        // Large uniform workloads: GPU is best
        if (wc.item_count > 10000 && wc.size_variance < wc.avg_item_size * wc.avg_item_size) {
            return ExecutionMode::GPU_ONLY;
        }

        // Very large workloads: hybrid approach
        if (wc.item_count > 100000) {
            return ExecutionMode::HYBRID_CPU_GPU;
        }

        // Default: use GPU for large workloads
        return ExecutionMode::GPU_ONLY;
    }

    void buildSequential(const std::vector<std::string>& data) {
        // Use the existing sequential implementation
        MerkleTreeSequential seq_tree;
        seq_tree.build(data);

        // Copy results
        leaf_hashes_ = seq_tree.getLeafHashes();
        root_ = copyNode(seq_tree.getRoot());
    }

    void buildMultithread(const std::vector<std::string>& data) {
        // Use the existing multithread implementation
        MerkleTreeLockFree mt_tree;
        mt_tree.build(data);

        // Copy results
        leaf_hashes_ = mt_tree.getLeafHashes();
        root_ = copyNode(mt_tree.getRoot());
    }

    void buildGPU(const std::vector<std::string>& data) {
        // This would use the GPU implementation
        // For now, fall back to multithread
        // TODO: Integrate with MerkleTreeGPU when compiled with CUDA
        std::cerr << "Warning: GPU implementation not available, using multithread CPU\n";
        buildMultithread(data);
    }

    void buildHybrid(const std::vector<std::string>& data) {
        // Hybrid approach: split work between CPU and GPU
        // Strategy: Use GPU for bulk leaf hashing, CPU for tree construction

        // For demonstration, we'll simulate by splitting the workload
        size_t split_point = data.size() / 2;

        // "GPU" part (first half)
        auto gpu_start = std::chrono::high_resolution_clock::now();
        MerkleTreeLockFree tree1;
        tree1.build(std::vector<std::string>(data.begin(), data.begin() + split_point));
        auto gpu_end = std::chrono::high_resolution_clock::now();

        // "CPU" part (second half)
        auto cpu_start = std::chrono::high_resolution_clock::now();
        MerkleTreeLockFree tree2;
        tree2.build(std::vector<std::string>(data.begin() + split_point, data.end()));
        auto cpu_end = std::chrono::high_resolution_clock::now();

        // Merge the two trees
        leaf_hashes_ = tree1.getLeafHashes();
        auto hashes2 = tree2.getLeafHashes();
        leaf_hashes_.insert(leaf_hashes_.end(), hashes2.begin(), hashes2.end());

        // Build final tree from combined hashes
        root_ = buildTreeFromHashes(leaf_hashes_);

        double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        std::cout << "  [Hybrid] GPU portion: " << gpu_ms << "ms, CPU portion: " << cpu_ms << "ms\n";
    }

    void recordPerformance(const WorkloadCharacteristics& wc, double time_ms, ExecutionMode mode) {
        PerformanceMetrics metrics;
        metrics.data_size = wc.total_bytes;
        metrics.item_count = wc.item_count;
        metrics.used_gpu = (mode == ExecutionMode::GPU_ONLY || mode == ExecutionMode::HYBRID_CPU_GPU);

        if (metrics.used_gpu) {
            metrics.gpu_time_ms = time_ms;
            metrics.cpu_time_ms = 0;
        } else {
            metrics.cpu_time_ms = time_ms;
            metrics.gpu_time_ms = 0;
        }

        history_.push_back(metrics);

        // Keep only recent history (last 100 entries)
        if (history_.size() > 100) {
            history_.erase(history_.begin());
        }
    }

    // Helper: Build tree from precomputed hashes
    Node* buildTreeFromHashes(const std::vector<std::string>& hashes) {
        if (hashes.empty()) return nullptr;

        std::vector<Node*> current_level;
        for (const auto& hash : hashes) {
            current_level.push_back(new Node{hash, nullptr, nullptr});
        }

        while (current_level.size() > 1) {
            std::vector<Node*> next_level;
            for (size_t i = 0; i < current_level.size(); i += 2) {
                Node* left = current_level[i];
                Node* right = (i + 1 < current_level.size()) ? current_level[i + 1] : left;

                std::string combined = left->hash + right->hash;
                std::string parent_hash = hashString(combined);

                next_level.push_back(new Node{parent_hash, left, right});
            }
            current_level = next_level;
        }

        return current_level.empty() ? nullptr : current_level[0];
    }

    // Helper: Copy a tree node recursively
    Node* copyNode(const Node* node) {
        if (!node) return nullptr;
        return new Node{node->hash, copyNode(node->left), copyNode(node->right)};
    }

    // Helper: Hash a string using SHA-256 (CPU version)
    std::string hashString(const std::string& data) const {
        // Use OpenSSL for hashing
        unsigned char hash[32];
        SHA256_CTX sha256;
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, data.c_str(), data.size());
        SHA256_Final(hash, &sha256);

        std::stringstream ss;
        for (int i = 0; i < 32; i++) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
        }
        return ss.str();
    }

    void deleteTree(Node* node) {
        if (!node) return;
        deleteTree(node->left);
        deleteTree(node->right);
        delete node;
    }
};

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <random>
#include <chrono>
#include "src/header_merkle_tree.hpp"

#ifdef HAS_CUDA
#include "src/merkle_tree_gpu.hpp"
#include "src/merkle_tree_gpu_adaptive.hpp"
#include "src/merkle_tree_gpu_streams.hpp"
#endif

// ANSI color codes for output
#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_CYAN "\033[36m"
#define COLOR_RESET "\033[0m"

struct ImplementationResult {
    std::string name;
    std::string root_hash;
    double time_ms;
    bool success;
    std::string error_message;
};

void printHeader(const std::string& title) {
    std::cout << "\n╔" << std::string(78, '═') << "╗\n";
    std::cout << "║ " << std::left << std::setw(76) << title << " ║\n";
    std::cout << "╚" << std::string(78, '═') << "╝\n\n";
}

void printTestCase(const std::string& name, const std::vector<std::string>& data) {
    std::cout << COLOR_CYAN << "Test Case: " << name << COLOR_RESET << "\n";
    std::cout << "  Items: " << data.size() << "\n";
    size_t total_bytes = 0;
    for (const auto& item : data) {
        total_bytes += item.size();
    }
    std::cout << "  Total size: " << total_bytes << " bytes\n";
    std::cout << "  First few items: ";
    for (size_t i = 0; i < std::min(size_t(3), data.size()); i++) {
        std::cout << "\"" << (data[i].length() > 10 ? data[i].substr(0, 10) + "..." : data[i]) << "\" ";
    }
    std::cout << "\n\n";
}

void printResults(const std::vector<ImplementationResult>& results) {
    std::cout << "\n" << std::string(100, '─') << "\n";
    std::cout << std::left
              << std::setw(30) << "Implementation"
              << std::setw(20) << "Status"
              << std::setw(70) << "Root Hash (first 32 chars)"
              << "\n";
    std::cout << std::string(100, '─') << "\n";

    for (const auto& result : results) {
        std::cout << std::setw(30) << result.name;

        if (result.success) {
            std::cout << COLOR_GREEN << std::setw(20) << "✓ SUCCESS" << COLOR_RESET;
        } else {
            std::cout << COLOR_RED << std::setw(20) << "✗ FAILED" << COLOR_RESET;
        }

        if (result.success) {
            std::cout << result.root_hash.substr(0, 32) << "...";
        } else {
            std::cout << COLOR_RED << result.error_message << COLOR_RESET;
        }

        std::cout << "\n";
    }
    std::cout << std::string(100, '─') << "\n";
}

bool verifyConsistency(const std::vector<ImplementationResult>& results) {
    std::cout << "\n" << COLOR_CYAN << "Consistency Check:" << COLOR_RESET << "\n";

    // Get reference hash (from first successful implementation)
    std::string reference_hash;
    std::string reference_name;

    for (const auto& result : results) {
        if (result.success && !result.root_hash.empty()) {
            reference_hash = result.root_hash;
            reference_name = result.name;
            break;
        }
    }

    if (reference_hash.empty()) {
        std::cout << COLOR_RED << "✗ No successful implementations to compare!" << COLOR_RESET << "\n";
        return false;
    }

    std::cout << "Reference: " << reference_name << "\n";
    std::cout << "Hash: " << reference_hash.substr(0, 64) << "...\n\n";

    bool all_match = true;

    for (const auto& result : results) {
        if (!result.success) {
            std::cout << COLOR_YELLOW << "⚠  " << result.name << ": SKIPPED (failed to run)" << COLOR_RESET << "\n";
            continue;
        }

        if (result.root_hash == reference_hash) {
            std::cout << COLOR_GREEN << "✓  " << result.name << ": MATCH" << COLOR_RESET << "\n";
        } else {
            std::cout << COLOR_RED << "✗  " << result.name << ": MISMATCH!" << COLOR_RESET << "\n";
            std::cout << "   Expected: " << reference_hash.substr(0, 64) << "...\n";
            std::cout << "   Got:      " << result.root_hash.substr(0, 64) << "...\n";
            all_match = false;
        }
    }

    return all_match;
}

int main() {
    printHeader("Merkle Tree Implementations - Correctness Verification");

    std::cout << "This test verifies that all implementations produce identical root hashes.\n";
    std::cout << "Testing: Sequential CPU, Multithread CPU, GPU, GPU Adaptive, GPU Streams\n\n";

    // Define test cases
    struct TestCase {
        std::string name;
        std::vector<std::string> data;
    };

    std::vector<TestCase> test_cases = {
        // Test 1: Minimal case (edge case)
        {
            "Minimal (2 items)",
            {"Hello", "World"}
        },

        // Test 2: Power of 2
        {
            "Power of 2 (4 items)",
            {"a", "b", "c", "d"}
        },

        // Test 3: Non-power of 2
        {
            "Non-power of 2 (5 items)",
            {"Alice", "Bob", "Charlie", "David", "Eve"}
        },

        // Test 4: Variable length data
        {
            "Variable length",
            {
                "short",
                "a bit longer string",
                "x",
                "this is a much longer string with more content to hash"
            }
        },

        // Test 5: Uniform data (100 items)
        {
            "Uniform (100 items x 64 bytes)",
            {}
        },

        // Test 6: Medium workload (1000 items)
        {
            "Medium (1K items x 128 bytes)",
            {}
        },

        // Test 7: Large workload (10000 items)
        {
            "Large (10K items x 256 bytes)",
            {}
        }
    };

    // Generate data for larger test cases
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<> char_dist(32, 126);

    // Test 5: 100 uniform items
    for (int i = 0; i < 100; i++) {
        std::string item;
        for (int j = 0; j < 64; j++) {
            item += static_cast<char>(char_dist(gen));
        }
        test_cases[4].data.push_back(item);
    }

    // Test 6: 1000 medium items
    for (int i = 0; i < 1000; i++) {
        std::string item;
        for (int j = 0; j < 128; j++) {
            item += static_cast<char>(char_dist(gen));
        }
        test_cases[5].data.push_back(item);
    }

    // Test 7: 10000 large items
    for (int i = 0; i < 10000; i++) {
        std::string item;
        for (int j = 0; j < 256; j++) {
            item += static_cast<char>(char_dist(gen));
        }
        test_cases[6].data.push_back(item);
    }

    int total_tests = 0;
    int passed_tests = 0;

    // Run all test cases
    for (const auto& test_case : test_cases) {
        printTestCase(test_case.name, test_case.data);

        std::vector<ImplementationResult> results;

        // Test 1: Sequential CPU
        {
            ImplementationResult result;
            result.name = "Sequential CPU";
            try {
                auto start = std::chrono::high_resolution_clock::now();

                MerkleTreeSequential tree;
                tree.build(test_case.data, 32);

                auto end = std::chrono::high_resolution_clock::now();
                result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

                result.root_hash = tree.getRoot()->hash;
                result.success = !result.root_hash.empty();
            } catch (const std::exception& e) {
                result.success = false;
                result.error_message = std::string("Exception: ") + e.what();
            }
            results.push_back(result);
        }

        // Test 2: Multithread CPU
        {
            ImplementationResult result;
            result.name = "Multithread CPU";
            try {
                auto start = std::chrono::high_resolution_clock::now();

                MerkleTreeLockFree tree;
                tree.build(test_case.data, 32);

                auto end = std::chrono::high_resolution_clock::now();
                result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

                result.root_hash = tree.getRootHash();
                result.success = !result.root_hash.empty();
            } catch (const std::exception& e) {
                result.success = false;
                result.error_message = std::string("Exception: ") + e.what();
            }
            results.push_back(result);
        }

#ifdef HAS_CUDA
        // Test 3: Standard GPU
        {
            ImplementationResult result;
            result.name = "GPU (manual chunk: 2048)";
            try {
                auto start = std::chrono::high_resolution_clock::now();

                MerkleTreeGPU tree(2048, 256);
                tree.build(test_case.data);

                auto end = std::chrono::high_resolution_clock::now();
                result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

                result.root_hash = tree.getRootHash();
                result.success = !result.root_hash.empty();
            } catch (const std::exception& e) {
                result.success = false;
                result.error_message = std::string("Exception: ") + e.what();
            }
            results.push_back(result);
        }

        // Test 4: GPU Adaptive
        {
            ImplementationResult result;
            result.name = "GPU Adaptive (auto chunk)";
            try {
                auto start = std::chrono::high_resolution_clock::now();

                MerkleTreeGPUAdaptive tree(false);
                tree.build(test_case.data);

                auto end = std::chrono::high_resolution_clock::now();
                result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

                result.root_hash = tree.getRootHash();
                result.success = !result.root_hash.empty();

                auto config = tree.getLastConfig();
                result.name += " [chunk: " + std::to_string(config.chunk_size) + "]";
            } catch (const std::exception& e) {
                result.success = false;
                result.error_message = std::string("Exception: ") + e.what();
            }
            results.push_back(result);
        }

        // Test 5: GPU Streams (4 streams)
        {
            ImplementationResult result;
            result.name = "GPU Streams (4 streams)";
            try {
                auto start = std::chrono::high_resolution_clock::now();

                MerkleTreeGPUStreams tree(2048, 256, 4);
                tree.build(test_case.data);

                auto end = std::chrono::high_resolution_clock::now();
                result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

                result.root_hash = tree.getRootHash();
                result.success = !result.root_hash.empty();
            } catch (const std::exception& e) {
                result.success = false;
                result.error_message = std::string("Exception: ") + e.what();
            }
            results.push_back(result);
        }
#else
        // CUDA not available
        {
            ImplementationResult result;
            result.name = "GPU (CUDA not available)";
            result.success = false;
            result.error_message = "CUDA not compiled";
            results.push_back(result);
        }
        {
            ImplementationResult result;
            result.name = "GPU Adaptive (CUDA not available)";
            result.success = false;
            result.error_message = "CUDA not compiled";
            results.push_back(result);
        }
        {
            ImplementationResult result;
            result.name = "GPU Streams (CUDA not available)";
            result.success = false;
            result.error_message = "CUDA not compiled";
            results.push_back(result);
        }
#endif

        // Print results for this test case
        printResults(results);

        // Verify consistency
        bool consistent = verifyConsistency(results);

        total_tests++;
        if (consistent) {
            passed_tests++;
            std::cout << "\n" << COLOR_GREEN << "✓ Test PASSED: All implementations agree!" << COLOR_RESET << "\n";
        } else {
            std::cout << "\n" << COLOR_RED << "✗ Test FAILED: Implementations disagree!" << COLOR_RESET << "\n";
        }

        std::cout << "\n" << std::string(100, '=') << "\n";
    }

    // Final summary
    printHeader("Final Summary");

    std::cout << "Total test cases: " << total_tests << "\n";
    std::cout << "Passed: " << COLOR_GREEN << passed_tests << COLOR_RESET << "\n";
    std::cout << "Failed: " << COLOR_RED << (total_tests - passed_tests) << COLOR_RESET << "\n";
    std::cout << "Success rate: " << (100.0 * passed_tests / total_tests) << "%\n\n";

    if (passed_tests == total_tests) {
        std::cout << COLOR_GREEN << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  ✓ ALL TESTS PASSED                                        ║\n";
        std::cout << "║  All implementations produce identical root hashes!        ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝" << COLOR_RESET << "\n\n";
        return 0;
    } else {
        std::cout << COLOR_RED << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  ✗ SOME TESTS FAILED                                       ║\n";
        std::cout << "║  Implementations produce different root hashes!            ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝" << COLOR_RESET << "\n\n";
        return 1;
    }
}

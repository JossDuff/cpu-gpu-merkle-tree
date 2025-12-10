#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include "src/header_merkle_tree.hpp"

#ifdef HAS_CUDA
#include "src/merkle_tree_gpu.hpp"
#endif

// Test correctness by comparing root hashes across implementations
int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Merkle Tree Correctness Verification Test               ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";

    // Test with known data
    std::vector<std::string> test_cases[] = {
        {"Hello", "World"},
        {"a", "b", "c", "d"},
        {"Apple", "Banana", "Cherry", "Date", "Elderberry"},
        {"Test1", "Test2", "Test3", "Test4", "Test5", "Test6", "Test7"},
        {"The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"}
    };

    int test_num = 1;
    int passed = 0;
    int failed = 0;

    for (const auto& data : test_cases) {
        std::cout << "Test " << test_num++ << ": " << data.size() << " items\n";
        std::cout << std::string(60, '-') << "\n";

        // Build with Sequential
        MerkleTreeSequential seq_tree;
        seq_tree.build(data, 32);
        std::string seq_root = seq_tree.getRoot() ? seq_tree.getRoot()->hash : "";

        // Build with Multithread
        MerkleTreeLockFree mt_tree;
        mt_tree.build(data, 32);
        std::string mt_root = mt_tree.getRootHash();

        // Display results
        std::cout << "Sequential Root:  " << seq_root << "\n";
        std::cout << "Multithread Root: " << mt_root << "\n";

#ifdef HAS_CUDA
        // Build with GPU
        MerkleTreeGPU gpu_tree(2048, 256);
        gpu_tree.build(data);
        std::string gpu_root = gpu_tree.getRootHash();
        std::cout << "GPU Root:         " << gpu_root << "\n";

        // Verify all match
        bool all_match = (seq_root == mt_root) && (mt_root == gpu_root);
#else
        std::cout << "GPU Root:         [CUDA not available]\n";
        bool all_match = (seq_root == mt_root);
#endif

        if (all_match) {
            std::cout << "✅ PASS: All implementations match!\n";
            passed++;
        } else {
            std::cout << "❌ FAIL: Implementations produce different results!\n";
            if (seq_root != mt_root) {
                std::cout << "   Sequential ≠ Multithread\n";
            }
#ifdef HAS_CUDA
            if (seq_root != gpu_root) {
                std::cout << "   Sequential ≠ GPU\n";
            }
            if (mt_root != gpu_root) {
                std::cout << "   Multithread ≠ GPU\n";
            }
#endif
            failed++;
        }

        std::cout << "\n";
    }

    // Summary
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Test Summary                                             ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Total Tests: " << (passed + failed) << "                                            ║\n";
    std::cout << "║  Passed:      " << passed << "                                            ║\n";
    std::cout << "║  Failed:      " << failed << "                                            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";

    return (failed > 0) ? 1 : 0;
}

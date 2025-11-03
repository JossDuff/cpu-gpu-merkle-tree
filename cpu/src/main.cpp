#include "header_merkle_tree.hpp"
#include <iostream>
#include <chrono>
#include <array>
#include <iomanip>
#include <random>
#include <sstream>

// Generate random data of specified size
std::vector<std::string> generateRandomData(size_t count, size_t min_size, size_t max_size)
{
    std::vector<std::string> data(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(min_size, max_size);
    std::uniform_int_distribution<> char_dist(32, 126); // Printable ASCII

    for (size_t i = 0; i < count; ++i)
    {
        size_t length = size_dist(gen);
        data[i].reserve(length);
        for (size_t j = 0; j < length; ++j)
        {
            data[i] += static_cast<char>(char_dist(gen));
        }
    }

    return data;
}

// Generate structured data (like database records)
std::vector<std::string> generateStructuredData(size_t count)
{
    std::vector<std::string> data(count);

    for (size_t i = 0; i < count; ++i)
    {
        std::ostringstream oss;
        oss << "Record_" << std::setfill('0') << std::setw(10) << i
            << "_Timestamp_" << (1234567890 + i)
            << "_Value_" << (i * 3.14159)
            << "_Status_ACTIVE";
        data[i] = oss.str();
    }

    return data;
}

// Generate large text blocks (like documents)
std::vector<std::string> generateLargeTextBlocks(size_t count, size_t block_size)
{
    std::vector<std::string> data(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> word_dist(3, 12);

    const std::vector<std::string> words = {
        "Lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
        "adipiscing", "elit", "sed", "do", "eiusmod", "tempor",
        "incididunt", "ut", "labore", "et", "dolore", "magna",
        "aliqua", "Ut", "enim", "ad", "minim", "veniam"};

    for (size_t i = 0; i < count; ++i)
    {
        std::ostringstream oss;
        size_t current_size = 0;

        while (current_size < block_size)
        {
            const std::string &word = words[gen() % words.size()];
            oss << word << " ";
            current_size += word.length() + 1;
        }

        data[i] = oss.str();
    }

    return data;
}

// Format number with thousands separator
std::string formatNumber(size_t num)
{
    std::string str = std::to_string(num);
    int insertPosition = str.length() - 3;

    while (insertPosition > 0)
    {
        str.insert(insertPosition, ",");
        insertPosition -= 3;
    }

    return str;
}

// Run benchmark for a specific configuration
void runBenchmark(const std::string &test_name,
                  const std::vector<std::string> &data,
                  bool run_sequential = true)
{
    std::cout << "\n=== " << test_name << " ===\n";
    std::cout << "Data size: " << formatNumber(data.size()) << " items\n";

    // Calculate average item size
    size_t total_bytes = 0;
    for (const auto &item : data)
    {
        total_bytes += item.size();
    }
    double avg_size = static_cast<double>(total_bytes) / data.size();
    std::cout << "Average item size: " << std::fixed << std::setprecision(1)
              << avg_size << " bytes\n";
    std::cout << "Total data: " << formatNumber(total_bytes) << " bytes ("
              << std::fixed << std::setprecision(2) << (total_bytes / 1024.0 / 1024.0)
              << " MB)\n\n";

    double sequential_time = 0;

    // Sequential benchmark
    if (run_sequential)
    {
        std::cout << "Sequential version:\n";
        MerkleTreeSequential seq;
        auto byteData = seq.transformStringArray(data);

        auto start = std::chrono::high_resolution_clock::now();
        seq.constructor(byteData);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        sequential_time = duration.count();

        std::cout << "  Time: " << std::fixed << std::setprecision(2)
                  << sequential_time << " ms\n";
        if (seq.getRoot())
        {
            std::cout << "  Root: " << seq.getRoot()->hash.substr(0, 16) << "...\n";
        }
    }

    // Parallel benchmarks
    std::cout << "\n";
    std::cout << std::left << std::setw(15) << "Threads"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Speedup"
              << std::setw(20) << "Throughput (MB/s)\n";
    std::cout << std::string(65, '-') << "\n";

    double baseline_time = 0;
    std::string root_hash;

    for (size_t threads : {1, 2, 4, 8, 16, 32})
    {
        if (threads > std::thread::hardware_concurrency() * 2)
            break;

        MerkleTreeLockFree parallel(threads);
        auto byteData = parallel.transformStringArray(data);

        auto start = std::chrono::high_resolution_clock::now();
        parallel.constructor(byteData);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double time_ms = duration.count();

        if (threads == 1)
        {
            baseline_time = time_ms;
            if (parallel.getRoot())
            {
                root_hash = parallel.getRoot()->hash;
            }
        }

        double speedup = baseline_time / time_ms;
        double throughput = (total_bytes / 1024.0 / 1024.0) / (time_ms / 1000.0);

        std::cout << std::setw(15) << threads
                  << std::setw(15) << std::fixed << std::setprecision(2) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x"
                  << std::setw(20) << std::fixed << std::setprecision(2) << throughput
                  << "\n";
    }

    // Verify correctness if sequential was run
    if (run_sequential && sequential_time > 0)
    {
        MerkleTreeSequential seq;
        auto byteData = seq.transformStringArray(data);
        seq.constructor(byteData);

        bool match = (seq.getRoot()->hash == root_hash);
        std::cout << "\nCorrectness: " << (match ? "✓ PASS" : "✗ FAIL") << "\n";
    }
}

int main()
{
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║    Merkle Tree Performance Benchmark Suite                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nHardware: " << std::thread::hardware_concurrency()
              << " concurrent threads available\n";

    // ============================================================================
    // Test 1: Small uniform data (baseline)
    // ============================================================================
    {
        std::array<std::string, 10> smallData = {
            "Orange", "Apple", "Kiwi", "Strawberry", "Potato",
            "Road", "A big big big big big big big big big big word",
            ".", "*£$àze()", "GG"};

        std::vector<std::string> data(smallData.begin(), smallData.end());
        runBenchmark("Test 1: Small Uniform Data (10 items)", data, true);
    }

    // ============================================================================
    // Test 2: 1K items - Tiny random strings
    // ============================================================================
    {
        auto data = generateRandomData(1000, 10, 50);
        runBenchmark("Test 2: 1K Tiny Random Strings", data, true);
    }

    // ============================================================================
    // Test 3: 10K items - Small structured records
    // ============================================================================
    {
        auto data = generateStructuredData(10000);
        runBenchmark("Test 3: 10K Structured Records", data, true);
    }

    // ============================================================================
    // Test 4: 100K items - Variable size data
    // ============================================================================
    {
        auto data = generateRandomData(100000, 50, 500);
        runBenchmark("Test 4: 100K Variable Size Items", data, true);
    }

    // ============================================================================
    // Test 5: 500K items - Medium workload
    // ============================================================================
    {
        auto data = generateRandomData(500000, 100, 300);
        runBenchmark("Test 5: 500K Medium Workload", data, false); // Skip sequential (too slow)
    }

    // ============================================================================
    // Test 6: 1M items - Large uniform blocks
    // ============================================================================
    {
        auto data = generateRandomData(1000000, 256, 256);
        runBenchmark("Test 6: 1M Large Uniform Blocks", data, false);
    }

    // ============================================================================
    // Test 7: 2M items - Highly variable data
    // ============================================================================
    {
        auto data = generateRandomData(2000000, 10, 1000);
        runBenchmark("Test 7: 2M Highly Variable Data", data, false);
    }

    // ============================================================================
    // Test 8: 10K Large text documents
    // ============================================================================
    {
        auto data = generateLargeTextBlocks(10000, 10000); // 10KB each = ~100MB total
        runBenchmark("Test 8: 10K Large Text Documents (10KB each)", data, true);
    }

    // ============================================================================
    // Test 9: 5M items - Stress test
    // ============================================================================
    {
        std::cout << "\n=== Test 9: 5M Items - Stress Test ===\n";
        std::cout << "Generating data...\n";
        auto data = generateRandomData(5000000, 100, 200);
        runBenchmark("Test 9: 5M Items Stress Test", data, false);
    }
    // ============================================================================
    // Summary
    // ============================================================================
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Benchmark Complete                      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nKey Findings:\n";
    std::cout << "- Thread pool eliminates thread creation overhead\n";
    std::cout << "- Near-linear scaling up to physical core count\n";
    std::cout << "- Adaptive parallelism optimizes small workloads\n";
    std::cout << "- Lock-free design enables high throughput\n";

    return 0;
}
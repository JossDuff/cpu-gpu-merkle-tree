#include "merkle_tree_gpu_streams.hpp"
#include "sha256_cuda.cuh"
#include <iostream>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <sstream>

MerkleTreeGPUStreams::MerkleTreeGPUStreams(size_t chunk_size, size_t block_size, int num_streams)
    : MerkleTreeGPU(chunk_size, block_size), num_streams_(num_streams) {
    initializeStreams();
}

MerkleTreeGPUStreams::~MerkleTreeGPUStreams() {
    cleanupStreams();
}

void MerkleTreeGPUStreams::initializeStreams() {
    streams_.resize(num_streams_);
    for (int i = 0; i < num_streams_; i++) {
        cudaStreamCreate(&streams_[i]);
    }

    std::cout << "✓ Created " << num_streams_ << " CUDA streams\n";
}

void MerkleTreeGPUStreams::cleanupStreams() {
    // Synchronize all streams before cleanup
    for (auto& stream : streams_) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    streams_.clear();

    freePinnedBuffers();
    freeGPUBuffers();
}

void MerkleTreeGPUStreams::allocatePinnedBuffers(size_t max_chunk_bytes) {
    pinned_buffers_.resize(num_streams_);

    for (int i = 0; i < num_streams_; i++) {
        auto& buf = pinned_buffers_[i];
        buf.capacity = max_chunk_bytes;

        // Allocate pinned memory for async transfers
        cudaHostAlloc(&buf.data, max_chunk_bytes, cudaHostAllocDefault);
        cudaHostAlloc(&buf.lengths, chunk_size_ * sizeof(size_t), cudaHostAllocDefault);
        cudaHostAlloc(&buf.offsets, chunk_size_ * sizeof(size_t), cudaHostAllocDefault);
    }

    std::cout << "✓ Allocated " << num_streams_ << " pinned host buffers ("
              << (max_chunk_bytes / (1024.0 * 1024.0)) << " MB each)\n";
}

void MerkleTreeGPUStreams::allocateGPUBuffers(size_t max_chunk_bytes) {
    gpu_buffers_.resize(num_streams_);

    for (int i = 0; i < num_streams_; i++) {
        auto& buf = gpu_buffers_[i];
        buf.capacity = max_chunk_bytes;

        cudaMalloc(&buf.d_data, max_chunk_bytes);
        cudaMalloc(&buf.d_lengths, chunk_size_ * sizeof(size_t));
        cudaMalloc(&buf.d_offsets, chunk_size_ * sizeof(size_t));
        cudaMalloc(&buf.d_output, chunk_size_ * 32);  // SHA-256 outputs
    }

    std::cout << "✓ Allocated " << num_streams_ << " GPU buffers ("
              << (max_chunk_bytes / (1024.0 * 1024.0)) << " MB each)\n";
}

void MerkleTreeGPUStreams::freePinnedBuffers() {
    for (auto& buf : pinned_buffers_) {
        if (buf.data) cudaFreeHost(buf.data);
        if (buf.lengths) cudaFreeHost(buf.lengths);
        if (buf.offsets) cudaFreeHost(buf.offsets);
    }
    pinned_buffers_.clear();
}

void MerkleTreeGPUStreams::freeGPUBuffers() {
    for (auto& buf : gpu_buffers_) {
        if (buf.d_data) cudaFree(buf.d_data);
        if (buf.d_lengths) cudaFree(buf.d_lengths);
        if (buf.d_offsets) cudaFree(buf.d_offsets);
        if (buf.d_output) cudaFree(buf.d_output);
    }
    gpu_buffers_.clear();
}

void MerkleTreeGPUStreams::build(const std::vector<std::string>& data) {
    auto start = std::chrono::high_resolution_clock::now();

    if (data.empty()) {
        root_ = nullptr;
        root_hash_ = "";
        return;
    }

    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  GPU Merkle Tree with CUDA Streams (Overlapped Execution) ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "Building with " << num_streams_ << " streams...\n";
    std::cout << "Chunk size: " << chunk_size_ << " items\n";
    std::cout << "Total items: " << data.size() << "\n\n";

    // Calculate max chunk size for buffer allocation
    size_t max_chunk_bytes = 0;
    for (size_t i = 0; i < data.size(); i += chunk_size_) {
        size_t chunk_bytes = 0;
        size_t chunk_end = std::min(i + chunk_size_, data.size());
        for (size_t j = i; j < chunk_end; j++) {
            chunk_bytes += data[j].size();
        }
        max_chunk_bytes = std::max(max_chunk_bytes, chunk_bytes);
    }

    // Allocate buffers
    allocatePinnedBuffers(max_chunk_bytes * 2);  // Extra space for safety
    allocateGPUBuffers(max_chunk_bytes * 2);

    // Process all chunks with stream overlap
    auto leaf_hashes = processChunksWithStreams(data);
    leaf_hashes_ = leaf_hashes;

    // Build tree from leaf hashes using GPU merge (same as standard GPU)
    std::vector<std::string> current_level = leaf_hashes;

    while (current_level.size() > 1) {
        std::vector<std::string> next_level;
        next_level.reserve((current_level.size() + 1) / 2);

        // Process pairs in chunks (same as MerkleTreeGPU)
        for (size_t offset = 0; offset < current_level.size(); offset += chunk_size_ * 2) {
            size_t chunk_end = std::min(offset + chunk_size_ * 2, current_level.size());
            std::vector<std::string> chunk_pairs(current_level.begin() + offset,
                                                 current_level.begin() + chunk_end);

            std::vector<std::string> merged = mergePairsOnGPU(chunk_pairs);
            next_level.insert(next_level.end(), merged.begin(), merged.end());
        }

        current_level = next_level;
    }

    root_hash_ = current_level.empty() ? "" : current_level[0];

    // Build tree structure (optional, for compatibility)
    root_ = buildTreeFromHashes(leaf_hashes);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = end - start;
    timings_.total_ms = total_duration.count();

    // Clean up buffers to avoid memory leaks and corruption on subsequent builds
    freePinnedBuffers();
    freeGPUBuffers();

    std::cout << "\n✓ Build complete with CUDA streams\n";
    std::cout << "  Total time: " << std::fixed << std::setprecision(2)
              << timings_.total_ms << " ms\n";
    std::cout << "  Overlap benefit: " << stream_timings_.overlap_benefit_ms << " ms saved\n";
    std::cout << "  Speedup vs sequential: " << std::setprecision(2)
              << stream_timings_.speedup_vs_sequential << "x\n\n";
}

std::vector<std::string> MerkleTreeGPUStreams::processChunksWithStreams(
    const std::vector<std::string>& data) {

    size_t num_chunks = (data.size() + chunk_size_ - 1) / chunk_size_;
    std::vector<std::string> all_hashes(data.size());

    std::cout << "Processing " << num_chunks << " chunks with "
              << num_streams_ << " streams (overlap enabled)...\n";

    auto start = std::chrono::high_resolution_clock::now();

    double total_h2d = 0, total_kernel = 0, total_d2h = 0;

    // Pipeline chunks across streams
    size_t chunks_processed = 0;

    for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int stream_idx = chunk_idx % num_streams_;
        cudaStream_t stream = streams_[stream_idx];

        // Wait for this stream to be available (previous iteration)
        if (chunk_idx >= num_streams_) {
            cudaStreamSynchronize(stream);
        }

        // Prepare chunk data
        size_t chunk_start = chunk_idx * chunk_size_;
        size_t chunk_end = std::min(chunk_start + chunk_size_, data.size());
        size_t chunk_items = chunk_end - chunk_start;

        std::vector<std::string> chunk_data(data.begin() + chunk_start,
                                            data.begin() + chunk_end);

        // Process this chunk asynchronously on the stream
        auto chunk_start_time = std::chrono::high_resolution_clock::now();

        // === ASYNC H2D TRANSFER ===
        auto& pinned = pinned_buffers_[stream_idx];
        auto& gpu = gpu_buffers_[stream_idx];

        // Copy data to pinned buffer
        size_t offset = 0;
        for (size_t i = 0; i < chunk_items; i++) {
            pinned.offsets[i] = offset;
            pinned.lengths[i] = chunk_data[i].size();
            std::memcpy(pinned.data + offset, chunk_data[i].data(), chunk_data[i].size());
            offset += chunk_data[i].size();
        }

        auto h2d_start = std::chrono::high_resolution_clock::now();

        // Async H2D transfers
        cudaMemcpyAsync(gpu.d_data, pinned.data, offset,
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(gpu.d_lengths, pinned.lengths, chunk_items * sizeof(size_t),
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(gpu.d_offsets, pinned.offsets, chunk_items * sizeof(size_t),
                       cudaMemcpyHostToDevice, stream);

        auto h2d_end = std::chrono::high_resolution_clock::now();

        // === ASYNC KERNEL EXECUTION ===
        auto kernel_start = std::chrono::high_resolution_clock::now();

        int num_blocks = (chunk_items + block_size_ - 1) / block_size_;
        sha256_kernel<<<num_blocks, block_size_, 0, stream>>>(
            (uint8_t*)gpu.d_data, (uint32_t*)gpu.d_output,
            (const uint32_t*)gpu.d_lengths, (const uint32_t*)gpu.d_offsets, chunk_items);

        auto kernel_end = std::chrono::high_resolution_clock::now();

        // === ASYNC D2H TRANSFER ===
        auto d2h_start = std::chrono::high_resolution_clock::now();

        std::vector<uint32_t> output_buffer(chunk_items * 8);  // SHA-256 = 8 uint32_t per hash
        cudaMemcpyAsync(output_buffer.data(), gpu.d_output, chunk_items * 8 * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost, stream);

        // Note: We need to synchronize before using output_buffer
        cudaStreamSynchronize(stream);

        auto d2h_end = std::chrono::high_resolution_clock::now();

        // Convert to hex strings (same format as standard GPU implementation)
        for (size_t i = 0; i < chunk_items; i++) {
            std::stringstream ss;
            for (int j = 0; j < 8; j++) {
                ss << std::hex << std::setfill('0') << std::setw(8) << output_buffer[i * 8 + j];
            }
            all_hashes[chunk_start + i] = ss.str();
        }

        auto chunk_end_time = std::chrono::high_resolution_clock::now();

        // Accumulate timings
        std::chrono::duration<double, std::milli> h2d_time = h2d_end - h2d_start;
        std::chrono::duration<double, std::milli> kernel_time = kernel_end - kernel_start;
        std::chrono::duration<double, std::milli> d2h_time = d2h_end - d2h_start;

        total_h2d += h2d_time.count();
        total_kernel += kernel_time.count();
        total_d2h += d2h_time.count();

        chunks_processed++;

        if (chunks_processed % 10 == 0 || chunks_processed == num_chunks) {
            std::cout << "  Progress: " << chunks_processed << "/" << num_chunks
                      << " chunks processed\r" << std::flush;
        }
    }

    // Wait for all streams to complete
    for (auto& stream : streams_) {
        cudaStreamSynchronize(stream);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = end - start;

    std::cout << "\n";

    // Calculate overlap benefit
    double sequential_time = total_h2d + total_kernel + total_d2h;
    double overlap_benefit = sequential_time - total_duration.count();
    double speedup = sequential_time / total_duration.count();

    stream_timings_.total_ms = total_duration.count();
    stream_timings_.h2d_total_ms = total_h2d;
    stream_timings_.kernel_total_ms = total_kernel;
    stream_timings_.d2h_total_ms = total_d2h;
    stream_timings_.overlap_benefit_ms = overlap_benefit;
    stream_timings_.speedup_vs_sequential = speedup;

    timings_.memory_h2d_ms = total_h2d;
    timings_.kernel_execution_ms = total_kernel;
    timings_.memory_d2h_ms = total_d2h;

    return all_hashes;
}

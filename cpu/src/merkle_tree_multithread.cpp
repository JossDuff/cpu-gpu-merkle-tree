#include "header_merkle_tree.hpp"
#include <openssl/evp.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <future>

// ============================================================================
// Node Implementation
// ============================================================================

Node::Node(std::string value)
    : hash(value), left(nullptr), right(nullptr)
{
}

// ============================================================================
// MerkleTreeLockFree Implementation (with Thread Pool)
// ============================================================================

MerkleTreeLockFree::MerkleTreeLockFree(size_t num_threads)
    : root(nullptr),
      thread_count_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads)
{
    if (thread_count_ == 0)
    {
        thread_count_ = 1;
    }

    // Create thread pool once (reused for all operations)
    thread_pool_ = std::make_unique<ThreadPool>(thread_count_);
}

MerkleTreeLockFree::~MerkleTreeLockFree()
{
    deleteTree(root);
    // thread_pool_ automatically destroyed by unique_ptr
}

std::array<unsigned char, 32> MerkleTreeLockFree::hash256(std::vector<std::byte> data)
{
    std::array<unsigned char, EVP_MAX_MD_SIZE> hash;
    unsigned int hash_len;

    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mdctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(mdctx, data.data(), data.size());
    EVP_DigestFinal_ex(mdctx, hash.data(), &hash_len);
    EVP_MD_CTX_free(mdctx);

    std::array<unsigned char, 32> result;
    std::copy(hash.begin(), hash.begin() + 32, result.begin());
    return result;
}

std::string MerkleTreeLockFree::hashToString(const std::array<unsigned char, 32> &hash)
{
    std::stringstream ss;
    for (int i = 0; i < 32; i++)
    {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

Node *MerkleTreeLockFree::getRoot() const
{
    return root;
}

const std::string &MerkleTreeLockFree::getRootHash() const
{
    return root_hash;
}

void MerkleTreeLockFree::constructor(std::vector<std::byte> data, size_t chunkSize)
{
    if (data.empty())
        return;

    // Split data into chunks with pre-allocation
    std::vector<std::vector<std::byte>> chunks;
    chunks.reserve(data.size() / chunkSize + 1);

    for (size_t i = 0; i < data.size(); i += chunkSize)
    {
        size_t end = std::min(i + chunkSize, data.size());
        chunks.emplace_back(data.begin() + i, data.begin() + end);
    }

    // Build tree using thread pool
    buildTreeWithPool(chunks);
}

// Sequential fallback for small workloads
void MerkleTreeLockFree::hashLeavesSequential(const std::vector<std::vector<std::byte>> &chunks)
{
    for (size_t i = 0; i < chunks.size(); ++i)
    {
        auto hash = hash256(chunks[i]);
        leaf_hashes_[i] = hashToString(hash);
    }
}

void MerkleTreeLockFree::buildLayerSequential(const std::vector<std::string> &current_layer,
                                              std::vector<std::string> &next_layer)
{
    for (size_t i = 0; i < current_layer.size(); i += 2)
    {
        size_t pair_idx = i / 2;
        std::string left_hash = current_layer[i];
        std::string right_hash;

        if (i + 1 < current_layer.size())
        {
            right_hash = current_layer[i + 1];
        }
        else
        {
            right_hash = left_hash;
        }

        // Concatenate the hash STRINGS (hex representation)
        std::string combined = left_hash + right_hash;

        // Convert the combined hex string to bytes
        std::vector<std::byte> combinedBytes;
        combinedBytes.reserve(combined.size());

        for (char c : combined)
        {
            combinedBytes.push_back(static_cast<std::byte>(c));
        }

        // Hash the combined bytes
        auto parentHash = hash256(combinedBytes);
        next_layer[pair_idx] = hashToString(parentHash);
    }
}

// Optimized tree building with thread pool and adaptive parallelism
void MerkleTreeLockFree::buildTreeWithPool(const std::vector<std::vector<std::byte>> &chunks)
{
    if (chunks.empty())
        return;

    leaf_hashes_.resize(chunks.size());

    // Step 1: Hash all leaf nodes
    // Determine if parallelization is beneficial
    size_t work_per_thread = chunks.size() / thread_count_;
    bool use_parallel_leaves = (work_per_thread >= MIN_WORK_PER_THREAD) && (thread_count_ > 1);

    if (use_parallel_leaves)
    {
        // Parallel leaf hashing using thread pool
        std::vector<std::future<void>> futures;
        size_t chunk_size = (chunks.size() + thread_count_ - 1) / thread_count_;

        for (size_t t = 0; t < thread_count_; ++t)
        {
            futures.push_back(thread_pool_->submit([this, &chunks, t, chunk_size]()
                                                   {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, chunks.size());

                for (size_t i = start; i < end; ++i)
                {
                    auto hash = this->hash256(chunks[i]);
                    this->leaf_hashes_[i] = this->hashToString(hash);
                } }));
        }

        // Wait for all leaf hashing to complete
        for (auto &future : futures)
        {
            future.get();
        }
    }
    else
    {
        // Sequential for small workloads (avoid thread overhead)
        hashLeavesSequential(chunks);
    }

    // Step 2: Build tree layer by layer
    std::vector<std::string> current_layer = leaf_hashes_;

    while (current_layer.size() > 1)
    {
        size_t next_size = (current_layer.size() + 1) / 2;
        std::vector<std::string> next_layer(next_size);

        // Adaptive parallelism: only parallelize if beneficial
        size_t pairs = current_layer.size() / 2;
        size_t pairs_per_thread = pairs / thread_count_;
        bool use_parallel_layer = (pairs_per_thread >= MIN_WORK_PER_THREAD / 4) && (thread_count_ > 1);

        if (use_parallel_layer)
        {
            // Parallel layer construction using thread pool
            std::vector<std::future<void>> futures;
            size_t pairs_per_chunk = (pairs + thread_count_ - 1) / thread_count_;

            for (size_t t = 0; t < thread_count_; ++t)
            {
                futures.push_back(thread_pool_->submit([this, &current_layer, &next_layer, t, pairs_per_chunk, pairs]()
                                                       {
                    size_t start_pair = t * pairs_per_chunk;
                    size_t end_pair = std::min(start_pair + pairs_per_chunk, pairs);

                    for (size_t pair = start_pair; pair < end_pair; ++pair)
                    {
                        size_t i = pair * 2;
                        std::string left_hash = current_layer[i];
                        std::string right_hash = current_layer[i + 1];

                        // Concatenate the hash STRINGS (hex representation)
                        std::string combined = left_hash + right_hash;

                        // Convert the combined hex string to bytes
                        std::vector<std::byte> combinedBytes;
                        combinedBytes.reserve(combined.size());
                        
                        for (char c : combined)
                        {
                            combinedBytes.push_back(static_cast<std::byte>(c));
                        }

                        // Hash the combined bytes
                        auto parentHash = this->hash256(combinedBytes);
                        next_layer[pair] = this->hashToString(parentHash);
                    } }));
            }

            // Handle odd node if present (in main thread to avoid race)
            if (current_layer.size() % 2 == 1)
            {
                next_layer.back() = current_layer.back();
            }

            // Wait for all tasks to complete
            for (auto &future : futures)
            {
                future.get();
            }
        }
        else
        {
            // Sequential for small layers (avoid thread pool overhead)
            buildLayerSequential(current_layer, next_layer);
        }

        current_layer = std::move(next_layer);
    }

    root_hash = current_layer[0];

    // Build Node tree structure from flat hashes
    root = constructNodeTree();
}

Node *MerkleTreeLockFree::constructNodeTree()
{
    if (leaf_hashes_.empty())
        return nullptr;

    // Build tree structure from bottom up
    std::vector<Node *> current_nodes;
    current_nodes.reserve(leaf_hashes_.size());

    // Create leaf nodes
    for (const auto &hash : leaf_hashes_)
    {
        current_nodes.push_back(new Node(hash));
    }

    // Build parent layers
    while (current_nodes.size() > 1)
    {
        std::vector<Node *> next_nodes;
        next_nodes.reserve((current_nodes.size() + 1) / 2);

        for (size_t i = 0; i < current_nodes.size(); i += 2)
        {
            Node *left = current_nodes[i];
            Node *right = nullptr;

            if (i + 1 < current_nodes.size())
            {
                right = current_nodes[i + 1];
            }
            else
            {
                right = new Node(left->hash);
            }

            // Concatenate and hash
            std::string combined = left->hash + right->hash;
            std::vector<std::byte> combinedBytes;
            combinedBytes.reserve(combined.size());

            for (char c : combined)
            {
                combinedBytes.push_back(static_cast<std::byte>(c));
            }

            auto parentHash = hash256(combinedBytes);
            Node *parent = new Node(hashToString(parentHash));
            parent->left = left;
            parent->right = right;
            next_nodes.push_back(parent);
        }

        current_nodes = std::move(next_nodes);
    }

    return current_nodes[0];
}

void MerkleTreeLockFree::deleteTree(Node *node)
{
    if (node)
    {
        deleteTree(node->left);
        deleteTree(node->right);
        delete node;
    }
}

// ============================================================================
// MerkleTreeSequential Implementation (Unchanged)
// ============================================================================

MerkleTreeSequential::MerkleTreeSequential()
    : root(nullptr)
{
}

MerkleTreeSequential::~MerkleTreeSequential()
{
    deleteTree(root);
}

std::array<unsigned char, 32> MerkleTreeSequential::hash256(std::vector<std::byte> data)
{
    std::array<unsigned char, EVP_MAX_MD_SIZE> hash;
    unsigned int hash_len;

    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mdctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(mdctx, data.data(), data.size());
    EVP_DigestFinal_ex(mdctx, hash.data(), &hash_len);
    EVP_MD_CTX_free(mdctx);

    std::array<unsigned char, 32> result;
    std::copy(hash.begin(), hash.begin() + 32, result.begin());
    return result;
}

std::string MerkleTreeSequential::hashToString(const std::array<unsigned char, 32> &hash)
{
    std::stringstream ss;
    for (int i = 0; i < 32; i++)
    {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

Node *MerkleTreeSequential::getRoot() const
{
    return root;
}

void MerkleTreeSequential::constructor(std::vector<std::byte> data, size_t chunkSize)
{
    if (data.empty())
        return;

    std::vector<Node *> leaves;

    for (size_t i = 0; i < data.size(); i += chunkSize)
    {
        size_t end = std::min(i + chunkSize, data.size());
        std::vector<std::byte> chunk(data.begin() + i, data.begin() + end);

        auto hash = hash256(chunk);
        leaves.push_back(new Node(hashToString(hash)));
    }

    root = buildTree(leaves);
}

Node *MerkleTreeSequential::buildTree(std::vector<Node *> &nodes)
{
    if (nodes.empty())
        return nullptr;

    if (nodes.size() == 1)
        return nodes[0];

    std::vector<Node *> parents;

    for (size_t i = 0; i < nodes.size(); i += 2)
    {
        Node *left = nodes[i];
        Node *right = nullptr;

        if (i + 1 < nodes.size())
        {
            right = nodes[i + 1];
        }
        else
        {
            right = new Node(left->hash);
        }

        std::string combined = left->hash + right->hash;
        std::vector<std::byte> combinedBytes;
        for (char c : combined)
        {
            combinedBytes.push_back(static_cast<std::byte>(c));
        }

        auto parentHash = hash256(combinedBytes);
        Node *parent = new Node(hashToString(parentHash));
        parent->left = left;
        parent->right = right;
        parents.push_back(parent);
    }

    return buildTree(parents);
}

void MerkleTreeSequential::deleteTree(Node *node)
{
    if (node)
    {
        deleteTree(node->left);
        deleteTree(node->right);
        delete node;
    }
}
#ifndef HEADER_MERKLE_TREE_HPP
#define HEADER_MERKLE_TREE_HPP

#include <string>
#include <vector>
#include <cstddef>
#include <array>
#include <memory>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <stdexcept>
#include <type_traits>

// ============================================================================
// Thread Pool Implementation (embedded)
// ============================================================================

class ThreadPool
{
public:
    // Constructor: creates a thread pool with specified number of threads
    // If num_threads == 0, uses hardware_concurrency()
    explicit ThreadPool(size_t num_threads = 0)
        : stop_(false), active_tasks_(0)
    {
        if (num_threads == 0)
        {
            num_threads = std::thread::hardware_concurrency();
        }
        if (num_threads == 0)
        {
            num_threads = 1;
        }

        workers_.reserve(num_threads);

        for (size_t i = 0; i < num_threads; ++i)
        {
            workers_.emplace_back([this]
                                  {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock, [this] { 
                            return this->stop_ || !this->tasks_.empty(); 
                        });
                        
                        if (this->stop_ && this->tasks_.empty()) {
                            return;
                        }
                        
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    
                    active_tasks_++;
                    task();
                    active_tasks_--;
                    wait_condition_.notify_all();
                } });
        }
    }

    // Destructor: stops all threads and waits for completion
    ~ThreadPool()
    {
        stop_ = true;
        condition_.notify_all();

        for (std::thread &worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    // Delete copy and move constructors/assignment operators
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
    ThreadPool(ThreadPool &&) = delete;
    ThreadPool &operator=(ThreadPool &&) = delete;

    // Submit a task and get a future for the result
    // FIXED: Use std::invoke_result instead of std::result_of (C++17/20 compatible)
    template <typename F, typename... Args>
    auto submit(F &&f, Args &&...args)
        -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> result = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            if (stop_)
            {
                throw std::runtime_error("Cannot submit task to stopped ThreadPool");
            }

            tasks_.emplace([task]()
                           { (*task)(); });
        }

        condition_.notify_one();
        return result;
    }

    // Wait for all currently queued tasks to complete
    void wait()
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        wait_condition_.wait(lock, [this]
                             { return tasks_.empty() && active_tasks_ == 0; });
    }

    // Get number of worker threads in the pool
    size_t size() const { return workers_.size(); }

    // Check if pool has been stopped
    bool stopped() const { return stop_.load(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable wait_condition_;

    std::atomic<bool> stop_;
    std::atomic<size_t> active_tasks_;
};

// ============================================================================
// Merkle Tree Node
// ============================================================================

class Node
{
public:
    std::string hash;
    Node *left;
    Node *right;

    Node(std::string value);
};

// ============================================================================
// Lock-Free Merkle Tree with Thread Pool
// ============================================================================

class MerkleTreeLockFree
{
public:
    // Constructor with optional thread count
    explicit MerkleTreeLockFree(size_t num_threads = 0);

    // Destructor
    ~MerkleTreeLockFree();

    // Hash function using SHA-256
    std::array<unsigned char, 32> hash256(std::vector<std::byte> data);

    // Template function to transform string array to byte vector
    // Must be in header because it's a template
    template <typename T>
    std::vector<std::byte> transformStringArray(const T &strings)
    {
        std::vector<std::byte> result;
        result.reserve(strings.size() * 32); // Pre-allocate for efficiency

        for (size_t i = 0; i < strings.size(); i++)
        {
            const char *c = strings[i].c_str();
            size_t len = strings[i].length();

            for (size_t j = 0; j < len; j++)
            {
                result.push_back(static_cast<std::byte>(c[j]));
            }
        }

        return result;
    }

    // Build the Merkle tree from data
    void constructor(std::vector<std::byte> data, size_t chunkSize = 32);

    // Get root node
    Node *getRoot() const;

    // Get root hash as string
    const std::string &getRootHash() const;

    // High-level build interface for compatibility
    void build(const std::vector<std::string>& data, size_t chunkSize = 32);

    // Get leaf hashes
    const std::vector<std::string>& getLeafHashes() const;

private:
    Node *root;
    std::string root_hash;
    size_t thread_count_;

    // Thread pool for efficient parallel execution
    std::unique_ptr<ThreadPool> thread_pool_;

    // Flat storage for lock-free construction
    std::vector<std::string> leaf_hashes_;

    // Minimum work per thread to justify parallelization overhead
    static constexpr size_t MIN_WORK_PER_THREAD = 128;

    // Private helper methods
    std::string hashToString(const std::array<unsigned char, 32> &hash);
    std::array<unsigned char, 32> hexStringToBytes(const std::string& hex);
    void buildTreeWithPool(const std::vector<std::vector<std::byte>> &chunks);
    Node *constructNodeTree();
    void deleteTree(Node *node);

    // Sequential fallback methods for small workloads
    void hashLeavesSequential(const std::vector<std::vector<std::byte>> &chunks);
    void buildLayerSequential(const std::vector<std::string> &current_layer,
                              std::vector<std::string> &next_layer);
};

// ============================================================================
// Sequential Merkle Tree (for comparison)
// ============================================================================

class MerkleTreeSequential
{
public:
    // Constructor
    MerkleTreeSequential();

    // Destructor
    ~MerkleTreeSequential();

    // Hash function using SHA-256
    std::array<unsigned char, 32> hash256(std::vector<std::byte> data);

    // Template function to transform string array to byte vector
    // Must be in header because it's a template
    template <typename T>
    std::vector<std::byte> transformStringArray(const T &strings)
    {
        std::vector<std::byte> result;
        result.reserve(strings.size() * 32);

        for (size_t i = 0; i < strings.size(); i++)
        {
            const char *c = strings[i].c_str();
            size_t len = strings[i].length();

            for (size_t j = 0; j < len; j++)
            {
                result.push_back(static_cast<std::byte>(c[j]));
            }
        }

        return result;
    }

    // Build the Merkle tree from data
    void constructor(std::vector<std::byte> data, size_t chunkSize = 32);

    // Get root node
    Node *getRoot() const;

    // High-level build interface for compatibility
    void build(const std::vector<std::string>& data, size_t chunkSize = 32);

    // Get leaf hashes
    const std::vector<std::string>& getLeafHashes() const;

private:
    Node *root;
    std::vector<std::string> leaf_hashes_;

    // Private helper methods
    std::string hashToString(const std::array<unsigned char, 32> &hash);
    std::array<unsigned char, 32> hexStringToBytes(const std::string& hex);
    Node *buildTree(std::vector<Node *> &nodes);
    void deleteTree(Node *node);
};

#endif // HEADER_MERKLE_TREE_HPP
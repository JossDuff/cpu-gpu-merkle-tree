#include <iostream>
#include <string>
#include <vector>
#include <cstddef>
#include <array>
#include <openssl/evp.h>
#include <sstream>
#include <iomanip>
#include "header_merkle_tree.hpp"

class Node
{
public:
    std::string hash;
    Node *left;
    Node *right;

    Node(std::string value) : hash(value),
                              left(nullptr),
                              right(nullptr) {};
};

class MerkleTreeSequential
{
public:
    auto hash256(std::vector<std::byte> data)
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

    template <typename T>
    std::vector<std::byte> transformStringArray(const T &strings)
    {
        std::vector<std::byte> result;

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

    Node *getRoot() const { return root; }

    MerkleTreeSequential() : root(nullptr) {}

    void constructor(std::vector<std::byte> data, size_t chunkSize = 32)
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

    ~MerkleTreeSequential()
    {
        deleteTree(root);
    }

private:
    Node *root;

    std::string hashToString(const std::array<unsigned char, 32> &hash)
    {
        std::stringstream ss;
        for (int i = 0; i < 32; i++)
        {
            ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
        }
        return ss.str();
    }

    Node *buildTree(std::vector<Node *> &nodes)
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

            // Concatenate the hash STRINGS (hex representation)
            std::string combined = left->hash + right->hash;

            // Convert the combined hex string to bytes
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

    void deleteTree(Node *node)
    {
        if (node)
        {
            deleteTree(node->left);
            deleteTree(node->right);
            delete node;
        }
    }
};

int main()
{
    std::array<std::string, 10> RowData = {"Orange", "Apple", "Kiwi", "Strawberry", "Potato", "Road", "A big big big big big big big big big big word", ".", "*£$àze()", "GG"};

    MerkleTreeSequential merkletree;

    auto byteData = merkletree.transformStringArray(RowData);
    merkletree.constructor(byteData);

    std::vector<std::byte> testData;
    for (char c : RowData[0])
    {
        testData.push_back(static_cast<std::byte>(c));
    }
    auto test = merkletree.hash256(testData);

    std::cout << "Hash of '" << RowData[0] << "': ";
    for (auto byte : test)
    {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
    }
    std::cout << std::endl;

    if (merkletree.getRoot())
    {
        std::cout << "Merkle root: " << merkletree.getRoot()->hash << std::endl;
    }

    return 0;
}
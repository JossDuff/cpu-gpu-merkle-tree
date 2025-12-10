#ifndef SHA256_CUDA_CUH
#define SHA256_CUDA_CUH

#include <cuda_runtime.h>
#include <cstdint>

// SHA-256 constants
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 helper functions
__device__ inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ inline uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ inline uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ inline uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ inline uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// SHA-256 kernel for hashing data chunks
static __global__ void sha256_kernel(const uint8_t* input, uint32_t* output,
                                     const uint32_t* lengths, const uint32_t* offsets,
                                     uint32_t num_items) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_items) return;

    // Get precomputed offset
    uint32_t offset = offsets[idx];
    const uint8_t* data = input + offset;
    uint32_t len = lengths[idx];

    // Initialize hash values (first 32 bits of fractional parts of square roots of first 8 primes)
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Prepare message schedule array
    uint32_t w[64];

    // Process message in 512-bit chunks
    uint64_t bit_len = (uint64_t)len * 8;

    // Add padding
    uint32_t padded_len = len + 1; // +1 for the '1' bit
    while ((padded_len % 64) != 56) {
        padded_len++;
    }
    padded_len += 8; // +8 for the length field

    // Process each 512-bit chunk
    for (uint32_t chunk_start = 0; chunk_start < padded_len; chunk_start += 64) {
        // Prepare the message schedule
        for (int i = 0; i < 16; i++) {
            uint32_t byte_idx = chunk_start + i * 4;
            w[i] = 0;

            for (int j = 0; j < 4; j++) {
                if (byte_idx + j < len) {
                    w[i] |= ((uint32_t)data[byte_idx + j]) << (24 - j * 8);
                } else if (byte_idx + j == len) {
                    w[i] |= 0x80000000 >> (j * 8); // Append '1' bit
                } else if (byte_idx + j >= padded_len - 8 && byte_idx + j < padded_len - 4) {
                    // High 32 bits of length (always 0 for our use case)
                    w[i] |= 0;
                } else if (byte_idx + j >= padded_len - 4) {
                    // Low 32 bits of length
                    int shift = (3 - (byte_idx + j - (padded_len - 4))) * 8;
                    w[i] |= ((bit_len >> shift) & 0xFF) << (24 - j * 8);
                }
            }
        }

        // Extend the first 16 words into the remaining 48 words
        for (int i = 16; i < 64; i++) {
            w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
        }

        // Initialize working variables
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

        // Main loop
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + sigma1(e) + ch(e, f, g) + K[i] + w[i];
            uint32_t t2 = sigma0(a) + maj(a, b, c);
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        // Add compressed chunk to current hash value
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    // Store result
    for (int i = 0; i < 8; i++) {
        output[idx * 8 + i] = h[i];
    }
}

// Kernel for merging hashes in Merkle tree (hash of concatenated hashes)
static __global__ void merkle_merge_kernel(const uint32_t* left_hashes, const uint32_t* right_hashes,
                                           uint32_t* output, uint32_t num_pairs) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_pairs) return;

    // Concatenate left and right hashes (8 uint32_t each = 32 bytes)
    uint8_t combined[64];

    // Copy left hash
    for (int i = 0; i < 8; i++) {
        uint32_t val = left_hashes[idx * 8 + i];
        combined[i * 4 + 0] = (val >> 24) & 0xFF;
        combined[i * 4 + 1] = (val >> 16) & 0xFF;
        combined[i * 4 + 2] = (val >> 8) & 0xFF;
        combined[i * 4 + 3] = val & 0xFF;
    }

    // Copy right hash
    for (int i = 0; i < 8; i++) {
        uint32_t val = right_hashes[idx * 8 + i];
        combined[32 + i * 4 + 0] = (val >> 24) & 0xFF;
        combined[32 + i * 4 + 1] = (val >> 16) & 0xFF;
        combined[32 + i * 4 + 2] = (val >> 8) & 0xFF;
        combined[32 + i * 4 + 3] = val & 0xFF;
    }

    // Hash the combined data (64 bytes needs 2 blocks with padding)
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint32_t w[64];

    // ========== BLOCK 1: Process the 64-byte input (512 bits) ==========
    // First 16 words from the 64-byte input
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)combined[i * 4] << 24) |
               ((uint32_t)combined[i * 4 + 1] << 16) |
               ((uint32_t)combined[i * 4 + 2] << 8) |
               ((uint32_t)combined[i * 4 + 3]);
    }

    // Extend to 64 words
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
    }

    // Initialize working variables
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

    // Main loop for block 1
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = hh + sigma1(e) + ch(e, f, g) + K[i] + w[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Add compressed chunk to current hash value
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;

    // ========== BLOCK 2: Padding block ==========
    // Padding: 0x80, then zeros, then length (512 bits = 0x200)
    // Block 2 layout: [0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000000, 0x00000200]
    w[0] = 0x80000000;  // Padding bit
    for (int i = 1; i < 15; i++) {
        w[i] = 0;
    }
    w[14] = 0x00000000;  // Length high bits (64 bytes = 512 bits = 0x0000000000000200)
    w[15] = 0x00000200;  // Length low bits

    // Extend to 64 words
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
    }

    // Reset working variables
    a = h[0]; b = h[1]; c = h[2]; d = h[3];
    e = h[4]; f = h[5]; g = h[6]; hh = h[7];

    // Main loop for block 2
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = hh + sigma1(e) + ch(e, f, g) + K[i] + w[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Final hash value
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;

    // Store result
    for (int i = 0; i < 8; i++) {
        output[idx * 8 + i] = h[i];
    }
}

// Helper function to convert hash to hex string on device
static __device__ inline void hash_to_hex(const uint32_t* hash, char* hex_output) {
    const char hex_chars[] = "0123456789abcdef";

    for (int i = 0; i < 8; i++) {
        uint32_t val = hash[i];
        for (int j = 0; j < 8; j++) {
            hex_output[i * 8 + j] = hex_chars[(val >> (28 - j * 4)) & 0xF];
        }
    }
    hex_output[64] = '\0';
}

#endif // SHA256_CUDA_CUH

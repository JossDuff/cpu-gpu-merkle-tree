# Makefile for Merkle Tree Project (CPU + GPU support)

CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 -march=native -pthread -Wall -Wextra
NVCCFLAGS = -std=c++17 -O3 -arch=sm_60 -Xcompiler -pthread
LDFLAGS =

# Check for CUDA support
CUDA_AVAILABLE := $(shell which nvcc 2>/dev/null)
ifdef CUDA_AVAILABLE
    HAS_CUDA = 1
    $(info ✓ CUDA detected: $(CUDA_AVAILABLE))

    # Detect CUDA installation path
    CUDA_PATH := $(shell dirname $(shell dirname $(CUDA_AVAILABLE)))
    CUDA_LIB_PATH := $(CUDA_PATH)/lib64

    # Check if lib64 exists, otherwise try lib
    ifeq ($(wildcard $(CUDA_LIB_PATH)),)
        CUDA_LIB_PATH := $(CUDA_PATH)/lib
    endif

    $(info CUDA library path: $(CUDA_LIB_PATH))
else
    HAS_CUDA = 0
    $(info ⚠ CUDA not found - GPU features will be simulated)
endif 

# Auto-detect OpenSSL location for pkg-config on macOS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    BREW_OPENSSL := $(shell brew --prefix openssl@3 2>/dev/null || brew --prefix openssl 2>/dev/null)
    ifneq ($(BREW_OPENSSL),)
        export PKG_CONFIG_PATH := $(BREW_OPENSSL)/lib/pkgconfig:$(PKG_CONFIG_PATH)
    endif
endif

# Use pkg-config to find OpenSSL
OPENSSL_CFLAGS := $(shell pkg-config --cflags openssl 2>/dev/null)
OPENSSL_LIBS := $(shell pkg-config --libs openssl 2>/dev/null)

# Check if pkg-config found OpenSSL
ifeq ($(OPENSSL_LIBS),)
    $(error pkg-config could not find OpenSSL. Run 'make help' for solutions)
endif

# Add OpenSSL flags
CXXFLAGS += $(OPENSSL_CFLAGS)
LDFLAGS += $(OPENSSL_LIBS)

# Directories and files
SRC_DIR = src
DEPS = $(SRC_DIR)/header_merkle_tree.hpp

# CPU-only targets
CPU_SOURCES = $(SRC_DIR)/merkle_tree_multithread.cpp $(SRC_DIR)/main.cpp
CPU_OBJECTS = $(CPU_SOURCES:.cpp=.o)
CPU_TARGET = merkle_comparison

# GPU benchmark targets
GPU_BENCH_SOURCES = $(SRC_DIR)/merkle_tree_multithread.cpp $(SRC_DIR)/main_gpu_benchmark.cpp
GPU_BENCH_OBJECTS = $(GPU_BENCH_SOURCES:.cpp=.o)
GPU_BENCH_TARGET = gpu_benchmark

# CUDA sources (if available)
ifeq ($(HAS_CUDA),1)
    # CUDA kernel files (compiled with nvcc)
    CUDA_SOURCES = $(SRC_DIR)/merkle_tree_gpu.cu $(SRC_DIR)/merkle_tree_gpu_streams.cu
    CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

    # C++ files that include CUDA headers but don't have kernel calls (compiled with g++)
    CUDA_CPP_SOURCES = $(SRC_DIR)/merkle_tree_gpu_adaptive.cpp $(SRC_DIR)/adaptive_chunk_optimizer.cpp
    CUDA_CPP_OBJECTS = $(CUDA_CPP_SOURCES:.cpp=.o)

    # Add CUDA include directory for C++ files that include CUDA headers
    CUDA_INCLUDE_PATH := $(CUDA_PATH)/include
    CXXFLAGS += -DHAS_CUDA -I$(CUDA_INCLUDE_PATH)

    # Disable relocatable device code to avoid needing cudadevrt
    NVCCFLAGS += -DHAS_CUDA -cudart shared -rdc=false
    LDFLAGS += -L$(CUDA_LIB_PATH) -lcudart -Wl,-rpath,$(CUDA_LIB_PATH)
    # NVCC linker flags (simplified to match working command)
    NVCC_LDFLAGS = -cudart shared -L$(CUDA_LIB_PATH)
endif

# Correctness test targets
CORRECTNESS_TEST_SOURCES = test_correctness.cpp $(SRC_DIR)/merkle_tree_multithread.cpp
CORRECTNESS_TEST_OBJECTS = $(CORRECTNESS_TEST_SOURCES:.cpp=.o)
CORRECTNESS_TEST_TARGET = test_correctness

# Comprehensive correctness test (all implementations)
ALL_CORRECTNESS_TEST_SOURCES = test_all_correctness.cpp $(SRC_DIR)/merkle_tree_multithread.cpp
ALL_CORRECTNESS_TEST_OBJECTS = $(ALL_CORRECTNESS_TEST_SOURCES:.cpp=.o)
ALL_CORRECTNESS_TEST_TARGET = test_all_correctness

# Demo targets
DEMO_ADAPTIVE_SOURCES = demo_adaptive.cpp $(SRC_DIR)/merkle_tree_multithread.cpp
DEMO_ADAPTIVE_OBJECTS = $(DEMO_ADAPTIVE_SOURCES:.cpp=.o)
DEMO_ADAPTIVE_TARGET = demo_adaptive

DEMO_STREAMS_SOURCES = demo_streams.cpp $(SRC_DIR)/merkle_tree_multithread.cpp
DEMO_STREAMS_OBJECTS = $(DEMO_STREAMS_SOURCES:.cpp=.o)
DEMO_STREAMS_TARGET = demo_streams

# Sweet spot benchmark
SWEET_SPOT_SOURCES = $(SRC_DIR)/sweet_spot_benchmark.cpp $(SRC_DIR)/merkle_tree_multithread.cpp
SWEET_SPOT_OBJECTS = $(SWEET_SPOT_SOURCES:.cpp=.o)
SWEET_SPOT_TARGET = sweet_spot_benchmark

# Note: demos use CUDA_OBJECTS which includes merkle_tree_gpu_streams.cu

# Default target
all: $(CPU_TARGET)

# Build everything (CPU + GPU benchmarks)
full: $(CPU_TARGET) $(GPU_BENCH_TARGET)
	@echo "✓ Full build complete!"

# Build all tests
tests: $(CORRECTNESS_TEST_TARGET) $(ALL_CORRECTNESS_TEST_TARGET)
	@echo "✓ All tests built!"

# Build original correctness test
test: $(CORRECTNESS_TEST_TARGET)
	@echo "✓ Correctness test built: ./$(CORRECTNESS_TEST_TARGET)"

# Build comprehensive correctness test (all implementations)
test-all: $(ALL_CORRECTNESS_TEST_TARGET)
	@echo "✓ Comprehensive test built: ./$(ALL_CORRECTNESS_TEST_TARGET)"

# Build all demos
demos: $(DEMO_ADAPTIVE_TARGET) $(DEMO_STREAMS_TARGET)
	@echo "✓ All demos built!"

# Build adaptive demo
demo-adaptive: $(DEMO_ADAPTIVE_TARGET)
	@echo "✓ Adaptive demo built: ./$(DEMO_ADAPTIVE_TARGET)"

# Build streams demo
demo-streams: $(DEMO_STREAMS_TARGET)
	@echo "✓ Streams demo built: ./$(DEMO_STREAMS_TARGET)"

# Build sweet spot benchmark
sweet-spot: $(SWEET_SPOT_TARGET)
	@echo "✓ Sweet spot benchmark built: ./$(SWEET_SPOT_TARGET)"

# Link correctness test executable
$(CORRECTNESS_TEST_TARGET): $(CORRECTNESS_TEST_OBJECTS) $(CUDA_OBJECTS)
	@echo "Linking $(CORRECTNESS_TEST_TARGET)..."
ifeq ($(HAS_CUDA),1)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
else
	$(CXX) $(CXXFLAGS) -o $@ $(CORRECTNESS_TEST_OBJECTS) $(LDFLAGS)
endif
	@echo "✓ Build complete: ./$(CORRECTNESS_TEST_TARGET)"

# Link CPU comparison executable
$(CPU_TARGET): $(CPU_OBJECTS)
	@echo "Linking $(CPU_TARGET)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Build complete: ./$(CPU_TARGET)"

# Link GPU benchmark executable
$(GPU_BENCH_TARGET): $(GPU_BENCH_OBJECTS) $(CUDA_OBJECTS) $(CUDA_CPP_OBJECTS)
	@echo "Linking $(GPU_BENCH_TARGET)..."
ifeq ($(HAS_CUDA),1)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
else
	$(CXX) $(CXXFLAGS) -o $@ $(GPU_BENCH_OBJECTS) $(LDFLAGS)
endif
	@echo "✓ Build complete: ./$(GPU_BENCH_TARGET)"

# Link comprehensive correctness test
$(ALL_CORRECTNESS_TEST_TARGET): $(ALL_CORRECTNESS_TEST_OBJECTS) $(CUDA_OBJECTS) $(CUDA_CPP_OBJECTS)
	@echo "Linking $(ALL_CORRECTNESS_TEST_TARGET)..."
ifeq ($(HAS_CUDA),1)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
else
	$(CXX) $(CXXFLAGS) -o $@ $(ALL_CORRECTNESS_TEST_OBJECTS) $(LDFLAGS)
endif
	@echo "✓ Build complete: ./$(ALL_CORRECTNESS_TEST_TARGET)"

# Link adaptive demo
$(DEMO_ADAPTIVE_TARGET): $(DEMO_ADAPTIVE_OBJECTS) $(CUDA_OBJECTS) $(CUDA_CPP_OBJECTS)
	@echo "Linking $(DEMO_ADAPTIVE_TARGET)..."
ifeq ($(HAS_CUDA),1)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
else
	@echo "⚠  Skipping $(DEMO_ADAPTIVE_TARGET) - requires CUDA"
endif
	@echo "✓ Build complete: ./$(DEMO_ADAPTIVE_TARGET)"

# Link streams demo
$(DEMO_STREAMS_TARGET): $(DEMO_STREAMS_OBJECTS) $(CUDA_OBJECTS) $(CUDA_CPP_OBJECTS)
	@echo "Linking $(DEMO_STREAMS_TARGET)..."
ifeq ($(HAS_CUDA),1)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
else
	@echo "⚠  Skipping $(DEMO_STREAMS_TARGET) - requires CUDA"
endif
	@echo "✓ Build complete: ./$(DEMO_STREAMS_TARGET)"

# Link sweet spot benchmark
$(SWEET_SPOT_TARGET): $(SWEET_SPOT_OBJECTS) $(CUDA_OBJECTS) $(CUDA_CPP_OBJECTS)
	@echo "Linking $(SWEET_SPOT_TARGET)..."
ifeq ($(HAS_CUDA),1)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
else
	@echo "⚠  Skipping $(SWEET_SPOT_TARGET) - requires CUDA"
endif
	@echo "✓ Build complete: ./$(SWEET_SPOT_TARGET)"

# Compile CPU source files
$(SRC_DIR)/merkle_tree_multithread.o: $(SRC_DIR)/merkle_tree_multithread.cpp $(DEPS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/main.o: $(SRC_DIR)/main.cpp $(DEPS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/main_gpu_benchmark.o: $(SRC_DIR)/main_gpu_benchmark.cpp $(DEPS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

test_correctness.o: test_correctness.cpp $(DEPS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

test_all_correctness.o: test_all_correctness.cpp $(DEPS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

demo_adaptive.o: demo_adaptive.cpp $(DEPS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

demo_streams.o: demo_streams.cpp $(DEPS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/sweet_spot_benchmark.o: $(SRC_DIR)/sweet_spot_benchmark.cpp $(DEPS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files
ifeq ($(HAS_CUDA),1)
$(SRC_DIR)/merkle_tree_gpu.o: $(SRC_DIR)/merkle_tree_gpu.cu $(SRC_DIR)/sha256_cuda.cuh $(DEPS)
	@echo "Compiling CUDA $<..."
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(SRC_DIR)/merkle_tree_gpu_streams.o: $(SRC_DIR)/merkle_tree_gpu_streams.cu $(SRC_DIR)/merkle_tree_gpu_streams.hpp $(SRC_DIR)/sha256_cuda.cuh $(DEPS)
	@echo "Compiling CUDA $<..."
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(SRC_DIR)/merkle_tree_gpu_adaptive.o: $(SRC_DIR)/merkle_tree_gpu_adaptive.cpp $(SRC_DIR)/merkle_tree_gpu_adaptive.hpp $(DEPS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/adaptive_chunk_optimizer.o: $(SRC_DIR)/adaptive_chunk_optimizer.cpp $(SRC_DIR)/adaptive_chunk_optimizer.hpp $(DEPS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -f $(SRC_DIR)/*.o $(CPU_TARGET) $(GPU_BENCH_TARGET) $(CORRECTNESS_TEST_TARGET) $(ALL_CORRECTNESS_TEST_TARGET) \
		$(DEMO_ADAPTIVE_TARGET) $(DEMO_STREAMS_TARGET) \
		test_correctness.o test_all_correctness.o demo_adaptive.o demo_streams.o *.csv
	@echo "✓ Clean complete!"

# Run the CPU comparison
run: $(CPU_TARGET)
	@echo "Running $(CPU_TARGET)..."
	@./$(CPU_TARGET)

# Run the GPU benchmark
benchmark: $(GPU_BENCH_TARGET)
	@echo "Running GPU benchmark..."
	@./$(GPU_BENCH_TARGET)

# Run benchmark and visualize results
visualize: benchmark
	@echo "Generating visualizations..."
	@chmod +x visualize_results.py
	@python3 visualize_results.py || echo "Note: Install matplotlib, pandas, seaborn to visualize results"

# Show build configuration
config:
	@echo "=== Build Configuration ==="
	@echo "System:         $(UNAME_S)"
	@echo "CXX:            $(CXX)"
	@echo "CUDA Support:   $(HAS_CUDA)"
ifeq ($(HAS_CUDA),1)
	@echo "NVCC:           $(CUDA_AVAILABLE)"
	@echo "CUDA Path:      $(CUDA_PATH)"
	@echo "CUDA Lib Path:  $(CUDA_LIB_PATH)"
endif
	@echo "Source Dir:     $(SRC_DIR)"
	@echo "OpenSSL CFLAGS: $(OPENSSL_CFLAGS)"
	@echo "OpenSSL LIBS:   $(OPENSSL_LIBS)"
	@echo "CPU Target:     $(CPU_TARGET)"
	@echo "GPU Bench:      $(GPU_BENCH_TARGET)"
	@echo "=========================="

# Test if pkg-config can find OpenSSL
test-pkg-config:
	@echo "Testing pkg-config for OpenSSL..."
	@pkg-config --exists openssl && \
		(echo "✓ pkg-config found OpenSSL!"; \
		 echo "  Version: $$(pkg-config --modversion openssl)"; \
		 echo "  CFLAGS:  $(OPENSSL_CFLAGS)"; \
		 echo "  LIBS:    $(OPENSSL_LIBS)") || \
		(echo "✗ pkg-config cannot find OpenSSL!"; exit 1)

help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  Merkle Tree GPU Project - Build System                     ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Build Targets:"
	@echo "  make               - Build CPU comparison (default)"
	@echo "  make full          - Build all executables (CPU + GPU benchmark)"
	@echo "  make tests         - Build all correctness tests"
	@echo "  make demos         - Build all demonstration programs"
	@echo "  make clean         - Remove build artifacts"
	@echo ""
	@echo "Test Targets:"
	@echo "  make test          - Build original correctness test"
	@echo "  make test-all      - Build comprehensive correctness test (all implementations)"
	@echo ""
	@echo "Demo Targets:"
	@echo "  make demo-adaptive - Build adaptive chunk size demonstration"
	@echo "  make demo-streams  - Build CUDA streams demonstration"
	@echo ""
	@echo "Run Targets:"
	@echo "  make run           - Build and run CPU comparison"
	@echo "  make benchmark     - Build and run GPU benchmarks"
	@echo "  make visualize     - Run benchmarks and generate plots"
	@echo ""
	@echo "Utility Targets:"
	@echo "  make config        - Show build configuration"
	@echo "  make test-pkg-config - Test if pkg-config finds OpenSSL"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Executables:"
	@echo "  ./$(CPU_TARGET)                 - CPU single vs multithread comparison"
	@echo "  ./$(GPU_BENCH_TARGET)           - GPU benchmark (includes adaptive + streams)"
	@echo "  ./$(CORRECTNESS_TEST_TARGET)    - Original correctness test (2 implementations)"
	@echo "  ./$(ALL_CORRECTNESS_TEST_TARGET) - Comprehensive test (all 5 implementations)"
ifeq ($(HAS_CUDA),1)
	@echo "  ./$(DEMO_ADAPTIVE_TARGET)       - Adaptive chunk size demo"
	@echo "  ./$(DEMO_STREAMS_TARGET)        - CUDA streams demo"
endif
	@echo ""
	@echo "CUDA Status:"
ifeq ($(HAS_CUDA),1)
	@echo "  ✓ CUDA is available and will be used"
	@echo "  All GPU features enabled (Adaptive, Streams, etc.)"
else
	@echo "  ⚠ CUDA not found - GPU will be simulated in benchmarks"
	@echo "  Install CUDA toolkit to enable real GPU acceleration"
	@echo "  Demos and GPU tests require CUDA and will be skipped"
endif
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make test-all && ./test_all_correctness  - Verify correctness"
	@echo "  2. make benchmark                            - Run performance tests"
	@echo "  3. make visualize                            - Generate plots"

.PHONY: all full clean run benchmark visualize config test-pkg-config help \
        tests test test-all demos demo-adaptive demo-streams
# Makefile for Merkle Tree Project using pkg-config

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -pthread -Wall -Wextra
LDFLAGS = 

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
SOURCES = $(SRC_DIR)/merkle_tree_multithread.cpp $(SRC_DIR)/main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
DEPS = $(SRC_DIR)/header_merkle_tree.hpp
TARGET = merkle_comparison

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Build complete: ./$(TARGET)"

# Compile source files
$(SRC_DIR)/merkle_tree_multithread.o: $(SRC_DIR)/merkle_tree_multithread.cpp $(DEPS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/main.o: $(SRC_DIR)/main.cpp $(DEPS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -f $(SRC_DIR)/*.o $(TARGET)
	@echo "✓ Clean complete!"

# Run the program
run: $(TARGET)
	@echo "Running $(TARGET)..."
	@./$(TARGET)

# Show build configuration
config:
	@echo "=== Build Configuration ==="
	@echo "System:         $(UNAME_S)"
	@echo "CXX:            $(CXX)"
	@echo "Source Dir:     $(SRC_DIR)"
	@echo "OpenSSL CFLAGS: $(OPENSSL_CFLAGS)"
	@echo "OpenSSL LIBS:   $(OPENSSL_LIBS)"
	@echo "Header:         $(DEPS)"
	@echo "Sources:        $(SOURCES)"
	@echo "Objects:        $(OBJECTS)"
	@echo "Target:         $(TARGET)"
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
	@echo "Available targets:"
	@echo "  make               - Build the project"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make run           - Build and run the program"
	@echo "  make config        - Show build configuration"
	@echo "  make test-pkg-config - Test if pkg-config finds OpenSSL"
	@echo "  make help          - Show this help message"

.PHONY: all clean run config test-pkg-config help
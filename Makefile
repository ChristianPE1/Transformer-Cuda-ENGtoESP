NVCC = nvcc
CXXFLAGS = -std=c++14 -O2
CUDAFLAGS = -arch=sm_70 -rdc=true
INCLUDES = -Iinclude -Isrc

# Source files
SRCDIR = src
BUILDDIR = build

# Find all .cu files
DATA_SOURCES = $(wildcard $(SRCDIR)/data/*.cu)
UTILS_SOURCES = $(wildcard $(SRCDIR)/utils/*.cu)
LAYERS_SOURCES = $(wildcard $(SRCDIR)/layers/*.cu)
TRAINING_SOURCES = $(wildcard $(SRCDIR)/training/*.cu)
TRANSFORMER_SOURCES = $(wildcard $(SRCDIR)/transformer/*.cu)

ALL_SOURCES = $(DATA_SOURCES) $(UTILS_SOURCES) $(LAYERS_SOURCES) $(TRAINING_SOURCES) $(TRANSFORMER_SOURCES)

# Main targets
all: test_simple main_dataset

# Simple CUDA test
test_simple: test_simple.cu
    $(NVCC) $(CUDAFLAGS) $(INCLUDES) $< -o $@

# Main program with dataset
main_dataset: src/main.cu $(ALL_SOURCES)
    $(NVCC) $(CUDAFLAGS) $(INCLUDES) $^ -o $@

# Clean
clean:
    rm -f test_simple main_dataset

.PHONY: all clean
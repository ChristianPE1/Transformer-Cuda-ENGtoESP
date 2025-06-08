NVCC = nvcc
CXXFLAGS = -std=c++14 -O2
CUDAFLAGS = -arch=sm_70 -rdc=true
INCLUDES = -Iinclude -Isrc

# Directorios
SRCDIR = src
BUILDDIR = build
SOURCES = $(shell find $(SRCDIR) -name "*.cu")
OBJECTS = $(SOURCES:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)

# Targets
all: test_simple transformer

test_simple: test_simple.cu
    $(NVCC) $(CUDAFLAGS) $(INCLUDES) $< -o $@

transformer: $(OBJECTS)
    $(NVCC) $(CUDAFLAGS) $(OBJECTS) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
    @mkdir -p $(dir $@)
    $(NVCC) $(CUDAFLAGS) $(INCLUDES) -c $< -o $@

clean:
    rm -rf $(BUILDDIR) test_simple transformer

.PHONY: all clean
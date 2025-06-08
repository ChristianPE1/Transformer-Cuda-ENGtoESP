project(CudaTransformer VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

# Include directories
include_directories(include src)

# Source files
file(GLOB_RECURSE CUDA_SOURCES
    "src/*.cu"
)

# Add executable
add_executable(cuda_transformer ${CUDA_SOURCES})

# Link libraries if needed (e.g., for cuBLAS, cuDNN)
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(cuda_transformer PRIVATE ${CUDAToolkit_LIBRARIES})

# Set the output directory for the executable
set_target_properties(cuda_transformer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
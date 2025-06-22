cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0048 NEW)
project(CudaTransformer VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

# Incluir CMake modules
include_directories(include src)

# Incluir archivos CUDA 
file(GLOB_RECURSE CUDA_SOURCES
    "src/*.cu"
)

# Add executable
add_executable(cuda_transformer ${CUDA_SOURCES})

# Set the output directory for the executable
set_target_properties(cuda_transformer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Comando custom main_transformer
add_custom_command(
    OUTPUT main_transformer
    COMMAND ${CMAKE_CUDA_COMPILER} ${CUDAFLAGS} -std=c++17 ${INCLUDES} $<TARGET_OBJECTS:cuda_transformer> -o $@
    DEPENDS cuda_transformer
)

# Custom objetivo 
add_custom_target(clean
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/bin/test_simple ${CMAKE_BINARY_DIR}/bin/main_transformer
)